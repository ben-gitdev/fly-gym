import math
from typing import Tuple, Optional, List, Dict
import numpy as np
import torch
import pandas as pd

# ----------------------------
# Small utils
# ----------------------------
def clamp(value, lower, upper):
    return lower if value < lower else upper if value > upper else value

def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def rot2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)

def box_to_disc_radius(box_half_extents_xy: Tuple[float, float]) -> float:
    hx, hy = box_half_extents_xy
    return float(math.sqrt(hx * hx + hy * hy))

# def _safe_geom_name(model, geom_id: int) -> str:
#     try:
#         g = model.geom(geom_id)
#         return (g.name or "").lower()
#     except Exception:
#         return ""



def progress_stalled(xy_prev: Optional[np.ndarray], xy_cur: np.ndarray, min_progress: float = 0.002) -> bool:
    if xy_prev is None:
        return False
    return float(np.linalg.norm(xy_cur - xy_prev)) < float(min_progress)
    
# -----------------------------
# Utils: device & dtype
# -----------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# -----------------------------
# Data Loading
# -----------------------------

def load_edge_list(path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, int]]:
    """Return (rows, cols, vals, sorted_node_ids) from CSV/Parquet.
    rows = postsynaptic IDs, cols = presynaptic IDs, vals = weights.
    """

    pre_col = "pre"; post_col = "post"; weight_col = "weight"

    suffix = path.lower()
    if suffix.endswith(".csv"): filetype = "csv"
    elif suffix.endswith(".parquet") or suffix.endswith(".pq"): filetype = "parquet"
    else: raise ValueError(f"Unsupported file extension for path: {path}")

    df = pd.read_csv(path) if filetype == "csv" else pd.read_parquet(path)

    if not {"pre","post","weight"}.issubset(df.columns):
        aliases = [
            {"pre":"Presynaptic_ID", "post":"Postsynaptic_ID", "weight":"Excitatory x Connectivity"},
            {"pre":"Pre", "post":"Post", "weight":"Weight"},
            {"pre":"pre_root_id", "post":"post_root_id", "weight":"syn_count"},
        ]
        for m in aliases:
            if all(k in df.columns for k in m.values()):
                pre_col, post_col, weight_col = m["pre"], m["post"], m["weight"]
                break

    if pre_col not in df.columns or post_col not in df.columns or weight_col not in df.columns:
        raise KeyError(f"Columns not found in {path}. Have {list(df.columns)[:10]}... expected pre/post/weight or known aliases.")

    pre = df[pre_col].astype(int).to_numpy()
    post = df[post_col].astype(int).to_numpy()
    w = df[weight_col].astype(float).to_numpy()
    node_ids = sorted(set(pre.tolist()) | set(post.tolist()))
    id2idx = {nid:i for i, nid in enumerate(node_ids)} # map original (root_ID in connectome) IDs to 0..N-1 (idx used in tensors)

    rows = torch.tensor([id2idx[p] for p in post], dtype=torch.int64)
    cols = torch.tensor([id2idx[p] for p in pre], dtype=torch.int64)
    vals = torch.tensor(w, dtype=torch.float32)

    print(f"[edge_list] Loaded {len(vals)} edges with {len(node_ids)} unique nodes.")

    return rows, cols, vals, id2idx

def load_photoreceptors(path: str, id2idx: dict, type: str = 'L1') :
    df = pd.read_csv(path)
    df = df[df['type']==type]  # only use R1-R6 photoreceptors
    ids = df["root_id"].astype(int).tolist()
    pos_x = df["x"].astype(float).tolist()
    pos_y = df["y"].astype(float).tolist()

    idx: List[int] = []
    positions: List[Tuple[float, float]] = []
    skipped = 0
    for nid, x, y in zip(ids, pos_x, pos_y):
        if nid not in id2idx:
            skipped += 1
            continue
        idx.append(id2idx[nid])
        positions.append((x, y))

    if skipped:
        print(f"[photoreceptors] skipped {skipped} IDs not present in connectome (kept {len(idx)}).")
    if not idx:
        raise ValueError(
            f"No photoreceptors from {path} match the provided connectome IDs."
        )

    return idx, positions

def load_sensory_neurons(path: str, id2idx: dict) :
    df = pd.read_csv(path)
    ids = df["root_id"].astype(int).tolist()
    idx: List[int] = []
    skipped = 0
    for nid in ids:
        if nid not in id2idx:
            skipped += 1
            continue
        idx.append(id2idx[nid])

    if skipped:
        print(f"[Sensory neurons] skipped {skipped} IDs not present in connectome (kept {len(idx)}).")
    if not idx:
        raise ValueError(
            f"No sensory neurons from {path} match the provided connectome IDs."
        )

    return idx


def load_descending_neurons(path: str, id2idx: dict) -> List[int]:
    df = pd.read_csv(path)
    ids = df["root_id"].astype(int).tolist()
    idx = [id2idx[nid] for nid in ids if nid in id2idx]
    skipped = len(ids) - len(idx)
    if skipped:
        print(f"[descending_neurons] skipped {skipped} IDs not present in connectome (kept {len(idx)}).")
    
    return idx


# -----------------------------
# Sparse builders
# -----------------------------

def build_sparse_matrix(rows: torch.Tensor, cols: torch.Tensor, vals: torch.Tensor,
                        shape: Tuple[int,int], dtype: torch.dtype) -> torch.Tensor:
    return torch.sparse_coo_tensor(torch.stack([rows, cols], dim=0),
                                   vals.to(dtype), size=shape).coalesce()


# -----------------------------
# Spectral radius control
# -----------------------------

def spectral_radius_power_iter(W: torch.Tensor, iters: int = 50) -> float:
    assert W.layout == torch.sparse_coo
    N = W.shape[0]; v = torch.randn(N, device=W.device); v = v/(v.norm()+1e-12)
    for _ in range(iters):
        v = torch.sparse.mm(W, v.unsqueeze(1)).squeeze(1)
        v = v/(v.norm()+1e-12)
    wv = torch.sparse.mm(W, v.unsqueeze(1)).squeeze(1)
    lam = torch.dot(v, wv)/(v.dot(v)+1e-12)
    return float(lam.abs().item())

def rescale_spectral_radius_(W: torch.Tensor, target: float = 0.9):
    with torch.no_grad():
        rho = spectral_radius_power_iter(W)
        if not (math.isfinite(rho) and rho>0):
            print(f"[spectral] skip scaling: invalid rho={rho}"); return
        scale = target/rho; W._values().mul_(scale)
        print(f"[spectral] scaled W by {scale:.4f} to target {target} (est {rho:.4f})")


def load_cell_types(path: str, id2idx: dict) -> Tuple[torch.Tensor, int]:
    """
    Loads cell types from CSV and maps every neuron in the RNN to a type index.
    
    Returns:
        neuron_type_ids: Tensor (N,) of integer type IDs.
        num_types: Total number of unique types found (including 'Unknown').
    """
    import pandas as pd
    
    # 1. Load CSV
    # Expected columns: "root_id", "primary_type" (or similar)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"[cell_types] Warning: {path} not found. Defaulting to single type.")
        return torch.zeros(len(id2idx), dtype=torch.long), 1

    # Handle column naming variations
    if "root_id" not in df.columns:
        raise KeyError(f"Column 'root_id' missing in {path}")
    
    type_col = "primary_type" if "primary_type" in df.columns else "type"
    if type_col not in df.columns:
         # Fallback if no type column, treat all as one type
         print(f"[cell_types] Column '{type_col}' missing. Defaulting to single type.")
         return torch.zeros(len(id2idx), dtype=torch.long), 1

    # 2. Build Mappings
    # Filter to only neurons that exist in our connectome (id2idx)
    df = df[df["root_id"].isin(id2idx.keys())].copy()
    
    # Get unique string types and map to 1..K (0 is reserved for 'Unknown')
    unique_strings = sorted(df[type_col].astype(str).unique().tolist())
    str2int = {s: i + 1 for i, s in enumerate(unique_strings)} # 1-based index
    
    # 3. Create the N-sized array
    N = len(id2idx)
    neuron_type_ids = torch.zeros(N, dtype=torch.int64) # Default 0 (Unknown)
    
    found_count = 0
    for _, row in df.iterrows():
        rid = int(row["root_id"])
        stype = str(row[type_col])
        if rid in id2idx:
            idx = id2idx[rid]
            neuron_type_ids[idx] = str2int[stype]
            found_count += 1
            
    num_types = len(unique_strings) + 1 # +1 for Unknown (0)
    print(f"[cell_types] Loaded {len(unique_strings)} unique types.")
    print(f"[cell_types] Assigned types to {found_count}/{N} neurons. (Rest are 'Unknown')")
    
    return torch.tensor(neuron_type_ids, dtype=torch.long), num_types


# -----------------------------
# Model Building Utilities
# -----------------------------

def build_connectome_cell(
    edge_path: str,
    device: torch.device,
    dtype: torch.dtype,
    photoreceptor_left_csv: str,
    photoreceptor_right_csv: str,
    tactile_left_csv: str,
    tactile_right_csv: str,
    descending_neurons_csv: str,
    cell_types_csv: str,
    wind_sensing_csv: str = None,  # NEW
    target_rho: float = 0.9,
    leak_alpha: float = 0.2,
    activation: str = "tanh",
    train_rnn_weights: bool = False,
    train_readout_head: bool = True,
    batch_chunk: int = 8,
    row_tile_size: int = 69320,
):
    """
    Build a connectome RNN cell with all necessary neuron mappings.
    
    Returns:
        cell: LeakyConnectomeRNNCell instance
        pr_positions: List of photoreceptor positions
        input_splits: Dict mapping sensor names to (start, end) indices
    """
    from models.connectome_rnn_model import LeakyConnectomeRNNCell
    print("Edge path: ", edge_path)
    rows, cols, vals, id2idx = load_edge_list(edge_path)
    N = len(id2idx)
    W = build_sparse_matrix(rows, cols, vals, (N, N), dtype=torch.float32).to(device)
    print("Built sparse W matrix")
    rescale_spectral_radius_(W, target=target_rho)

    pr_L1_left_idx, pr_L1_left_pos = load_photoreceptors(photoreceptor_left_csv, id2idx, type='L1')
    pr_L2_left_idx, pr_L2_left_pos = load_photoreceptors(photoreceptor_left_csv, id2idx, type='L2')
    pr_L3_left_idx, pr_L3_left_pos = load_photoreceptors(photoreceptor_left_csv, id2idx, type='L3')
    
    pr_L1_right_idx, pr_L1_right_pos = load_photoreceptors(photoreceptor_right_csv, id2idx, type='L1')
    pr_L2_right_idx, pr_L2_right_pos = load_photoreceptors(photoreceptor_right_csv, id2idx, type='L2')
    pr_L3_right_idx, pr_L3_right_pos = load_photoreceptors(photoreceptor_right_csv, id2idx, type='L3')
    
    tactile_left_idx = load_sensory_neurons(tactile_left_csv, id2idx)
    tactile_right_idx = load_sensory_neurons(tactile_right_csv, id2idx)
    
    # NEW: Load wind sensing neurons if path provided
    wind_idx = []
    if wind_sensing_csv:
        wind_idx = load_sensory_neurons(wind_sensing_csv, id2idx)
        print(f"[io] Loaded {len(wind_idx)} wind sensing neurons.")

    input_nodes = pr_L1_left_idx + pr_L2_left_idx + pr_L3_left_idx + pr_L1_right_idx + pr_L2_right_idx + pr_L3_right_idx + tactile_left_idx + tactile_right_idx + wind_idx
    pr_positions = pr_L1_left_pos + pr_L2_left_pos + pr_L3_left_pos + pr_L1_right_pos + pr_L2_right_pos + pr_L3_right_pos

    # Load Cell Types
    print(f"[io] Loading cell types from {cell_types_csv}...")
    neuron_type_ids, num_types = load_cell_types(cell_types_csv, id2idx)
    neuron_type_ids = neuron_type_ids.to(device)

    input_splits = {}
    start = 0
    for name, length in [
        ("pr_L1_left", len(pr_L1_left_idx)),
        ("pr_L2_left", len(pr_L2_left_idx)),
        ("pr_L3_left", len(pr_L3_left_idx)),
        ("pr_L1_right", len(pr_L1_right_idx)),
        ("pr_L2_right", len(pr_L2_right_idx)),
        ("pr_L3_right", len(pr_L3_right_idx)),
        ("tactile_left", len(tactile_left_idx)),
        ("tactile_right", len(tactile_right_idx)),
        ("wind", len(wind_idx)),  # NEW
    ]:
        end = start + length
        input_splits[name] = (start, end)
        start = end

    dn_nodes = load_descending_neurons(descending_neurons_csv, id2idx)
    if not dn_nodes:
        dn_nodes = list(range(min(256, N)))
        print(f"[io] No descending neuron IDs matched; using first {len(dn_nodes)} nodes as outputs.")

    print(f"N={N} | Nin={len(input_nodes)} | outputs={len(dn_nodes)}")
    cell = LeakyConnectomeRNNCell(
        W, input_nodes, dn_nodes,
        neuron_type_ids=neuron_type_ids,
        num_cell_types=num_types,
        leak_alpha=leak_alpha, 
        activation=activation,
        train_rnn_weights=train_rnn_weights, 
        train_readout_head=train_readout_head,
        dtype=dtype, 
        batch_chunk=batch_chunk, 
        row_tile_size=row_tile_size,
    ).to(device)
    return cell, pr_positions, input_splits, id2idx


def obs_to_torch(obs: Dict, device: torch.device, dtype: torch.dtype, vision: List[bool] = [True, True]) -> Dict:
    """Convert environment observation dict to torch tensors."""
    cam_left = torch.from_numpy(obs["cam_left"]).permute(2, 0, 1).unsqueeze(0).to(device=device)
    cam_right = torch.from_numpy(obs["cam_right"]).permute(2, 0, 1).unsqueeze(0).to(device=device)
    if not vision[0]:
        cam_left = torch.zeros_like(cam_left)
    if not vision[1]:
        cam_right = torch.zeros_like(cam_right)
    sensors_np = obs["sensors"]
    vec_to_goal = torch.as_tensor(
        sensors_np["vec_to_goal"], device=device, dtype=dtype
    ).unsqueeze(0)
    vec_left_to_goal = torch.as_tensor(
        sensors_np.get("vec_left_to_goal", sensors_np["vec_to_goal"]), device=device, dtype=dtype
    ).unsqueeze(0)
    vec_right_to_goal = torch.as_tensor(
        sensors_np.get("vec_right_to_goal", sensors_np["vec_to_goal"]), device=device, dtype=dtype
    ).unsqueeze(0)
    collision_angle = torch.as_tensor(
        sensors_np.get("collision_angle", 0.0) or 0.0, device=device, dtype=dtype
    ).view(1)
    collision = torch.as_tensor(
        sensors_np.get("collision", 0.0) or 0.0, device=device, dtype=dtype
    ).view(1)
    # Wind direction for Johnston's organ neurons
    wind_direction = torch.as_tensor(
        sensors_np.get("wind_direction", np.zeros(2, dtype=np.float32)), device=device, dtype=dtype
    ).view(1, 2)
    sensors = {
        "vec_to_goal": vec_to_goal,
        "vec_left_to_goal": vec_left_to_goal,
        "vec_right_to_goal": vec_right_to_goal,
        "collision": collision,
        "collision_angle": collision_angle,
        "wind_direction": wind_direction,
    }
    return {
        "cam_left": cam_left.to(dtype=torch.float32),
        "cam_right": cam_right.to(dtype=torch.float32),
        "sensors": sensors,
    }