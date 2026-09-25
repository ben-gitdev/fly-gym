"""
Roll out a trained connectome RNN policy from a saved checkpoint.

This mirrors the architecture and preprocessing used during DAgger training
in `train_connectome_rnn_dagger.py`, so checkpoints from that script can be
evaluated directly.
"""

from __future__ import annotations

import argparse
import os
import time
import csv
import numpy as np
from typing import Optional, Tuple, List
import pandas as pd
try:
    import cv2
except ImportError:
    cv2 = None
import torch

from agents.connectome_rnn_agent import ConnectomeAgent
from agents.teacher_analytic_agent import PlannerAnalyticTeacher
from core.utils import get_device
from environment.mujoco_two_cam_env_random_obstacles import MuJoCoTwoCamEnv
from train_connectome_rnn_dagger import (
    ARENA_HALF_EXTENT,
    DTYPE,
    EDGE_PATH,
    ENV_HEIGHT,
    ENV_WIDTH,
    INPUT_SCALE_INIT,
    MAX_EPISODE_STEPS,
    N_OBSTACLES,
    TARGET_RHO,
    LEAK_ALPHA,
    ACTIVATION,
    BATCH_CHUNK,
    ROW_TILE_SIZE,
)
from core.utils import build_connectome_cell, obs_to_torch

# Reward-shaping constants below match the values a since-removed train_connectome_rnn_rl.py
# (an abandoned PPO/critic training path) used to import them from. Only
# time_penalty/prog_scale/ctrl_penalty are actually used by _make_env() below; goal_bonus/
# contact_penalty are passed as 0 there regardless.
CTRL_PENALTY = 0.001
TIME_PENALTY = 0.01
PROG_SCALE = 10.0


# CHECKPOINT = "checkpoints/connectome_rnn_dagger_princeton_random_full_vision.pt"
# CHECKPOINT = "checkpoints/connectome_rnn_dagger_princeton_3.pt"
RECORD_CSV = None#"connectomes/drosophila adult connectome/moonwalker_neurons.csv"       # e.g., "neurons_to_record.csv"
OVERWRITE_CSV = None#"connectomes/drosophila adult connectome/moonwalker_neurons.csv"    # e.g., "neurons_to_overwrite.csv"
END_ON_COLLISION = False

def maybe_show_cameras(obs, vision=(True, True)):
    """Show the two eye cameras, blacking out any eye disabled by `vision` ([left, right])
    so the window matches what the agent actually receives."""
    if cv2 is None: return
    try:
        left_gray = cv2.cvtColor(obs["cam_left"], cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(obs["cam_right"], cv2.COLOR_RGB2GRAY)
        if not vision[0]:
            left_gray = np.zeros_like(left_gray)
        if not vision[1]:
            right_gray = np.zeros_like(right_gray)
        frame = np.hstack([left_gray, right_gray])
        cv2.imshow("Agent View (Left | Right)", frame)
        cv2.waitKey(1)
    except Exception:
        pass

class PhotoreceptorView:
    """Show the per-column intensities the connectome agent actually samples from each eye.

    ConnectomeAgent.obs_to_x grayscales each eye image, applies a 5x5 average pool, and
    bilinearly samples it at every photoreceptor's grid position (L1/L2/L3 of a column share
    one position).  This draws each sampled value as a dot at its grid position, so the window
    shows the agent's visual input *before* the virtual-retina temporal filtering.
    """

    EYE_SEGMENTS = {
        "left": ("pr_L1_left", "pr_L2_left", "pr_L3_left"),
        "right": ("pr_L1_right", "pr_L2_right", "pr_L3_right"),
    }

    def __init__(self, agent: ConnectomeAgent, eye_size: int = 256, background: int = 40):
        self.eye_size = eye_size
        self.background = background
        self.eyes = {}
        for eye, segments in self.EYE_SEGMENTS.items():
            grids = [getattr(agent, f"grid_{seg[3:]}") for seg in segments]
            # Grid is (1, N, 1, 2) in image coords (x right, y down), align_corners=True.
            px = [((g.view(-1, 2).cpu().numpy() + 1.0) / 2.0 * (eye_size - 1)).round().astype(int)
                  if g is not None else np.zeros((0, 2), dtype=int) for g in grids]
            self.eyes[eye] = (segments, np.concatenate(px, axis=0))
        # Dot radius from the nearest-neighbour spacing of the lattice.
        pts = self.eyes["left"][1] if len(self.eyes["left"][1]) else self.eyes["right"][1]
        uniq = np.unique(pts, axis=0).astype(float)
        if len(uniq) > 1:
            d = np.sqrt(((uniq[:, None, :] - uniq[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            self.radius = max(1, int(np.median(d.min(axis=1)) / 2))
        else:
            self.radius = 2
        self.agent = agent

        # Dot positions are fixed, so precompute a label map per eye (pixel -> index into that
        # eye's sampled values, or -1 for background).  show() is then one gather per eye
        # instead of thousands of cv2.circle calls per frame.
        r = self.radius
        dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
        in_disk = dx ** 2 + dy ** 2 <= r ** 2
        dx, dy = dx[in_disk], dy[in_disk]
        self.labels = {}
        for eye, (_, pts) in self.eyes.items():
            labels = np.full((eye_size, eye_size), -1, dtype=np.int64)
            for i, (cx, cy) in enumerate(pts):
                xs, ys = cx + dx, cy + dy
                ok = (xs >= 0) & (xs < eye_size) & (ys >= 0) & (ys < eye_size)
                labels[ys[ok], xs[ok]] = i
            self.labels[eye] = labels

    _cache: dict = {}

    @classmethod
    def for_agent(cls, agent: ConnectomeAgent) -> "PhotoreceptorView":
        """Build once per agent and reuse across episodes (the label maps take ~0.15 s)."""
        if id(agent) not in cls._cache:
            cls._cache[id(agent)] = cls(agent)
        return cls._cache[id(agent)]

    def render(self, x: torch.Tensor) -> np.ndarray:
        """Return both eyes as one (eye_size, 2 * eye_size + 4) uint8 image (left | right)."""
        x_np = x[0].detach().float().cpu().numpy()
        panels = []
        for eye in ("left", "right"):
            segments, _ = self.eyes[eye]
            vals = np.concatenate([
                x_np[slice(*self.agent.input_splits[seg])] if seg in self.agent.input_splits
                else np.zeros(0) for seg in segments
            ])
            # Append the background as the last entry so label -1 maps to it.
            lut = np.append(np.clip(vals, 0.0, 1.0) * 255.0, self.background).astype(np.uint8)
            panels.append(lut[self.labels[eye]])
        sep = np.full((self.eye_size, 4), 255, dtype=np.uint8)
        return np.hstack([panels[0], sep, panels[1]])

    def show(self, x: torch.Tensor) -> None:
        if cv2 is None: return
        try:
            cv2.imshow("Agent Input: photoreceptors (Left | Right)", self.render(x))
            cv2.waitKey(1)
        except Exception:
            pass


# Filenames every connectome directory uses for the neuron-set CSVs.  They are identical
# across connectomes (FLYNN, SmallWorldNet, ...), which is what lets them be resolved from
# the edge list's own directory instead of being configured separately -- a --connectome
# pointing at one directory can't then silently pick up another directory's neuron IDs.
_CONNECTOME_CSV_NAMES = {
    "photoreceptor_left_csv": "visual_column_L1_L2_L3_rear_view_left.csv",
    "photoreceptor_right_csv": "visual_column_L1_L2_L3_rear_view_right.csv",
    "tactile_left_csv": "head_bristles_left.csv",
    "tactile_right_csv": "head_bristles_right.csv",
    "descending_neurons_csv": "descending_neurons.csv",
    "cell_types_csv": "consolidated_cell_types.csv",
    "wind_sensing_csv": "JO-C_and_JO-E.csv",
}


def resolve_connectome_path(edge_path: Optional[str]) -> str:
    """Return the connectome edge list to build the model from.

    When *edge_path* is None this falls back to shared_config's EDGE_PATH -- i.e. whichever
    connectome the *training* config currently points at, which is not necessarily the one
    this checkpoint was trained on.  A .pt records no provenance, so a mismatch would only
    be caught if the shapes happened to differ; hence the warning.
    """
    if edge_path is not None:
        return edge_path
    print(
        "[warn] No --connectome given; falling back to the shared_config default:\n"
        f"[warn]   {EDGE_PATH}\n"
        "[warn] That is the connectome the training config points at today, which may\n"
        "[warn] NOT be the one this checkpoint was trained on. Checkpoints record no\n"
        "[warn] provenance, so pass --connectome explicitly if you are not certain."
    )
    return EDGE_PATH


def connectome_csv_paths(edge_path: str) -> dict:
    """Resolve the neuron-set CSVs that accompany *edge_path*, from its own directory."""
    base = os.path.dirname(edge_path)
    return {key: os.path.join(base, name) for key, name in _CONNECTOME_CSV_NAMES.items()}


def _make_env(render_mode: Optional[str], n_obstacles: int, texture_mode: str = "checker") -> MuJoCoTwoCamEnv:
    return MuJoCoTwoCamEnv(
        width=ENV_WIDTH,
        height=ENV_HEIGHT,
        max_episode_steps=MAX_EPISODE_STEPS,
        n_obstacles=n_obstacles,
        arena_half_extent=ARENA_HALF_EXTENT,
        render_mode=render_mode,
        end_on_collision = END_ON_COLLISION,
        goal_bonus=0,
        contact_penalty=0,
        time_penalty=TIME_PENALTY,
        prog_scale=PROG_SCALE,
        ctrl_penalty=CTRL_PENALTY,
        texture_mode=texture_mode,
    )


def _load_agent(checkpoint_path: str, device: torch.device, dtype: torch.dtype,
                edge_path: str) -> ConnectomeAgent:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.isfile(edge_path):
        raise FileNotFoundError(f"Connectome edge list not found: {edge_path}")

    cell, pr_positions, input_splits, id2idx = build_connectome_cell(
        edge_path=edge_path,
        device=device,
        dtype=dtype,
        **connectome_csv_paths(edge_path),
        target_rho=TARGET_RHO,
        leak_alpha=LEAK_ALPHA,
        activation=ACTIVATION,
        train_rnn_weights=False,
        train_readout_head=True,
        batch_chunk=BATCH_CHUNK,
        row_tile_size=ROW_TILE_SIZE,
    )
    agent = ConnectomeAgent(
        cell,
        photoreceptor_positions=pr_positions,
        input_splits=input_splits,
        dtype=dtype,
        input_scale_init=INPUT_SCALE_INIT,
    ).to(device)
    print("Checkpoint path: ", checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent, id2idx


def load_neuron_indices(csv_path: str, id2idx: dict) -> List[int]:
    """Load neuron indices from a CSV file containing root_ids."""
    if not csv_path or not os.path.exists(csv_path):
        return []
    
    try:
        df = pd.read_csv(csv_path)
        if "root_id" not in df.columns:
            print(f"[warn] {csv_path} missing 'root_id' column.")
            return []
        
        ids = df["root_id"].astype(int).tolist()
        indices = [id2idx[nid] for nid in ids if nid in id2idx]
        print(f"[io] Loaded {len(indices)} neuron indices from {csv_path} (out of {len(ids)} requested).")
        return indices
    except Exception as e:
        print(f"[error] Failed to load neuron indices from {csv_path}: {e}")
        return []


def load_neuron_overwrite_config(
    csv_path: str, id2idx: dict, default_value: float = 0.0,
    device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32,
) -> Tuple[List[int], Optional[torch.Tensor]]:
    """Load neuron indices and per-neuron overwrite values from a CSV.

    The CSV must have a ``root_id`` column.  If it also contains a ``value``
    column, each neuron gets its own overwrite value; otherwise all neurons
    use *default_value*.

    Returns
    -------
    indices : list[int]
        Network-internal indices of the neurons to overwrite.
    values : torch.Tensor or None
        Shape ``(len(indices),)`` tensor of per-neuron overwrite values,
        or *None* when *csv_path* is empty / missing.
    """
    if not csv_path or not os.path.exists(csv_path):
        return [], None

    try:
        df = pd.read_csv(csv_path)
        if "root_id" not in df.columns:
            print(f"[warn] {csv_path} missing 'root_id' column.")
            return [], None

        has_value_col = "value" in df.columns
        ids = df["root_id"].astype(int).tolist()
        values_raw = df["value"].tolist() if has_value_col else None

        indices: List[int] = []
        values_list: List[float] = []
        for i, nid in enumerate(ids):
            if nid in id2idx:
                indices.append(id2idx[nid])
                values_list.append(values_raw[i] if has_value_col else default_value)

        values_tensor = torch.tensor(values_list, device=device, dtype=dtype)
        print(
            f"[io] Loaded {len(indices)} overwrite neurons from {csv_path} "
            f"(out of {len(ids)} requested, per-neuron values: {has_value_col})."
        )
        return indices, values_tensor
    except Exception as e:
        print(f"[error] Failed to load overwrite config from {csv_path}: {e}")
        return [], None



@torch.no_grad()
def rollout_episode(
    env: MuJoCoTwoCamEnv,
    agent: ConnectomeAgent,
    teacher: PlannerAnalyticTeacher,
    device: torch.device,
    dtype: torch.dtype,
    render: bool,
    show_path: bool,
    episode_idx: int,
    save_dir: str,
    record_indices: Optional[List[int]] = None,
    overwrite_indices: Optional[List[int]] = None,
    overwrite_values: Optional[torch.Tensor] = None,
    overwrite_interval: int = 1,
    overwrite_steps: int = 1,
    seed: Optional[int] = None,
    vision: List[bool] = [True, True],
    save_hidden_states: bool = False,
) -> Tuple[float, int, bool, bool, bool, np.ndarray]:
    if record_indices is None:
        record_indices = []
    if overwrite_indices is None:
        overwrite_indices = []
    
    obs, _ = env.reset(seed=seed)
    
    # --- Record Goal Location ---
    goal_xy = env._goal_xy.copy()
    # --------------------------

    # --- Record Obstacles ---
    # env._obstacle_xy is (N, 2)
    # We only care about the active obstacles
    active_obstacles = env._obstacle_xy[:env.n_obstacles]
    obs_file = os.path.join(save_dir, f"obstacles_{episode_idx}.txt")
    np.savetxt(obs_file, active_obstacles, fmt="%.4f", delimiter=",")
    # ------------------------

    agent.reset_vision_state()
    teacher.reset()
    pr_view = PhotoreceptorView.for_agent(agent) if render else None
    h = torch.zeros(1, agent.cell.N, device=device, dtype=dtype)
    done = False
    trunc = False
    ep_ret = 0.0
    steps = 0
    
    trajectory = []
    recorded_activity = []
    hidden_states_list = []

    while not (done or trunc):
        # Update teacher path visualization (side effect on env)
        if show_path:
            _ = teacher.act(env)

        obs_t = obs_to_torch(obs, device=device, dtype=dtype, vision=vision)
        # Compute the photoreceptor input once so the display shows exactly what the agent gets.
        x = agent.obs_to_x(obs_t)
        h, action = agent.step(h, obs_t, x=x)
        
        if render:
            maybe_show_cameras(obs, vision=vision)
            pr_view.show(x)

        # --- Overwrite Neuron Activity AFTER step to clamp values ---
        if overwrite_indices and overwrite_values is not None and (steps % overwrite_interval < overwrite_steps):
            h[:, overwrite_indices] = overwrite_values
        # ------------------------------------------------------------
        
        # Now h is the NEW state (with overwrites applied). Record it.
        if record_indices:
            with torch.no_grad():
                act = h[:, record_indices].cpu().numpy().flatten()
                recorded_activity.append(act)

        # --- Save full hidden state ---
        if save_hidden_states:
            hidden_states_list.append(h.squeeze(0).cpu().numpy())

        action_np = action.squeeze(0).cpu().numpy()

        # --- Record Robot Position and Collision ---
        robot_pos = env._base_xy()
        collision = obs["sensors"]["collision"]  # Get collision flag from current obs
        trajectory.append([robot_pos[0], robot_pos[1], int(collision)])
        # --------------------------------------------

        obs, reward, done, trunc, _ = env.step(action_np)
        ep_ret += float(reward)
        steps += 1

        if render:
            env.render()
            
    # --- Save Trajectory ---
    traj_file = os.path.join(save_dir, f"trajectory_{episode_idx}.csv")
    with open(traj_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "x", "y", "collision"])
        for i, row in enumerate(trajectory):
            writer.writerow([i, row[0], row[1], row[2]])
    
    # --- Save Activity ---
    if recorded_activity:
        act_file = os.path.join(save_dir, f"activity_{episode_idx}.csv")
        # Save as CSV: header could be indices, but just raw valid_values
        # Or no header. Let's use numpy for speed
        np.savetxt(act_file, np.array(recorded_activity), fmt="%.6f", delimiter=",")
    # -----------------------

    # --- Save Hidden States ---
    if hidden_states_list:
        hs_file = os.path.join(save_dir, f"hidden_states_{episode_idx}.npy")
        np.save(hs_file, np.stack(hidden_states_list))  # shape (T, N)
        print(f"[io] Saved hidden states to {hs_file} "
              f"(shape={len(hidden_states_list)}x{hidden_states_list[0].shape[0]})")
    # --------------------------

    # --- Determine Goal Reached ---
    final_xy = env._base_xy()
    final_dist = float(np.linalg.norm(env._goal_xy - final_xy))
    goal_reached = final_dist < env.goal_radius
    # ------------------------------

    return ep_ret, steps, bool(done), bool(trunc), bool(goal_reached), goal_xy



def run_one_configuration(checkpoint, vision, rendermode = None, edge_path = None, texture_mode = "checker"):
    device = get_device()
    dtype = DTYPE
    
    # --- Configuration ---
    # checkpoint = CHECKPOINT
    # Episode ep is rolled out with env seed=ep, so the 500 eval layouts are fixed and
    # identical across vision conditions and checkpoints.  Training uses seeds starting at
    # TRAIN_ENV_SEED_BASE (100_000), so these layouts are never seen during training.
    episodes = 500
    render_mode = rendermode# Set to None for faster headless run
    vision = vision
    
    # Neuron Manipulation Config
    record_csv = RECORD_CSV #"connectomes/drosophila adult connectome/moonwalker_descending_neurons.csv"       # e.g., "neurons_to_record.csv"
    overwrite_csv = OVERWRITE_CSV #"connectomes/drosophila adult connectome/moonwalker_descending_neurons.csv"    # e.g., "neurons_to_overwrite.csv"
    overwrite_val = 0.8
    overwrite_int = 150
    overwrite_steps = 80
    # Hidden State Recording Config
    save_hidden_state_episodes = [53]  # e.g., [1, 5, 53] to save those episodes
    # ---------------------

    edge_path = resolve_connectome_path(edge_path)
    print(f"[main] Connectome: {edge_path}")

    env = _make_env(render_mode=render_mode, n_obstacles=N_OBSTACLES, texture_mode=texture_mode)
    agent, id2idx = _load_agent(checkpoint, device=device, dtype=dtype, edge_path=edge_path)

    # Load indices
    record_indices = load_neuron_indices(record_csv, id2idx)
    overwrite_indices, overwrite_values = load_neuron_overwrite_config(
        overwrite_csv, id2idx, default_value=overwrite_val, device=device, dtype=dtype,
    )
    
    if overwrite_indices:
        print(f"[main] Overwriting {len(overwrite_indices)} neurons every {overwrite_int} steps.")
        print(f"       Per-neuron values: {overwrite_values}")

    teacher = PlannerAnalyticTeacher(
        arena_half_extent=env.arena,
        cell_size=0.1,
        robot_radius=0.2,
        safety_margin=0.01,
        obstacle_box_half=(0.4, 0.4),
        k_nearest_obs=5,
        device="cpu",
    )

    # Prepare data directory
    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("eval_data", f"connectome_rnn_{timestr}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[main] Saving episode data to: {save_dir}")

    episode_summaries = []
    try:
        # for ep in range (53, 54):
        for ep in range (1, episodes + 1):
            ret, steps, done, trunc, goal_reached, goal_xy = rollout_episode(
                env, agent, teacher, device=device, dtype=dtype, 
                render=render_mode == "human", show_path=False,
                episode_idx=ep, save_dir=save_dir,
                record_indices=record_indices,
                overwrite_indices=overwrite_indices,
                overwrite_values=overwrite_values,
                overwrite_interval=overwrite_int,
                overwrite_steps=overwrite_steps,
                seed=ep,
                vision=vision,
                save_hidden_states=False,#(ep in save_hidden_state_episodes),
            )
            episode_summaries.append({
                "episode": ep,
                "goal_x": goal_xy[0],
                "goal_y": goal_xy[1],
                "goal_reached": goal_reached,
                "return": ret,
                "steps": steps,
                "done": done,
                "trunc": trunc,
            })
            print(
                f"[eval] episode {ep}/{episodes} | return={ret:.3f} | steps={steps} "
                f"| done={done} | trunc={trunc} | goal_reached={goal_reached} "
                f"| goal=({goal_xy[0]:.2f}, {goal_xy[1]:.2f})"
            )
    finally:
        # --- Save Episode Summary CSV ---
        summary_file = os.path.join(save_dir, "episode_summary.csv")
        with open(summary_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "goal_x", "goal_y", "goal_reached", "return", "steps", "done", "trunc"])
            writer.writeheader()
            writer.writerows(episode_summaries)
        print(f"[main] Episode summary saved to: {summary_file}")
        # --------------------------------
        env.close()
    return save_dir

# Vision condition codes -> [left_eye_enabled, right_eye_enabled] (order used by obs_to_torch).
VISION_CONDITIONS = {
    "11": [True, True],    # full vision
    "10": [True, False],   # left eye only
    "01": [False, True],   # right eye only
    "00": [False, False],  # blind
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Roll out a trained connectome RNN (FLYNN/SmallWorldNet) checkpoint across "
                     "the selected vision conditions (full vision, left-eye-only, right-eye-only, blind)."
    )
    parser.add_argument(
        "checkpoint",
        help="Path to the .pt checkpoint to evaluate. Required"
    )
    parser.add_argument(
        "--connectome",
        default=None,
        metavar="EDGE_CSV",
        help="Connectome edge-list CSV to build the model from, e.g. "
             "'connectomes/drosophila adult connectome/connections_princeton.csv'. The "
             "photoreceptor, bristle, descending-neuron, cell-type and wind CSVs are read "
             "from the same directory. Defaults to shared_config's EDGE_PATH, with a warning "
             "that it may not match the checkpoint.",
    )
    parser.add_argument(
        "--vision",
        nargs="+",
        choices=list(VISION_CONDITIONS),
        default=list(VISION_CONDITIONS),
        help="Vision condition(s) to evaluate, as two digits <left><right> where 1 = eye "
             "enabled and 0 = eye blind: 11 = full vision, 10 = left eye only, "
             "01 = right eye only, 00 = blind. Multiple values may be given; defaults to "
             "all four.",
    )
    parser.add_argument(
        "--texture-mode",
        choices=["checker", "realistic"],
        default="checker",
        help="Scene texture: 'checker' (default; matches training and the in-distribution "
             "eval) or 'realistic' (photo-realistic PNG textures, for the out-of-distribution "
             "generalization eval).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    checkpoint = args.checkpoint
    # Resolved once so the fallback warning is printed a single time rather than per condition.
    edge_path = resolve_connectome_path(args.connectome)
    # while not os.path.exists(checkpoint):
    #     print(f"Waiting for checkpoint {checkpoint} to be created...")
    #     time.sleep(60)
    # time.sleep(60)
    rendermode = "human"
    condition_dirs = {}
    for code in args.vision:
        print(f"[main] Vision condition {code} (left, right) = {VISION_CONDITIONS[code]}")
        condition_dirs[code] = run_one_configuration(
            checkpoint, vision=VISION_CONDITIONS[code], rendermode=rendermode, edge_path=edge_path,
            texture_mode=args.texture_mode,
        )

    # # from analysis_pca_statistics import analysis_pca_cka
    # condition_folders = [(dir1, "Full vision"), (dir2, "Right eye only"), (dir3, "Left eye only"), (dir4, "Total blindness")]
    # analysis_pca_cka(condition_folders)
