"""
generate_ws_network.py
======================
Construct a topology-matched Watts–Strogatz (WS) small-world control network
whose sensory→DN routing statistics approximate those of a biological fly
connectome.

Steps:
  1. Load fly connectome
  2. Compute fly IO path statistics (BFS-based)
  3. Generate WS graph, assign per-edge syn_count weights
  4. Sample DN pool
  5. Compute distance-to-DN on WS graph
  6. Louvain community detection on WS graph
  7. Match WS communities to fly sensory modalities (optimization)
  8. Sample sensory neurons within matched communities
  9. Sanity check — print fly vs WS comparison

Libraries: numpy, networkx, pandas, community (python-louvain), scipy
"""

import os
import sys
import time
from collections import deque
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain  # python-louvain

# ============================================================================
# Configuration
# ============================================================================
RANDOM_SEED = 42
BETA = 0.1  # WS rewiring probability

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Read directly from the real adult-connectome data folder -- there is no
# separate "connectome/" copy checked in alongside this script, and copying
# the ~270MB source files would just create a second, driftable copy of the
# same data.
CONNECTOME_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "drosophila adult connectome")
)
OUTPUT_DIR = SCRIPT_DIR  # output CSVs go alongside the script

# Sensory modality file names and short labels (order matters for matching)
MODALITY_FILES = [
    ("head_bristles_left.csv", "HB_L"),
    ("head_bristles_right.csv", "HB_R"),
    ("JO-C_and_JO-E.csv", "JO"),
    ("visual_column_L1_L2_L3_rear_view_left.csv", "Vis_L"),
    ("visual_column_L1_L2_L3_rear_view_right.csv", "Vis_R"),
]

np.random.seed(RANDOM_SEED)


# ============================================================================
# Helper: Multi-source BFS on an undirected adjacency list
# ============================================================================
def multi_source_bfs(adj, sources):
    """
    Run BFS from *all* source nodes simultaneously on an undirected graph
    represented as an adjacency list (dict of lists / sets).

    Returns
    -------
    dist : dict[int, int]
        Shortest-path distance from the source set to every reachable node.
    """
    dist = {}
    queue = deque()
    for s in sources:
        if s in adj:
            dist[s] = 0
            queue.append(s)
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def build_undirected_adj(edges, nodes=None):
    """Build an adjacency-list dict from an edge array (N×2)."""
    adj = {}
    if nodes is not None:
        for n in nodes:
            adj[n] = []
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    return adj


# ============================================================================
# STEP 1 — Load Fly Connectome
# ============================================================================
def load_fly_connectome():
    """
    Load the Princeton connectome edge list and modality neuron IDs.
    Remap all root_ids to contiguous 0-based integers.

    Returns
    -------
    fly_edges : np.ndarray (E, 2) — directed edge list (0-based)
    fly_adj   : dict — undirected adjacency list (0-based)
    dn_ids    : set of int — 0-based DN neuron IDs
    modality_ids : list of (label, set_of_0based_ids)
    id_map    : dict original_root_id -> 0-based_id
    N         : int — total number of neurons
    E         : int — total number of directed edges
    """
    print("=" * 70)
    print("STEP 1: Loading fly connectome")
    print("=" * 70)
    t0 = time.time()

    # --- Load edges ---
    print("  Loading connections_princeton.csv...")
    edge_df = pd.read_csv(
        os.path.join(CONNECTOME_DIR, "connections_princeton.csv"),
        usecols=["pre_root_id", "post_root_id"],
    )
    print(f"  Loaded {len(edge_df):,} edges in {time.time()-t0:.1f}s")
    all_nodes = set(edge_df["pre_root_id"]).union(edge_df["post_root_id"])

    # --- Load DN IDs ---
    dn_df = pd.read_csv(os.path.join(CONNECTOME_DIR, "descending_neurons.csv"))
    dn_root_ids = set(dn_df["root_id"].values)
    all_nodes.update(dn_root_ids)

    # --- Load modality IDs ---
    modality_root_ids = []
    for fname, label in MODALITY_FILES:
        mdf = pd.read_csv(os.path.join(CONNECTOME_DIR, fname))
        ids = set(mdf["root_id"].values)
        modality_root_ids.append((label, ids))
        all_nodes.update(ids)

    # --- Build 0-based mapping ---
    print("  Building 0-based ID mapping...")
    sorted_nodes = sorted(all_nodes)
    id_map = {rid: idx for idx, rid in enumerate(sorted_nodes)}
    N = len(sorted_nodes)

    # Map edges
    print("  Mapping edges to 0-based IDs...")
    src = edge_df["pre_root_id"].map(id_map).values.astype(np.int32)
    dst = edge_df["post_root_id"].map(id_map).values.astype(np.int32)
    fly_edges = np.column_stack([src, dst])
    E = len(fly_edges)

    # Map DN
    dn_ids = {id_map[r] for r in dn_root_ids if r in id_map}

    # Map modalities
    modality_ids = []
    for label, rids in modality_root_ids:
        mapped = {id_map[r] for r in rids if r in id_map}
        modality_ids.append((label, mapped))

    # Build undirected adjacency
    print(f"  Building undirected adjacency list for {N:,} nodes, {E:,} edges...")
    fly_adj = build_undirected_adj(fly_edges, nodes=range(N))
    print(f"  Adjacency list built in {time.time()-t0:.1f}s")

    dt = time.time() - t0
    print(f"  Neurons  N = {N:,}")
    print(f"  Edges    E = {E:,}")
    print(f"  DN count   = {len(dn_ids):,}")
    for label, ids in modality_ids:
        print(f"  {label:8s}   = {len(ids):,}")
    print(f"  Loaded in {dt:.1f}s\n")

    return fly_edges, fly_adj, dn_ids, modality_ids, id_map, N, E


# ============================================================================
# STEP 2 — Compute Fly IO Path Statistics
# ============================================================================
def compute_path_stats(adj, source_sets, dn_ids, labels):
    """
    Compute shortest-path distance statistics between each sensory group
    and the DN pool, plus inter-modality separations.

    Uses multi-source BFS — never computes all-pairs shortest path.

    Parameters
    ----------
    adj         : undirected adjacency list
    source_sets : list of sets (one per modality)
    dn_ids      : set of DN node IDs
    labels      : list of str (modality names)

    Returns
    -------
    stats : dict with keys:
        'sensory_to_dn' : list of dicts with mean/median/std per modality
        'inter_modality' : list of dicts for each pair, with mean/median/std
    """
    n_mod = len(source_sets)

    # --- sensory → DN distances ---
    # BFS from DN, then read off distances at sensory nodes
    print(f"  Running multi-source BFS from {len(dn_ids):,} DN neurons...")
    t_bfs = time.time()
    dn_dist = multi_source_bfs(adj, dn_ids)
    print(f"  DN BFS done in {time.time()-t_bfs:.1f}s, reached {len(dn_dist):,} nodes")

    sensory_to_dn = []
    for label, srcs in zip(labels, source_sets):
        dists = [dn_dist[n] for n in srcs if n in dn_dist]
        if len(dists) == 0:
            sensory_to_dn.append(
                {"label": label, "mean": np.nan, "median": np.nan, "std": np.nan, "n": 0}
            )
        else:
            arr = np.array(dists, dtype=np.float64)
            sensory_to_dn.append(
                {
                    "label": label,
                    "mean": float(np.mean(arr)),
                    "median": float(np.median(arr)),
                    "std": float(np.std(arr)),
                    "n": len(arr),
                }
            )

    # --- inter-modality separations ---
    # For each pair (A, B): BFS from A, read distances at B nodes
    inter_modality = []
    pair_list = list(combinations(range(n_mod), 2))
    for pair_idx, (i, j) in enumerate(pair_list):
        label_pair = f"{labels[i]} <-> {labels[j]}"
        print(f"  Inter-modality BFS {pair_idx+1}/{len(pair_list)}: {label_pair}...")
        t_bfs = time.time()
        dist_from_i = multi_source_bfs(adj, source_sets[i])
        print(f"    BFS done in {time.time()-t_bfs:.1f}s")
        dists = [dist_from_i[n] for n in source_sets[j] if n in dist_from_i]
        if len(dists) == 0:
            inter_modality.append(
                {"pair": label_pair, "i": i, "j": j,
                 "mean": np.nan, "median": np.nan, "std": np.nan, "n": 0}
            )
        else:
            arr = np.array(dists, dtype=np.float64)
            inter_modality.append(
                {
                    "pair": label_pair, "i": i, "j": j,
                    "mean": float(np.mean(arr)),
                    "median": float(np.median(arr)),
                    "std": float(np.std(arr)),
                    "n": len(arr),
                }
            )

    return {"sensory_to_dn": sensory_to_dn, "inter_modality": inter_modality}


def print_path_stats(stats, title="Path Statistics"):
    """Pretty-print path statistics."""
    print(f"\n  --- {title} ---")
    print(f"  {'Modality':30s}  {'Mean':>8s}  {'Median':>8s}  {'Std':>8s}  {'N':>6s}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")
    for s in stats["sensory_to_dn"]:
        print(f"  {s['label']:30s}  {s['mean']:8.2f}  {s['median']:8.2f}  {s['std']:8.2f}  {s['n']:6d}")
    print()
    print(f"  {'Inter-modality pair':30s}  {'Mean':>8s}  {'Median':>8s}  {'Std':>8s}  {'N':>6s}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")
    for s in stats["inter_modality"]:
        print(f"  {s['pair']:30s}  {s['mean']:8.2f}  {s['median']:8.2f}  {s['std']:8.2f}  {s['n']:6d}")
    print()


# ============================================================================
# STEP 3 — Generate WS Graph
# ============================================================================
def generate_ws_graph(N, E):
    """
    Generate a Watts-Strogatz small-world graph.

    Parameters
    ----------
    N : int — number of fly neurons
    E : int — number of fly directed edges

    Returns
    -------
    ws_edges : np.ndarray (2*E_ws, 2) — directed edge list (bidirectional)
    ws_adj   : dict  — undirected adjacency list
    ws_G_undirected : nx.Graph — for Louvain
    """
    print("=" * 70)
    print("STEP 3: Generating WS graph")
    print("=" * 70)

    k = int(round((2 * E) / N))
    # k must be even for WS
    if k % 2 != 0:
        k += 1
    # k must be >= 2
    k = max(k, 2)

    print(f"  N = {N:,},  E = {E:,}")
    print(f"  k = {k}  (each node connects to k nearest neighbours)")
    print(f"  beta = {BETA}")

    print("  Generating WS graph (this may take a moment)...")
    t_ws = time.time()
    ws_G = nx.watts_strogatz_graph(N, k, BETA, seed=RANDOM_SEED)
    n_undirected_edges = ws_G.number_of_edges()
    print(f"  Undirected edges: {n_undirected_edges:,}  (generated in {time.time()-t_ws:.1f}s)")

    # Convert to directed: each undirected edge -> one randomly oriented edge
    # This preserves the directed edge count (≈ E) and is more biologically
    # realistic than making every connection bidirectional.
    print("  Randomly orienting edges...")
    directed_edges = []
    for u, v in ws_G.edges():
        if np.random.rand() < 0.5:
            directed_edges.append((u, v))
        else:
            directed_edges.append((v, u))
    ws_edges = np.array(directed_edges, dtype=np.int32)
    print(f"  Directed edges:   {len(ws_edges):,}")

    # Assign a per-edge connection-strength weight. Folded in from the
    # former standalone add_weight.py post-process step so the edge list
    # is written in one deterministic pass under RANDOM_SEED instead of a
    # second, unseeded script run. core/utils.py's load_edge_list() looks
    # for this exact column name via its pre_root_id/post_root_id/syn_count
    # alias.
    print("  Assigning syn_count weights...")
    syn_count = np.random.uniform(0.0, 1.0, size=len(ws_edges))

    # Save edge list
    out_path = os.path.join(OUTPUT_DIR, "connections_ws_small_world.csv")
    pd.DataFrame({
        "pre_root_id": ws_edges[:, 0],
        "post_root_id": ws_edges[:, 1],
        "syn_count": syn_count,
    }).to_csv(out_path, index=False)
    print(f"  Saved: {out_path}\n")

    # Build adjacency list
    print("  Building WS undirected adjacency list...")
    ws_adj = build_undirected_adj(ws_edges, nodes=range(N))
    print(f"  Step 3 total time: {time.time()-t_ws:.1f}s")

    return ws_edges, ws_adj, ws_G


# ============================================================================
# STEP 4 — DN Pool Sampling
# ============================================================================
def sample_dn_pool(N, fly_dn_count, fly_N):
    """
    Sample WS descending neurons proportionally.

    Returns
    -------
    ws_dn : set of int — 0-based WS DN node IDs
    """
    print("=" * 70)
    print("STEP 4: Sampling DN pool")
    print("=" * 70)

    dn_fraction = fly_dn_count / fly_N
    ws_dn_count = max(1, int(round(dn_fraction * N)))
    ws_dn = set(np.random.choice(N, size=ws_dn_count, replace=False))

    out_path = os.path.join(OUTPUT_DIR, "descending_neurons.csv")
    pd.DataFrame({"root_id": sorted(ws_dn)}).to_csv(out_path, index=False)
    print(f"  Fly DN fraction: {dn_fraction:.6f}")
    print(f"  WS DN count:     {ws_dn_count}")
    print(f"  Saved: {out_path}\n")

    return ws_dn


# ============================================================================
# STEP 5 — Distance from DN on WS graph
# ============================================================================
def compute_distance_to_dn(ws_adj, ws_dn):
    """
    Multi-source BFS from WS DN neurons.

    Returns
    -------
    dist_to_dn : dict[int, int]
    """
    print("=" * 70)
    print("STEP 5: Computing distance-to-DN on WS graph")
    print("=" * 70)
    t0 = time.time()
    dist_to_dn = multi_source_bfs(ws_adj, ws_dn)
    dt = time.time() - t0
    print(f"  Reachable nodes: {len(dist_to_dn):,}")
    print(f"  BFS completed in {dt:.2f}s\n")
    return dist_to_dn


# ============================================================================
# STEP 6 — Community Detection (Louvain)
# ============================================================================
def detect_communities(ws_G):
    """
    Louvain clustering on the undirected WS graph.

    Returns
    -------
    partition : dict[int, int] — node -> community_id
    communities : dict[int, set] — community_id -> set of nodes
    """
    print("=" * 70)
    print("STEP 6: Community detection (Louvain)")
    print("=" * 70)
    t0 = time.time()
    partition = community_louvain.best_partition(ws_G, random_state=RANDOM_SEED)
    dt = time.time() - t0

    # Group by community
    communities = {}
    for node, cid in partition.items():
        communities.setdefault(cid, set()).add(node)

    n_communities = len(communities)
    sizes = sorted([len(v) for v in communities.values()], reverse=True)
    print(f"  Number of communities: {n_communities}")
    print(f"  Sizes (top-10): {sizes[:10]}")
    print(f"  Louvain completed in {dt:.2f}s\n")

    return partition, communities


# ============================================================================
# STEP 7 — Match Sensory Regions to Fly Statistics
# ============================================================================
def match_sensory_regions(
    ws_adj, ws_dn, communities, fly_stats, fly_modality_counts, labels,
    dn_dist=None,
):
    """
    Select 5 communities (one per modality) such that the WS sensory→DN
    distances and inter-modality separations best match the fly statistics.

    Strategy
    --------
    1. Pre-compute: for each community, mean distance to DN.
    2. Pre-compute: for each pair of communities, mean inter-community distance.
    3. Score all valid 5-community tuples (using greedy + random restarts).
    4. Return the best assignment.

    Returns
    -------
    best_assignment : list of int — community IDs for each modality
    """
    print("=" * 70)
    print("STEP 7: Matching sensory regions to fly statistics")
    print("=" * 70)
    t0 = time.time()

    n_mod = len(labels)

    # --- Target values ---
    fly_dn_means = [s["mean"] for s in fly_stats["sensory_to_dn"]]
    fly_inter_means = [s["mean"] for s in fly_stats["inter_modality"]]
    fly_inter_pairs = [(s["i"], s["j"]) for s in fly_stats["inter_modality"]]

    # --- Pre-compute community → DN mean distance ---
    # Reuse BFS from Step 5 if available, otherwise compute here
    if dn_dist is None:
        dn_dist = multi_source_bfs(ws_adj, ws_dn)

    comm_ids = sorted(communities.keys())
    comm_dn_mean = {}
    for cid in comm_ids:
        dists = [dn_dist[n] for n in communities[cid] if n in dn_dist]
        if dists:
            comm_dn_mean[cid] = np.mean(dists)
        else:
            comm_dn_mean[cid] = np.inf

    # Filter communities that are large enough.
    # Each modality needs at least as many neurons as fly_modality_counts[i].
    min_needed = min(fly_modality_counts)
    eligible_cids = [
        cid for cid in comm_ids
        if len(communities[cid]) >= min_needed and comm_dn_mean[cid] < np.inf
    ]
    print(f"  Eligible communities (size >= {min_needed}): {len(eligible_cids)}")

    # --- Pre-compute pairwise inter-community mean distances ---
    # For each community, run BFS from all its members (sampling if too large)
    MAX_BFS_SOURCES = 200  # cap to limit compute
    comm_bfs_dist = {}

    n_elig = len(eligible_cids)
    print(f"  Pre-computing inter-community distances ({n_elig} communities)...")
    for idx_c, cid in enumerate(eligible_cids):
        members = list(communities[cid])
        if len(members) > MAX_BFS_SOURCES:
            sampled = list(np.random.choice(members, MAX_BFS_SOURCES, replace=False))
        else:
            sampled = members
        t_bfs = time.time()
        comm_bfs_dist[cid] = multi_source_bfs(ws_adj, sampled)
        print(f"    Community {cid} ({idx_c+1}/{n_elig}): "
              f"BFS from {len(sampled)} sources in {time.time()-t_bfs:.1f}s")

    # Pairwise mean distance between eligible communities
    inter_comm_mean = {}
    for ci, cj in combinations(eligible_cids, 2):
        dists = [comm_bfs_dist[ci][n] for n in communities[cj] if n in comm_bfs_dist[ci]]
        if dists:
            inter_comm_mean[(ci, cj)] = np.mean(dists)
            inter_comm_mean[(cj, ci)] = np.mean(dists)
        else:
            inter_comm_mean[(ci, cj)] = np.inf
            inter_comm_mean[(cj, ci)] = np.inf

    # Also need intra-community "distance" (community to itself) = 0-ish but
    # we won't need that for inter-modality matching.

    # --- Scoring function ---
    def score_assignment(assignment):
        """
        Total squared error between WS stats and fly stats for a given
        assignment of communities to modalities.
        assignment : list of 5 community IDs (one per modality)
        """
        err = 0.0
        # Sensory→DN error
        for i in range(n_mod):
            err += (comm_dn_mean[assignment[i]] - fly_dn_means[i]) ** 2

        # Inter-modality error
        for idx, (mi, mj) in enumerate(fly_inter_pairs):
            ci = assignment[mi]
            cj = assignment[mj]
            if ci == cj:
                # Same community — distance ≈ 0, big penalty
                d = 0.0
            else:
                key = (ci, cj)
                d = inter_comm_mean.get(key, np.inf)
            err += (d - fly_inter_means[idx]) ** 2

        return err

    # --- Search strategy ---
    # If eligible communities are few enough, try brute force;
    # otherwise use greedy + random restarts.
    from itertools import permutations

    n_eligible = len(eligible_cids)
    print(f"  Searching over {n_eligible} eligible communities for best 5-assignment...")

    best_score = np.inf
    best_assignment = None

    if n_eligible <= 20:
        # Feasible to try all 5-permutations
        count = 0
        for combo in combinations(eligible_cids, n_mod):
            for perm in permutations(combo):
                # Check size constraints
                valid = True
                for i in range(n_mod):
                    if len(communities[perm[i]]) < fly_modality_counts[i]:
                        valid = False
                        break
                if not valid:
                    continue
                s = score_assignment(list(perm))
                if s < best_score:
                    best_score = s
                    best_assignment = list(perm)
                count += 1
        print(f"  Evaluated {count:,} valid permutations")
    else:
        # Greedy: for each modality, pick the community whose DN distance
        # is closest to target, then refine with random restarts.

        # Sort communities by DN distance
        sorted_by_dn = sorted(eligible_cids, key=lambda c: comm_dn_mean[c])

        # Greedy assignment
        def greedy_assign(order):
            """Assign modalities in given order, greedily."""
            used = set()
            assignment = [None] * n_mod
            for idx in order:
                target_dn = fly_dn_means[idx]
                best_c = None
                best_err = np.inf
                for cid in sorted_by_dn:
                    if cid in used:
                        continue
                    if len(communities[cid]) < fly_modality_counts[idx]:
                        continue
                    e = abs(comm_dn_mean[cid] - target_dn)
                    if e < best_err:
                        best_err = e
                        best_c = cid
                if best_c is not None:
                    assignment[idx] = best_c
                    used.add(best_c)
            return assignment

        # Try multiple orderings
        N_RESTARTS = 500
        for trial in range(N_RESTARTS):
            if trial == 0:
                order = list(range(n_mod))
            else:
                order = list(np.random.permutation(n_mod))
            asgn = greedy_assign(order)
            if None in asgn:
                continue
            s = score_assignment(asgn)
            if s < best_score:
                best_score = s
                best_assignment = asgn

        # Local refinement: try swapping pairs
        improved = True
        while improved:
            improved = False
            for i, j in combinations(range(n_mod), 2):
                trial = best_assignment.copy()
                trial[i], trial[j] = trial[j], trial[i]
                # Check size constraints
                valid = True
                for m in range(n_mod):
                    if len(communities[trial[m]]) < fly_modality_counts[m]:
                        valid = False
                        break
                if not valid:
                    continue
                s = score_assignment(trial)
                if s < best_score:
                    best_score = s
                    best_assignment = trial
                    improved = True

        # Also try replacing one community with another eligible one
        for i in range(n_mod):
            for cid in eligible_cids:
                if cid in best_assignment:
                    continue
                if len(communities[cid]) < fly_modality_counts[i]:
                    continue
                trial = best_assignment.copy()
                trial[i] = cid
                # Check uniqueness
                if len(set(trial)) < n_mod:
                    continue
                s = score_assignment(trial)
                if s < best_score:
                    best_score = s
                    best_assignment = trial

        print(f"  Evaluated {N_RESTARTS} greedy restarts + local refinement")

    dt = time.time() - t0
    print(f"  Best score (sum squared error): {best_score:.4f}")
    for i, label in enumerate(labels):
        cid = best_assignment[i]
        print(
            f"    {label:8s} -> community {cid:3d}  "
            f"(size={len(communities[cid]):,},  "
            f"DN_dist={comm_dn_mean[cid]:.2f},  "
            f"target={fly_dn_means[i]:.2f})"
        )
    print(f"  Matching completed in {dt:.1f}s\n")

    return best_assignment


# ============================================================================
# STEP 8 — Sample Sensory Neurons
# ============================================================================
def sample_sensory_neurons(best_assignment, communities, fly_modality_counts, labels, ws_dn=None):
    """
    Within each assigned community, randomly sample the same number of neurons
    as in the corresponding fly modality.

    Returns
    -------
    ws_modality_ids : list of (label, set_of_ids)
    """
    print("=" * 70)
    print("STEP 8: Sampling sensory neurons")
    print("=" * 70)

    ws_modality_ids = []
    output_filenames = [fname for fname, _ in MODALITY_FILES]

    for i, label in enumerate(labels):
        cid = best_assignment[i]
        # Exclude DN neurons from the sensory sampling pool
        pool_set = communities[cid]
        if ws_dn is not None:
            pool_set = pool_set - ws_dn
        pool = sorted(pool_set)
        n_sample = fly_modality_counts[i]

        if n_sample > len(pool):
            print(f"  WARNING: community {cid} has only {len(pool)} nodes, "
                  f"need {n_sample} for {label}. Sampling all.")
            sampled = pool
        else:
            sampled = list(np.random.choice(pool, size=n_sample, replace=False))

        sampled_set = set(sampled)
        ws_modality_ids.append((label, sampled_set))

        out_path = os.path.join(OUTPUT_DIR, output_filenames[i])
        pd.DataFrame({"root_id": sorted(sampled)}).to_csv(out_path, index=False)
        print(f"  {label:8s}: sampled {len(sampled):,} from community {cid} -> {out_path}")

    print()
    return ws_modality_ids


# ============================================================================
# STEP 9 — Sanity Check
# ============================================================================
def sanity_check(ws_adj, ws_dn, ws_modality_ids, fly_stats, labels):
    """
    Recompute WS path stats and print comparison with fly stats.
    """
    print("=" * 70)
    print("STEP 9: Sanity check — Fly vs WS comparison")
    print("=" * 70)

    ws_source_sets = [ids for _, ids in ws_modality_ids]
    ws_stats = compute_path_stats(ws_adj, ws_source_sets, ws_dn, labels)

    # Print comparison table
    print(f"\n  {'Modality':30s}  {'Fly Mean':>10s}  {'WS Mean':>10s}  {'Delta':>8s}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*8}")
    for f_s, w_s in zip(fly_stats["sensory_to_dn"], ws_stats["sensory_to_dn"]):
        delta = w_s["mean"] - f_s["mean"]
        print(f"  {f_s['label']:30s}  {f_s['mean']:10.2f}  {w_s['mean']:10.2f}  {delta:+8.2f}")

    print()
    print(f"  {'Inter-modality pair':30s}  {'Fly Mean':>10s}  {'WS Mean':>10s}  {'Delta':>8s}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*8}")
    for f_s, w_s in zip(fly_stats["inter_modality"], ws_stats["inter_modality"]):
        delta = w_s["mean"] - f_s["mean"]
        print(f"  {f_s['pair']:30s}  {f_s['mean']:10.2f}  {w_s['mean']:10.2f}  {delta:+8.2f}")

    print()
    return ws_stats


# ============================================================================
# STEP 10 — Assign Neuron Types
# ============================================================================
def compute_participation_coefficient(ws_G, partition):
    """
    Compute the participation coefficient for each node.
    P_i = 1 - sum_c (k_ic / k_i)^2
    where k_ic is degree within community c and k_i is total degree.
    High P = projection neuron (edges across many communities).
    Low P = local neuron (edges mostly within own community).
    """
    pc = {}
    for node in ws_G.nodes():
        k_i = ws_G.degree(node)
        if k_i == 0:
            pc[node] = 0.0
            continue
        # Count edges per community
        comm_degree = {}
        for neighbor in ws_G.neighbors(node):
            c = partition[neighbor]
            comm_degree[c] = comm_degree.get(c, 0) + 1
        p = 1.0 - sum((kc / k_i) ** 2 for kc in comm_degree.values())
        pc[node] = p
    return pc


def assign_neuron_types(fly_adj, fly_dn_ids, id_map, dn_dist_ws, partition, ws_G, N):
    """
    Assign fly primary_type labels to WS neurons using topological matching.

    Strategy:
    1. Load fly type distribution and compute each type's mean distance-to-DN
       in the fly connectome to establish a "depth ordering" of types.
    2. Compute participation coefficient for each WS neuron.
    3. Sort WS neurons by (distance_to_DN, -participation_coefficient) to
       create a topological ordering from motor-proximal to sensory-distal.
    4. Sort fly types by their mean distance-to-DN.
    5. Assign fly types to WS neurons in matching rank order, preserving
       the count of each type.

    Returns
    -------
    type_df : pd.DataFrame with columns [root_id, primary_type]
    """
    print("=" * 70)
    print("STEP 10: Assigning neuron types")
    print("=" * 70)
    t0 = time.time()

    # --- Load fly types ---
    fly_types_df = pd.read_csv(
        os.path.join(CONNECTOME_DIR, "consolidated_cell_types.csv"),
        usecols=["root_id", "primary_type"],
    )
    print(f"  Fly types loaded: {len(fly_types_df):,} neurons, "
          f"{fly_types_df['primary_type'].nunique():,} unique types")

    # --- Compute fly distance-to-DN for each neuron that has a type ---
    print("  Computing fly distance-to-DN via BFS...")
    t_bfs = time.time()
    fly_dn_dist = multi_source_bfs(fly_adj, fly_dn_ids)
    print(f"  Fly DN BFS done in {time.time()-t_bfs:.1f}s")

    # Map fly root_ids to 0-based and get distances
    fly_types_df["node_id"] = fly_types_df["root_id"].map(id_map)
    fly_types_df["dist_to_dn"] = fly_types_df["node_id"].map(
        lambda x: fly_dn_dist.get(x, np.nan) if pd.notna(x) else np.nan
    )

    # --- Compute mean distance-to-DN per fly type ---
    type_mean_dist = (
        fly_types_df.groupby("primary_type")["dist_to_dn"]
        .mean()
        .sort_values()
    )
    # For types with NaN distances, put them in the middle
    median_dist = type_mean_dist.median()
    type_mean_dist = type_mean_dist.fillna(median_dist)

    # --- Count per type (this is what we need to reproduce in WS) ---
    type_counts = fly_types_df["primary_type"].value_counts()
    # Only keep types that sum to <= N; if fly has more neurons than WS,
    # we proportionally scale down
    total_fly_typed = type_counts.sum()
    if total_fly_typed > N:
        # Scale proportionally
        scale = N / total_fly_typed
        type_counts_scaled = (type_counts * scale).round().astype(int)
        # Fix rounding: adjust the largest type
        diff = N - type_counts_scaled.sum()
        type_counts_scaled.iloc[0] += diff
        type_counts = type_counts_scaled
    elif total_fly_typed < N:
        # Add generic "untyped" for remaining neurons
        type_counts["untyped"] = N - total_fly_typed
        type_mean_dist["untyped"] = median_dist

    # --- Sort types by fly distance-to-DN ---
    sorted_types = type_mean_dist.sort_values().index.tolist()
    # Build ordered list of type labels (repeating by count)
    type_labels_ordered = []
    for t in sorted_types:
        if t in type_counts.index:
            type_labels_ordered.extend([t] * int(type_counts[t]))

    # Trim or pad to exactly N
    if len(type_labels_ordered) > N:
        type_labels_ordered = type_labels_ordered[:N]
    elif len(type_labels_ordered) < N:
        # Pad with the most common type
        most_common = type_counts.index[0]
        type_labels_ordered.extend(
            [most_common] * (N - len(type_labels_ordered))
        )

    # --- Compute WS participation coefficient ---
    print("  Computing participation coefficients...")
    pc = compute_participation_coefficient(ws_G, partition)

    # --- Sort WS neurons by (distance_to_DN, -participation_coefficient) ---
    # Neurons close to DN (low dist) get motor-proximal types;
    # neurons far from DN (high dist) get sensory types.
    ws_neurons = list(range(N))
    max_dist = max(dn_dist_ws.values()) if dn_dist_ws else 0
    ws_neurons.sort(
        key=lambda n: (
            dn_dist_ws.get(n, max_dist + 1),
            -pc.get(n, 0.0),
        )
    )

    # --- Assign types ---
    neuron_types = {}
    for i, node in enumerate(ws_neurons):
        neuron_types[node] = type_labels_ordered[i]

    # --- Save ---
    type_df = pd.DataFrame([
        {"root_id": node, "primary_type": neuron_types[node]}
        for node in sorted(neuron_types.keys())
    ])
    out_path = os.path.join(OUTPUT_DIR, "consolidated_cell_types.csv")
    type_df.to_csv(out_path, index=False)

    dt = time.time() - t0
    n_unique_assigned = type_df["primary_type"].nunique()
    print(f"  Assigned {n_unique_assigned:,} unique types to {N:,} WS neurons")
    print(f"  Saved: {out_path}")
    print(f"  Step 10 completed in {dt:.1f}s\n")

    return type_df


# ============================================================================
# Main
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("  WS Small-World Control Network Generation")
    print("=" * 70 + "\n")

    # Step 1
    fly_edges, fly_adj, dn_ids, modality_ids, id_map, N, E = load_fly_connectome()
    labels = [label for label, _ in modality_ids]
    source_sets = [ids for _, ids in modality_ids]
    fly_modality_counts = [len(ids) for _, ids in modality_ids]

    # Step 2
    print("=" * 70)
    print("STEP 2: Computing fly IO path statistics")
    print("=" * 70)
    t0 = time.time()
    fly_stats = compute_path_stats(fly_adj, source_sets, dn_ids, labels)
    dt = time.time() - t0
    print_path_stats(fly_stats, title="Fly Path Statistics")
    print(f"  Fly stats computed in {dt:.1f}s\n")

    # Step 3
    ws_edges, ws_adj, ws_G = generate_ws_graph(N, E)

    # Step 4
    ws_dn = sample_dn_pool(N, len(dn_ids), N)

    # Step 5
    dn_dist = compute_distance_to_dn(ws_adj, ws_dn)

    # Step 6
    partition, communities = detect_communities(ws_G)

    # Step 7
    best_assignment = match_sensory_regions(
        ws_adj, ws_dn, communities, fly_stats, fly_modality_counts, labels,
        dn_dist=dn_dist,
    )

    # Step 8
    ws_modality_ids = sample_sensory_neurons(
        best_assignment, communities, fly_modality_counts, labels, ws_dn=ws_dn
    )

    # Step 9
    ws_stats = sanity_check(ws_adj, ws_dn, ws_modality_ids, fly_stats, labels)

    # Step 10
    type_df = assign_neuron_types(
        fly_adj, dn_ids, id_map, dn_dist, partition, ws_G, N
    )

    print("=" * 70)
    print("  All output files generated successfully!")
    print("=" * 70)
    output_files = ["connections_ws_small_world.csv", "descending_neurons.csv",
                    "consolidated_cell_types.csv"] + [
        fname for fname, _ in MODALITY_FILES
    ]
    for f in output_files:
        full = os.path.join(OUTPUT_DIR, f)
        exists = os.path.isfile(full)
        size = os.path.getsize(full) if exists else 0
        print(f"  {'[OK]' if exists else '[MISSING]'} {f}  ({size:,} bytes)")
    print()


if __name__ == "__main__":
    main()
