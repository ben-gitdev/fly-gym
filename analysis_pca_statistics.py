"""
Statistical comparison of PCA distances across episodes and conditions.

For each episode, loads hidden_states_x.npy from each condition folder,
performs joint PCA, computes the average Euclidean distance (in 10-PC space)
of each condition's trajectory to the first condition, then runs paired
t-tests across episodes and visualises the results as a bar plot.
"""

import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA

# ---- Global plot style ----
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 24

# ---- Configuration ----
# Each entry: (folder_path, display_label)
# # The FIRST condition is the reference – distances are measured FROM it.
# CONDITION_FOLDERS = [
#     (r"path_to\eval_data\connectome_full_vision_2", "Full vision"),
#     (r"path_to\eval_data\connectome_left_eye_2", "Left eye only"),
#     (r"path_to\eval_data\connectome_right_eye_2", "Right eye only"),
#     (r"path_to\eval_data\connectome_total_blind_2", "Total blind"),
# ]

CONDITION_FOLDERS = [
    (r"path_to\eval_data\small_world_full_vision_states", "Full vision"),
    (r"path_to\eval_data\small_world_left_eye_states", "Left eye only"),
    (r"path_to\eval_data\small_world_right_eye_states", "Right eye only"),
    (r"path_to\eval_data\small_world_blind_states", "Total blind"),
    # (r"path_to\eval_data\small_world_textured_env", "Testing env"),
]

# CONDITION_FOLDERS = [
#     # (r"path_to\eval_data\connectome_full_vision_4", "Training env"),
#     # (r"path_to\eval_data\connectome_textured_env", "Testing env"),
#     (r"path_to\eval_data\small_world_full_vision_states", "Training env"),
#     (r"path_to\eval_data\small_world_textured_env", "Testing env"),
# ]

# CONDITION_FOLDERS = [
#     (r"path_to\eval_data\connectome_full_vision_4", "Full vision"),
#     (r"path_to\eval_data\connectome_left_eye_4", "Left eye only"),
#     (r"path_to\eval_data\connectome_right_eye_4", "Right eye only"),
#     (r"path_to\eval_data\connectome_blind_4", "Total blind"),
    
#     (r"path_to\eval_data\connectome_textured_env", "Testing env"),
# ]

# Colors for each condition (skip index 0 since it's the reference)
# COLORS = ["#2196F3", "#FF9800", "#F44336", "#4CAF50"]
COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#F44336", "#C203FC"]

N_PCS = 2  # number of principal components for distance calculation


def discover_episodes(condition_folders: list, max_episodes: int = None) -> list:
    """Find episode numbers common to ALL condition folders.

    Each folder is expected to contain files named ``hidden_states_<x>.npy``
    where *x* is the episode number.

    Returns a sorted list of episode numbers present in every folder.
    """
    episode_sets = []
    for folder, _ in condition_folders:
        files = glob.glob(os.path.join(folder, "hidden_states_*.npy"))
        eps = set()
        for f in files:
            basename = os.path.splitext(os.path.basename(f))[0]
            # basename is "hidden_states_<x>"
            ep_num = basename.replace("hidden_states_", "")
            try:
                eps.add(int(ep_num))
            except ValueError:
                continue
        episode_sets.append(eps)
        print(f"[io] {folder}: found {len(eps)} episodes")

    common = episode_sets[0]
    for s in episode_sets[1:]:
        common = common & s
    common = sorted(common)
    print(f"[io] {len(common)} episodes common across all conditions")
    return common[:max_episodes] if max_episodes else common


def compute_pca_distances(condition_folders: list,
                          episodes: list,
                          n_pcs: int = N_PCS) -> np.ndarray:
    """Compute average PCA distance for each episode and non-reference condition.

    Parameters
    ----------
    condition_folders : list of (path, label) tuples
    episodes : list of int, episode numbers
    n_pcs : int, number of PCA components

    Returns
    -------
    distances : np.ndarray, shape (n_episodes, n_conditions - 1)
        ``distances[i, j]`` is the average Euclidean distance (over time)
        between condition 0 and condition j+1 in the *n_pcs*-dimensional
        PCA space for episode *episodes[i]*.
    """
    n_conds = len(condition_folders)
    distances = np.zeros((len(episodes), n_conds - 1))

    for ep_idx, ep_num in enumerate(episodes):
        # Load hidden states for this episode from every condition
        arrays = []
        for folder, label in condition_folders:
            fpath = os.path.join(folder, f"hidden_states_{ep_num}.npy")
            arr = np.load(fpath)  # (T, N)
            arrays.append(arr)

        # Concatenate for joint PCA fitting
        all_states = np.concatenate(arrays, axis=0)  # (sum_T, N)
        pca = PCA(n_components=n_pcs)
        pca.fit(all_states)

        # Project each condition
        projected = [pca.transform(a) for a in arrays]  # list of (T, n_pcs)

        # Reference is the first condition
        ref = projected[0]
        for cond_idx in range(1, n_conds):
            other = projected[cond_idx]
            min_T = min(ref.shape[0], other.shape[0])
            dists = np.linalg.norm(ref[:min_T] - other[:min_T], axis=1)
            distances[ep_idx, cond_idx - 1] = np.mean(dists)

        print(f"[pca] Episode {ep_num}: distances = {distances[ep_idx]}")

    return distances

def linear_cka(X, Y):
    """
    Standard Linear CKA (Kornblith et al., 2019).
    X, Y: (T, N) — T time steps (samples), N neurons (features).
    """

    # Center across samples (time steps)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Linear kernels (T x T)
    K = X @ X.T
    L = Y @ Y.T

    def center_gram(G):
        n = G.shape[0]
        H = np.eye(n) - np.ones((n,n))/n
        return H @ G @ H

    Kc = center_gram(K)
    Lc = center_gram(L)

    return np.sum(Kc*Lc)/np.sqrt(np.sum(Kc*Kc)*np.sum(Lc*Lc))

def compute_cka_similarities(condition_folders, episodes):
    n_conds = len(condition_folders)
    similarities = np.zeros((len(episodes), n_conds - 1))

    for ep_idx, ep_num in enumerate(episodes):
        arrays = []
        for folder, _ in condition_folders:
            fpath = os.path.join(folder, f"hidden_states_{ep_num}.npy")
            arrays.append(np.load(fpath)) # (T, N)

        # Reference is the first condition (Full Vision)
        ref = arrays[0]
        for cond_idx in range(1, n_conds):
            other = arrays[cond_idx]
            # Ensure time steps match
            min_T = min(ref.shape[0], other.shape[0])
            score = linear_cka(ref[:min_T], other[:min_T])
            similarities[ep_idx, cond_idx - 1] = score

        print(f"[cka] Episode {ep_num}: scores = {similarities[ep_idx]}")
    return similarities

def run_paired_ttests(distances: np.ndarray,
                      condition_labels: list) -> list:
    """Run paired t-tests between all non-reference condition pairs.

    Parameters
    ----------
    distances : np.ndarray, shape (n_episodes, n_conditions - 1)
    condition_labels : list of str, labels for non-reference conditions

    Returns
    -------
    results : list of dict with keys 'pair', 't_stat', 'p_value'
    """
    n_conds = distances.shape[1]
    results = []
    for i in range(n_conds):
        for j in range(i + 1, n_conds):
            t_stat, p_val = stats.ttest_rel(distances[:, i], distances[:, j])
            pair_label = f"{condition_labels[i]} vs {condition_labels[j]}"
            results.append({
                "pair": pair_label,
                "cond_i": i,
                "cond_j": j,
                "t_stat": t_stat,
                "p_value": p_val,
            })
            print(f"[stat] {pair_label}: t={t_stat:.4f}, p={p_val:.4e}")
    return results


def format_p_value(p: float) -> str:
    """Return a human-readable string for a p-value."""
    # if p < 0.001:
    #     return "p < 0.001"
    # elif p < 0.01:
    #     return f"p = {p:.3f}"
    # elif p < 0.05:
    #     return f"p = {p:.3f}"
    # else:
    #     return f"p = {p:.3f} (n.s.)"

    return f"p = {p:.5f}"


def significance_marker(p: float) -> str:
    """Return stars for significance level."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


def plot_results(distances: np.ndarray,
                 condition_labels: list,
                 ref_label: str,
                 ttest_results: list) -> None:
    """Create a bar plot with significance brackets.

    Parameters
    ----------
    distances : np.ndarray, shape (n_episodes, n_conditions - 1)
    condition_labels : list of str, labels for non-reference conditions
    ref_label : str, label of the reference (first) condition
    ttest_results : list of dict from run_paired_ttests
    """
    n_conds = distances.shape[1]
    means = np.mean(distances, axis=0)
    sems = stats.sem(distances, axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    x_pos = np.arange(n_conds)
    bar_colors = [COLORS[i + 1] for i in range(n_conds)]

    ax.bar(x_pos, means, yerr=sems, capsize=6,
           color=bar_colors, edgecolor='k', width=0.5, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(condition_labels)
    ax.set_ylabel(f"Avg Euclidean Distance ({N_PCS} PCs)")
    ax.set_title(f"PCA Distance to {ref_label}")
    ax.grid(True, alpha=0.3, axis='y')

    # Draw significance brackets
    y_max = max(means + sems) if len(means) > 0 else 1.0
    bracket_offset = y_max * 0.08
    bracket_y = y_max + bracket_offset

    for result in ttest_results:
        i, j = result["cond_i"], result["cond_j"]
        p = result["p_value"]
        marker = significance_marker(p)

        # Draw bracket line
        ax.plot([i, i, j, j],
                [bracket_y, bracket_y + bracket_offset * 0.3,
                 bracket_y + bracket_offset * 0.3, bracket_y],
                color='k', linewidth=1.2)
        # Add p-value text
        ax.text((i + j) / 2, bracket_y + bracket_offset * 0.4,
                f"{marker}\n{format_p_value(p)}",
                ha='center', va='bottom', fontsize=14)

        bracket_y += bracket_offset * 2.5  # offset for next bracket

    fig.tight_layout()
    plt.show()


def plot_pca_kde(condition_folders: list, episodes: list,
                 n_episodes: int = None, n_fit_episodes: int = 5, n_pcs: int = N_PCS) -> dict:
    """2D KDE contour plot of PC1 vs PC2 overlaying multiple episodes.

    To save memory on large datasets, PCA is fitted on a subset of 
    episodes (n_fit_episodes), and then all selected episodes are 
    projected into this shared subspace one by one.

    Parameters
    ----------
    condition_folders : list of (path, label)
    episodes : list of int
    n_episodes : int or None, number of episodes to overlay (None = all)
    n_fit_episodes: int, number of episodes to use for PCA fitting
    n_pcs: int, number of components to project and calculate KDE peak
    """
    import seaborn as sns
    from sklearn.decomposition import PCA
    from scipy.optimize import minimize
    
    sel_episodes = episodes[:n_episodes] if n_episodes else episodes
    fit_episodes = sel_episodes[:n_fit_episodes]
    labels = [label for _, label in condition_folders]
    n_conds = len(condition_folders)

    print(f"[pca] Fitting PCA on subspace from first {len(fit_episodes)} episodes...")
    # 1. Collect a subset of data for fitting PCA
    fit_arrays = []
    for ep_num in fit_episodes:
        for folder, _ in condition_folders:
            fpath = os.path.join(folder, f"hidden_states_{ep_num}.npy")
            fit_arrays.append(np.load(fpath))

    fit_states = np.concatenate(fit_arrays, axis=0)
    pca = PCA(n_components=n_pcs)
    pca.fit(fit_states)
    
    total_var = np.sum(pca.explained_variance_ratio_)
    print(f"[pca] Total explained variance by {n_pcs} PCs: {total_var * 100:.2f}%")
    
    # Clear large arrays from memory
    del fit_arrays
    del fit_states

    print(f"[pca] Projecting all {len(sel_episodes)} episodes to 2D...")
    # 2. Project all selected episodes step-by-step
    proj_data = [[] for _ in range(n_conds)]
    for ep_num in sel_episodes:
        print(f"[pca] Projecting episode {ep_num}")
        for cond_idx, (folder, _) in enumerate(condition_folders):
            fpath = os.path.join(folder, f"hidden_states_{ep_num}.npy")
            arr = np.load(fpath)  # (T, N)
            proj = pca.transform(arr)  # (T, 2)
            proj_data[cond_idx].append(proj)

    fig, ax = plt.subplots(figsize=(8, 7))

    max_points_for_kde = 100000  # Cap to prevent KDE calculation from hanging
    
    # Dictionary to hold the calculated KDE values for each condition
    kde_results = {}

    for cond_idx in range(n_conds):
        # Concatenate 2D projections for this condition
        cond_proj = np.concatenate(proj_data[cond_idx], axis=0) # (sum_T, 2)
        
        # # Downsample for faster KDE estimation if needed
        # if cond_proj.shape[0] > max_points_for_kde:
        #     print(f"[pca] Downsampling condition '{labels[cond_idx]}' from "
        #           f"{cond_proj.shape[0]} to {max_points_for_kde} points for KDE computation...")
        #     rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        #     idx = rng.choice(cond_proj.shape[0], size=max_points_for_kde, replace=False)
        #     cond_proj = cond_proj[idx]

        # sns.kdeplot(
        #     x=cond_proj[:, 0], y=cond_proj[:, 1],
        #     ax=ax,
        #     color=COLORS[cond_idx],
        #     label=labels[cond_idx],
        #     levels=6,
        #     linewidths=2,
        #     alpha=0.8
        # )
        
        # Extract x and y coordinates
        x_val = cond_proj[:, 0]
        y_val = cond_proj[:, 1]
        
        # 1. Fit 2D KDE manually using scipy for contour plot
        kde = stats.gaussian_kde(np.vstack([x_val, y_val]))
        
        # 2. Evaluate on a 100x100 grid spanning the data
        xmin, xmax = x_val.min(), x_val.max()
        ymin, ymax = y_val.min(), y_val.max()
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        
        # Compute the density values and reshape back to grid dimensions
        Z = np.reshape(kde(positions).T, X.shape)
        
        # 3. Fit N-dimensional KDE to find the true peak in N-D space
        kde_nd = stats.gaussian_kde(cond_proj.T)
        
        # Evaluate on a subset of points to find a good initialization for the peak
        n_eval = min(10000, cond_proj.shape[0])
        rng = np.random.default_rng(42)
        idx = rng.choice(cond_proj.shape[0], size=n_eval, replace=False)
        eval_pts = cond_proj[idx]
        
        densities = kde_nd(eval_pts.T)
        peak_nd_init = eval_pts[np.argmax(densities)]
        
        # Refine the peak using optimization
        res = minimize(lambda x: -kde_nd(x)[0], peak_nd_init, method='L-BFGS-B')
        peak_nd = res.x

        # Store the actual values in the dictionary
        kde_results[labels[cond_idx]] = {
            'X': X,
            'Y': Y,
            'Z': Z,  # These are the actual 2D KDE density values
            'peak_nd': peak_nd # N-dimensional peak 
        }
        
        # 3. Plot the contours using the evaluated values
        ax.contour(X, Y, Z, levels=6, colors=[COLORS[cond_idx]], linewidths=2, alpha=0.8)
        
        # Add a dummy line to show up in the legend
        ax.plot([], [], color=COLORS[cond_idx], label=labels[cond_idx], linewidth=2, alpha=0.8)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"PCA Density (KDE) — {len(sel_episodes)} episodes")
    # ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    # fig.tight_layout()
    plt.show()
    
    return kde_results


def analysis_pca_cka(condition_folders: list):
    """Main entry point."""
    # Validate configuration
    # assert len(condition_folders) >= 2, "Need at least 2 conditions"
    for folder, label in condition_folders:
        assert os.path.isdir(folder), f"Folder not found: {folder}"

    # Discover common episodes
    episodes = discover_episodes(condition_folders, max_episodes=500)
    # assert len(episodes) >= 2, (
    #     f"Need at least 2 common episodes for paired t-test, found {len(episodes)}"
    # )

    # # Compute PCA distances
    # ref_label = condition_folders[0][1]
    # non_ref_labels = [label for _, label in condition_folders[1:]]
    
    # # distances = compute_pca_distances(CONDITION_FOLDERS, episodes)
    # # csv_path = os.path.join(condition_folders[0][0], "pca_distances.csv")

    # distances = compute_cka_similarities(condition_folders, episodes)
    # csv_path = os.path.join(condition_folders[0][0], "cka_distances.csv")


    # with open(csv_path, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["episode"] + non_ref_labels)
    #     for ep_idx, ep_num in enumerate(episodes):
    #         writer.writerow([ep_num] + list(distances[ep_idx]))
    # print(f"[io] Distances saved to {csv_path}")

    # print(f"\n[summary] Reference condition: {ref_label}")
    # print(f"[summary] Episodes analysed: {len(episodes)}")
    # print(f"[summary] Mean distances: "
    #       f"{dict(zip(non_ref_labels, np.mean(distances, axis=0)))}")

    # # Paired t-tests between non-reference conditions
    # ttest_results = run_paired_ttests(distances, non_ref_labels)

    # # Plot
    # plot_results(distances, non_ref_labels, ref_label, ttest_results)

    # 2D PCA KDE contour plot
    kde_results = plot_pca_kde(condition_folders, episodes, n_fit_episodes=20)
    
    print("\n[pca] Analyzing KDE peaks...")
    peaks = {}
    for label, res in kde_results.items():
        peak_nd = res['peak_nd']
        peaks[label] = peak_nd
        print(f"  Peak for '{label}': {np.round(peak_nd[:2], 3)}... (showing first 2 of {len(peak_nd)} PCs)")
    
    required_keys = ["Full vision", "Left eye only", "Right eye only", "Total blind"]
    if all(k in peaks for k in required_keys):
        pa = peaks["Full vision"]
        pb = peaks["Left eye only"]
        pc = peaks["Right eye only"]
        pd = peaks["Total blind"]
        
        # Vectors from 'Total blind' (pd) to other peaks
        v1 = pc - pd  # pd -> pc (Right eye)
        v2 = pb - pd  # pd -> pb (Left eye)
        v3 = pa - pd  # pd -> pa (Full vision)
        
        v1_plus_v2 = v1 + v2
        
        mag_v1_plus_v2 = np.linalg.norm(v1_plus_v2)
        mag_v3 = np.linalg.norm(v3)
        
        if mag_v1_plus_v2 > 0 and mag_v3 > 0:
            cosine_sim = np.dot(v1_plus_v2, v3) / (mag_v1_plus_v2 * mag_v3)
        else:
            cosine_sim = float('nan')
            
        print("\n[pca] Vector Analysis (Origin: Total blind):")
        print(f"  v1 (pd -> Right eye)  : [{v1[0]:.4f}, {v1[1]:.4f}]")
        print(f"  v2 (pd -> Left eye)   : [{v2[0]:.4f}, {v2[1]:.4f}]")
        print(f"  v3 (pd -> Full vision): [{v3[0]:.4f}, {v3[1]:.4f}]")
        print(f"  v1 + v2               : [{v1_plus_v2[0]:.4f}, {v1_plus_v2[1]:.4f}]")
        
        print("\n[pca] Linearity Metrics:")
        print(f"  Cosine similarity between (v1+v2) and v3: {cosine_sim:.8f}")
        print(f"  Magnitude of (v1+v2): {mag_v1_plus_v2:.4f}")
        print(f"  Magnitude of v3: {mag_v3:.4f}")
    else:
        print("\n[pca] Warning: Not all 4 required conditions found for vector analysis.")

if __name__ == "__main__":
    analysis_pca_cka(CONDITION_FOLDERS)
