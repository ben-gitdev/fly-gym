"""
Compare trajectories from multiple model folders on one plot per episode.

Reads trajectory and obstacle files from subdirectories within a base folder,
then overlays trajectories from all models on a single plot for each episode ID.
"""

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import csv


# ── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = (
    r"D:\Benquan\OneDrive MSState\OneDrive - Mississippi State University\Publications\IROS2026\materials\trajectories\small_world"
)

# Subfolder name → legend label (order determines draw & legend order)
FOLDER_LABELS = {
    "full_vision":        "Full Vision",
    "right_eye_only": "Right Eye Only",
    "left_eye_only":    "Left Eye Only",
    "blind":      "Blind",
}

# Distinct colors for each model
COLORS = ["#2196F3", "#FF9800", "#F44336"]

# ── Helpers ──────────────────────────────────────────────────────────────────

def read_obstacles(obs_file):
    """Return obstacle positions as an (N, 2+) numpy array."""
    obstacles = np.loadtxt(obs_file, delimiter=",")
    if obstacles.ndim == 1 and obstacles.size > 0:
        obstacles = obstacles.reshape(1, -1)
    return obstacles


def read_trajectory(traj_file):
    """Return trajectory as an (T, 2) numpy array of (x, y)."""
    trajectory = []
    with open(traj_file, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            trajectory.append([float(row[1]), float(row[2])])
    return np.array(trajectory)


def get_episode_ids(folder):
    """Return sorted list of episode IDs found in *folder*."""
    traj_files = glob.glob(os.path.join(folder, "trajectory_*.csv"))
    ids = []
    for tf in traj_files:
        fname = os.path.basename(tf)
        try:
            ep_id = int(fname.replace("trajectory_", "").replace(".csv", ""))
            ids.append(ep_id)
        except ValueError:
            continue
    return sorted(ids)


def load_goal_positions(folder):
    """Return dict  episode_id → (goal_x, goal_y)  from episode_summary.csv."""
    summary = os.path.join(folder, "episode_summary.csv")
    goals = {}
    if os.path.exists(summary):
        with open(summary, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ep_id = int(row["episode"])
                goals[ep_id] = (float(row["goal_x"]), float(row["goal_y"]))
    return goals


# ── Main routine ─────────────────────────────────────────────────────────────

def compare_episode(ep_id, folders, labels, colors, output_dir):
    """
    Plot trajectories from every folder for the given episode on one figure
    and save the result to *output_dir*.
    """
    # Use the first folder's obstacle file (identical across folders)
    first_folder = list(folders.values())[0]
    obs_file = os.path.join(first_folder, f"obstacles_{ep_id}.txt")
    if not os.path.exists(obs_file):
        print(f"  [skip] Missing obstacle file for episode {ep_id}")
        return

    obstacles = read_obstacles(obs_file)

    # Also load goal from the first folder that has it
    target_x, target_y = None, None
    for folder_path in folders.values():
        goals = load_goal_positions(folder_path)
        if ep_id in goals:
            target_x, target_y = goals[ep_id]
            break

    fig, ax = plt.subplots(figsize=(8, 8))

    # Walls
    arena = 7.0
    wx = [-arena, arena, arena, -arena, -arena]
    wy = [-arena, -arena, arena, arena, -arena]
    ax.plot(wx, wy, "k-", linewidth=2)

    # Obstacles
    if obstacles.size > 0:
        for obs in obstacles:
            circle = plt.Circle((obs[0], obs[1]), 0.4, color="gray", alpha=0.7)
            ax.add_patch(circle)

    # Target
    if target_x is not None:
        ax.plot(target_x, target_y, "m*", markersize=14, label="Target", zorder=10)

    # Trajectories
    for (name, folder_path), label, color in zip(folders.items(), labels, colors):
        traj_file = os.path.join(folder_path, f"trajectory_{ep_id}.csv")
        if not os.path.exists(traj_file):
            print(f"  [skip] {label}: no trajectory for episode {ep_id}")
            continue

        traj = read_trajectory(traj_file)
        if len(traj) == 0:
            continue

        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.5,
                label=label, alpha=0.85)
        # Start marker
        ax.plot(traj[0, 0], traj[0, 1], "o", color=color, markersize=7,
                zorder=5)
        # End marker
        ax.plot(traj[-1, 0], traj[-1, 1], "s", color=color, markersize=7,
                zorder=5)

    ax.set_aspect("equal")
    ax.set_xlim(-arena - 0.1, arena + 0.1)
    ax.set_ylim(-arena - 0.1, arena + 0.1)
    # ax.legend(loc="upper right", fontsize=9)
    # ax.set_title(f"Episode {ep_id}", fontsize=14)
    ax.grid(False)
    ax.axis("off")

    out_path = os.path.join(output_dir, f"compare_{ep_id}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    # Build ordered mapping  name → full path
    folders = {}
    labels = []
    for name, label in FOLDER_LABELS.items():
        path = os.path.join(BASE_DIR, name)
        if not os.path.isdir(path):
            print(f"Warning: folder not found – {path}")
            continue
        folders[name] = path
        labels.append(label)

    if not folders:
        print("No valid folders found. Exiting.")
        return

    colors = COLORS[: len(folders)]

    # Discover episode IDs from the first folder
    first_folder = list(folders.values())[0]
    episode_ids = get_episode_ids(first_folder)
    print(f"Found {len(episode_ids)} episodes: {episode_ids}")

    # Output directory (sibling to the subfolders)
    output_dir = os.path.join(BASE_DIR, "comparisons")
    os.makedirs(output_dir, exist_ok=True)

    for ep_id in episode_ids:
        print(f"Processing episode {ep_id} ...")
        compare_episode(ep_id, folders, labels, colors, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
