"""
Plot per-episode top-down trajectories (walls, obstacles, path, start/end, target) from a
directory of eval rollout data produced by run_connectome_rnn_checkpoint.py /
run_vision_agent_checkpoint.py.

For each `trajectory_<episode>.csv` file found in the target directory, reads its matching
`obstacles_<episode>.txt`, and, if `episode_summary.csv` is present, that episode's goal
position, then saves a `visualization_<episode>.png` plot.
"""

import argparse
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import csv

def visualize_episode(obs_file, traj_file, output_file, target_x=None, target_y=None):
    """
    Visualizes a single episode.
    """
    # 1. Read Obstacles
    try:
        obstacles = np.loadtxt(obs_file, delimiter=",")
    except Exception as e:
        print(f"Error reading obstacles from {obs_file}: {e}")
        return

    # Handle case where there might be no obstacles or single obstacle
    if obstacles.ndim == 1 and obstacles.size > 0:
        obstacles = obstacles.reshape(1, -1)
    
    # 2. Read Trajectory
    trajectory = []
    try:
        with open(traj_file, "r") as f:
            reader = csv.reader(f)
            next(reader) # skip header
            for row in reader:
                # step, x, y
                trajectory.append([float(row[1]), float(row[2])])
        trajectory = np.array(trajectory)
    except Exception as e:
        print(f"Error reading trajectory from {traj_file}: {e}")
        return

    if len(trajectory) == 0:
        print(f"Warning: Empty trajectory in {traj_file}")
        return

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw Walls (Square from -7 to 7)
    arena_half_extent = 7.0
    wall_x = [-arena_half_extent, arena_half_extent, arena_half_extent, -arena_half_extent, -arena_half_extent]
    wall_y = [-arena_half_extent, -arena_half_extent, arena_half_extent, arena_half_extent, -arena_half_extent]
    ax.plot(wall_x, wall_y, "k-", linewidth=2, label="Walls")
    
    # Draw Obstacles (Circles, radius 0.4)
    if obstacles.size > 0:
        for obs in obstacles:
            circle = plt.Circle((obs[0], obs[1]), 0.4, color="gray", alpha=0.7)
            ax.add_patch(circle)
    
    # Draw Trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], "b-", linewidth=1.5, label="Trajectory", alpha=0.8)
    
    # Start and End points
    ax.plot(trajectory[0, 0], trajectory[0, 1], "go", label="Start", markersize=8)
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], "ro", label="End", markersize=8)

    # Target position
    if target_x is not None and target_y is not None:
        ax.plot(target_x, target_y, "m*", label="Target", markersize=14)
    
    # Settings
    ax.set_aspect("equal")
    ax.set_xlim(-arena_half_extent - 0.1, arena_half_extent + 0.1)
    ax.set_ylim(-arena_half_extent - 0.1, arena_half_extent + 0.1)
    # ax.set_title(f"Episode Visualization\n{os.path.basename(traj_file)}")
    # ax.legend(loc="upper right")
    ax.grid(False, linestyle="--", alpha=0.3)
    ax.axis('off')
    
    # 4. Save
    plt.savefig(output_file, dpi=100)
    plt.close(fig)
    print(f"Saved visualization to {output_file}")

def main():
   
    target_dir = "eval_data/small_world_textured_env"
    if not os.path.exists(target_dir):
        print(f"Directory not found: {target_dir}")
        return

    # Load episode summary for target positions
    summary_file = os.path.join(target_dir, "episode_summary.csv")
    goal_positions = {}  # ep_id -> (goal_x, goal_y)
    if os.path.exists(summary_file):
        with open(summary_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ep_id = int(row["episode"])
                goal_positions[ep_id] = (float(row["goal_x"]), float(row["goal_y"]))
        print(f"Loaded {len(goal_positions)} goal positions from episode_summary.csv")
    else:
        print(f"Warning: {summary_file} not found, targets will not be drawn.")

    # Find pairs of obstacle and trajectory files
    # Assuming naming convention: obstacles_N.txt and trajectory_N.csv
    traj_files = glob.glob(os.path.join(target_dir, "trajectory_*.csv"))
    
    if not traj_files:
        print(f"No trajectory files found in {target_dir}")
        return

    print(f"Found {len(traj_files)} episodes to process...")

    for traj_file in traj_files:
        # Extract ID
        filename = os.path.basename(traj_file)
        # expected format: trajectory_123.csv
        try:
            ep_id_str = filename.replace("trajectory_", "").replace(".csv", "")
            ep_id = int(ep_id_str)
        except ValueError:
            print(f"Skipping malformed filename: {filename}")
            continue
            
        obs_file = os.path.join(target_dir, f"obstacles_{ep_id}.txt")
        output_file = os.path.join(target_dir, f"visualization_{ep_id}.png")
        
        if not os.path.exists(obs_file):
            print(f"Missing obstacle file for episode {ep_id}: {obs_file}")
            continue

        # Get target position for this episode
        target_x, target_y = goal_positions.get(ep_id, (None, None))
        visualize_episode(obs_file, traj_file, output_file, target_x, target_y)

if __name__ == "__main__":
    main()
