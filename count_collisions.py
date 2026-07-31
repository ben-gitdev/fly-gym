"""
Count collisions from trajectory files and add to episode_summary.csv.

Collisions within 10 steps of each other are counted as a single collision event.
"""

import os
import csv
import pandas as pd
import numpy as np
import math
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.astar import AStarGridPlanner, GridSpec

EVAL_DIR = os.path.join(os.path.dirname(__file__), "eval_data", "connectome_full_vision_3")


def count_collisions(trajectory_path, min_gap=10):
    """Count collision events in a trajectory file.
    
    Consecutive collisions (within `min_gap` steps) are merged into one event.
    
    Returns:
        (count, first_collision_step): tuple of collision count and the step
        number of the first collision (-1 if no collisions).
    """
    df = pd.read_csv(trajectory_path)
    collision_steps = df.index[df["collision"] == 1].tolist()
    
    if not collision_steps:
        return 0, -1
    
    first_collision_step = collision_steps[0]
    
    count = 1
    last_step = collision_steps[0]
    for step in collision_steps[1:]:
        if step - last_step >= min_gap:
            count += 1
        last_step = step
    
    return count, first_collision_step


def count_one_folder(eval_dir = EVAL_DIR):
    summary_path = os.path.join(eval_dir, "episode_summary.csv")
    
    # Read existing summary
    summary_df = pd.read_csv(summary_path)
    
    # Count collisions for each episode and calculate SPL
    collision_counts = []
    first_collision_steps = []
    spl_list = []
    avg_speed_list = []
    obstacle_in_line_list = []
    
    # Initialize planner once
    planner = AStarGridPlanner(GridSpec(arena_half_extent=7.0, cell_size=0.1, obstacle_inflate=0.1))
    
    for _, row in summary_df.iterrows():
        ep = int(row["episode"])
        traj_path = os.path.join(eval_dir, f"trajectory_{ep}.csv")
        obs_path = os.path.join(eval_dir, f"obstacles_{ep}.txt")
        
        goal_reached = bool(row.get("goal_reached", 0))
        num_steps = row.get("steps", 600) if goal_reached else 600
        
        if os.path.exists(traj_path):
            count, first_step = count_collisions(traj_path)
            collision_counts.append(count)
            first_collision_steps.append(first_step)
            
            # Load trajectory and calculate actual length
            traj_df = pd.read_csv(traj_path)
            if len(traj_df) > 0:
                start_xy = np.array([traj_df.iloc[0]["x"], traj_df.iloc[0]["y"]], dtype=np.float32)
                goal_xy = np.array([row["goal_x"], row["goal_y"]], dtype=np.float32)
                
                # Actual path length
                actual_len = 0.0
                coords = traj_df[["x", "y"]].values
                if len(coords) > 1:
                    diffs = np.diff(coords, axis=0)
                    actual_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
                
                avg_speed = actual_len / num_steps if num_steps > 0 else 0.0
                avg_speed_list.append(avg_speed)
                
                # Load obstacles
                obstacles_xy = []
                if os.path.exists(obs_path):
                    with open(obs_path, "r") as f:
                        for line in f:
                            if line.strip():
                                parts = line.strip().split(',')
                                if len(parts) >= 2:
                                    obstacles_xy.append([float(parts[0]), float(parts[1])])
                obstacles_xy = np.array(obstacles_xy, dtype=np.float32)
                if len(obstacles_xy) == 0:
                    obstacles_xy = np.zeros((0, 2), dtype=np.float32)
                
                # Check if any obstacle is directly in line between start and goal
                if len(obstacles_xy) > 0:
                    v = goal_xy - start_xy
                    w = obstacles_xy - start_xy
                    c1 = np.sum(w * v, axis=1)
                    c2 = np.sum(v * v)
                    if c2 > 0:
                        b = np.clip(c1 / c2, 0.0, 1.0)
                        proj = start_xy + b[:, np.newaxis] * v
                        dists = np.linalg.norm(obstacles_xy - proj, axis=1)
                        obstacle_in_line = bool(np.min(dists) < 0.6)
                    else:
                        dists = np.linalg.norm(obstacles_xy - start_xy, axis=1)
                        obstacle_in_line = bool(np.min(dists) < 0.6)
                else:
                    obstacle_in_line = False
                obstacle_in_line_list.append(obstacle_in_line)
                
                # Shortest path length using A*
                path = planner.plan(
                    start_xy=start_xy,
                    goal_xy=goal_xy,
                    obstacles_xy=obstacles_xy,
                    obstacle_radius=0.4,
                    smooth_path=True
                )
                
                if path is None or len(path) < 2:
                    print(f"Warning: Path not found for episode {ep}")
                    shortest_len = math.hypot(goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1])
                else:
                    path[0] = start_xy
                    shortest_len = 0.0
                    for i in range(len(path) - 1):
                        shortest_len += math.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
                shortest_len -= 0.8 # take away goal radius
                success = float(row.get("goal_reached", 0))
                if max(shortest_len, actual_len) > 0:
                    spl = success * (shortest_len / max(shortest_len, actual_len))
                else:
                    spl = 0.0
                spl_list.append(spl)
            else:
                spl_list.append(0.0)
                avg_speed_list.append(0.0)
                obstacle_in_line_list.append(False)
        else:
            print(f"Warning: {traj_path} not found, setting collisions to 0, SPL to 0")
            collision_counts.append(0)
            first_collision_steps.append(-1)
            spl_list.append(0.0)
            avg_speed_list.append(0.0)
            obstacle_in_line_list.append(False)
    
    # Convert goal_reached from true/false to 1/0
    summary_df["goal_reached"] = summary_df["goal_reached"].astype(int)
    
    # Add columns and save
    summary_df["collisions"] = collision_counts
    summary_df["first_collision_step"] = first_collision_steps
    summary_df["SPL"] = spl_list
    summary_df["average_speed"] = avg_speed_list
    summary_df["obstacle_in_line"] = obstacle_in_line_list
    summary_df.to_csv(summary_path, index=False)
    print(f"Updated {summary_path} with collision counts, SPL, avg speed, and obstacle_in_line for {len(summary_df)} episodes.")


if __name__ == "__main__":
    base_dir = "eval_data"
    
    # folders =   [
    #     "connectome_full_vision_4", "connectome_left_eye_4", "connectome_right_eye_4", "connectome_blind_4",
    #     "vision_efficientnet_robust_full_vision", "vision_efficientnet_robust_left_eye", "vision_efficientnet_robust_right_eye", "vision_efficientnet_robust_blind",
    #     "vision_mobilenet_robust_full_vision", "vision_mobilenet_robust_left_eye", "vision_mobilenet_robust_right_eye", "vision_mobilenet_robust_blind",
    #     "small_world_full_vision", "small_world_left_eye", "small_world_right_eye", "small_world_total_blind",
    # ]
    # folders = ["connectome_textured_env", "small_world_textured_env", "vision_efficientnet_robust_textured_env", "vision_mobilenet_robust_textured_env"]
    folders = [] # list of eval result folders
    for folder in folders:
        count_one_folder(os.path.join(base_dir, folder))
    
