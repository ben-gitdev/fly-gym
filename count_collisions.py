"""
Count collisions from trajectory files and add to episode_summary.csv.

Collisions within 10 steps of each other are counted as a single collision event.
"""

import os
import csv
import pandas as pd

EVAL_DIR = os.path.join(os.path.dirname(__file__), "eval_data", "small_world_total_blind")


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


def main():
    summary_path = os.path.join(EVAL_DIR, "episode_summary.csv")
    
    # Read existing summary
    summary_df = pd.read_csv(summary_path)
    
    # Count collisions for each episode
    collision_counts = []
    first_collision_steps = []
    for _, row in summary_df.iterrows():
        ep = int(row["episode"])
        traj_path = os.path.join(EVAL_DIR, f"trajectory_{ep}.csv")
        if os.path.exists(traj_path):
            count, first_step = count_collisions(traj_path)
            collision_counts.append(count)
            first_collision_steps.append(first_step)
        else:
            print(f"Warning: {traj_path} not found, setting collisions to 0")
            collision_counts.append(0)
            first_collision_steps.append(-1)
    
    # Convert goal_reached from true/false to 1/0
    summary_df["goal_reached"] = summary_df["goal_reached"].astype(int)
    
    # Add collisions and first collision step columns and save
    summary_df["collisions"] = collision_counts
    summary_df["first_collision_step"] = first_collision_steps
    summary_df.to_csv(summary_path, index=False)
    print(f"Updated {summary_path} with collision counts for {len(summary_df)} episodes.")


if __name__ == "__main__":
    main()
