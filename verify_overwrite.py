import os
import torch
import numpy as np
import pandas as pd
from run_connectome_rnn_checkpoint import (
    _make_env, _load_agent, rollout_episode, get_device, DTYPE, PlannerAnalyticTeacher
)

def test_overwrite():
    print("Setting up verification...")
    device = get_device()
    dtype = DTYPE
    checkpoint = "checkpoints/connectome_rnn_dagger.pt"
    
    # Setup
    env = _make_env(render_mode=None, n_obstacles=20)
    agent, id2idx = _load_agent(checkpoint, device=device, dtype=dtype)
    teacher = PlannerAnalyticTeacher(
        arena_half_extent=env.arena, cell_size=0.1, robot_radius=0.2,
        safety_margin=0.01, obstacle_box_half=(0.4, 0.4), k_nearest_obs=5, device="cpu"
    )
    
    # Pick a neuron to overwrite (arbitrary index if load fails, or just 0)
    # We'll valid indices from the agent
    target_idx = 0
    overwrite_val = 0.6
    
    save_dir = "test_verification_data"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Running episode with overwrite: Index {target_idx} -> {overwrite_val}")
    
    rollout_episode(
        env, agent, teacher, device, dtype, 
        render=False, show_path=False,
        episode_idx=1, save_dir=save_dir,
        record_indices=[target_idx],
        overwrite_indices=[target_idx],
        overwrite_value=overwrite_val,
        overwrite_interval=1,
        overwrite_steps=1000 # Always overwrite
    )
    
    # Check Result
    csv_path = os.path.join(save_dir, "activity_1.csv")
    if not os.path.exists(csv_path):
        print("Error: Activity file not generated.")
        return

    data = np.loadtxt(csv_path)
    # data is (Steps, 1) usually
    if data.ndim == 1: data = data[:, None]
    
    mean_val = np.mean(data)
    print(f"Recorded Mean: {mean_val:.4f}")
    print(f"Target Value:  {overwrite_val:.4f}")
    print(f"Difference:    {mean_val - overwrite_val:.4f}")
    
    if abs(mean_val - overwrite_val) < 0.1:
        print("SUCCESS: Recorded value is close to target (accounting for single-step drift).")
    else:
        print("WARNING: high drift/divergence observed.")

if __name__ == "__main__":
    test_overwrite()
