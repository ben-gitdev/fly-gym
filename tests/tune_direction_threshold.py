
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

# This script lives in tests/, but imports repo-root modules directly (no package
# prefix) -- add the repo root (this file's parent directory) to sys.path so that
# works regardless of the caller's current working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import everything needed from the training script
from train_connectome_rnn_dagger import (
    PathAnalyzer,
    get_device,
    build_connectome_cell,
    ConnectomeAgent,
    PlannerAnalyticTeacher,
    MuJoCoTwoCamEnv,
    # Constants
    ENV_WIDTH, ENV_HEIGHT, MAX_EPISODE_STEPS, N_OBSTACLES, ARENA_HALF_EXTENT, RENDER_MODE, END_ON_COLLISION,
    DIRECTION_THRESHOLD_DEG, CONSECUTIVE_SEGMENTS_N, COLLISION_LOOKBACK, COLLISION_RECOVERY_WINDOW,
    EDGE_PATH, PHOTORECEPTOR_LEFT_CSV, PHOTORECEPTOR_RIGHT_CSV,
    TACTILE_LEFT_CSV, TACTILE_RIGHT_CSV, DESCENDING_NEURONS_CSV, CELL_TYPES_CSV, WIND_SENSING_CSV,
    TARGET_RHO, LEAK_ALPHA, ACTIVATION, BATCH_CHUNK, ROW_TILE_SIZE, INPUT_SCALE_INIT, DTYPE, TRAIN_RNN_WEIGHTS, TRAIN_READOUT_HEAD
)

END_ON_COLLISION = False
DIRECTION_THRESHOLD_DEG = 1.0


def rollout_episode(env, teacher, agent, beta, device, dtype, beta_noise):
    """Single-episode, pure-teacher-drive rollout for this diagnostic script.

    train_connectome_rnn_dagger.py used to have a rollout_episode() with this exact
    signature; it was replaced by the batched, multi-env rollout_and_collect_balanced()
    used for real DAgger training/collection, which has a completely different interface
    (buffer-filling across N envs, not "give me back one episode's raw trajectory") and
    isn't a drop-in substitute for this script's needs. This script only ever calls with
    beta=1.0/beta_noise=0.0 (pure teacher drive, for a clean path to tune
    DIRECTION_THRESHOLD_DEG against), so it only implements that case -- `agent` is
    accepted for call-site compatibility but unused, since the teacher alone drives beta=1.0.
    """
    if beta != 1.0 or beta_noise != 0.0:
        raise NotImplementedError(
            "This script's rollout_episode() only supports pure teacher-drive "
            "(beta=1.0, beta_noise=0.0); it does not reimplement agent-blended rollout "
            "or noise injection from rollout_and_collect_balanced()."
        )
    obs, _ = env.reset()
    teacher.reset()
    raw_obs_list = [obs]
    teacher_actions = []
    for _ in range(MAX_EPISODE_STEPS):
        action = teacher.act(env)
        if action is None:
            return raw_obs_list, teacher_actions, len(teacher_actions) > 0
        teacher_actions.append(action.copy())
        obs, _, done, trunc, _ = env.step(action)
        if done or trunc:
            break
        raw_obs_list.append(obs)
    return raw_obs_list, teacher_actions, True


def main():
    device = get_device()
    dtype = DTYPE
    print(f"[device] Using {device} ({dtype})")

    # 1. Setup Environment
    env = MuJoCoTwoCamEnv(
        width=ENV_WIDTH,
        height=ENV_HEIGHT,
        max_episode_steps=MAX_EPISODE_STEPS,
        n_obstacles=N_OBSTACLES,
        arena_half_extent=ARENA_HALF_EXTENT,
        render_mode=RENDER_MODE,
        end_on_collision=END_ON_COLLISION,
    )

    # 2. Setup Agent (Dummy weights are fine, we rely on Teacher for path)
    print("Building cell...")
    cell, pr_positions, input_splits, _id2idx = build_connectome_cell(
        edge_path=EDGE_PATH,
        device=device,
        dtype=dtype,
        photoreceptor_left_csv=PHOTORECEPTOR_LEFT_CSV,
        photoreceptor_right_csv=PHOTORECEPTOR_RIGHT_CSV,
        tactile_left_csv=TACTILE_LEFT_CSV,
        tactile_right_csv=TACTILE_RIGHT_CSV,
        descending_neurons_csv=DESCENDING_NEURONS_CSV,
        cell_types_csv=CELL_TYPES_CSV,
        wind_sensing_csv=WIND_SENSING_CSV,
        target_rho=TARGET_RHO,
        leak_alpha=LEAK_ALPHA,
        activation=ACTIVATION,
        train_rnn_weights=TRAIN_RNN_WEIGHTS,
        train_readout_head=TRAIN_READOUT_HEAD,
        batch_chunk=BATCH_CHUNK,
        row_tile_size=ROW_TILE_SIZE,
    )
    
    agent = ConnectomeAgent(
        cell,
        photoreceptor_positions=pr_positions,
        input_splits=input_splits,
        dtype=dtype,
        input_scale_init=INPUT_SCALE_INIT
    ).to(device)

    # 3. Setup Teacher
    teacher = PlannerAnalyticTeacher(
        arena_half_extent=env.arena,
        cell_size=0.1,
        robot_radius=0.2,
        safety_margin=0.01,
        obstacle_box_half=(0.4, 0.4),
        k_nearest_obs=5,
        device="cpu",
    )

    # 4. Initialize Analyzer
    analyzer = PathAnalyzer(
        direction_threshold_deg=DIRECTION_THRESHOLD_DEG,
        consecutive_n=CONSECUTIVE_SEGMENTS_N,
        collision_lookback=COLLISION_LOOKBACK,
        collision_recovery=COLLISION_RECOVERY_WINDOW,
    )
    
    print(f"\nCurrent Settings:")
    print(f"  DIRECTION_THRESHOLD_DEG: {DIRECTION_THRESHOLD_DEG}")
    print(f"  CONSECUTIVE_SEGMENTS_N:  {CONSECUTIVE_SEGMENTS_N}")
    print("\nRunning collection episode (Teacher Only)...")

    # 5. Run Rollout
    # Use beta=1.0 to get pure Teacher behavior (ideal paths)
    # Use beta_noise=0.0 for clean paths
    raw_obs_list, teacher_actions, valid = rollout_episode(
        env, teacher, agent, beta=1.0, device=device, dtype=dtype, beta_noise=0.0
    )

    if not valid:
        print("Episode failed (Teacher could not act). Retrying might be needed.")
        env.close()
        return

    n_steps = len(raw_obs_list)
    print(f"Collected {n_steps} steps.")

    # 6. Analyze Direction Changes
    xy_path = analyzer.extract_xy_path(raw_obs_list)
    
    # Calculate "Max Angle Change" for each time step i
    # This mirrors Logic in PathAnalyzer._classify_segment
    metric_values = []
    
    for i in range(len(xy_path)):
        # _classify_segment logic: look ahead consecutive_n steps
        max_change = 0.0
        # Re-implement inner loop
        for k in range(i, i + analyzer.consecutive_n):
            if k + 2 >= len(xy_path):
                break
            v1 = xy_path[k + 1] - xy_path[k]
            v2 = xy_path[k + 2] - xy_path[k + 1]
            angle_change = abs(analyzer._angle_between_vectors(v1, v2))
            max_change = max(max_change, angle_change)
        
        metric_values.append(np.rad2deg(max_change))

    metric_values = np.array(metric_values)
    
    # 7. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Trajectory
    x = xy_path[:, 0]
    y = xy_path[:, 1]
    
    # Color segments based on current threshold classification
    above_threshold = metric_values > DIRECTION_THRESHOLD_DEG
    
    ax1.plot(x, y, color='gray', alpha=0.5, label='Path')
    ax1.scatter(x[above_threshold], y[above_threshold], color='red', s=10, label=f'Turn (> {DIRECTION_THRESHOLD_DEG}°)')
    ax1.scatter(x[~above_threshold], y[~above_threshold], color='blue', s=2, label='Straight')
    ax1.scatter(x[0], y[0], color='green', marker='^', s=100, label='Start')
    ax1.scatter(x[-1], y[-1], color='black', marker='x', s=100, label='End')
    ax1.set_title("Trajectory Classification")
    ax1.set_aspect('equal')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Metric vs Time
    steps = np.arange(len(metric_values))
    ax2.plot(steps, metric_values, label='Max Angle Change (5-step window)')
    ax2.axhline(y=DIRECTION_THRESHOLD_DEG, color='r', linestyle='--', label=f'Current Threshold ({DIRECTION_THRESHOLD_DEG}°)')
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Max Angle Change (deg)")
    ax2.set_title(f"Turn Metric Analysis\nN={CONSECUTIVE_SEGMENTS_N} segments lookahead")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    save_path = "direction_tuning_plot.png"
    plt.savefig(save_path)
    print(f"\nPlot saved to {save_path}")
    
    # Statistics
    print("\nStatistics:")
    print(f"  Mean Change: {np.mean(metric_values):.4f}")
    print(f"  Median Change: {np.median(metric_values):.4f}")
    print(f"  Max Change: {np.max(metric_values):.4f}")
    print(f"  Steps > Threshold: {np.sum(metric_values > DIRECTION_THRESHOLD_DEG)} / {len(metric_values)} ({np.mean(metric_values > DIRECTION_THRESHOLD_DEG)*100:.1f}%)")
    
    # Interactive show if possible
    # plt.show() # Commented out for headless environments, remove if running locally with UI

    env.close()

if __name__ == "__main__":
    main()
