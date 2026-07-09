"""
Roll out a trained connectome RNN policy from a saved checkpoint.

This mirrors the architecture and preprocessing used during DAgger training
in `train_connectome_rnn_dagger.py`, so checkpoints from that script can be
evaluated directly.
"""

from __future__ import annotations

import os
import time
import csv
import numpy as np
from typing import Optional, Tuple, List
import pandas as pd

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
    PHOTORECEPTOR_LEFT_CSV,
    PHOTORECEPTOR_RIGHT_CSV,
    TACTILE_LEFT_CSV,
    TACTILE_RIGHT_CSV,
    DESCENDING_NEURONS_CSV,
    CELL_TYPES_CSV,
    WIND_SENSING_CSV, # NEW
    TARGET_RHO,
    LEAK_ALPHA,
    ACTIVATION,
    BATCH_CHUNK,
    ROW_TILE_SIZE,
)
from core.utils import build_connectome_cell, obs_to_torch


# EDGE_PATH = "connectomes/drosophila adult connectome/connections_princeton_random.csv"
# CHECKPOINT = "checkpoints/connectome_rnn_dagger_princeton_random_full_vision.pt"
# EDGE_PATH = "connectomes/drosophila adult connectome/connections_princeton.csv"
CHECKPOINT = "checkpoints/connectome_rnn_dagger_princeton.pt"
# CHECKPOINT = "checkpoints/connectome_rnn_dagger_small_world_full_vision.pt"
RECORD_CSV = None#"connectomes/drosophila adult connectome/moonwalker_neurons.csv"       # e.g., "neurons_to_record.csv"
OVERWRITE_CSV = None#"connectomes/drosophila adult connectome/moonwalker_neurons.csv"    # e.g., "neurons_to_overwrite.csv"
END_ON_COLLISION = False
MAX_EPISODE_STEPS = 600

# Reward-shaping constants for env construction during evaluation rollouts
# (kept in sync with the values used during DAgger training).
CTRL_PENALTY = 0.001
TIME_PENALTY = 0.01
PROG_SCALE = 10.0


def _make_env(render_mode: Optional[str], n_obstacles: int) -> MuJoCoTwoCamEnv:
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
    )


def _load_agent(checkpoint_path: str, device: torch.device, dtype: torch.dtype) -> ConnectomeAgent:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cell, pr_positions, input_splits, id2idx = build_connectome_cell(
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
        h, action = agent.step(h, obs_t)
        
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



def run_one_configuration(checkpoint, vision, rendermode = None):
    device = get_device()
    dtype = DTYPE
    
    # --- Configuration ---
    # checkpoint = CHECKPOINT
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

    env = _make_env(render_mode=render_mode, n_obstacles=20)
    agent, id2idx = _load_agent(checkpoint, device=device, dtype=dtype)

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

if __name__ == "__main__":
    checkpoint = CHECKPOINT
    rendermode = "human"
    # dir1 = run_one_configuration(checkpoint, vision=[True, True], rendermode = rendermode)
    # dir2 = run_one_configuration(checkpoint, vision=[False, True], rendermode = rendermode)
    # dir3 = run_one_configuration(checkpoint, vision=[True, False], rendermode = rendermode)
    dir4 = run_one_configuration(checkpoint, vision=[False, False], rendermode = rendermode)

    # from analysis_pca_statistics import analysis_pca_cka
    # condition_folders = [(dir1, "Full vision"), (dir2, "Right eye only"),(dir3, "Left eye only"), (dir4, "Total blindness")]
    # analysis_pca_cka(condition_folders)
