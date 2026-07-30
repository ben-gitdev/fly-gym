"""
Script to run trained vision-based agents (EfficientNet, MobileNet, Dual-Backbone)
from checkpoints.
Uses the same preprocessing pipeline as train_visionnet_dagger.py to ensure
observation processing matches training exactly.
"""

import argparse
import os
import time
import csv
import numpy as np
import torch
import cv2
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.mujoco_two_cam_env_random_obstacles import MuJoCoTwoCamEnv
from agents.efficientnet_agent import EfficientNetAgent
from agents.mobilenet_agent import MobileNetAgent
from agents.dual_backbone_agent import DualBackboneAgent
from core.utils import get_device

# Import config constants
from shared_config import (
    ENV_WIDTH,
    ENV_HEIGHT,
    MAX_EPISODE_STEPS,
    N_OBSTACLES,
    ARENA_HALF_EXTENT,
    DTYPE,
    CHECKPOINT_DIR,
)


# ---- Activation Recorder (forward-hook based) ----
class ActivationRecorder:
    """Register forward hooks on selected modules to capture activations."""
    def __init__(self):
        self.activations = {}   # name -> tensor (last step)
        self._hooks = []

    def register(self, model, module_names=None):
        """Hook into named modules. If module_names is None, hook all."""
        for name, module in model.named_modules():
            if module_names is None or name in module_names:
                self._hooks.append(
                    module.register_forward_hook(self._make_hook(name))
                )

    def _make_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach().cpu()
        return hook

    def snapshot(self):
        """Return a copy of current activations as numpy dict."""
        return {k: v.squeeze(0).numpy() for k, v in self.activations.items()}

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---- Shared sensor preprocessing ----
def _preprocess_sensors(obs, dtype=np.float32):
    sensors = obs["sensors"]
    wind_dir = sensors.get("wind_direction", np.zeros(2, dtype=dtype))
    col = sensors.get("collision", np.array([0.0], dtype=dtype))
    wind_dir = np.asarray(wind_dir, dtype=dtype)
    col = np.asarray(col, dtype=dtype)
    if wind_dir.ndim == 0: wind_dir = np.expand_dims(wind_dir, axis=0)
    if col.ndim == 0: col = np.expand_dims(col, axis=0)
    return wind_dir.astype(dtype), col.astype(dtype)

# ---- Preprocessing for single-backbone agents ----
def preprocess_obs_cpu(obs, dtype=np.float32, vision=[True, True]):
    left_small = cv2.resize(obs["cam_left"], (30, 30), interpolation=cv2.INTER_AREA)
    right_small = cv2.resize(obs["cam_right"], (30, 30), interpolation=cv2.INTER_AREA)
    left_gray = cv2.cvtColor(left_small, cv2.COLOR_RGB2GRAY) * 0.0 if not vision[0] else cv2.cvtColor(left_small, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right_small, cv2.COLOR_RGB2GRAY) * 0.0 if not vision[1] else cv2.cvtColor(right_small, cv2.COLOR_RGB2GRAY)
    combined = np.hstack([left_gray, right_gray])
    img_out = combined[np.newaxis, :, :]
    wind_dir, col = _preprocess_sensors(obs, dtype)
    return {
        "img": img_out.astype(np.uint8),
        "wind_direction": wind_dir,
        "collision": col,
    }

# ---- Preprocessing for dual-backbone agents ----
def preprocess_obs_cpu_dual(obs, dtype=np.float32, vision=[True, True]):
    left_small = cv2.resize(obs["cam_left"], (30, 30), interpolation=cv2.INTER_AREA)
    right_small = cv2.resize(obs["cam_right"], (30, 30), interpolation=cv2.INTER_AREA)
    left_gray = cv2.cvtColor(left_small, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right_small, cv2.COLOR_RGB2GRAY)
    img_left = left_gray[np.newaxis, :, :] * 0.0 if not vision[0] else left_gray[np.newaxis, :, :]
    img_right = right_gray[np.newaxis, :, :] * 0.0 if not vision[1] else right_gray[np.newaxis, :, :]
    wind_dir, col = _preprocess_sensors(obs, dtype)
    return {
        "img_left": img_left.astype(np.uint8),
        "img_right": img_right.astype(np.uint8),
        "wind_direction": wind_dir,
        "collision": col,
    }


def maybe_show_cameras(obs):
    if cv2 is None: return
    try:
        left_gray = cv2.cvtColor(obs["cam_left"], cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(obs["cam_right"], cv2.COLOR_RGB2GRAY)
        frame = np.hstack([left_gray, right_gray])
        cv2.imshow("Agent View (Left | Right)", frame)
        cv2.waitKey(1)
    except Exception:
        pass

def run_episode(env, agent, device, dtype, render=False, render_skip=1,
                episode_idx=0, save_dir=None, seed=None, vision=None,
                save_internal_states="none"):
    obs, info = env.reset(seed=seed)
    if hasattr(agent, "reset_vision_state"):
        agent.reset_vision_state()
    
    is_dual = isinstance(agent, DualBackboneAgent)
    preprocess_fn = preprocess_obs_cpu_dual if is_dual else preprocess_obs_cpu

    # --- Record Goal Location ---
    goal_xy = env._goal_xy.copy()
    # --------------------------

    # --- Record Obstacles ---
    if save_dir is not None:
        active_obstacles = env._obstacle_xy[:env.n_obstacles]
        obs_file = os.path.join(save_dir, f"obstacles_{episode_idx}.txt")
        np.savetxt(obs_file, active_obstacles, fmt="%.4f", delimiter=",")
    # ------------------------

    # Initialize hidden state (1, hidden_size)
    h = torch.zeros(1, agent.hidden_size, device=device, dtype=dtype)
    
    steps = 0
    total_reward = 0.0
    trajectory = []
    internal_state_snapshots = []

    # --- Set up activation hooks ---
    recorder = None
    if save_internal_states != "none" and save_dir is not None:
        recorder = ActivationRecorder()
        # Determine which modules to hook
        hook_names = []
        if save_internal_states == "all":
            # Backbone feature blocks
            for name, _ in agent.backbone.features.named_children():
                hook_names.append(f"backbone.features.{name}")
            # Average pool
            hook_names.append("backbone.avgpool")
        # GRU and policy head (always included when recording)
        hook_names.append("gru")
        hook_names.append("policy_head")
        recorder.register(agent, module_names=hook_names)
        print(f"[io] Recording internal states ({save_internal_states}) for episode "
              f"{episode_idx} ({len(hook_names)} modules hooked)")
    # --------------------------------
    
    np_dtype = np.float32 if dtype == torch.float32 else np.float16
    use_cuda = device.type == 'cuda'
    
    # Pre-allocate pinned memory buffers for lower-latency GPU transfers
    if use_cuda:
        if is_dual:
            pin_img_l = torch.empty(1, 1, 30, 30, dtype=torch.uint8).pin_memory()
            pin_img_r = torch.empty(1, 1, 30, 30, dtype=torch.uint8).pin_memory()
        else:
            pin_img = torch.empty(1, 1, 30, 60, dtype=torch.uint8).pin_memory()
        pin_wind = torch.empty(1, 2, dtype=torch.float32).pin_memory()
        pin_col = torch.empty(1, 1, dtype=torch.float32).pin_memory()
    
    # Match training loop: step first with initial zero action
    action_exec = np.zeros(2, dtype=np.float32)
    
    if render:
        env.render()
        
    while steps < MAX_EPISODE_STEPS:
        # Step environment first (matches training rollout_episode)
        obs, reward, done, trunc, info = env.step(action_exec)
        steps += 1
        total_reward += reward
        
        # --- Record Robot Position and Collision ---
        robot_pos = env._base_xy()
        collision = obs["sensors"]["collision"]
        trajectory.append([robot_pos[0], robot_pos[1], int(collision)])
        # --------------------------------------------
        
        if render and steps % render_skip == 0:
            env.render()
            maybe_show_cameras(obs)
        
        # CPU preprocessing (matches training pipeline)
        xs_cpu = preprocess_fn(obs, dtype=np_dtype, vision=vision)
        
        # Transfer to GPU
        if use_cuda:
            if is_dual:
                pin_img_l[0] = torch.from_numpy(xs_cpu["img_left"])
                pin_img_r[0] = torch.from_numpy(xs_cpu["img_right"])
            else:
                pin_img[0] = torch.from_numpy(xs_cpu["img"])
            pin_wind[0] = torch.from_numpy(xs_cpu["wind_direction"])
            pin_col[0] = torch.from_numpy(xs_cpu["collision"])
            if is_dual:
                xs_gpu = {
                    "img_left": pin_img_l.to(device, non_blocking=True),
                    "img_right": pin_img_r.to(device, non_blocking=True),
                    "wind_direction": pin_wind.to(device, non_blocking=True),
                    "collision": pin_col.to(device, non_blocking=True),
                }
            else:
                xs_gpu = {
                    "img": pin_img.to(device, non_blocking=True),
                    "wind_direction": pin_wind.to(device, non_blocking=True),
                    "collision": pin_col.to(device, non_blocking=True),
                }
        else:
            if is_dual:
                xs_gpu = {
                    "img_left": torch.from_numpy(xs_cpu["img_left"]).unsqueeze(0).to(device),
                    "img_right": torch.from_numpy(xs_cpu["img_right"]).unsqueeze(0).to(device),
                    "wind_direction": torch.from_numpy(xs_cpu["wind_direction"]).unsqueeze(0).to(device, dtype=dtype),
                    "collision": torch.from_numpy(xs_cpu["collision"]).unsqueeze(0).to(device, dtype=dtype),
                }
            else:
                xs_gpu = {
                    "img": torch.from_numpy(xs_cpu["img"]).unsqueeze(0).to(device),
                    "wind_direction": torch.from_numpy(xs_cpu["wind_direction"]).unsqueeze(0).to(device, dtype=dtype),
                    "collision": torch.from_numpy(xs_cpu["collision"]).unsqueeze(0).to(device, dtype=dtype),
                }
        
        # Inference step
        with torch.no_grad():
            h, action = agent.step(h, None, x=xs_gpu)

        # --- Capture activations ---
        if recorder is not None:
            internal_state_snapshots.append(recorder.snapshot())
        # ----------------------------

        action_exec = action.squeeze(0).cpu().numpy()
        
        if done or trunc:
            break
    
    # --- Save Trajectory ---
    if save_dir is not None:
        traj_file = os.path.join(save_dir, f"trajectory_{episode_idx}.csv")
        with open(traj_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "x", "y", "collision"])
            for i, row in enumerate(trajectory):
                writer.writerow([i, row[0], row[1], row[2]])
    # -----------------------

    # --- Save Internal States ---
    if recorder is not None:
        recorder.remove()
        if internal_state_snapshots:
            # Stack per-module arrays across timesteps: {name: (T, ...)}
            all_keys = internal_state_snapshots[0].keys()
            stacked = {k: np.stack([s[k] for s in internal_state_snapshots], axis=0)
                       for k in all_keys}
            is_file = os.path.join(save_dir, f"internal_states_{episode_idx}.npz")
            np.savez_compressed(is_file, **stacked)
            print(f"[io] Saved internal states to {is_file}")
            for k, v in stacked.items():
                print(f"     {k}: {v.shape}")
    # ----------------------------

    # --- Determine Goal Reached ---
    final_xy = env._base_xy()
    final_dist = float(np.linalg.norm(env._goal_xy - final_xy))
    goal_reached = final_dist < env.goal_radius
    # ------------------------------

    return info, total_reward, steps, goal_reached, goal_xy

def _detect_model_type(filename: str) -> str:
    """Infer model_type from checkpoint filename."""
    name = filename.lower()
    if "dual_mobilenet" in name:
        return "dual_mobilenet"
    elif "dual_efficientnet" in name:
        return "dual_efficientnet"
    elif "mobilenet" in name:
        return "mobilenet"
    else:
        return "efficientnet"

def _build_agent(model_type: str, device, dtype):
    """Construct the correct agent class for a given model_type."""
    common = dict(action_dim=2, hidden_size=256, dtype=dtype)
    if model_type == "efficientnet":
        agent = EfficientNetAgent(**common)
    elif model_type == "mobilenet":
        agent = MobileNetAgent(**common)
    elif model_type == "dual_efficientnet":
        agent = DualBackboneAgent(backbone_type="efficientnet", **common)
    elif model_type == "dual_mobilenet":
        agent = DualBackboneAgent(backbone_type="mobilenet", **common)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return agent.to(device=device, dtype=dtype)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation rollouts for a trained vision-based agent "
                     "(EfficientNet/MobileNet/Dual-Backbone) checkpoint."
    )
    parser.add_argument(
        "checkpoint",
        help="Path to the .pt checkpoint to evaluate. Required: this used to default to "
             "'efficientnet_dagger_final_robust.pt' with a silent 'pick any checkpoint found' "
             "fallback, which made it ambiguous which trained model actually produced a given "
             "result (see CLEANUP_PLAN.md item #6). Now the caller must say explicitly.",
    )
    parser.add_argument(
        "--model-type",
        choices=["efficientnet", "mobilenet", "dual_efficientnet", "dual_mobilenet"],
        default=None,
        help="Model architecture. Default: auto-detected from the checkpoint filename "
             "(see _detect_model_type).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path!r}. Available checkpoints in "
            f"{CHECKPOINT_DIR}/: {[f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')]}"
        )
    model_type = args.model_type or _detect_model_type(os.path.basename(checkpoint_path))
    vision = [False, False]

    episodes = 500
    render = True
    # Internal State Recording Config
    save_internal_state_episodes = [53]  # e.g., [1, 5, 10] to save those episodes
    internal_state_mode = "gru_mlp"  # "none", "gru_mlp", or "all"
    # ---------------------

    device = get_device()
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path} (model_type={model_type})")

    # 1. Initialize Environment
    render_mode = "human" if render else None
    
    env = MuJoCoTwoCamEnv(
        width=ENV_WIDTH,
        height=ENV_HEIGHT,
        max_episode_steps=MAX_EPISODE_STEPS,
        n_obstacles=N_OBSTACLES,
        arena_half_extent=ARENA_HALF_EXTENT,
        render_mode=render_mode,
    )
    
    # 2. Initialize Agent
    print(f"Initializing {model_type} agent...")
    agent = _build_agent(model_type, device, DTYPE)
        
    # 3. Load Checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(checkpoint)
    agent.eval()

    # 4. Prepare data directory
    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("eval_data", f"vision_{model_type}_{timestr}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[main] Saving episode data to: {save_dir}")

    # 5. Run Loop
    success_count = 0
    distances = []
    episode_summaries = []
    
    print(f"Running {episodes} episodes...")
    
    render_skip = 1
    
    try:
        # for i in range(52,53):
        for i in range(episodes):
            ep = i + 1
            print(f"Episode {ep}/{episodes}...", end=" ", flush=True)
            
            info, reward, steps, goal_reached, goal_xy = run_episode(
                env, agent, device, DTYPE, render, render_skip,
                episode_idx=ep, save_dir=save_dir, seed=ep, vision=vision,
                save_internal_states=internal_state_mode if ep in save_internal_state_episodes else "none",
            )
            
            dist = info["dist_to_goal"]
            status = "SUCCESS" if goal_reached else "FAIL"
            if info.get("stalled"): status = "STALLED"
            
            print(
                f"[{status}] Steps: {steps}, Reward: {reward:.2f}, Final Dist: {dist:.2f} "
                f"| goal=({goal_xy[0]:.2f}, {goal_xy[1]:.2f})"
            )
            
            if goal_reached:
                success_count += 1
            distances.append(dist)
            episode_summaries.append({
                "episode": ep,
                "goal_x": goal_xy[0],
                "goal_y": goal_xy[1],
                "goal_reached": goal_reached,
                "return": reward,
                "steps": steps,
                "final_dist": dist,
            })
            
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # --- Save Episode Summary CSV ---
        summary_file = os.path.join(save_dir, "episode_summary.csv")
        with open(summary_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "goal_x", "goal_y", "goal_reached", "return", "steps", "final_dist"])
            writer.writeheader()
            writer.writerows(episode_summaries)
        print(f"[main] Episode summary saved to: {summary_file}")
        # --------------------------------
        env.close()
        if render and cv2 is not None:
             cv2.destroyAllWindows()
             
    # 6. Summary
    if len(distances) > 0:
        print("\n--- Summary ---")
        print(f"Success Rate: {success_count}/{len(distances)} ({success_count/len(distances)*100:.1f}%)")
        print(f"Avg Final Distance: {np.mean(distances):.2f}")
    
if __name__ == "__main__":
    main()
