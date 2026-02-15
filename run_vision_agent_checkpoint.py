"""
Script to run trained vision-based agents (EfficientNet, MobileNet) from checkpoints.
Uses the same preprocessing pipeline as train_visionnet_dagger.py to ensure
observation processing matches training exactly.
"""

import os
import time
import numpy as np
import torch
import cv2
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.mujoco_two_cam_env_random_obstacles import MuJoCoTwoCamEnv
from agents.efficientnet_agent import EfficientNetAgent
from agents.mobilenet_agent import MobileNetAgent
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

# ---- Inline preprocessing (matches train_visionnet_dagger.preprocess_obs_cpu) ----
def preprocess_obs_cpu(obs, dtype=np.float32):
    """
    Process observation on CPU to match the training pipeline exactly.
    Resizes images to 30x30, converts to grayscale, stacks them, 
    and extracts wind_direction + collision sensors.
    Returns dict of numpy arrays ready for GPU transfer.
    """
    # 1. Image Processing (CPU OpenCV)
    left_small = cv2.resize(obs["cam_left"], (30, 30), interpolation=cv2.INTER_AREA)
    right_small = cv2.resize(obs["cam_right"], (30, 30), interpolation=cv2.INTER_AREA)
    
    left_gray = cv2.cvtColor(left_small, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right_small, cv2.COLOR_RGB2GRAY)
    
    # Format: (1, 30, 60) — horizontal stack with channel dim
    combined = np.hstack([left_gray, right_gray])
    img_out = combined[np.newaxis, :, :]
    
    # 2. Sensor Processing
    sensors = obs["sensors"]
    wind_dir = sensors.get("wind_direction", np.zeros(2, dtype=dtype))
    
    col = sensors.get("collision", np.array([0.0], dtype=dtype))
    
    wind_dir = np.asarray(wind_dir, dtype=dtype)
    col = np.asarray(col, dtype=dtype)
    
    if wind_dir.ndim == 0: wind_dir = np.expand_dims(wind_dir, axis=0)
    if col.ndim == 0: col = np.expand_dims(col, axis=0)
    
    return {
        "img": img_out.astype(np.uint8),
        "wind_direction": wind_dir.astype(dtype),
        "collision": col.astype(dtype),
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

def run_episode(env, agent, device, dtype, render=False, render_skip=1):
    obs, info = env.reset()
    if hasattr(agent, "reset_vision_state"):
        agent.reset_vision_state()
    
    # Initialize hidden state (1, hidden_size)
    h = torch.zeros(1, agent.hidden_size, device=device, dtype=dtype)
    
    steps = 0
    total_reward = 0.0
    
    np_dtype = np.float32 if dtype == torch.float32 else np.float16
    use_cuda = device.type == 'cuda'
    
    # Pre-allocate pinned memory buffers for lower-latency GPU transfers
    if use_cuda:
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
        
        if render and steps % render_skip == 0:
            env.render()
            maybe_show_cameras(obs)
        
        # CPU preprocessing (matches training pipeline)
        xs_cpu = preprocess_obs_cpu(obs, dtype=np_dtype)
        
        # Transfer to GPU
        if use_cuda:
            pin_img[0] = torch.from_numpy(xs_cpu["img"])
            pin_wind[0] = torch.from_numpy(xs_cpu["wind_direction"])
            pin_col[0] = torch.from_numpy(xs_cpu["collision"])
            xs_gpu = {
                "img": pin_img.to(device, non_blocking=True),
                "wind_direction": pin_wind.to(device, non_blocking=True),
                "collision": pin_col.to(device, non_blocking=True),
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
            
        action_exec = action.squeeze(0).cpu().numpy()
        
        if done or trunc:
            break
            
    return info, total_reward, steps

def main():
    
    # --- Configuration ---
    # checkpoint_path = os.path.join(CHECKPOINT_DIR, "efficientnet_dagger_final.pt")
    # model_type = "efficientnet" 
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "efficientnet_dagger_final.pt")
    model_type = "efficientnet"

    episodes = 1000
    render = True
    # ---------------------
    
    device = get_device()
    print(f"Using device: {device}")

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        # Try finding *any* checkpoint
        print(f"Checkpoint not found at {checkpoint_path}")
        available = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
        if available:
            print(f"Found checkpoints: {available}")
            checkpoint_path = os.path.join(CHECKPOINT_DIR, available[0])
            print(f"Defaulting to {checkpoint_path}")
            if "mobilenet" in checkpoint_path:
                model_type = "mobilenet"
            else:
                model_type = "efficientnet"
        else:
            print(f"No checkpoints found in {CHECKPOINT_DIR}/. Exiting.")
            return

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
    if model_type == "efficientnet":
        agent = EfficientNetAgent(
            action_dim=2,
            hidden_size=256,
            dtype=DTYPE
        ).to(device=device, dtype=DTYPE)
    elif model_type == "mobilenet":
        agent = MobileNetAgent(
            action_dim=2,
            hidden_size=256,
            dtype=DTYPE
        ).to(device=device, dtype=DTYPE)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    # 3. Load Checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(checkpoint)
    agent.eval()

    # 4. Run Loop
    success_count = 0
    distances = []
    
    print(f"Running {episodes} episodes...")
    
    render_skip = 1
    
    try:
        for i in range(episodes):
            print(f"Episode {i+1}/{episodes}...", end=" ", flush=True)
            
            info, reward, steps = run_episode(env, agent, device, DTYPE, render, render_skip)
            
            # Check success (dist_to_goal < goal_radius which is 0.8)
            dist = info["dist_to_goal"]
            is_success = dist < 0.8
            status = "SUCCESS" if is_success else "FAIL"
            if info.get("stalled"): status = "STALLED"
            
            print(f"[{status}] Steps: {steps}, Reward: {reward:.2f}, Final Dist: {dist:.2f}")
            
            if is_success:
                success_count += 1
            distances.append(dist)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        env.close()
        if render and cv2 is not None:
             cv2.destroyAllWindows()
             
    # 5. Summary
    if len(distances) > 0:
        print("\n--- Summary ---")
        print(f"Success Rate: {success_count}/{len(distances)} ({success_count/len(distances)*100:.1f}%)")
        print(f"Avg Final Distance: {np.mean(distances):.2f}")
    
if __name__ == "__main__":
    main()
