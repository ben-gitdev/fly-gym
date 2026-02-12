"""
Script to run trained vision-based agents (EfficientNet, MobileNet) from checkpoints.
"""

import argparse
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
from core.utils import get_device, obs_to_torch

# Import config constants
from shared_config import (
    ENV_WIDTH,
    ENV_HEIGHT,
    MAX_EPISODE_STEPS,
    N_OBSTACLES,
    ARENA_HALF_EXTENT,
    DTYPE,
)

def maybe_show_cameras(obs):
    # Only show if cv2 is available and we want to see it
    # For now, let's keep it simple
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
    done = False
    trunc = False
    
    if render:
        env.render()
        
    while not (done or trunc):
        # Prepare observation
        obs_t = obs_to_torch(obs, device=device, dtype=dtype)
        
        # Inference step
        with torch.no_grad():
            x = agent.obs_to_x(obs_t)
            h_next, action = agent.step(h, obs_t, x=x)
            h = h_next
            
        action_np = action.squeeze(0).cpu().numpy()
        
        # Environment step
        obs, reward, done, trunc, info = env.step(action_np)
        
        total_reward += reward
        steps += 1
        
        if render and steps % render_skip == 0:
            env.render()
            maybe_show_cameras(obs)
            
    return info, total_reward, steps

def main():

    checkpoint = "checkpoints/efficientnet_dagger_final.pt"
    model_type = "efficientnet"
    episodes = 1000
    render = True
    
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Initialize Environment
    # If rendering, we set render_mode="human" so the env spawns a window
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
        
    # 3. Load Checkpoint
    if os.path.isfile(checkpoint):
        print(f"Loading checkpoint: {checkpoint}")
        checkpoint = torch.load(checkpoint, map_location=device)
        agent.load_state_dict(checkpoint)
        agent.eval()
    else:
        print(f"Error: Checkpoint file not found at {checkpoint}")
        return

    # 4. Run Loop
    success_count = 0
    distances = []
    
    print(f"Running {episodes} episodes...")
    
    render_skip = 10
    
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
