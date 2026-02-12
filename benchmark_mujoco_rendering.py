import time
import numpy as np
import os

import ctypes

# 1. Force Windows to load the correct OpenGL DLL (fixes 100% Bus Load)
try:
    ctypes.windll.LoadLibrary('opengl32.dll')
except Exception:
    print("Failed to load OpenGL DLL")
    pass

# 2. Tell MuJoCo to use GLFW (The Windows GPU Backend)

os.environ['MUJOCO_GL'] = 'glfw'  # Force GLFW instead of EGL/OSMesa

# Adjust import path if necessary based on your folder structure
# Assumes mujoco_two_cam_env_random_obstacles.py is in 'environment' folder
# or directly available. 
try:
    from environment.mujoco_two_cam_env_random_obstacles import MuJoCoTwoCamEnv
except ImportError:
    # Fallback if file is in the same directory
    from mujoco_two_cam_env_random_obstacles import MuJoCoTwoCamEnv

def benchmark(render_mode, steps=5000):
    print(f"\n--- Benchmarking render_mode='{render_mode}' ---")
    
    # 1. Initialize Environment
    # Using small resolution to match training defaults
    env = MuJoCoTwoCamEnv(
        width=256,
        height=256,
        n_obstacles=20,
        max_episode_steps=600,
        arena_half_extent=7.0,
        render_mode=render_mode
    )
    
    action = np.zeros(2, dtype=np.float32)
    env.reset()
    
    # 2. Warmup (Compile kernels / Allocate memory)
    print("Warming up (100 steps)...")
    for _ in range(100):
        env.step(action)
        
    # 3. Run Benchmark
    print(f"Running {steps} steps...")
    start_time = time.time()
    
    for _ in range(steps):
        # Random action to ensure physics is actually working
        # (Though your env uses fixed action in your example, random is safer for benchmarking)
        action = np.random.uniform(-1, 1, size=2)
        _, _, done, trunc, _ = env.step(action)
        
        if done or trunc:
            env.reset()
            
    end_time = time.time()
    
    # 4. Results
    duration = end_time - start_time
    sps = steps / duration
    print(f"Done.")
    print(f"Time: {duration:.2f}s")
    print(f"SPS:  {sps:.1f} steps/sec")
    
    env.close()

if __name__ == "__main__":
    # Benchmark 1: Headless (Physics Only)
    # This represents the maximum possible training speed if rendering wasn't the bottleneck.
    benchmark(render_mode=None, steps=5000)

    # Benchmark 2: Human (Windowed)
    # This will likely be capped at 60 SPS (60Hz monitor) or the window draw speed.
    # We use fewer steps so you don't have to wait too long.
    benchmark(render_mode="human", steps=1000)

    print("\n[Analysis]")
    print("If 'None' is fast (>500 SPS) but your training is slow (<300 SPS),")
    print("it implies the camera rendering (which runs even in 'None' mode for observations)")
    print("is the bottleneck. Try enabling EGL/GPU rendering as discussed.")