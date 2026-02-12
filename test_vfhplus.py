import numpy as np
import time
import cv2
import mujoco

from environment.mujoco_two_cam_env_random_obstacles import MuJoCoTwoCamEnv
from agents.teacher_analytic_agent import PlannerAnalyticTeacher

def test_vfh_teacher():
    print("[Test] Initializing Environment...")
    env = MuJoCoTwoCamEnv(
        width=256,
        height=256,
        n_obstacles=20,
        max_episode_steps=1000, 
        render_mode="human", # Revert to human for user
        arena_half_extent=7.0,
        goal_radius=0.8,
        contact_penalty=0.5,
        end_on_collision=False
    )
    
    print("[Test] Initializing PlannerAnalyticTeacher with VFH*...")
    teacher = PlannerAnalyticTeacher(
        arena_half_extent=env.arena,
        cell_size=0.1, 
        robot_radius=0.2,
        safety_margin=0.1,
        obstacle_box_half=(0.4, 0.4),
        k_nearest_obs=5,
        device="cpu",
        include_collision_flag=True
    )

    num_episodes = 50
    success_count = 0
    
    try:
        for ep in range(num_episodes):
            print(f"\n--- Episode {ep+1}/{num_episodes} ---")
            obs, _ = env.reset()
            teacher.reset()
            
            start_time = time.time()
            steps = 0
            
            while True:
                # 1. Teacher Action
                action = teacher.act(env)
                
                if action is None:
                    print("Teacher returned None action (stuck or no plan).")
                    action = np.zeros(2, dtype=np.float32) 
                
                # 2. Step Env
                obs, reward, done, trunc, info = env.step(action)
                steps += 1
                
                if done or trunc:
                    dist = info.get("dist_to_goal", 100)
                    is_success = dist < env.goal_radius
                    result = "Success" if is_success else "Timeout/Collision"
                    if is_success: success_count += 1
                    print(f"Episode {ep+1} finished: {result} in {steps} steps. Dist: {dist:.2f}")
                    break
        
        print(f"\nTest finished. Success Rate: {success_count}/{num_episodes}")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        env.close()
        # if cv2 is not None:
        #     cv2.destroyAllWindows()

if __name__ == "__main__":
    test_vfh_teacher()
