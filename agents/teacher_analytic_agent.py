import numpy as np
import math
import torch
from typing import Tuple, Optional
import mujoco

from core.utils import rot2d, box_to_disc_radius, wrap_pi
from core.vfhplus import VFHPlusPlanner
from models.teacher_analytic_model import pid_line_follower, PIDState


# ----------------------------
# Teacher: Planner + MLP + Robust Recovery
# ----------------------------
class PlannerAnalyticTeacher:
    def __init__(self, 
                 arena_half_extent: float, 
                 cell_size: float = 0.1,
                 robot_radius: float = 0.2, 
                 safety_margin: float = 0.1,
                 obstacle_box_half: Tuple[float, float] = (0.4, 0.4),
                 obstacle_cylinder_radius: float = 0.4,
                 k_nearest_obs: int = 5, 
                 device: str = "cpu",
                 recovery_turn_steps: int = 5, 
                 recovery_back_steps: int = 40,
                 recovery_forward_steps: int = 6, 
                 recovery_cooldown_steps: int = 15,
                 stuck_min_progress: float = 0.002, 
                 stuck_requires_forward_cmd: bool = True,
                 include_collision_flag: bool = False):
        
        self.device = torch.device(device)
        self.k = int(k_nearest_obs)
        self.include_collision_flag = bool(include_collision_flag)
        self.obstacle_disc_r = obstacle_cylinder_radius#float(box_to_disc_radius(obstacle_box_half))
        # VFH+ Planner (standalone, no longer uses GridSpec)
        self.planner = VFHPlusPlanner(
            arena_half_extent=float(arena_half_extent),
            robot_radius=robot_radius, 
            safety_dist=safety_margin,
            goal_threshold=float(cell_size)  # Use cell_size as goal threshold
        )
        
        self._prev_bad_contact = False
        self._rec_phase = None
        self._rec_step_count = 0
        self._rec_turn_dir = 1.0
        self.contact_point = None
        self.recovery_turn_steps = int(recovery_turn_steps)
        self.recovery_back_steps = int(recovery_back_steps)
        self.recovery_forward_steps = int(recovery_forward_steps)
        self.stuck_min_progress = float(stuck_min_progress)
        self.pid_state = PIDState()
        
        self._xy_prev = None
        self._path = None # For visualization compatibility

    @staticmethod
    def _get_base_xy_yaw(env) -> Tuple[np.ndarray, float]:
        xy = env._base_xy().astype(np.float32)
        free_id = env.model.jnt_qposadr[env.model.joint("base_free").id]
        q = np.array(env.data.qpos[free_id + 3: free_id + 7], dtype=np.float32)
        w, x, y, z = map(float, q)
        yaw = float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        return xy, yaw

    @staticmethod
    def _get_goal_xy(env) -> np.ndarray: return np.array(env._goal_xy, dtype=np.float32)
    @staticmethod
    def _get_obstacles_xy(env) -> np.ndarray: return np.array(env._obstacle_xy[: env.n_obstacles], dtype=np.float32)

    def _filter_obstacles_fov(self, xy: np.ndarray, yaw: float, obstacles_xy: np.ndarray, fov_angle: float) -> np.ndarray:
        """
        Filter obstacles to only those within fov_angle degrees FOV (+/- fov_angle/2 degrees) in front of the robot.
        """
        if len(obstacles_xy) == 0:
            return obstacles_xy
            
        diff = obstacles_xy - xy
        # Calculate angles of obstacles
        obs_angles = np.arctan2(diff[:, 1], diff[:, 0])
        
        # Relative angle to robot heading
        # wrap to [-pi, pi]
        rel_angles = (obs_angles - yaw + np.pi) % (2 * np.pi) - np.pi
        
        # Filter: absolute relative angle < 60 degrees
        mask = np.abs(rel_angles) < np.deg2rad(fov_angle/2)
        
        return obstacles_xy[mask]

    def _bad_contact_edge(self, env):
        bad, self.contact_point, self.contact_angle = env._has_collision()
        rising = bad and (not self._prev_bad_contact)
        self._prev_bad_contact = bad
        return bad, rising

    def _choose_recovery_turn_dir(self, xy, yaw, obstacles_xy):
        return -1.0 if self.contact_angle > 0 else 1.0

    def _recovery_action(self, phase, turn_dir):
        if phase == "turn":
            v, w = 0.0, 1.2 * turn_dir
        elif phase == "back":
            v, w = -0.35, 0.1 * turn_dir
        else:
            v, w = 0.7, 0.0
        return np.array([v, w], dtype=np.float32)

    def _start_recovery(self, xy, yaw, obstacles_xy):
        self._rec_turn_dir = self._choose_recovery_turn_dir(xy, yaw, obstacles_xy)
        self._rec_phase = "back"
        self._rec_step_count = 0 

    def _step_recovery(self):
        a = self._recovery_action(self._rec_phase, self._rec_turn_dir)
        replan = False
        self._rec_step_count += 1
        if self._rec_step_count < self.recovery_back_steps :
            self._rec_phase = "back"
        if self._rec_step_count >= self.recovery_back_steps:
            self._rec_phase = "back" 
        if self._rec_step_count == self.recovery_back_steps + self.recovery_turn_steps:
            self._rec_phase = None
            self._rec_step_count = 0
            replan = True
            
        return a, replan

    def _should_trigger_recovery(self, env):
        if self._rec_phase is not None : return False
        bad_now, rising = self._bad_contact_edge(env)
        if rising: return True
        return False

    def teleport_robot_safely(self, env, max_tries=20):
        """Teleport robot to a random nearby position (simplified version without occupancy check)."""
        curr_xy, curr_yaw = self._get_base_xy_yaw(env)
        
        for _ in range(max_tries):
            # Generate random perturbation
            pos_noise = np.random.uniform(-0.6, 0.6, size=2)
            yaw_noise = np.random.uniform(-math.pi/2, math.pi/2)
            
            test_xy = curr_xy + pos_noise
            test_yaw = wrap_pi(curr_yaw + yaw_noise)
            
            # Check if within arena bounds
            arena = self.planner.arena_half_extent - 0.5
            if abs(test_xy[0]) < arena and abs(test_xy[1]) < arena:
                qpos = env.data.qpos.copy()
                qvel = env.data.qvel.copy()
                env.place_robot(test_xy[0], test_xy[1], test_yaw, qpos, qvel)
                mujoco.mj_forward(env.model, env.data)
                return True
        return False

    # @torch.no_grad()
    def act(self, env) -> np.ndarray:
        xy, yaw = self._get_base_xy_yaw(env)
        goal = self._get_goal_xy(env)
        obs_xy = self._get_obstacles_xy(env)
        dt = env.model.opt.timestep * env.frame_skip
        
        # internal path var for drawing
        self._path = None
        
        # Check recovery
        if self._rec_phase is not None:
            a_cmd, _ = self._step_recovery()
        else:
            # Normal Planning Mode - VFH+
            # 1. Filter obstacles
            # FOV = 230.0
            # visible_obs = self._filter_obstacles_fov(xy, yaw, obs_xy, fov_angle=FOV)
            
            # 2. Plan (every step)
            path = self.planner.plan(start_xy=xy, goal_xy=goal, obstacles_xy=obs_xy, 
                                     obstacle_radius=self.obstacle_disc_r, current_yaw=yaw)
            
            if path is None or len(path) < 2:
                # Fallback
                wp = goal
                start_wp = xy
            else:
                start_wp = path[0]
                wp = path[1]
                self._path = path # Set path for visualization
                
            # 3. Control
            a_cmd = pid_line_follower(xy=xy, yaw=yaw, start_xy=start_wp, target_wp=wp, state=self.pid_state, dt=dt)
            
            # Visualization
            env.path_to_draw = self._path if self._path is not None else [xy, wp]
            env.target_to_draw = wp

        # Check for new collisions to trigger recovery
        if self._should_trigger_recovery(env):
            self._start_recovery(xy, yaw, obs_xy) 
            a_cmd, _ = self._step_recovery()

        self._xy_prev = xy.copy()
        
        return a_cmd

    def reset_pid(self):
        self.pid_state = PIDState()

    def reset(self):
        self._xy_prev = None
        self._prev_bad_contact = False
        self._rec_phase = None
        self._rec_step_count = 0
        self.pid_state = PIDState()
        self._path = None
        if hasattr(self.planner, 'prev_steering_dir'):
            self.planner.prev_steering_dir = 0.0
