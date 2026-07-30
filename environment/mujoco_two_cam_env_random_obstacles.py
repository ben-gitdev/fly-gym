import math
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import mujoco
from mujoco import MjModel, MjData, Renderer
import glfw
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.astar import AStarGridPlanner, GridSpec


def _quat_to_yaw(q):
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def vel_angle_to_action(vel: float, pred_heading_angle: float) -> np.ndarray:
    # simple P-controller: turn_rate = Kp * error
    # Since 'pred_heading_angle' is relative to self (0 is straight), the error is just the angle.
    kp_turn = 0.5  # Tunable gain
    turn = kp_turn * pred_heading_angle
    
    left = vel - turn
    right = vel + turn
    max_val = max(abs(left), abs(right))
    if max_val > 1.0:
        left /= max_val
        right /= max_val
    return np.array([left, right], dtype=np.float32)

class MuJoCoTwoCamEnv(gym.Env):
    """Two-wheeled robot with two cameras, obstacles, and visibility-aware rewards (MuJoCo 3.x)."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    # Two scene variants, matching the paper's two evaluated texture conditions:
    #   "checker"   -- procedural checkerboard textures, used for TRAINING and the
    #                  in-distribution checkerboard evaluation (default).
    #   "realistic" -- photo-realistic PNG textures (grass/concrete/sky), used ONLY
    #                  for the out-of-distribution generalization evaluation.
    TEXTURE_XML_BY_MODE = {
        "checker": "mujoco_model_random_obstacles.xml",
        "realistic": "mujoco_model_random_obstacles_realistic.xml",
    }

    def __init__(self,
                 width=84,
                 height=84,
                 max_episode_steps=600,
                 n_obstacles=20,
                 goal_radius=0.8,
                 goal_bonus=10.0,
                 ctrl_penalty=1e-3,
                 contact_penalty=0.5,
                 arena_half_extent=5.0,
                 seed=None,
                 render_mode=None,
                 explore_binsize=0.2,
                 time_penalty=5e-3,
                 prog_scale=5.0,
                 frame_skip=10,
                 stall_threshold=0.002,
                 stall_limit=50,
                 end_on_collision=False,
                 texture_mode="checker"):
        super().__init__()

        if texture_mode not in self.TEXTURE_XML_BY_MODE:
            raise ValueError(
                f"Unknown texture_mode={texture_mode!r}; expected one of "
                f"{sorted(self.TEXTURE_XML_BY_MODE)}"
            )
        self.texture_mode = texture_mode

        self.W, self.H = int(width), int(height)
        self.max_steps = int(max_episode_steps)
        max_placeholders = 20
        if n_obstacles > max_placeholders:
            raise ValueError(f"n_obstacles={n_obstacles} exceeds XML placeholders ({max_placeholders}).")
        self.n_obstacles = int(n_obstacles)
        self.goal_radius, self.goal_bonus = float(goal_radius), float(goal_bonus)
        self.ctrl_penalty, self.contact_penalty = float(ctrl_penalty), float(contact_penalty)
        self.arena = float(arena_half_extent)
        self.render_mode = render_mode
        self.explore_binsize = float(explore_binsize)
        self.time_penalty = float(time_penalty)
        self.prog_scale = float(prog_scale)
        self.frame_skip = int(frame_skip)
        self.stall_threshold = float(stall_threshold)
        self.stall_limit = int(stall_limit)
        self.end_on_collision = bool(end_on_collision)

        xml_name = self.TEXTURE_XML_BY_MODE[texture_mode]
        self.model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name(xml_name)))
        self.data = MjData(self.model)
        
        # Renderer for sensors (cameras)
        self.renderer = Renderer(self.model, self.W, self.H)

        # --- Manual Rendering Setup ---
        self.window = None
        self._view_cam = mujoco.MjvCamera()
        self._view_opt = mujoco.MjvOption()
        self._view_scn = mujoco.MjvScene(self.model, maxgeom=10000) # Increased maxgeom for path lines
        
        # NOTE: Do NOT initialize MjrContext here (no window yet)
        self._view_ctx = None
        
        # Initialize camera view
        mujoco.mjv_defaultCamera(self._view_cam)
        mujoco.mjv_defaultOption(self._view_opt)
        self._view_cam.distance = 20.0  # Zoom out to see arena
        self._view_cam.lookat[:] = [0, 0, 0]
        self._view_cam.elevation = -90  # Top-down view
        self._view_cam.azimuth = 90

        # --- Path Drawing Attribute ---
        self.path_to_draw = None 
        self.target_to_draw = None

        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._obstacle_xy = np.zeros((self.n_obstacles, 2), np.float32)

        # IDs
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal_site")
        self.left_hinge_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_hinge")
        self.right_hinge_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_hinge")
        self.cam_left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_left")
        self.cam_right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_right")

        self.obstacle_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle{i}")
                             for i in range(self.n_obstacles)]
        self.obstacle_body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle{i}_body")
                                  for i in range(self.n_obstacles)]
        self.obstacle_x_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"obstacle{i}_x")
                                     for i in range(self.n_obstacles)]
        self.obstacle_y_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"obstacle{i}_y")
                                     for i in range(self.n_obstacles)]
        self._obstacle_x_qposadr = np.array(
            [self.model.jnt_qposadr[jid] for jid in self.obstacle_x_joint_ids],
            dtype=np.int32
        )
        self._obstacle_y_qposadr = np.array(
            [self.model.jnt_qposadr[jid] for jid in self.obstacle_y_joint_ids],
            dtype=np.int32
        )
        self._obstacle_xy[:, 0] = self.model.qpos0[self._obstacle_x_qposadr]
        self._obstacle_xy[:, 1] = self.model.qpos0[self._obstacle_y_qposadr]

        # Spaces
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = spaces.Dict({
            "cam_left":  spaces.Box(0, 255, (self.H, self.W, 3), np.uint8),
            "cam_right": spaces.Box(0, 255, (self.H, self.W, 3), np.uint8),
            "sensors":   spaces.Dict({
                "vec_to_goal": spaces.Box(-np.inf, np.inf, (2,), np.float32),
                "collision": spaces.Box(0.0, 1.0, (1,), np.float32),
                "wind_direction": spaces.Box(-1.0, 1.0, (2,), np.float32), # NEW
            }),
            "privileged": spaces.Box(-np.inf, np.inf, (17,), np.float32), # 2(pos)+2(yaw)+2(goal)+1(vel)+10(obs)
        })

        self.k_nearest = 5

        # State
        self._goal_xy = np.zeros(2, np.float32)
        self._prev_xy = np.zeros(2, np.float32)
        self._speed = 0.0
        self._t = 0
        self.visited = set()
        self._last_action = np.zeros(2, np.float32)

        # Control scaling
        ctrl_range = np.array(self.model.actuator_ctrlrange, dtype=np.float32)
        if ctrl_range.size:
            ctrl_high = np.max(np.abs(ctrl_range), axis=1)
            self._action_scale = float(np.min(ctrl_high))
        else:
            self._action_scale = 1.0

        # A* Planner for distance calculation
        self.planner = AStarGridPlanner(
            GridSpec(
                arena_half_extent=self.arena,
                cell_size=0.2, 
                obstacle_inflate=0.2 # Roughly robot radius
            )
        )

        self.reset(seed=seed)

    # ---------------- Core API ----------------
    def place_obstacles(self, qpos, min_spacing=1.5, wall_margin=0.6, max_attempts=1024, lock_eps=1e-4):
        if not self.obstacle_ids:
            return
        lo = -self.arena + wall_margin
        hi = self.arena - wall_margin
        placed = []
        for idx in range(self.n_obstacles):
            for _ in range(max_attempts):
                ox = self.np_random.uniform(lo, hi)
                oy = self.np_random.uniform(lo, hi)
                if all(np.hypot(ox - px, oy - py) >= min_spacing for px, py in placed):
                    qpos[self._obstacle_x_qposadr[idx]] = ox
                    qpos[self._obstacle_y_qposadr[idx]] = oy
                    self._obstacle_xy[idx] = [ox, oy]
                    x_jid = self.obstacle_x_joint_ids[idx]
                    y_jid = self.obstacle_y_joint_ids[idx]
                    self.model.jnt_range[x_jid, :] = [ox - lock_eps, ox + lock_eps]
                    self.model.jnt_range[y_jid, :] = [oy - lock_eps, oy + lock_eps]
                    placed.append((ox, oy))
                    break
            else:
                raise RuntimeError("Failed to place obstacles.")
            
    def place_robot(self, x, y, yaw, qpos, qvel):
        free_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")
        base_qpos_start = self.model.jnt_qposadr[free_id]
        qpos[base_qpos_start + 0:base_qpos_start + 3] = [x, y, 0.1]
        half = yaw * 0.5
        qpos[base_qpos_start + 3:base_qpos_start + 7] = [math.cos(half), 0, 0, math.sin(half)]
        qpos[self.model.jnt_qposadr[self.left_hinge_id]] = 0.0
        qpos[self.model.jnt_qposadr[self.right_hinge_id]] = 0.0

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel * 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        def sample_xy(margin=1.2):
            obstacle_positions = self._obstacle_xy[:len(self.obstacle_ids)].tolist()
            for _ in range(512):
                gx = self.np_random.uniform(-self.arena + 1.5, self.arena - 1.5)
                gy = self.np_random.uniform(-self.arena + 1.5, self.arena - 1.5)
                if any(np.hypot(gx - ox, gy - oy) < margin for ox, oy in obstacle_positions):
                    continue
                return gx, gy
            return 0.0, 0.0

        self.place_obstacles(qpos, min_spacing=1.5)

        x, y = sample_xy(margin=1.2)
        yaw = self.np_random.uniform(-math.pi, math.pi)

        self.place_robot(x, y, yaw, qpos, qvel)


       # random obstacle color
        for gid in self.obstacle_ids:
            self.model.geom_rgba[gid, :] = [
                self.np_random.uniform(0.2, 1.0),
                self.np_random.uniform(0.2, 1.0),
                self.np_random.uniform(0.2, 1.0),
                1.0,
            ]
           
        # gx, gy = sample_xy(margin=1.0)
        # find gx, gy so that distance betwee goal [gx, gy] and robot [x,y] is larger than 1/3 of arena size
        gx, gy = 0.0, 0.0
        for _ in range(1024):
            gx, gy = sample_xy(margin=1.2)
            if np.hypot(gx - x, gy - y) > self.arena*2 / 3.0:
                break

        self._goal_xy[:] = [gx, gy]
        self.model.site_pos[self.goal_site_id, :2] = [gx, gy]

        # # move the first obstable to position in-between robot and goal
        # ox, oy = (x + gx) / 2, (y + gy) / 2
        # # calculate distances between [ox, oy] and all obstacles
        # dists = [np.hypot(ox - ox_i, oy - oy_i) for ox_i, oy_i in self._obstacle_xy[1:]]
        # if min(dists) > 1.132:
        #     qpos[self._obstacle_x_qposadr[0]] = ox
        #     qpos[self._obstacle_y_qposadr[0]] = oy
        #     self._obstacle_xy[0] = [ox, oy]
        #     x_jid = self.obstacle_x_joint_ids[0]
        #     y_jid = self.obstacle_y_joint_ids[0]
        #     self.model.jnt_range[x_jid, :] = [ox - 1e-4, ox + 1e-4]
        #     self.model.jnt_range[y_jid, :] = [oy - 1e-4, oy + 1e-4]
        # self.data.qpos[:] = qpos
        # self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self._prev_xy[:] = [x, y]
        self._speed = 0.0
        self._last_action[:] = 0.0
        self._stall_steps = 0
        self._reset_exploration()

        obs = self._get_obs()

        # --- FIX: FORCE RENDER ON RESET ---
        if self.render_mode == "human":
            self.render()

        return obs, {"dist_to_goal": self._dist_to_goal(self._prev_xy, dist_mode="euclidean")}

    # ---------------- Step ----------------
    def step(self, action):
        self._t += 1
        a = vel_angle_to_action(vel=action[0], pred_heading_angle=action[1])
        a = np.clip(np.asarray(a, np.float32), -1, 1) * self._action_scale
        self.data.ctrl[:] = a
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        xy = self._base_xy()
        prev_d = self._dist_to_goal(self._prev_xy, dist_mode="euclidean")
        new_d = self._dist_to_goal(xy, dist_mode="euclidean")
        dt = self.model.opt.timestep * self.frame_skip
        self._speed = np.linalg.norm(xy - self._prev_xy) / max(dt, 1e-6)

        # Stall detection
        dist_moved = np.linalg.norm(xy - self._prev_xy)
        if dist_moved < self.stall_threshold and abs(action[0]) > 0.05:
            # print(f"Stall detected: dist_moved={dist_moved}, action={action[0]}")
            self._stall_steps += 1
        else:
            self._stall_steps = 0


        goal_vis = False# self._goal_visible()
        r_prog = ((prev_d - new_d) / self.arena) * self.prog_scale #if goal_vis else 0.0
        r_explore = self._explore_bonus(xy, binsize=self.explore_binsize)# if not goal_vis else 0.0
        r_ctrl = - self.ctrl_penalty * float(abs(action[1])) # penalize heading angle only
        r_time = -self.time_penalty
        r_col = -self.contact_penalty if self._has_collision()[0] else 0.0
        # r_cruise = 0.02*(min(self._speed, 1.0)-0.5) #if self._speed > 0.5 else 0.0
        r_danger = self._danger_penalty(safety_margin=0.7, ratio=0.3)

        # print(f"r_prog: {r_prog} | r_explore: {r_explore} | r_ctrl: {r_ctrl} | r_time: {r_time} | r_col: {r_col} | r_danger: {r_danger}")

        reward = r_prog + r_explore + r_ctrl + r_time + r_col + r_danger
        goal_reached = new_d < self.goal_radius
        stalled = self._stall_steps >= self.stall_limit
        collision = self._has_collision()[0]
        # if(stalled): print("stalled")
        done = goal_reached or stalled or (collision and self.end_on_collision) # terminate as soon as collision

        if goal_reached:
            reward += self.goal_bonus
        trunc = self._t >= self.max_steps
        self._prev_xy[:] = xy
        self._last_action[:] = a

        obs = self._get_obs()
        info = {
            "dist_to_goal": new_d,
            "goal_visible": bool(goal_vis),
            "stalled": bool(stalled),
        }

        # Call Manual Render if mode is human
        if self.render_mode == "human":
            self.render()

        return obs, float(np.clip(reward, -50.0, 50.0)), bool(done), bool(trunc), info
    
    # mouse interaction setup
    def _setup_mouse_interaction(self):
        self._button_left_pressed = False
        self._button_right_pressed = False
        self._last_mouse_x = 0
        self._last_mouse_y = 0

        def mouse_button_callback(window, button, action, mods):
            self._button_left_pressed = (button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS)
            self._button_right_pressed = (button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS)

        def cursor_pos_callback(window, xpos, ypos):
            dx = xpos - self._last_mouse_x
            dy = ypos - self._last_mouse_y
            self._last_mouse_x = xpos
            self._last_mouse_y = ypos

            if not (self._button_left_pressed or self._button_right_pressed):
                return

            # Determine action based on buttons
            action = mujoco.mjtMouse.mjMOUSE_ZOOM if self._button_right_pressed else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            
            # Use MuJoCo's built-in camera mover
            mujoco.mjv_moveCamera(
                self.model, 
                action, 
                dx / self.H, 
                dy / self.H, 
                self._view_scn, 
                self._view_cam
            )

        def scroll_callback(window, xoffset, yoffset):
             mujoco.mjv_moveCamera(
                self.model, 
                mujoco.mjtMouse.mjMOUSE_ZOOM, 
                0, 
                -0.05 * yoffset, 
                self._view_scn, 
                self._view_cam
            )

        glfw.set_mouse_button_callback(self.window, mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, cursor_pos_callback)
        glfw.set_scroll_callback(self.window, scroll_callback)


    # ---------------- Manual Rendering ----------------
    def render(self, mode="human"):
        mode = self.render_mode or mode
        if mode == "human":
            if not glfw:
                return

            # 1. Initialize Window/Context if not exists
            if self.window is None:
                if not glfw.init():
                    raise Exception("Could not initialize GLFW")
                
                # --- NEW: Hint to make sure window is visible and floating ---
                glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
                glfw.window_hint(glfw.FOCUSED, glfw.TRUE)
                
                self.window = glfw.create_window(1024, 768, "MuJoCo Manual Render", None, None)
                self._setup_mouse_interaction()
                if not self.window:
                    glfw.terminate()
                    raise Exception("Could not create GLFW window")
                
                # --- NEW: Force window position to top-left (100, 100) to ensure it's on screen ---
                glfw.set_window_pos(self.window, 100, 100)
                glfw.show_window(self.window)
                glfw.focus_window(self.window)
                
                glfw.make_context_current(self.window)
                self._view_ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

            # 2. Check window close flag
            if glfw.window_should_close(self.window):
                return

            # 3. Update Scene (Standard)
            glfw.make_context_current(self.window)
            viewport = mujoco.MjrRect(0, 0, 1024, 768)
            mujoco.mjv_updateScene(
                self.model, self.data, self._view_opt, None, self._view_cam,
                mujoco.mjtCatBit.mjCAT_ALL.value, self._view_scn
            )

            # 4. INJECT PATH GEOMS
            if self.path_to_draw is not None and len(self.path_to_draw) > 1:
                for i in range(len(self.path_to_draw) - 1):
                    if self._view_scn.ngeom >= self._view_scn.maxgeom:
                        break
                    
                    p1 = self.path_to_draw[i]
                    p2 = self.path_to_draw[i+1]
                    
                    # Points must be 3D numpy arrays for mjv_connector.
                    p1_3d = np.array([p1[0], p1[1], 0.05], dtype=np.float64)
                    p2_3d = np.array([p2[0], p2[1], 0.05], dtype=np.float64)
                    
                    # Using mjv_connector
                    mujoco.mjv_connector(
                        self._view_scn.geoms[self._view_scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_LINE,
                        3.0, # width
                        p1_3d,
                        p2_3d
                    )
                    
                    # Set Color (Red)
                    self._view_scn.geoms[self._view_scn.ngeom].rgba[:] = [1.0, 0.0, 0.0, 1.0]
                    self._view_scn.ngeom += 1
                if hasattr(self, "target_to_draw") and self.target_to_draw is not None:
                    if self._view_scn.ngeom < self._view_scn.maxgeom:
                        g = self._view_scn.geoms[self._view_scn.ngeom]
                        mujoco.mjv_initGeom(
                            g,
                            type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=np.array([0.15, 0.15, 0.15]), # 15cm sphere
                            pos=np.zeros(3),
                            mat=np.eye(3).flatten(),
                            rgba=np.array([0.0, 1.0, 0.0, 0.8], dtype=np.float32) # Green
                        )
                        g.pos[:] = np.array([self.target_to_draw[0], self.target_to_draw[1], 0.2])
                        self._view_scn.ngeom += 1

            # 5. Render to Window
            # Check if context matches window
            if glfw.get_current_context() != self.window:
                 glfw.make_context_current(self.window)
            
            mujoco.mjr_render(viewport, self._view_scn, self._view_ctx)
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        elif mode == "rgb_array":
            return self._render_camera("cam_left")

    def _render_camera(self, cam_name):
        self.renderer.update_scene(self.data, camera=cam_name)
        return self.renderer.render()

    def _get_obs(self):
        img_left = self._render_camera("cam_left")
        img_right = self._render_camera("cam_right")
        sensors = self._sensor_vec()
        priv = self._get_privileged_obs()
        return {"cam_left": img_left, "cam_right": img_right, "sensors": sensors, "privileged": priv}

    def _get_privileged_obs(self):
        # 1. Robot State
        xy = self._base_xy()
        yaw = self._base_yaw()
        c, s = math.cos(yaw), math.sin(yaw)
        # Velocity
        v_lin = self._speed
        # We don't track angular velocity explicitly in self variables, 
        # but we can get it from qvel if needed. 
        # For now, let's use the last action as a proxy or just qvel.
        # qvel[base_yaw_index]... let's stick to simple state.
        
        # 2. Goal
        gxy = self._goal_xy
        
        # 3. Obstacles (K nearest)
        # Calculate distances to all obstacles
        obs_xy = self._obstacle_xy[:self.n_obstacles]
        dists = np.linalg.norm(obs_xy - xy, axis=1)
        # Get indices of k nearest
        idx = np.argsort(dists)[:self.k_nearest]
        nearest_obs = obs_xy[idx]
        
        # Pad if fewer than k obstacles (unlikely with N=20, K=5)
        if len(nearest_obs) < self.k_nearest:
            pad = np.zeros((self.k_nearest - len(nearest_obs), 2), dtype=np.float32)
            nearest_obs = np.concatenate([nearest_obs, pad], axis=0)
            
        # Flatten
        # [Rx, Ry, Rc, Rs, V, Gx, Gy, Ox1, Oy1, ..., OxK, OyK]
        # 2 + 2 + 1 + 2 + 10 = 17? Wait.
        # Robot: x, y (2)
        # Yaw: c, s (2)
        # Vel: v (1) -> let's assume 1D speed for now
        # Goal: x, y (2)
        # Obs: 5*2 (10)
        # Total: 17. 
        
        # Let's adjust space definition to 17.
        
        priv = np.concatenate([
            xy, 
            np.array([c, s], dtype=np.float32),
            np.array([v_lin], dtype=np.float32),
            gxy, 
            nearest_obs.flatten()
        ])
        return priv.astype(np.float32)

    # ---------------- Visibility (FOV + occlusion) ----------------
    def _goal_visible(self):
        return self._goal_in_sight(self.cam_left_id) or self._goal_in_sight(self.cam_right_id)

    def _goal_in_sight(self, cam_id: int) -> bool:
        goal_w = self.data.site_xpos[self.goal_site_id]
        u, v, in_front, in_fov = self._project_world_to_cam_pixels(goal_w, cam_id)
        if not (in_front and in_fov):
            return False
        cam_pos = self.data.cam_xpos[cam_id]
        return not self._occluded(cam_pos, goal_w)

    def _get_camera_matrix(self, cam_id: int):
        cam_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
        self.renderer.update_scene(self.data, camera=cam_name)
        cam = self.renderer.scene.camera[0]
        pos = np.array(cam.pos, dtype=np.float64)
        z = np.array(cam.forward, dtype=np.float64)
        y = np.array(cam.up, dtype=np.float64)
        x = np.cross(y, z)
        x /= np.linalg.norm(x) + 1e-8
        y /= np.linalg.norm(y) + 1e-8
        z /= np.linalg.norm(z) + 1e-8
        rot3 = np.vstack([x, y, z])
        translation = np.eye(4, dtype=np.float64)
        translation[0:3, 3] = -pos
        rotation = np.eye(4, dtype=np.float64)
        rotation[0:3, 0:3] = rot3
        fovy = float(self.model.cam_fovy[cam_id]) * math.pi / 180.0
        focal_scaling = (1.0 / math.tan(fovy / 2.0)) * (self.H / 2.0)
        focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0.0])[0:3, :]
        image = np.eye(3, dtype=np.float64)
        image[0, 2] = (self.W - 1) / 2.0
        image[1, 2] = (self.H - 1) / 2.0
        M = image @ focal @ rotation @ translation
        return M

    def _project_world_to_cam_pixels(self, p_world, cam_id):
        M = self._get_camera_matrix(cam_id)
        pw_h = np.ones(4, dtype=np.float64)
        pw_h[0:3] = p_world
        xs, ys, s = M @ pw_h
        if s <= 1e-8:
            return None, None, False, False
        u = xs / s
        v = ys / s
        in_fov = (0.0 <= u < self.W) and (0.0 <= v < self.H)
        return float(u), float(v), True, in_fov

    def _occluded(self, p_world, q_world, eps=1e-4) -> bool:
        ray = q_world - p_world
        dist_goal = float(np.linalg.norm(ray))
        if dist_goal < eps:
            return False
        dir_world = ray / dist_goal
        geomid = np.zeros(1, dtype=np.int32)
        hit_dist = mujoco.mj_ray(self.model, self.data, p_world, dir_world, None, 1, -1, geomid)
        return (hit_dist >= 0) and ((hit_dist + eps) < dist_goal)

    def _base_xy(self):
        return np.array(self.data.xpos[self.base_body_id, :2], np.float32)

    def _base_yaw(self):
        free_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")
        base_qpos_start = self.model.jnt_qposadr[free_id]
        q = self.data.qpos[base_qpos_start + 3:base_qpos_start + 7]
        return _quat_to_yaw(q)

    def _dist_to_goal(self, xy, dist_mode="euclidean"):
        if dist_mode == "euclidean":
            return float(np.linalg.norm(self._goal_xy - xy))
        elif dist_mode == "astar":
            path = self.planner.plan(
                start_xy=xy, 
                goal_xy=self._goal_xy, 
                obstacles_xy=self._obstacle_xy[:self.n_obstacles], 
                obstacle_radius=0.2, # same as teacher setup
                smooth_path=True
            )
            if path is None or len(path) < 2:
                # Fallback if planning fails
                return float(np.linalg.norm(self._goal_xy - xy))
            
            # Sum up path segments
            # replace the first path point with the robot's current position
            path[0] = xy
            length = 0.0
            for i in range(len(path) - 1):
                length += float(np.linalg.norm(path[i+1] - path[i]))
            return length
        else:
            raise ValueError(f"Unknown dist_mode: {dist_mode}")

    
    def _has_collision(self) :
        obstacle_name_prefix = "obstacle"
        ground_name = "floor"
        wheel_tokens = "wheel"
        robot_tokens = ("base", "wheel", "caster")
        data = self.data
        model = self.model
        for i in range(int(getattr(data, "ncon", 0))):
            con = data.contact[i]
            g1 = int(con.geom1)
            g2 = int(con.geom2)
            n1 = model.geom(g1).name.lower() 
            n2 = model.geom(g2).name.lower() 
            if not n1 and not n2:
                continue
            ground1 = ground_name in n1
            ground2 = ground_name in n2
            is_robot1 = any(tok in n1 for tok in robot_tokens)
            is_robot2 = any(tok in n2 for tok in robot_tokens)
            if (is_robot1 or is_robot2) and not (ground1 or ground2) and not(is_robot1 and is_robot2):
                contact_point = con.pos.copy()
                # calculate angel between robot pos and contact point
                xy = self._base_xy()
                yaw = self._base_yaw()
                dx, dy = contact_point[0] - xy[0], contact_point[1] - xy[1]
                angle = math.atan2(dy, dx)
                contact_angle = wrap_pi(angle - yaw) # angle > 0 means contact is to the left of robot front
                # print(f"Collision at angle {math.degrees(contact_angle):.1f}deg")

                return True, contact_point, contact_angle
        return False, None, None

    def _sensor_vec(self):
        xy = self._base_xy()
        dx, dy = self._goal_xy - xy
        has_collision, collision_pos, collision_angle = self._has_collision()
        # collision_arr = np.array([1.0 if has_collision else 0.0], np.float32)
        vec_to_goal = np.array([dx, dy], np.float32)
        # get coordinate of cameras
        cam_left_pos = np.array(self.data.cam_xpos[self.cam_left_id])
        cam_right_pos = np.array(self.data.cam_xpos[self.cam_right_id])
        vec_left_to_goal = self._goal_xy - cam_left_pos[:2]
        vec_right_to_goal = self._goal_xy - cam_right_pos[:2]
        # print(f"Camera Left Pos: {cam_left_pos}, Right Pos: {cam_right_pos}")
        
        # NEW: Wind Direction (angle to goal relative to robot heading)
        wind_angle = math.atan2(dy, dx)
        rel_wind_angle = wrap_pi(wind_angle - self._base_yaw())
        wind_direction = np.array([math.cos(rel_wind_angle), math.sin(rel_wind_angle)], dtype=np.float32)

        return {
            "vec_to_goal": vec_to_goal, 
            "collision": has_collision, 
            "collision_pos": collision_pos, 
            "collision_angle": collision_angle, 
            "vec_left_to_goal": vec_left_to_goal, 
            "vec_right_to_goal": vec_right_to_goal,
            "wind_direction": wind_direction, # NEW
        }


    def _bin_of(self, xy, binsize=0.5):
        return (int(np.floor(xy[0]/binsize)), int(np.floor(xy[1]/binsize)))

    def _reset_exploration(self):
        self.visited = set()
        self.visited.add(self._bin_of(self._base_xy(), self.explore_binsize))

    def _explore_bonus(self, xy, binsize=0.2):
        b = self._bin_of(xy, binsize)
        if b not in self.visited:
            self.visited.add(b)
            return 0.1
        return 0.0#0.01 * float(self._speed)

    def _danger_penalty(self, safety_margin=0.35, ratio=0.5):
        r_danger = 0.0
        
        # --- 1. Get Robot State ---
        xy = self._base_xy()
        x, y = xy[0], xy[1]
        
        # Calculate velocity vector from speed & heading
        # (Assuming qpos[2] is yaw. Use self._data.qvel if you want exact physics velocity)
        yaw = self._base_yaw()
        vx = np.cos(yaw) * self._speed
        vy = np.sin(yaw) * self._speed
        
        # --- 2. Wall Penalty (The New Part) ---
        limit = self.arena
        
        # North Wall (+y)
        dist_n = limit - y
        if dist_n < safety_margin and vy > 0:
            r_danger += -ratio * vy  # Penalize positive velocity

        # South Wall (-y)
        dist_s = y - (-limit)
        if dist_s < safety_margin and vy < 0:
            r_danger += -ratio * abs(vy) # Penalize negative velocity

        # East Wall (+x)
        dist_e = limit - x
        if dist_e < safety_margin and vx > 0:
            r_danger += -ratio * vx

        # West Wall (-x)
        dist_w = x - (-limit)
        if dist_w < safety_margin and vx < 0:
            r_danger += -ratio * abs(vx)

        # --- 3. Cylinder Obstacle Penalty (Existing Logic) ---
        # Only check the SINGLE nearest obstacle to save compute
        diffs = self._obstacle_xy[:self.n_obstacles] - xy
        dists = np.linalg.norm(diffs, axis=1)
        nearest_idx = np.argmin(dists)
        
        min_dist = dists[nearest_idx]
        if min_dist < safety_margin:
            # Vector pointing TO the obstacle
            to_obs = diffs[nearest_idx] / (min_dist + 1e-8)
            # Project velocity onto that vector
            crash_speed = (vx * to_obs[0]) + (vy * to_obs[1])
            
            if crash_speed > 0:
                r_danger += -ratio * crash_speed

        # --- 4. Total Reward ---
        # Note: We sum them. If you drive into a corner, you get double penalty.
        # This is good! Corners are death traps.
        return r_danger

    def close(self):
        if self.window:
            try:
                glfw.destroy_window(self.window)
                glfw.terminate()
            except Exception as e:
                # Swallow error on exit
                pass
            self.window = None


def main():
    env = MuJoCoTwoCamEnv(
        width=256,
        height=256,
        n_obstacles=20,
        max_episode_steps=600,
        arena_half_extent=7.0,
        render_mode="human",
    )
    obs, info = env.reset()
    try:
        for _ in range(20000):
            # random policy with slight forward bias
            action = np.array([0.9, 0.2]) 
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
