import numpy as np
import math
from core.utils import wrap_pi


class PIDState:
    def __init__(self):
        self.integral = 0.0
        self.last_error = 0.0

def pid_line_follower(
    xy: np.ndarray,          # Current [x, y]
    yaw: float,             # Current heading
    start_xy: np.ndarray,    # Point A (where the segment started)
    target_wp: np.ndarray,   # Point B (the waypoint)
    state: PIDState,         # Persistent PID memory
    dt: float = 0.02          # Time step
) -> np.ndarray:
    # Guard against missing inputs (e.g., fallback goal-seek calls).
    if start_xy is None:
        start_xy = xy
    if state is None:
        state = PIDState()

    # 1. Calculate Vectors
    path_vec = target_wp - start_xy
    robot_vec = xy - start_xy
    
    path_len = np.linalg.norm(path_vec) + 1e-6
    path_unit_vec = path_vec / path_len

    # 2. Calculate Cross-Track Error (CTE)
    # Using 2D cross product: (x1*y2 - y1*x2)
    cte = (path_unit_vec[0] * robot_vec[1]) - (path_unit_vec[1] * robot_vec[0])

    # 3. Calculate Heading Error (to keep robot moving parallel to line)
    target_yaw = math.atan2(path_vec[1], path_vec[0])
    heading_err = wrap_pi(target_yaw - yaw)

    # # 4. PID Logic for CTE
    # # Gains (Tuned for a typical small robot)
    # Kp = 1.5   # Proportional: Corrects current drift
    # Ki = 0.1  # Integral: Corrects systematic bias (e.g., motor imbalance)
    # Kd = 0.3   # Derivative: Prevents overshooting the line
    
    # state.integral += cte * dt
    # derivative = (cte - state.last_error) / dt
    # state.last_error = cte

    # # Steering command: Combine line correction + heading alignment
    # # If CTE is positive (robot is left of line), we need to steer right (negative w)
    # w = (Kp * -cte) + (Ki * -state.integral) + (Kd * -derivative) + (1.0 * heading_err)

    # 5. Forward Velocity (Slow down if we are way off or turned sideways)
    # v_fwd = 0.6 * abs(math.cos(heading_err))
    # v_fwd = max(0.1, v_fwd) if abs(cte) < 0.5 else 0.1
    v_fwd = 0.7

    # # # 6. Differential Drive Output
    # # left = v_fwd - 0.45 * w
    # # right = v_fwd + 0.45 * w
    # w=w/0.45

    # return np.array([v_fwd, w], dtype=np.float32)

    # 1. Heading Error (Alignment with path)
    heading_err = wrap_pi(target_yaw - yaw)

    # 2. Cross-Track Error (Distance from path)
    # Note: Stanley controller uses arctan(k * error / velocity)
    # k_cte is a gain parameter, e.g., 2.0
    k_cte = 1.0
    cte_correction = math.atan2(k_cte * -cte, max(v_fwd, 0.1))

    # 3. Calculate Self-Centric Heading Angle (Steering Command)
    # This is the angle the robot *should* face relative to its current body
    steering_angle = wrap_pi(heading_err + cte_correction)

    # Output: [Velocity, Steering Angle]
    # Note: We do NOT multiply by gains to get 'w' here. We return the angle itself.
    return np.array([v_fwd, steering_angle], dtype=np.float32)