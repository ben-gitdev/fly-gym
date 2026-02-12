import numpy as np
import math
from typing import List, Tuple, Optional
from core.utils import wrap_pi
import heapq
from dataclasses import dataclass, field

@dataclass(order=True)
class SearchNode:
    f_score: float
    g_score: float = field(compare=False)
    x: float = field(compare=False)
    y: float = field(compare=False)
    yaw: float = field(compare=False)
    depth: int = field(compare=False)
    parent: Optional['SearchNode'] = field(compare=False, default=None)

class VFHPlusPlanner:
    def __init__(self, 
                 arena_half_extent: float,
                 robot_radius: float = 0.2,
                 safety_dist: float = 0.1,
                 active_window: float = 3, # meters, lookahead window
                 sector_angle: float = 2.0, # degrees
                 u_max: float = 10.0, # max certainty for histogram
                 wide_opening_width: int = 5, # number of sectors for wide opening
                 search_depth: int = 5, # A* search depth
                 step_size: float = 0.6, # A* step size
                 goal_threshold: float = 0.3): # distance threshold for goal reached
        
        self.arena_half_extent = float(arena_half_extent)
        self.goal_threshold = float(goal_threshold)
        self.robot_radius = robot_radius
        self.safety_dist = safety_dist
        self.active_window = active_window
        self.alpha = float(np.deg2rad(sector_angle))
        self.num_sectors = int(np.ceil(2 * np.pi / self.alpha))
        self.u_max = u_max
        self.wide_opening_width = wide_opening_width
        
        # Histograms
        self.primary_hist = np.zeros(self.num_sectors, dtype=np.float32)
        self.binary_hist = np.zeros(self.num_sectors, dtype=bool)
        
        # Heuristic weights
        self.mu1 = 10.0 # Target direction weight
        self.mu2 = 4.0 # Current direction weight (alignment with robot heading)
        self.mu3 = 10.0 # Previous direction weight (hysteresis) - increased to reduce jumping
        self.mu4 = 15.0 # Path consistency weight (deviation from previous path direction)
        
        # Direction stabilization parameters
        self.smoothing_factor = 0.3  # Exponential smoothing (lower = smoother, 0.1-0.5 range)
        self.max_angular_rate = np.deg2rad(5.0)  # Relaxed for search, we penalize in cost
        
        self.prev_steering_dir = 0.0
        self.prev_path_dir = None  # Direction to first waypoint of previous path
        
        # A* params
        self.search_depth = search_depth
        self.step_size = step_size

    def _update_histograms(self, start_xy: np.ndarray, obstacles_xy: np.ndarray, obstacle_radius: float,
                           current_yaw: float = None, fov: float = None):
        """
        Builds the Primary Polar Histogram based on raw obstacle positions.
        Considers obstacles within the active window (circular) around start_xy.
        If current_yaw and fov are provided, only considers obstacles within the FOV.
        """
        self.primary_hist.fill(0.0)
        
        if len(obstacles_xy) == 0:
            return

        # Vectorized distance calculation
        dx = obstacles_xy[:, 0] - start_xy[0]
        dy = obstacles_xy[:, 1] - start_xy[1]
        dist_sq = dx**2 + dy**2
        dist = np.sqrt(dist_sq)

        # Filter by active window (circular)
        mask = dist <= self.active_window
        
        # Filter by FOV if provided
        if current_yaw is not None and fov is not None:
            angles = np.arctan2(dy, dx)
            angle_diff = np.abs(wrap_pi(angles - current_yaw))
            fov_rad = np.deg2rad(fov) / 2.0
            mask = mask & (angle_diff <= fov_rad)
        
        valid_dx = dx[mask]
        valid_dy = dy[mask]
        valid_dist = dist[mask]
        
        # Safe radius for expansion: Robot Radius + Safety Margin + Obstacle Radius
        # We assume obstacle_radius is scalar for now (all obstacles same size) or max size
        safe_r = self.robot_radius + self.safety_dist + obstacle_radius

        # Pre-compute sector angles for vectorized operations
        sec_angles = np.arange(self.num_sectors) * self.alpha
        
        for i in range(len(valid_dist)):
            d = valid_dist[i]
            beta = math.atan2(valid_dy[i], valid_dx[i])
            
            # Calculate enlargement angle
            if d < safe_r:
                enlargement_angle = np.pi/2 # Block full circle
            else:
                enlargement_angle = math.asin(safe_r / d)
            
            # Magnitude profile
            m_val = (1.0 - (d / self.active_window)) * self.u_max
            if m_val < 0: m_val = 0.0

            # Vectorized sector update (replaces inner for-loop)
            diffs = np.abs(np.arctan2(np.sin(sec_angles - beta), np.cos(sec_angles - beta)))
            sector_mask = diffs <= enlargement_angle
            self.primary_hist[sector_mask] += m_val

        # Wall detection
        # Walls are at x = +/- arena_half_extent and y = +/- arena_half_extent
        A = self.arena_half_extent - 0.1
        
        # Ray casting for each sector to detect walls
        for k in range(self.num_sectors):
            angle = k * self.alpha
            # Ray direction
            rx = math.cos(angle)
            ry = math.sin(angle)
            
            dist_to_wall = float('inf')
            
            # Check intersection with 4 walls
            # 1. x = A (Right wall)
            if rx > 1e-3:
                d = (A - start_xy[0]) / rx
                if d >= 0: dist_to_wall = min(dist_to_wall, d)
            # 2. x = -A (Left wall)
            if rx < -1e-3:
                 d = (-A - start_xy[0]) / rx
                 if d >= 0: dist_to_wall = min(dist_to_wall, d)
            # 3. y = A (Top wall)
            if ry > 1e-3:
                d = (A - start_xy[1]) / ry
                if d >= 0: dist_to_wall = min(dist_to_wall, d)
            # 4. y = -A (Bottom wall)
            if ry < -1e-3:
                d = (-A - start_xy[1]) / ry
                if d >= 0: dist_to_wall = min(dist_to_wall, d)
                
            # If wall is within active window, add to histogram
            if dist_to_wall <= self.active_window:
                 # Magnitude logic same as obstacles
                 m_val = (1.0 - (dist_to_wall / self.active_window)) * self.u_max
                 if m_val < 0: m_val = 0.0
                 self.primary_hist[k] += m_val

    def _binary_polar_histogram(self, threshold: float):
        # Hysteresis thresholding
        # If bin > high -> occupied (1)
        # If bin < low -> free (0)
        # If low < bin < high -> keep previous (we'll just use high for simplicity if no history per step kept efficiently)
        # For this implementation, simplified single threshold or simple hysteresis if we kept state
        
        # To strictly follow VFH+, we should update based on previous binary hist, but we rebuild every step
        # So we just use one threshold or a conservative one.
        limit = threshold
        self.binary_hist = self.primary_hist > limit
        
    def _find_candidate_directions(self, target_dir: float) -> List[float]:
        # Identify valleys (sequences of free sectors)
        candidates = []
        
        # Find consecutive free sectors
        # Handle wrap around by concatenating or careful indexing
        # Let's verify sectors [0, n] and wrap
        
        is_free = ~self.binary_hist
        
        # Find runs of True in is_free
        # We can construct a list of valleys: (start_index, end_index)
        valleys = []
        if np.all(is_free):
            valleys.append((0, self.num_sectors + 0)) # Full circle
        elif np.any(is_free):
             # classic run finding
             # Rotate to start with a non-free to simplify or just standard logic
             # Let's find starts of valleys (False -> True transitions)
             extended = np.concatenate([is_free, is_free])
             
             # Locate openings
             for i in range(self.num_sectors):
                 prev_idx = (i - 1) % self.num_sectors  # Proper wraparound
                 if not is_free[prev_idx] and is_free[i]:
                     # Found start of a valley at i
                     # Trace forward with safety limit
                     j = i
                     max_trace = 2 * self.num_sectors  # Safety limit
                     while j < max_trace and extended[j]:
                         j += 1
                     # Valley is [i, j-1] (modulo num_sectors effectively)
                     valleys.append((i, j-1))
        
        # Process valleys to find candidate directions
        for start, end in valleys:
            width = end - start + 1
            # Sector indices, can be > num_sectors if wrapped in extended logic, need to normalize for angle calc
            
            # Wide valley
            if width > self.wide_opening_width:
                # "Wide" valley: logic is usually to follow the side that is closer to target, or target itself if within
                
                # Check if target is within this sector range
                # Target angle in indices
                target_idx = target_dir / self.alpha
                # Normalize target_idx to "match" the valley range (handle wrap)
                # This is tricky with indices. convert valley to angles.
                
                s_angle = wrap_pi(start * self.alpha)
                e_angle = wrap_pi(end * self.alpha) # careful, end might have wrapped
                
                # Simpler approach: check angles directly
                # Mid angle
                mid_angle = wrap_pi((s_angle + e_angle) / 2.0) # Careful with wrapping averagin
                # Actually, easier to use vector averaging or just stay in linear index space if we unwrapped carefully
                
                # Re-evaluate logic:
                # 1. Target direction is within the valley? then choose target direction.
                # 2. Else choose the edge of the valley closer to target.
                
                # Let's work with 'unwrapped' indices relative to start
                # valley indices: start ... end
                # target index (relative to start?)
                
                # Map target_dir to an index k such that start <= k <= end ?
                # We need to handle the wrap properly.
                # Convert target_dir to [0, 2pi) then to index
                t_idx = (wrap_pi(target_dir) if wrap_pi(target_dir) >= 0 else wrap_pi(target_dir) + 2*np.pi) / self.alpha
                
                # If t_idx is effectively inside [start, end] (modulo N)
                # ...
                
                # Heuristic: Generate 'left' and 'right' candidates (edges) + target if inside
                cand_angles = []
                
                # Left edge (start) + safety margin (some sectors)
                k_l = start + min(width//2, self.wide_opening_width//2) # somewhat inside
                ang_l = k_l * self.alpha
                cand_angles.append(ang_l)
                
                # Right edge (end) - safety
                k_r = end - min(width//2, self.wide_opening_width//2)
                ang_r = k_r * self.alpha
                cand_angles.append(ang_r)
                
                # Target if inside?
                # Simple check: is angle difference small?
                # or verify if target vector is between start vector and end vector
                
                for c in cand_angles:
                    candidates.append(wrap_pi(c))
                # Add target if it seems "safe" (i.e. is inside this valley)
                # We can check specific target angle against binary hist
                t_k = int((wrap_pi(target_dir) % (2*np.pi)) / self.alpha)
                if is_free[t_k]:
                    candidates.append(target_dir)
                    
            else:
                # Narrow valley: choose center
                center_idx = (start + end) / 2.0
                candidates.append(wrap_pi(center_idx * self.alpha))
                
        # Also always consider "target" if it's strictly free? (Already handled above)
        
        return candidates

    def _astar_search(self, start_xy: np.ndarray, goal_xy: np.ndarray, 
                      obstacles_xy: np.ndarray, obstacle_radius: float, 
                      start_yaw: float) -> Optional[List[np.ndarray]]:
        
        start_node = SearchNode(
            f_score=0.0, g_score=0.0,
            x=start_xy[0], y=start_xy[1], yaw=start_yaw,
            depth=0, parent=None
        )
        
        open_set = [start_node]
        heapq.heapify(open_set)
        
        # Track best node (closest to goal) in case we don't reach target perfectly
        closest_node = start_node
        min_dist = float('inf')
        
        max_nodes = 300 # Safety limit
        count = 0
        
        # Closed set for duplicate detection (discretized states)
        closed_set = set()

        dist_to_goal = math.hypot(goal_xy[0] - start_node.x, goal_xy[1] - start_node.y)
        search_dist = min(self.search_depth * self.step_size, dist_to_goal)
        search_depth = max(1, int(search_dist / self.step_size))
        
        # Limit candidates per expansion to control branching factor
        max_candidates = 5
        
        while open_set and count < max_nodes:
            count += 1
            current = heapq.heappop(open_set)
            
            # Skip if already visited (discretize to avoid floating point issues)
            state_key = (round(current.x, 2), round(current.y, 2), round(current.yaw, 2))
            if state_key in closed_set:
                continue
            closed_set.add(state_key)
            
            dist_to_goal = math.hypot(goal_xy[0] - current.x, goal_xy[1] - current.y)
            if dist_to_goal < min_dist:
                min_dist = dist_to_goal
                closest_node = current
            
            # Termination: Goal reached or Depth limit
            if dist_to_goal < self.goal_threshold or current.depth >= search_depth:
                return self._reconstruct_path(current)
            
            # --- Expand ---
            # 1. Update histogram at current node position
            pos = np.array([current.x, current.y])
            self._update_histograms(pos, obstacles_xy, obstacle_radius)
            self._binary_polar_histogram(threshold=0)
            
            # 2. Candidates
            dx = goal_xy[0] - current.x
            dy = goal_xy[1] - current.y
            target_dir = math.atan2(dy, dx)
            
            candidates = self._find_candidate_directions(target_dir)
            
            if not candidates:
                continue
            
            # Score and limit candidates to control branching factor
            scored_candidates = []
            for cand_dir in candidates:
                d_target = abs(wrap_pi(cand_dir - target_dir))
                d_yaw = abs(wrap_pi(cand_dir - current.yaw))
                score = self.mu1 * d_target + self.mu2 * d_yaw
                scored_candidates.append((score, cand_dir))
            scored_candidates.sort(key=lambda x: x[0])
            candidates = [c[1] for c in scored_candidates[:max_candidates]]
                
            for cand_dir in candidates:
                # 4. Next state (compute first to check closed set)
                next_x = current.x + self.step_size * math.cos(cand_dir)
                next_y = current.y + self.step_size * math.sin(cand_dir)
                
                # Skip if already visited (check BEFORE adding to heap)
                child_key = (round(next_x, 2), round(next_y, 2), round(cand_dir, 2))
                if child_key in closed_set:
                    continue
                
                # 3. Calculate Cost
                # Change in direction
                d_yaw = abs(wrap_pi(cand_dir - current.yaw))
                
                # Deviation from goal
                d_target = abs(wrap_pi(cand_dir - target_dir))
                
                # Deviation from previous direction (hysteresis)
                prev_yaw = current.parent.yaw if current.parent else current.yaw
                d_prev = abs(wrap_pi(cand_dir - prev_yaw))
                
                # Deviation from previous path direction (path consistency)
                # Only apply at depth 0 (first step) where path changes are most visible
                if current.depth == 0 and self.prev_path_dir is not None:
                    d_path = abs(wrap_pi(cand_dir - self.prev_path_dir))
                else:
                    d_path = 0.0
                
                # Heuristic Weights adapted from VFH+
                # step cost essentially penalizes length
                # we weight the angular costs
                
                edge_cost = (
                    self.step_size + 
                    self.mu1 * d_target * 0.1 + 
                    self.mu2 * d_yaw * 0.1 +
                    self.mu3 * d_prev * 0.1 +  # Hysteresis cost
                    self.mu4 * d_path * 0.1    # Path consistency cost
                )
                
                new_g = current.g_score + edge_cost
                
                # Heuristic
                h = math.hypot(goal_xy[0] - next_x, goal_xy[1] - next_y)
                
                new_f = new_g + h
                
                # Create Child
                child = SearchNode(
                    f_score=new_f, g_score=new_g,
                    x=next_x, y=next_y, yaw=cand_dir,
                    depth=current.depth + 1,
                    parent=current
                )
                heapq.heappush(open_set, child)
                
        # Return best path found if loop finishes
        return self._reconstruct_path(closest_node)

    def _reconstruct_path(self, node: SearchNode) -> List[np.ndarray]:
        path = []
        curr = node
        while curr is not None:
            path.append(np.array([curr.x, curr.y], dtype=np.float32))
            curr = curr.parent
        return path[::-1] # Reverse

    def plan(self, start_xy: np.ndarray, goal_xy: np.ndarray, obstacles_xy: np.ndarray, 
             obstacle_radius: float, current_yaw: float, fov: float = 160.0) -> Optional[List[np.ndarray]]:
        """
        VFH* local planning step (A* Lookahead).
        Returns a path: [start_xy, wp1, wp2, ...].
        """
        # Run A* Search using prev_steering_dir as starting orientation
        # This makes the search prefer directions consistent with previous steering
        path = self._astar_search(start_xy, goal_xy, obstacles_xy, obstacle_radius, self.prev_steering_dir)
        
        if path is None or len(path) < 2:
            # Fallback if stuck: simple rotation or stop
            # Try finding *any* free direction from start
            self._update_histograms(start_xy, obstacles_xy, obstacle_radius, current_yaw, fov)
            self._binary_polar_histogram(threshold=0)
            candidates = self._find_candidate_directions(current_yaw)
            if candidates:
                best_dir = candidates[0] # Pick first
                wx = start_xy[0] + self.step_size * math.cos(best_dir)
                wy = start_xy[1] + self.step_size * math.sin(best_dir)
                path = [start_xy, np.array([wx, wy], dtype=np.float32)]
            else:
                return [start_xy, start_xy] # Stop
        
        # Update prev steering dir based on first step with smoothing and rate limiting
        if len(path) > 1:
            p0 = path[0]
            p1 = path[1]
            raw_yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            
            # Store raw path direction for consistency cost in next call
            self.prev_path_dir = raw_yaw
            
            # Apply exponential smoothing
            smoothed_yaw = wrap_pi(
                self.prev_steering_dir + 
                self.smoothing_factor * wrap_pi(raw_yaw - self.prev_steering_dir)
            )
            
            # Apply rate limiting
            delta = wrap_pi(smoothed_yaw - self.prev_steering_dir)
            if abs(delta) > self.max_angular_rate:
                delta = np.sign(delta) * self.max_angular_rate
            
            self.prev_steering_dir = wrap_pi(self.prev_steering_dir + delta)

        return path
