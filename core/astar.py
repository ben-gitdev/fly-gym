from dataclasses import dataclass
import math
import heapq
from typing import List, Tuple, Optional
import numpy as np
from scipy.ndimage import distance_transform_edt
# ----------------------------
# A* planner on a grid
# ----------------------------
@dataclass
class GridSpec:
    arena_half_extent: float
    cell_size: float
    obstacle_inflate: float

class AStarGridPlanner:
    def __init__(self, spec: GridSpec):
        self.spec = spec
        self._cached_occ = None
        self._cached_obstacles_hash = None
        self._W = None
        self._H = None
        self._origin = None 
        self._cached_clearance = None

    def _world_to_cell(self, xy: np.ndarray) -> Tuple[int, int]:
        x, y = float(xy[0]), float(xy[1])
        ox, oy = self._origin
        c = int(round((x - ox) / self.spec.cell_size))
        r = int(round((y - oy) / self.spec.cell_size))
        return c, r

    def _cell_to_world(self, c: int, r: int) -> np.ndarray:
        ox, oy = self._origin
        x = ox + c * self.spec.cell_size
        y = oy + r * self.spec.cell_size
        return np.array([x, y], dtype=np.float32)

    def _build_occupancy(self, obstacles_xy: np.ndarray, obstacle_radius: float) -> np.ndarray:
        A = self.spec.arena_half_extent
        cs = self.spec.cell_size
        C = float(self.spec.obstacle_inflate)
        W = int(math.floor((2 * A) / cs)) + 1
        H = int(math.floor((2 * A) / cs)) + 1
        self._W, self._H = W, H
        self._origin = (-A, -A)
        occ = np.zeros((H, W), dtype=np.uint8)
        xs = np.linspace(-A, A, W, dtype=np.float32)
        ys = np.linspace(-A, A, H, dtype=np.float32)
        x_min, x_max = -A + C, A - C
        y_min, y_max = -A + C, A - C
        bad_x = (xs < x_min) | (xs > x_max)
        if np.any(bad_x): occ[:, bad_x] = 1
        bad_y = (ys < y_min) | (ys > y_max)
        if np.any(bad_y): occ[bad_y, :] = 1
        R = float(obstacle_radius + C)
        RR = R * R
        for (ox, oy) in obstacles_xy:
            dx = xs - float(ox)
            dy = ys - float(oy)
            dist2 = (dy[:, None] ** 2) + (dx[None, :] ** 2)
            occ |= (dist2 <= RR).astype(np.uint8)
        free = (occ == 0)
        clearance = distance_transform_edt(free) * cs
        self._cached_clearance = clearance
        return occ

    def _hash_obstacles(self, obstacles_xy: np.ndarray) -> int:
        q = np.round(obstacles_xy.astype(np.float32) * 1000.0).astype(np.int32)
        return hash(q.tobytes())

    def plan(self, start_xy, goal_xy, obstacles_xy, obstacle_radius, max_expand=20000, smooth_path: Optional[bool] = True):
        obs_hash = self._hash_obstacles(obstacles_xy)
        if self._cached_occ is None or obs_hash != self._cached_obstacles_hash:
            self._cached_occ = self._build_occupancy(obstacles_xy, obstacle_radius)
            self._cached_obstacles_hash = obs_hash
        occ = self._cached_occ
        W, H = self._W, self._H
        start = self._world_to_cell(start_xy)
        goal = self._world_to_cell(goal_xy)

        def inb(c: int, r: int) -> bool: return 0 <= c < W and 0 <= r < H
        def blocked(c: int, r: int) -> bool: return bool(occ[r, c])

        if not inb(*start) or not inb(*goal): 
            # print("Start or goal out of bounds, cannot plan.")
            return None

        def nearest_free(cell, max_rad=12):
            c0, r0 = cell
            if inb(c0, r0) and not blocked(c0, r0): return cell
            for rad in range(1, max_rad + 1):
                for dc in range(-rad, rad + 1):
                    for dr in range(-rad, rad + 1):
                        cc, rr = c0 + dc, r0 + dr
                        if inb(cc, rr) and not blocked(cc, rr): return (cc, rr)
            return None

        if blocked(*start): start = nearest_free(start)
        if blocked(*goal): goal = nearest_free(goal)
        if start is None or goal is None: return None

        nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        def h(c, r): return math.hypot(goal[0] - c, goal[1] - r)
        open_heap = []
        heapq.heappush(open_heap, (h(*start), 0.0, start))
        came = {start: None}
        gscore = {start: 0.0}
        expanded = 0

        while open_heap and expanded < max_expand:
            _, gcur, cur = heapq.heappop(open_heap)
            if cur == goal:
                path_cells = []
                node = cur
                while node is not None:
                    path_cells.append(node)
                    node = came[node]
                path_cells.reverse()
                _path = [self._cell_to_world(c, r) for (c, r) in path_cells]
                return self.smooth_path(_path) if smooth_path else _path
            expanded += 1
            cc, rr = cur
            for dc, dr in nbrs:
                nc, nr = cc + dc, rr + dr
                if not inb(nc, nr) or blocked(nc, nr): continue
                step = math.hypot(dc, dr) * self.spec.cell_size
                d_clear = float(self._cached_clearance[nr, nc])
                sigma = float(self.spec.obstacle_inflate)
                soft = 0.8 * math.exp(-d_clear / max(1e-6, sigma))
                ng = gcur + step + soft
                nxt = (nc, nr)
                if nxt not in gscore or ng < gscore[nxt]:
                    gscore[nxt] = ng
                    came[nxt] = cur
                    heapq.heappush(open_heap, (ng + h(nc, nr), ng, nxt))
        return None
    
    def smooth_path(self, path: List[np.ndarray]) -> List[np.ndarray]:
        if path is None or len(path) < 3: return path
        smoothed = [path[0]]
        current_idx = 0
        while current_idx < len(path) - 1:
            next_idx = current_idx + 1
            for i in range(len(path) - 1, current_idx, -1):
                if self.is_line_safe(path[current_idx], path[i]):
                    next_idx = i
                    break
            smoothed.append(path[next_idx])
            current_idx = next_idx
        return smoothed

    def is_line_safe(self, start_xy, end_xy) -> bool:
        p1 = self._world_to_cell(start_xy)
        p2 = self._world_to_cell(end_xy)
        x0, y0 = p1
        x1, y1 = p2
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if self.cached_occ_safe(x, y): return False
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if self.cached_occ_safe(x, y): return False
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        return True

    def cached_occ_safe(self, c, r):
        if 0 <= c < self._W and 0 <= r < self._H:
            return self._cached_occ[r, c] == 1
        return True
