"""
DAgger imitation learning for the connectome RNN agent using vision-only inputs.
Refactored to fix terminal sampling bias, action bounds, and precision issues.
"""

from __future__ import annotations
import time as _time
import csv
import os
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt

import numpy as np
import torch
import cv2


from environment.mujoco_two_cam_env_random_obstacles import MuJoCoTwoCamEnv
from agents.teacher_analytic_agent import PlannerAnalyticTeacher
from agents.connectome_rnn_agent import ConnectomeAgent
from core.utils import (
    get_device,
    build_connectome_cell,
    obs_to_torch,
)

# -----------------------------
# Paths (hardcoded)
# -----------------------------
from shared_config import (
    EDGE_PATH,
    PHOTORECEPTOR_LEFT_CSV,
    PHOTORECEPTOR_RIGHT_CSV,

    TACTILE_LEFT_CSV,
    TACTILE_RIGHT_CSV,
    DESCENDING_NEURONS_CSV,
    CELL_TYPES_CSV,
    WIND_SENSING_CSV,
    ENV_WIDTH,
    ENV_HEIGHT,
    MAX_EPISODE_STEPS,
    N_OBSTACLES,
    ARENA_HALF_EXTENT,
    RENDER_MODE,
    LEAK_ALPHA,
    ACTIVATION,
    TARGET_RHO,
    BATCH_CHUNK,
    ROW_TILE_SIZE,
    TRAIN_RNN_WEIGHTS,
    TRAIN_READOUT_HEAD,
    INPUT_SCALE_INIT,
    DTYPE,
    USE_GRADIENT_CHECKPOINT,
    CHECKPOINT_DIR,
    LOSS_DIR,
)

END_ON_COLLISION = False
# Import shared optimizer configuration
from shared_config import configure_optimizer

# -----------------------------
# DAgger Hyperparameters
# -----------------------------
N_DAGGER_ITERS = 1
EPISODES_PER_ITER = 650
TRAIN_STEPS_PER_ITER = 400
N_ENVS = 10  # Number of concurrent environments for vectorized rollout

BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 4  # Number of mini-batches to accumulate before optimizer step

T_UNROLL = 80
T_BURN = 50
LR = 3e-4

BETA_START = 0.0 # 1.0: teacher drive, 0.0: agent drive
BETA_END = 0.0
BETA_DECAY = 0.5
BETA_WARMUP = 0  # Number of iterations to keep beta=1.0 at start

STEERING_LOSS_SCALE = 2.0  # Prioritize steering accuracy over velocity

NOISE_INTERVAL = 10
START_NOISE = 0.5
NOISE_DECAY = 0.0


# -----------------------------
# Balanced Buffer Configuration
# -----------------------------
RATIO_STRAIGHT = 0.2
RATIO_TURN = 0.25
RATIO_COLLISION = 0.2
RATIO_PRE_COLLISION = 0.2
RATIO_START = 0.15

# Path Analysis Parameters
DIRECTION_THRESHOLD_DEG = 1.0  # Degrees threshold for straight vs turn
CONSECUTIVE_SEGMENTS_N = 5    # Number of segments to analyze for classification
COLLISION_LOOKBACK = T_BURN + 50       # Steps to look back before collision (matches T_UNROLL for full context)
COLLISION_RECOVERY_WINDOW = T_UNROLL + 50  # Steps after collision for recovery
MAX_CHUNKS_PER_CATEGORY = 10000  # Max chunks stored per category

LOSS_CSV_PATH = os.path.join(LOSS_DIR, "connectome_rnn_dagger_loss.csv")
FINAL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "connectome_rnn_dagger_princeton_base.pt")

# Path to checkpoint to resume from (set to None to train from scratch)
RESUME_CHECKPOINT_PATH = "checkpoints/connectome_rnn_dagger_princeton.pt"


def maybe_show_cameras(obs):
    if RENDER_MODE != "human" or cv2 is None: return
    left_gray = cv2.cvtColor(obs["cam_left"], cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(obs["cam_right"], cv2.COLOR_RGB2GRAY)
    frame = np.hstack([left_gray, right_gray])
    cv2.imshow("MuJoCo cams (left | right, gray)", frame)
    cv2.waitKey(1)

def beta_schedule(iter_idx):
    if iter_idx < BETA_WARMUP:
        beta = 1.0
    else: # linear decay
        beta = BETA_START - BETA_DECAY * (iter_idx - BETA_WARMUP)
    return max(BETA_END, beta)

def noise_schedule(iter_idx):
    if iter_idx < BETA_WARMUP:
        noise = 0.0
    else:
        noise = START_NOISE - NOISE_DECAY * (iter_idx - BETA_WARMUP)
    return max(0.0, noise)


def _make_env(render_mode=None, seed = None):
    """Create a single MuJoCo environment with shared settings."""
    return MuJoCoTwoCamEnv(
        width=ENV_WIDTH,
        height=ENV_HEIGHT,
        max_episode_steps=MAX_EPISODE_STEPS,
        n_obstacles=N_OBSTACLES,
        arena_half_extent=ARENA_HALF_EXTENT,
        render_mode=render_mode,
        end_on_collision=END_ON_COLLISION,
        seed=seed,
    )

def _make_teacher():
    """Create a single PlannerAnalyticTeacher with shared settings."""
    return PlannerAnalyticTeacher(
        arena_half_extent=ARENA_HALF_EXTENT,
        cell_size=0.1,
        robot_radius=0.2,
        safety_margin=0.1,
        obstacle_box_half=(0.4, 0.4),
        k_nearest_obs=5,
        device="cpu",
    )


class PathAnalyzer:
    """Analyze robot trajectory to identify straight, turn, and collision keypoints."""
    
    def __init__(
        self,
        direction_threshold_deg: float = DIRECTION_THRESHOLD_DEG,
        consecutive_n: int = CONSECUTIVE_SEGMENTS_N,
        collision_lookback: int = COLLISION_LOOKBACK,
        collision_recovery: int = COLLISION_RECOVERY_WINDOW,
    ):
        self.direction_threshold_rad = np.deg2rad(direction_threshold_deg)
        self.consecutive_n = consecutive_n
        self.collision_lookback = collision_lookback
        self.collision_recovery = collision_recovery
    
    def extract_xy_path(self, raw_obs_list: List[dict]) -> np.ndarray:
        """Extract (x, y) positions from privileged observations."""
        return np.array([obs["privileged"][:2] for obs in raw_obs_list], dtype=np.float32)
    
    def extract_collision_flags(self, raw_obs_list: List[dict]) -> np.ndarray:
        """Extract collision booleans from sensor data."""
        return np.array([obs["sensors"]["collision"] for obs in raw_obs_list], dtype=bool)
    
    def _angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate signed angle between two 2D vectors."""
        # Handle zero-length vectors
        len1, len2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if len1 < 1e-8 or len2 < 1e-8:
            return 0.0
        
        # Calculate angle using atan2 for proper sign
        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        diff = angle2 - angle1
        
        # Wrap to [-pi, pi]
        while diff > np.pi: diff -= 2 * np.pi
        while diff < -np.pi: diff += 2 * np.pi
        return diff
    
    def _classify_segment(self, xy_path: np.ndarray, start_idx: int) -> Optional[str]:
        """Classify a segment starting at start_idx as 'straight' or 'turn'."""
        if start_idx + self.consecutive_n + 1 >= len(xy_path):
            return None
        
        # Calculate direction changes over n consecutive segments
        max_change = 0.0
        for i in range(start_idx, start_idx + self.consecutive_n):
            if i + 2 >= len(xy_path):
                break
            v1 = xy_path[i + 1] - xy_path[i]
            v2 = xy_path[i + 2] - xy_path[i + 1]
            angle_change = abs(self._angle_between_vectors(v1, v2))
            max_change = max(max_change, angle_change)
        
        if max_change < self.direction_threshold_rad:
            return 'straight'
        else:
            return 'turn'
    
    def find_keypoints(self, raw_obs_list: List[dict], xy_path: np.ndarray, collisions: np.ndarray) -> dict:
        """
        Find keypoints marking the start of straight, turn, and collision events.
        
        Returns:
            dict with 'straight_starts', 'turn_starts', 'collision_points' lists
        """
        n = len(xy_path)
        keypoints = {
            'straight_starts': [],
            'turn_starts': [],
            'collision_points': [],
        }
        
        # Find collision points first
        collision_indices = np.where(collisions)[0]
        for idx in collision_indices:
            keypoints['collision_points'].append(idx)
        
        # Create a set of collision-affected regions (collision ± recovery window)
        collision_regions = set()
        for cp in keypoints['collision_points']:
            for i in range(max(0, cp - self.collision_lookback), 
                          min(n, cp + self.collision_recovery)):
                collision_regions.add(i)
        
        # Scan through trajectory to find straight/turn transitions
        prev_classification = None
        i = 0
        while i < n - self.consecutive_n - 1:
            # Skip collision regions
            if i in collision_regions:
                i += 1
                prev_classification = None
                continue
            
            classification = self._classify_segment(xy_path, i)
            if classification is None:
                break
            
            # Detect transitions
            if classification != prev_classification:
                if classification == 'straight':
                    keypoints['straight_starts'].append(i)
                elif classification == 'turn':
                    keypoints['turn_starts'].append(i)
            
            prev_classification = classification
            i += 1

        # Filter collision points to avoid repeated collision recordings at the same location
        unique_collision_points = []
        if keypoints['collision_points']:
            last_pos = None
            for cp in keypoints['collision_points']:
                # Extract position from raw observation
                curr_pos = np.array(raw_obs_list[cp]['privileged'][:2])
                
                if last_pos is None or np.linalg.norm(curr_pos - last_pos) > 1.0:
                    unique_collision_points.append(cp)
                    last_pos = curr_pos
        keypoints['collision_points'] = unique_collision_points

        return keypoints
    
    def _find_next_keypoint(self, idx: int, keypoints: dict, max_idx: int) -> int:
        """Find the next keypoint (straight, turn, or collision) after idx."""
        next_points = []
        for key in ['straight_starts', 'turn_starts', 'collision_points']:
            for p in keypoints[key]:
                if p > idx:
                    next_points.append(p)
        return min(next_points) if next_points else max_idx
    
    def segment_into_chunks(
        self, 
        processed_obs_list: List[dict],
        teacher_actions: np.ndarray,
        keypoints: dict,
        chunk_length: int,
        stride: int,
        protected_start_steps: int,
    ) -> dict:
        """
        Segment observations into 4 category chunks with proper overlap handling.
        
        Args:
            processed_obs_list: List of pre-processed observations from episode
            teacher_actions: Array of teacher actions (T, action_dim)
            keypoints: Dict with straight_starts, turn_starts, collision_points
            chunk_length: T_UNROLL + T_BURN
            stride: T_BURN (since overlap = T_UNROLL)
            protected_start_steps: Region protected for start chunk (chunk_length)
        
        Returns:
            dict with 'straight', 'turn', 'collision', 'start' lists of (obs_chunk, action_chunk)
        """
        n = len(processed_obs_list)
        chunks = {'straight': [], 'turn': [], 'collision': [], 'pre_collision': [], 'start': []}
        
        # 1. Extract START chunk (beginning of episode)
        if n >= chunk_length:
            obs_chunk = processed_obs_list[:chunk_length]
            act_chunk = teacher_actions[:chunk_length]
            chunks['start'].append((obs_chunk, act_chunk))
        
        # 2. Extract STRAIGHT chunks
        for start_idx in keypoints['straight_starts']:
            # Skip if in protected start region
            if start_idx < protected_start_steps:
                start_idx = protected_start_steps
            
            end_idx = self._find_next_keypoint(start_idx, keypoints, n)
            
            # Calculate number of chunks (round down)
            segment_length = end_idx - start_idx
            if segment_length < chunk_length:
                continue
            
            num_chunks = (segment_length - chunk_length) // stride + 1
            
            for i in range(num_chunks):
                chunk_start = start_idx + i * stride
                chunk_end = chunk_start + chunk_length
                if chunk_end <= n:
                    obs_chunk = processed_obs_list[chunk_start:chunk_end]
                    act_chunk = teacher_actions[chunk_start:chunk_end]
                    chunks['straight'].append((obs_chunk, act_chunk))
        
        # 3. Extract TURN chunks (shifted back by T_UNROLL from turn start)
        for turn_start in keypoints['turn_starts']:
            # Shift back T_UNROLL steps (chunk_length - stride) to capture approach to turn
            # chunk_length = T_BURN + T_UNROLL, stride = T_BURN
            actual_start = max(0, turn_start - (chunk_length - stride))
            
            # Skip if overlaps with protected start region
            if actual_start < protected_start_steps:
                actual_start = protected_start_steps
            
            end_idx = self._find_next_keypoint(turn_start, keypoints, n)
            
            # Calculate number of chunks (round up)
            segment_length = end_idx - actual_start
            if segment_length < chunk_length:
                # Still try to get at least one chunk if possible
                if actual_start + chunk_length <= n:
                    obs_chunk = processed_obs_list[actual_start:actual_start + chunk_length]
                    act_chunk = teacher_actions[actual_start:actual_start + chunk_length]
                    chunks['turn'].append((obs_chunk, act_chunk))
                continue
            
            num_chunks = -(-((segment_length - chunk_length)) // stride) + 1  # Ceiling division
            
            for i in range(num_chunks):
                chunk_start = actual_start + i * stride
                chunk_end = chunk_start + chunk_length
                if chunk_end <= n:
                    obs_chunk = processed_obs_list[chunk_start:chunk_end]
                    act_chunk = teacher_actions[chunk_start:chunk_end]
                    chunks['turn'].append((obs_chunk, act_chunk))
        
        # 4. Extract COLLISION chunks (shifted back by collision_lookback)
        for collision_point in keypoints['collision_points']:
            # Shift back to capture approach to collision
            actual_start = max(0, collision_point - self.collision_lookback)
            
            # Skip if overlaps with protected start region
            if actual_start < protected_start_steps:
                actual_start = protected_start_steps
            
            end_idx = min(n, collision_point + self.collision_recovery)
            
            # Calculate number of chunks (round up)
            segment_length = end_idx - actual_start
            if segment_length < chunk_length:
                continue
            
            num_chunks = -(-((segment_length - chunk_length)) // stride) + 1  # Ceiling division
            
            for i in range(num_chunks):
                chunk_start = actual_start + i * stride
                chunk_end = chunk_start + chunk_length
                if chunk_end <= n:
                    obs_chunk = processed_obs_list[chunk_start:chunk_end]
                    act_chunk = teacher_actions[chunk_start:chunk_end]
                    chunks['collision'].append((obs_chunk, act_chunk))
        
        # 5. Extract PRE-COLLISION chunks (ending at collision point t, i.e., up to t-1)
        for collision_point in keypoints['collision_points']:
            # Chunk ends at collision_point (exclusive), so it covers indices up to t-1
            chunk_end = collision_point
            chunk_start = chunk_end - chunk_length
            
            # Check validity: must start after protected region
            if chunk_start >= protected_start_steps:
                if chunk_end <= n:
                    obs_chunk = processed_obs_list[chunk_start:chunk_end]
                    act_chunk = teacher_actions[chunk_start:chunk_end]
                    chunks['pre_collision'].append((obs_chunk, act_chunk))
        
        return chunks


class BalancedDAggerBuffer:
    """4-category buffer for balanced training data sampling."""
    
    def __init__(self, capacity_per_category: int = MAX_CHUNKS_PER_CATEGORY):
        self.capacity = capacity_per_category
        
        # Each category stores processed chunks as lists of (xs, actions) tuples
        # xs: (chunk_length, Nin), actions: (chunk_length, action_dim)
        self.straight_chunks: List[Tuple[np.ndarray, np.ndarray]] = []
        self.turn_chunks: List[Tuple[np.ndarray, np.ndarray]] = []
        self.collision_chunks: List[Tuple[np.ndarray, np.ndarray]] = []
        self.pre_collision_chunks: List[Tuple[np.ndarray, np.ndarray]] = []
        self.start_chunks: List[Tuple[np.ndarray, np.ndarray]] = []
    
    def __len__(self) -> int:
        return (len(self.straight_chunks) + len(self.turn_chunks) + 
                len(self.collision_chunks) + len(self.pre_collision_chunks) + len(self.start_chunks))
    
    def get_counts(self) -> Dict[str, int]:
        """Return counts for each category."""
        return {
            'straight': len(self.straight_chunks),
            'turn': len(self.turn_chunks),
            'collision': len(self.collision_chunks),
            'pre_collision': len(self.pre_collision_chunks),
            'start': len(self.start_chunks),
        }
    
    def add_chunk(self, category: str, xs: np.ndarray, actions: np.ndarray):
        """Add a processed chunk to the appropriate buffer."""
        chunk = (xs.astype(np.float32), actions.astype(np.float32))
        
        target_list = getattr(self, f'{category}_chunks')
        target_list.append(chunk)
        
        # Enforce capacity limit (FIFO)
        if len(target_list) > self.capacity:
            target_list.pop(0)
    
    def sample_balanced_sequences(
        self, 
        batch_size: int,
        ratios: Tuple[float, float, float, float, float] = (RATIO_STRAIGHT, RATIO_TURN, RATIO_COLLISION, RATIO_PRE_COLLISION, RATIO_START),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from 5 buffers according to ratios.
        Ensures at least 1 sample per non-empty category.
        
        Returns:
            xs_arr: (T, B, Nin)
            act_arr: (T, B, Act)
        """
        r_str, r_turn, r_col, r_pre_col, r_start = ratios
        
        buffers = {
            'straight': self.straight_chunks,
            'turn': self.turn_chunks,
            'collision': self.collision_chunks,
            'pre_collision': self.pre_collision_chunks,
            'start': self.start_chunks,
        }
        ratio_map = {
            'straight': r_str,
            'turn': r_turn,
            'collision': r_col,
            'pre_collision': r_pre_col,
            'start': r_start,
        }
        
        # Calculate counts (ensure at least 1 if buffer non-empty)
        counts = {}
        total_available = 0
        for cat, buf in buffers.items():
            if len(buf) > 0:
                counts[cat] = max(1, int(batch_size * ratio_map[cat]))
                total_available += len(buf)
            else:
                counts[cat] = 0
        
        if total_available == 0:
            raise ValueError("All buffers are empty!")
        
        # Sample from each buffer
        samples = []
        for category, count in counts.items():
            buffer = buffers[category]
            if count > 0 and len(buffer) > 0:
                # Sample with replacement if needed
                indices = np.random.choice(len(buffer), size=min(count, len(buffer)), 
                                          replace=(count > len(buffer)))
                for idx in indices:
                    samples.append(buffer[idx])
        
        # Shuffle to mix categories
        np.random.shuffle(samples)
        
        # Stack into tensors
        xs_list = [s[0] for s in samples]
        act_list = [s[1] for s in samples]
        
        xs_arr = torch.from_numpy(np.stack(xs_list, axis=1))    # (T, B, Nin)
        act_arr = torch.from_numpy(np.stack(act_list, axis=1))  # (T, B, Act)
        
        return xs_arr, act_arr


def log_training_loss(iter_idx, mean_loss, batches, buffer_size, chunk_counts=None):
    os.makedirs(os.path.dirname(LOSS_CSV_PATH) or ".", exist_ok=True)
    write_header = not os.path.exists(LOSS_CSV_PATH)
    with open(LOSS_CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["dagger_iter", "mean_loss", "batches", "buffer_size", 
                           "straight_chunks", "turn_chunks", "collision_chunks", "pre_collision_chunks", "start_chunks"])
        if chunk_counts:
            writer.writerow([iter_idx, mean_loss, batches, buffer_size, 
                           chunk_counts.get('straight', 0), chunk_counts.get('turn', 0),
                           chunk_counts.get('collision', 0), chunk_counts.get('pre_collision', 0), chunk_counts.get('start', 0)])
        else:
            writer.writerow([iter_idx, mean_loss, batches, buffer_size, 0, 0, 0, 0, 0])

def save_checkpoint(agent, iter_idx):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"connectome_rnn_dagger_iter_{iter_idx}.pt")
    torch.save(agent.state_dict(), ckpt_path)
    return ckpt_path


def forward_policy_sequence(agent, xs):
    # Use the unified efficient sequence processing
    _, y_seq, _ = agent.forward_sequence(xs, checkpoint_steps=USE_GRADIENT_CHECKPOINT)
    return y_seq


def process_episode_to_chunks(
    agent,
    raw_obs_list: List[dict],
    processed_obs_list: List[np.ndarray],
    teacher_actions: np.ndarray,
    analyzer: PathAnalyzer,
    chunk_length: int,
    stride: int,
    device,
    dtype,
) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Process raw observations into categorized chunks.
    
    Args:
        agent: The agent (unused now, kept for API consistency)
        raw_obs_list: List of raw observation dicts (used for keypoint extraction)
        processed_obs_list: List of pre-computed obs_to_x numpy arrays (Nin,)
        teacher_actions: Array of teacher actions (T, action_dim)
        analyzer: PathAnalyzer instance
        chunk_length: T_UNROLL + T_BURN
        stride: T_BURN (overlap = T_UNROLL)
        device: Torch device
        dtype: Torch dtype
    
    Returns:
        Dict with 'straight', 'turn', 'collision', 'start' lists of (xs, actions) tuples
    """
    if len(raw_obs_list) < chunk_length:
        return {'straight': [], 'turn': [], 'collision': [], 'pre_collision': [], 'start': []}
    
    # 1. Extract trajectory and collision data
    xy_path = analyzer.extract_xy_path(raw_obs_list)
    collisions = analyzer.extract_collision_flags(raw_obs_list)
    
    # 2. Find keypoints
    keypoints = analyzer.find_keypoints(raw_obs_list, xy_path, collisions)
    
    # 3. Segment into chunks (respecting protected start region)
    # Uses processed_obs_list for slicing (already pre-computed numpy arrays)
    protected_start = chunk_length  # T_UNROLL + T_BURN
    obs_chunks_dict = analyzer.segment_into_chunks(
        processed_obs_list, teacher_actions, keypoints,
        chunk_length, stride, protected_start
    )
    
    # 4. Stack pre-computed numpy arrays into (T, Nin) chunks
    processed_chunks = {'straight': [], 'turn': [], 'collision': [], 'pre_collision': [], 'start': []}
    
    for category, chunks in obs_chunks_dict.items():
        for (xs_chunk, action_chunk) in chunks:
            if len(xs_chunk) != chunk_length:
                continue  # Skip invalid chunks
            
            # xs_chunk is a list of numpy arrays (Nin,) — just stack them
            xs_arr = np.stack(xs_chunk, axis=0)  # (T, Nin)
            processed_chunks[category].append((xs_arr, action_chunk))
    
    return processed_chunks


def rollout_and_collect_balanced(
    envs, teachers, agent, buffer: BalancedDAggerBuffer,
    beta: float, device, dtype, beta_noise: float,
    analyzer: PathAnalyzer, chunk_length: int, stride: int
):
    """
    Collect EPISODES_PER_ITER episodes using N concurrent envs with batched
    GPU inference.  Each env has its own teacher.

    The connectome agent's obs_to_x() is called per-env to maintain correct
    retinal temporal state.  The RNN step is batched across all active envs.
    """
    agent.eval()
    n_envs = len(envs)
    need_student = beta < 1.0

    total_chunks = {'straight': 0, 'turn': 0, 'collision': 0, 'pre_collision': 0, 'start': 0}
    episodes_done = 0

    # ---- per-env mutable state ----
    obs_list      = [None] * n_envs          # latest raw obs
    raw_obs_buf   = [[] for _ in range(n_envs)]  # raw obs history
    proc_obs_buf  = [[] for _ in range(n_envs)]  # pre-computed obs_to_x numpy arrays
    teacher_act_buf = [[] for _ in range(n_envs)]
    action_exec   = [np.zeros(2, dtype=np.float32) for _ in range(n_envs)]
    steps         = [0] * n_envs
    noise_disturbing = [False] * n_envs
    noise_steps   = [0] * n_envs
    noise_vals    = [np.zeros(2, dtype=np.float32) for _ in range(n_envs)]
    active        = [True] * n_envs
    done_flags    = [False] * n_envs
    trunc_flags   = [False] * n_envs

    # Per-env retinal vision states (dicts with L1/L2/L3 keys)
    vision_states_left  = [None] * n_envs
    vision_states_right = [None] * n_envs

    if need_student:
        h = torch.zeros(n_envs, agent.cell.N, device=device, dtype=dtype)

    # ---- reset all envs ----
    for i in range(n_envs):
        obs_list[i], _ = envs[i].reset()
        teachers[i].reset()

    t0 = _time.perf_counter()

    # ---- main loop: step all active envs until enough episodes collected ----
    while episodes_done < EPISODES_PER_ITER:
        # 1. MuJoCo step all active envs
        for i in range(n_envs):
            if not active[i]:
                continue
            obs, _, done_flags[i], trunc_flags[i], _ = envs[i].step(action_exec[i])
            obs_list[i] = obs
            steps[i] += 1
            raw_obs_buf[i].append(obs)

        # 2. Per-env preprocessing: obs_to_torch -> obs_to_x (stateful retinal processing)
        #    Must be sequential per env to maintain correct L2/L3 temporal state.
        #    Always run obs_to_x (even with beta=1.0) to populate proc_obs_buf for chunks.
        x_list = [None] * n_envs
        active_ids = [i for i in range(n_envs) if active[i]]
        gpu_ids = [i for i in active_ids if not done_flags[i] and not trunc_flags[i]]

        for i in active_ids:
            # Restore per-env retinal state into agent
            agent.state_vision_left = vision_states_left[i]
            agent.state_vision_right = vision_states_right[i]

            obs_t = obs_to_torch(obs_list[i], device=device, dtype=dtype)
            with torch.no_grad():
                xs = agent.obs_to_x(obs_t)  # (1, Nin)
            x_list[i] = xs

            # Save updated retinal state back
            vision_states_left[i] = agent.state_vision_left
            vision_states_right[i] = agent.state_vision_right

            # Store pre-computed obs_to_x result as numpy for chunk processing
            proc_obs_buf[i].append(xs.squeeze(0).cpu().numpy())

        # 3. Batched RNN step on GPU (all active envs at once)
        n_gpu = len(gpu_ids)
        if need_student and n_gpu > 0:
            # Batch the pre-computed x tensors
            x_batch = torch.cat([x_list[i] for i in gpu_ids], dim=0)  # (n_gpu, Nin)

            # Gather hidden states for active envs
            gpu_idx_tensor = torch.tensor(gpu_ids, device=device, dtype=torch.long)
            h_active = h[gpu_idx_tensor]

            with torch.no_grad():
                h_new, student_actions = agent.step(h_active, None, x=x_batch)

            student_actions_np = student_actions.cpu().numpy()

            # Write back hidden state
            h[gpu_idx_tensor] = h_new

        # 4. Teacher planning (CPU)
        for i in range(n_envs):
            if not active[i]:
                continue
            teacher_action = teachers[i].act(envs[i])
            if teacher_action is None:
                active[i] = False
            else:
                teacher_act_buf[i].append(teacher_action.copy())
            if done_flags[i] or trunc_flags[i] or not active[i]:
                active[i] = False

        # 5. Compute executed action per env
        for idx_a, i in enumerate(gpu_ids):
            if not active[i]:
                continue
            teacher_action = teacher_act_buf[i][-1]

            if teachers[i]._rec_phase is not None:
                action_exec[i] = teacher_action
            elif not need_student:
                action_exec[i] = teacher_action
            else:
                student_np = student_actions_np[idx_a]
                action_exec[i] = beta * teacher_action + (1.0 - beta) * student_np

            # Noise injection
            if (not noise_disturbing[i]
                    and teachers[i]._rec_phase is None
                    and steps[i] % NOISE_INTERVAL == 1):
                noise_disturbing[i] = True
                noise_steps[i] = 0
                noise_vals[i] = np.array(
                    [0.0, np.random.uniform(-1.0, 1.0)], dtype=np.float32
                ) * beta_noise
            if noise_disturbing[i]:
                action_exec[i] = action_exec[i] + noise_vals[i]
                noise_steps[i] += 1
                if noise_steps[i] >= 50:
                    noise_disturbing[i] = False

        # 6. Harvest finished episodes & reset
        reset_ids = []
        for i in range(n_envs):
            if active[i]:
                continue
            # This env finished its episode – process it
            raw_obs = raw_obs_buf[i]
            proc_obs = proc_obs_buf[i]
            t_acts = teacher_act_buf[i]

            valid = len(raw_obs) > 0 and len(t_acts) > 0
            if valid and len(raw_obs) >= chunk_length:
                teacher_actions_arr = np.array(t_acts, dtype=np.float32)
                processed = process_episode_to_chunks(
                    agent, raw_obs, proc_obs, teacher_actions_arr, analyzer,
                    chunk_length=chunk_length, stride=stride,
                    device=device, dtype=dtype
                )
                for category, chunks in processed.items():
                    for (xs, actions) in chunks:
                        buffer.add_chunk(category, xs, actions)
                        total_chunks[category] += 1

            episodes_done += 1
            if episodes_done % 10 == 0 or episodes_done == EPISODES_PER_ITER:
                elapsed = _time.perf_counter() - t0
                eps_per_sec = episodes_done / max(elapsed, 1e-6)
                print(
                    f"\r  Episodes: {episodes_done}/{EPISODES_PER_ITER}  "
                    f"({eps_per_sec:.2f} ep/s)",
                    end="", flush=True,
                )

            if episodes_done >= EPISODES_PER_ITER:
                break

            # Reset this env for a new episode
            raw_obs_buf[i] = []
            proc_obs_buf[i] = []
            teacher_act_buf[i] = []
            action_exec[i] = np.zeros(2, dtype=np.float32)
            steps[i] = 0
            done_flags[i] = False
            trunc_flags[i] = False
            noise_disturbing[i] = False
            noise_steps[i] = 0
            noise_vals[i] = np.zeros(2, dtype=np.float32)
            obs_list[i], _ = envs[i].reset(seed = episodes_done)
            # print("Episodes done: ", episodes_done)
            teachers[i].reset()
            vision_states_left[i] = None
            vision_states_right[i] = None
            reset_ids.append(i)
            active[i] = True

        # Batched hidden-state reset
        if need_student and reset_ids:
            h[reset_ids] = 0.0

        # If no envs are active and we still need episodes, reset all
        if not any(active) and episodes_done < EPISODES_PER_ITER:
            print("\n[warn] All envs finished but target not reached. Resetting all.")
            for i in range(n_envs):
                raw_obs_buf[i] = []
                proc_obs_buf[i] = []
                teacher_act_buf[i] = []
                action_exec[i] = np.zeros(2, dtype=np.float32)
                steps[i] = 0
                done_flags[i] = False
                trunc_flags[i] = False
                noise_disturbing[i] = False
                noise_steps[i] = 0
                noise_vals[i] = np.zeros(2, dtype=np.float32)
                obs_list[i], _ = envs[i].reset()
                teachers[i].reset()
                vision_states_left[i] = None
                vision_states_right[i] = None
                active[i] = True
            if need_student:
                h[:] = 0.0

    print()  # newline after progress
    return total_chunks


def train_step(agent, buffer, opt, accum_steps=GRAD_ACCUM_STEPS):
    """
    Single training step with gradient accumulation.
    
    Args:
        agent: The agent to train
        buffer: DAgger buffer to sample from
        opt: Optimizer
        accum_steps: Number of mini-batches to accumulate gradients over
    
    Returns:
        Average loss over all accumulation steps
    """
    agent.train()  # Training mode
    
    # Zero gradients at the start of accumulation
    opt.zero_grad(set_to_none=True)
    
    total_loss = 0.0
    
    for accum_idx in range(accum_steps):
        # Sample a batch using balanced sampling
        xs, ys = buffer.sample_balanced_sequences(batch_size=BATCH_SIZE)
        # print(f"Sampled batch of size {xs.shape[1]}", end="", flush=True)
        mu = forward_policy_sequence(agent, xs)
        
        ys = ys.to(device=mu.device, dtype=mu.dtype, non_blocking=True)

        # Slice for loss (ignore burn-in period)
        mu_tail = mu[T_BURN:]
        ys_tail = ys[T_BURN:]

        # Compute masked MSE
        squared_error_vel = (mu_tail[..., 0] - ys_tail[..., 0]) ** 2
        squared_error_angle = STEERING_LOSS_SCALE * (1 - torch.cos(mu_tail[..., 1] - ys_tail[..., 1]))
        squared_error = squared_error_vel + squared_error_angle

        loss = squared_error.mean()
        
        # Scale loss for gradient accumulation (average over accum steps)
        scaled_loss = loss / accum_steps
        
        # Backward pass (accumulates gradients)
        scaled_loss.backward()
        
        # Track unscaled loss for logging
        total_loss += loss.item()
        
        # Memory cleanup for this accumulation step
        del mu, xs, ys, mu_tail, ys_tail, squared_error, squared_error_vel, squared_error_angle, loss, scaled_loss
    
    # Gradient checks (after accumulation)
    if agent.cell.W_values.grad is None and agent.cell.W_values.requires_grad: 
        print("Warning: RNN weights have no gradients!")
        alphas = agent.cell.get_alphas().detach().cpu().numpy()
        print(f"  Debug: Alphas min/max/mean: {alphas.min():.4f}/{alphas.max():.4f}/{alphas.mean():.4f}")
    if agent.cell.bias.grad is None and agent.cell.bias.requires_grad: 
        print("Warning: RNN bias has no gradients!")
    
    # Clip gradients and update weights (once per train_step call)
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 2.0)
    opt.step()
    
    # Return average loss over accumulation steps
    return total_loss / accum_steps
    



def main():
    device = get_device()
    dtype = DTYPE
    print(f"[device] Using {device} ({dtype})")

    cell, pr_positions, input_splits, _ = build_connectome_cell(
        edge_path=EDGE_PATH,
        device=device,
        dtype=dtype,
        photoreceptor_left_csv=PHOTORECEPTOR_LEFT_CSV,
        photoreceptor_right_csv=PHOTORECEPTOR_RIGHT_CSV,

        tactile_left_csv=TACTILE_LEFT_CSV,
        tactile_right_csv=TACTILE_RIGHT_CSV,
        descending_neurons_csv=DESCENDING_NEURONS_CSV,
        cell_types_csv=CELL_TYPES_CSV,
        wind_sensing_csv=WIND_SENSING_CSV,
        target_rho=TARGET_RHO,
        leak_alpha=LEAK_ALPHA,
        activation=ACTIVATION,
        train_rnn_weights=TRAIN_RNN_WEIGHTS,
        train_readout_head=TRAIN_READOUT_HEAD,
        batch_chunk=BATCH_CHUNK,
        row_tile_size=ROW_TILE_SIZE,
    )
    
    agent = ConnectomeAgent(
        cell,
        photoreceptor_positions=pr_positions,
        input_splits=input_splits,
        dtype=dtype,
        input_scale_init=INPUT_SCALE_INIT
    ).to(device)

    # Load checkpoint if specified
    if RESUME_CHECKPOINT_PATH is not None:
        if os.path.exists(RESUME_CHECKPOINT_PATH):
            print(f"[ckpt] Loading checkpoint from {RESUME_CHECKPOINT_PATH}")
            checkpoint = torch.load(RESUME_CHECKPOINT_PATH, map_location=device)
            agent.load_state_dict(checkpoint)
            print(f"[ckpt] Successfully loaded checkpoint")
        else:
            print(f"[ckpt] Warning: Checkpoint not found at {RESUME_CHECKPOINT_PATH}, starting from scratch")

    # Create N_ENVS environments and teachers
    envs = [_make_env(render_mode=RENDER_MODE if i == 0 else None)
            for i in range(N_ENVS)]
    teachers = [_make_teacher() for _ in range(N_ENVS)]
    print(f"[env] Created {N_ENVS} concurrent environments")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Initialize balanced buffer and path analyzer
    buffer = BalancedDAggerBuffer(capacity_per_category=MAX_CHUNKS_PER_CATEGORY)
    analyzer = PathAnalyzer(
        direction_threshold_deg=DIRECTION_THRESHOLD_DEG,
        consecutive_n=CONSECUTIVE_SEGMENTS_N,
        collision_lookback=COLLISION_LOOKBACK,
        collision_recovery=COLLISION_RECOVERY_WINDOW,
    )
    chunk_length = T_BURN + T_UNROLL
    stride = T_BURN  # overlap = T_UNROLL
    
    opt = configure_optimizer(agent)

    try:
        for it in range(N_DAGGER_ITERS):
            beta = beta_schedule(it)
            beta_noise = noise_schedule(it)
            print(f"\n[DAgger] Iteration {it+1}/{N_DAGGER_ITERS} | beta={beta:.3f} | noise={beta_noise:.3f}")

            # Collect data with balanced buffer
            chunks_added = rollout_and_collect_balanced(
                envs, teachers, agent, buffer, beta, device, dtype, beta_noise,
                analyzer, chunk_length, stride
            )
            chunk_counts = buffer.get_counts()
            print(f"[data] Chunks added: Straight={chunks_added['straight']}, Turn={chunks_added['turn']}, Collision={chunks_added['collision']}, PreCol={chunks_added['pre_collision']}, Start={chunks_added['start']}")
            print(f"[data] Total buffer: Straight={chunk_counts['straight']}, Turn={chunk_counts['turn']}, Collision={chunk_counts['collision']}, PreCol={chunk_counts['pre_collision']}, Start={chunk_counts['start']}")
            
            losses = []
            for _ in range(TRAIN_STEPS_PER_ITER):
                
                try:
                    loss = train_step(agent, buffer, opt)
                    losses.append(loss)
                except ValueError:
                    break
                print(f"\r[train] Trained step {_+1}/{TRAIN_STEPS_PER_ITER}, loss={loss:.5f}. Dagger iter {it+1}/{N_DAGGER_ITERS}", end="", flush=True)
                if (_+1 == TRAIN_STEPS_PER_ITER): print()  # Newline after last step
            # plot losses curve
            if losses:
                plt.plot(losses)
                plt.title("Training Losses")
                plt.xlabel("Training Steps")
                plt.ylabel("Loss")
                plt.savefig(f"loss/losses_{it+1}.png")
                plt.close()
            mean_loss = float(np.mean(losses)) if losses else np.nan
            batches = len(losses)
            log_training_loss(it + 1, mean_loss, batches, len(buffer), chunk_counts)

            if losses:
                print(f"[train] mean loss={mean_loss:.5f} | batches={batches}")
            else:
                print("[train] skipped (insufficient buffer)")

            if (it + 1) % 1 == 0:
                ckpt_path = save_checkpoint(agent, it + 1)
                print(f"[ckpt] Saved checkpoint to {ckpt_path}")

    finally:
        for e in envs:
            e.close()
        if RENDER_MODE == "human" and cv2 is not None:
            cv2.destroyAllWindows()
        torch.save(agent.state_dict(), FINAL_CHECKPOINT_PATH)
        print(f"Saved trained model to {FINAL_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()

