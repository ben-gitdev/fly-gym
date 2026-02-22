"""
DAgger imitation learning for vision-based agents (EfficientNet-B0 / MobileNetV3-Large).

Usage:
    Set AGENT = "efficientnet" or "mobilenet" at the top of this file.
"""

from __future__ import annotations
import time as _time
import os
import csv
import cv2
import numpy as np
import torch


from environment.mujoco_two_cam_env_random_obstacles import MuJoCoTwoCamEnv
from agents.teacher_analytic_agent import PlannerAnalyticTeacher
from agents.efficientnet_agent import EfficientNetAgent
from agents.mobilenet_agent import MobileNetAgent
from agents.dual_backbone_agent import DualBackboneAgent
from core.utils import get_device

# -----------------------------
# Paths (hardcoded)
# -----------------------------
from shared_config import (
    ENV_WIDTH,
    ENV_HEIGHT,
    MAX_EPISODE_STEPS,
    N_OBSTACLES,
    ARENA_HALF_EXTENT,
    DTYPE,
    CHECKPOINT_DIR,
    LOSS_DIR,
)


# -----------------------------
# DAgger Hyperparameters
# -----------------------------

AGENT = "efficientnet" # "efficientnet" or "mobilenet"

N_DAGGER_ITERS = 4
EPISODES_PER_ITER = 500
TRAIN_STEPS_PER_ITER = 300
N_ENVS = 10  # Number of concurrent environments for vectorized rollout

BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2  # Number of mini-batches to accumulate before optimizer step

T_UNROLL = 40
T_BURN = 32
LR = 1e-4

BETA_START = 1.0 # 1.0: teacher drive, 0.0: agent drive
BETA_END = 0.0
BETA_DECAY = 0.5
BETA_WARMUP = 0  # Number of iterations to keep beta=1.0 at start

STEERING_LOSS_SCALE = 2.0  # Prioritize steering accuracy over velocity

# Camera dropout probabilities (per-chunk in each batch)
CAM_DROP_LEFT  = 0.20   # Probability of blacking out left camera only
CAM_DROP_RIGHT = 0.20   # Probability of blacking out right camera only
CAM_DROP_BOTH  = 0.20   # Probability of blacking out both cameras

NOISE_INTERVAL = 10
START_NOISE = 0.5
NOISE_DECAY = 0.2


# -----------------------------
# Balanced Buffer Configuration
# -----------------------------
RATIO_STRAIGHT = 0.25
RATIO_TURN = 0.25
RATIO_COLLISION = 0.2
RATIO_PRE_COLLISION = 0.2
RATIO_START = 0.1

# Path Analysis Parameters
DIRECTION_THRESHOLD_DEG = 1.0  # Degrees threshold for straight vs turn
CONSECUTIVE_SEGMENTS_N = 5    # Number of segments to analyze for classification
COLLISION_LOOKBACK = T_BURN + 50       # Steps to look back before collision (matches T_UNROLL for full context)
COLLISION_RECOVERY_WINDOW = T_UNROLL + 50  # Steps after collision for recovery
MAX_CHUNKS_PER_CATEGORY = 10000  # Max chunks stored per category

# Path to checkpoint to resume from (set to None to train from scratch)
RESUME_CHECKPOINT_PATH = None  

# -----------------------------
# Agent-specific configuration
# -----------------------------
AGENT_REGISTRY = {
    "efficientnet": {
        "class": EfficientNetAgent,
        "label": "EfficientNet-B0",
        "checkpoint_prefix": "efficientnet_dagger",
        "loss_csv": "efficientnet_dagger_loss.csv",
    },
    "mobilenet": {
        "class": MobileNetAgent,
        "label": "MobileNetV3-Large",
        "checkpoint_prefix": "mobilenet_dagger",
        "loss_csv": "mobilenet_dagger_loss.csv",
    },
    "dual_efficientnet": {
        "class": DualBackboneAgent,
        "label": "Dual-EfficientNet-B0",
        "checkpoint_prefix": "dual_efficientnet_dagger",
        "loss_csv": "dual_efficientnet_dagger_loss.csv",
        "backbone_type": "efficientnet",
    },
    "dual_mobilenet": {
        "class": DualBackboneAgent,
        "label": "Dual-MobileNetV3-Large",
        "checkpoint_prefix": "dual_mobilenet_dagger",
        "loss_csv": "dual_mobilenet_dagger_loss.csv",
        "backbone_type": "mobilenet",
    },
}

def _get_agent_config(agent_name: str) -> dict:
    """Return agent-specific config dict; raises if unknown."""
    name = agent_name.lower()
    if name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent '{agent_name}'. Choose from: {list(AGENT_REGISTRY.keys())}")
    return AGENT_REGISTRY[name]

def _make_env(render_mode=None):
    """Create a single MuJoCo environment with shared settings."""
    return MuJoCoTwoCamEnv(
        width=ENV_WIDTH,
        height=ENV_HEIGHT,
        max_episode_steps=MAX_EPISODE_STEPS,
        n_obstacles=N_OBSTACLES,
        arena_half_extent=ARENA_HALF_EXTENT,
        render_mode=render_mode,
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
            actual_start = max(0, turn_start - (chunk_length - stride))
            if actual_start < protected_start_steps:
                actual_start = protected_start_steps
            end_idx = self._find_next_keypoint(turn_start, keypoints, n)
            segment_length = end_idx - actual_start
            if segment_length < chunk_length:
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
            actual_start = max(0, collision_point - self.collision_lookback)
            if actual_start < protected_start_steps:
                actual_start = protected_start_steps
            end_idx = min(n, collision_point + self.collision_recovery)
            segment_length = end_idx - actual_start
            if segment_length < chunk_length:
                continue
            num_chunks = -(-((segment_length - chunk_length)) // stride) + 1
            for i in range(num_chunks):
                chunk_start = actual_start + i * stride
                chunk_end = chunk_start + chunk_length
                if chunk_end <= n:
                    obs_chunk = processed_obs_list[chunk_start:chunk_end]
                    act_chunk = teacher_actions[chunk_start:chunk_end]
                    chunks['collision'].append((obs_chunk, act_chunk))
        
        # 5. Extract PRE-COLLISION chunks
        for collision_point in keypoints['collision_points']:
            chunk_end = collision_point
            chunk_start = chunk_end - chunk_length
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
    
    def add_chunk(self, category: str, xs: Any, actions: np.ndarray):
        """Add a processed chunk to the appropriate buffer."""
        # xs is dict {'img': ..., 'wind_direction': ...}
        # actions is (T, 2)
        
        chunk = (xs, actions.astype(np.float32))
        
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
        """
        r_str, r_turn, r_col, r_pre_col, r_start = ratios
        
        buffers = {
            'straight': self.straight_chunks,
            'turn': self.turn_chunks,
            'collision': self.collision_chunks,
            'pre_collision': self.pre_collision_chunks,
            'start': self.start_chunks,
        }
        initial_ratio_map = {
            'straight': r_str,
            'turn': r_turn,
            'collision': r_col,
            'pre_collision': r_pre_col,
            'start': r_start,
        }
        
        # Identify non-empty buffers
        available_categories = [cat for cat, buf in buffers.items() if len(buf) > 0]
        
        if not available_categories:
            raise ValueError("All buffers are empty!")
            
        # Re-normalize ratios based on available categories
        current_total_ratio = sum(initial_ratio_map[cat] for cat in available_categories)
        if current_total_ratio == 0:
             # Fallback: uniform distribution if somehow ratios are 0
             ratio_map = {cat: 1.0/len(available_categories) for cat in available_categories}
        else:
             ratio_map = {cat: initial_ratio_map[cat] / current_total_ratio for cat in available_categories}

        # Calculate counts
        counts = {}
        samples = []
        
        # First pass: distribute batch size
        remaining_batch = batch_size
        for i, cat in enumerate(available_categories):
            if i == len(available_categories) - 1:
                # Last category takes the remainder to ensure sum == batch_size
                count = remaining_batch
            else:
                count = int(batch_size * ratio_map[cat])
                remaining_batch -= count
            
            counts[cat] = count
            
            # Sample
            buffer = buffers[cat]
            if count > 0:
                 # Sample with replacement if needed
                 indices = np.random.choice(len(buffer), size=count, replace=(count > len(buffer)))
                 for idx in indices:
                     samples.append(buffer[idx])

        # Shuffle to mix categories
        np.random.shuffle(samples)
        
        # Stack into tensors
        xs_list = [s[0] for s in samples]
        act_list = [s[1] for s in samples]
        
        # xs_list is list of dicts
        # We need to stack each key separately
        
        first_elem = xs_list[0] # dict
        keys = first_elem.keys()
        
        xs_batch = {}
        for k in keys:
            # stack along batch dim (axis=1) -> (T, B, ...)
            stacked = np.stack([x[k] for x in xs_list], axis=1)
            xs_batch[k] = torch.from_numpy(stacked)
            
        act_arr = torch.from_numpy(np.stack(act_list, axis=1))  # (T, B, Act)
        
        return xs_batch, act_arr


def log_training_loss(loss_csv_path, iter_idx, mean_loss, batches, buffer_size, chunk_counts=None):
    os.makedirs(os.path.dirname(loss_csv_path) or ".", exist_ok=True)
    write_header = not os.path.exists(loss_csv_path)
    with open(loss_csv_path, "a", newline="") as f:
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

def save_checkpoint(agent, iter_idx, checkpoint_prefix):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_prefix}_iter_{iter_idx}.pt")
    torch.save(agent.state_dict(), ckpt_path)
    return ckpt_path



def _preprocess_sensors(obs, dtype=np.float32):
    """Shared sensor preprocessing for all agent types."""
    sensors = obs["sensors"]
    wind_dir = sensors.get("wind_direction", np.zeros(2, dtype=dtype))
    col = sensors.get("collision", np.array([0.0], dtype=dtype))
    wind_dir = np.asarray(wind_dir, dtype=dtype)
    col = np.asarray(col, dtype=dtype)
    if wind_dir.ndim == 0: wind_dir = np.expand_dims(wind_dir, axis=0)
    if col.ndim == 0: col = np.expand_dims(col, axis=0)
    return wind_dir.astype(dtype), col.astype(dtype)

def preprocess_obs_cpu(obs, dtype=np.float32):
    """
    Process observation on CPU (single-backbone agents).
    Returns dict with combined 'img' (1, 30, 60).
    """
    left_small = cv2.resize(obs["cam_left"], (30, 30), interpolation=cv2.INTER_AREA)
    right_small = cv2.resize(obs["cam_right"], (30, 30), interpolation=cv2.INTER_AREA)
    left_gray = cv2.cvtColor(left_small, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right_small, cv2.COLOR_RGB2GRAY)
    combined = np.hstack([left_gray, right_gray])
    img_out = combined[np.newaxis, :, :]
    wind_dir, col = _preprocess_sensors(obs, dtype)
    return {
        "img": img_out.astype(np.uint8),
        "wind_direction": wind_dir,
        "collision": col,
    }

def preprocess_obs_cpu_dual(obs, dtype=np.float32):
    """
    Process observation on CPU (dual-backbone agents).
    Returns dict with separate 'img_left' and 'img_right' (each 1, 30, 30).
    """
    left_small = cv2.resize(obs["cam_left"], (30, 30), interpolation=cv2.INTER_AREA)
    right_small = cv2.resize(obs["cam_right"], (30, 30), interpolation=cv2.INTER_AREA)
    left_gray = cv2.cvtColor(left_small, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right_small, cv2.COLOR_RGB2GRAY)
    img_left = left_gray[np.newaxis, :, :]
    img_right = right_gray[np.newaxis, :, :]
    wind_dir, col = _preprocess_sensors(obs, dtype)
    return {
        "img_left": img_left.astype(np.uint8),
        "img_right": img_right.astype(np.uint8),
        "wind_direction": wind_dir,
        "collision": col,
    }

def forward_policy_sequence(agent, xs):
    """Wrapper for agent.forward_sequence for use in train_step."""
    _, y_seq, _ = agent.forward_sequence(xs, checkpoint_steps=False)
    return y_seq


def process_episode_to_chunks(
    agent,
    raw_obs_list: List[dict],
    processed_obs_list: List[dict],
    teacher_actions: np.ndarray,
    analyzer: PathAnalyzer,
    chunk_length: int,
    stride: int,
    device,
    dtype,
) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Process raw observations into categorized chunks.
    """
    if len(raw_obs_list) < chunk_length:
        return {'straight': [], 'turn': [], 'collision': [], 'pre_collision': [], 'start': []}
    
    # 1. Extract trajectory and collision data
    xy_path = analyzer.extract_xy_path(raw_obs_list)
    collisions = analyzer.extract_collision_flags(raw_obs_list)
    
    # 2. Find keypoints
    keypoints = analyzer.find_keypoints(raw_obs_list, xy_path, collisions)
    
    # 3. Segment into chunks (respecting protected start region)
    protected_start = chunk_length  # T_UNROLL + T_BURN
    # Use processed_obs_list for segmentation indices (same length as raw_obs_list)
    obs_chunks_dict = analyzer.segment_into_chunks(
        processed_obs_list, teacher_actions, keypoints,
        chunk_length, stride, protected_start
    )
    
    # 4. Process each chunk through agent.obs_to_x()
    processed_chunks = {'straight': [], 'turn': [], 'collision': [], 'pre_collision': [], 'start': []}
    
    for category, chunks in obs_chunks_dict.items():
        for (xs_chunk, action_chunk) in chunks:
            if len(xs_chunk) != chunk_length:
                continue  # Skip invalid chunks
            
            # xs_chunk is already a list of dicts with numpy arrays
            # We just need to stack them using the first element's keys
            xs_list = xs_chunk
            
            # Stack: xs_arr becomes dict of arrays
            # {'img': (T, 1, H, W), 'wind_direction': (T, 2), ...}
            xs_arr = {}
            for k in xs_list[0].keys():
                xs_arr[k] = np.stack([x[k] for x in xs_list], axis=0)
            
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

    Optimizations:
      1. Async GPU→CPU via pinned output buffer with deferred synchronization
      2. Reusable GPU index tensors for hidden-state gather/scatter
      3. Batched hidden-state resets (single write instead of per-env scalars)
      4. Pre-allocated pinned input buffers filled via numpy views
      5. CPU/GPU pipelining: GPU inference overlaps with teacher planning
      6. Merged env-step + preprocessing loop for cache locality
    """
    agent.eval()
    n_envs = len(envs)
    need_student = beta < 1.0
    np_dtype = np.float32 if dtype == torch.float32 else np.float16
    use_cuda = device.type == 'cuda'
    is_dual = isinstance(agent, DualBackboneAgent)
    preprocess_fn = preprocess_obs_cpu_dual if is_dual else preprocess_obs_cpu

    total_chunks = {'straight': 0, 'turn': 0, 'collision': 0, 'pre_collision': 0, 'start': 0}
    episodes_done = 0

    # ---- per-env mutable state ----
    obs_list      = [None] * n_envs          # latest raw obs
    raw_obs_buf   = [[] for _ in range(n_envs)]  # raw obs history
    proc_obs_buf  = [[] for _ in range(n_envs)]  # preprocessed obs history
    teacher_act_buf = [[] for _ in range(n_envs)]
    action_exec   = [np.zeros(2, dtype=np.float32) for _ in range(n_envs)]
    steps         = [0] * n_envs
    noise_disturbing = [False] * n_envs
    noise_steps   = [0] * n_envs
    noise_vals    = [np.zeros(2, dtype=np.float32) for _ in range(n_envs)]
    active        = [True] * n_envs          # env is running
    done_flags    = [False] * n_envs         # per-env done tracking
    trunc_flags   = [False] * n_envs         # per-env truncation tracking

    if need_student:
        h = torch.zeros(n_envs, agent.hidden_size, device=device, dtype=dtype)
        if use_cuda:
            # Opt #4: Pre-allocated pinned memory for CPU→GPU input transfers
            if is_dual:
                pin_imgs_l = torch.empty(n_envs, 1, 30, 30, dtype=torch.uint8).pin_memory()
                pin_imgs_r = torch.empty(n_envs, 1, 30, 30, dtype=torch.uint8).pin_memory()
                pin_imgs_l_np = pin_imgs_l.numpy()
                pin_imgs_r_np = pin_imgs_r.numpy()
            else:
                pin_imgs  = torch.empty(n_envs, 1, 30, 60, dtype=torch.uint8).pin_memory()
                pin_imgs_np  = pin_imgs.numpy()
            pin_winds = torch.empty(n_envs, 2, dtype=torch.float32).pin_memory()
            pin_cols  = torch.empty(n_envs, 1, dtype=torch.float32).pin_memory()
            pin_winds_np = pin_winds.numpy()
            pin_cols_np  = pin_cols.numpy()
            # Opt #1: Pre-allocated pinned memory for GPU→CPU result transfer
            pin_actions_out = torch.empty(n_envs, 2, dtype=torch.float32).pin_memory()
            pin_actions_out_np = pin_actions_out.numpy()

    # ---- reset all envs ----
    for i in range(n_envs):
        obs_list[i], _ = envs[i].reset()
        teachers[i].reset()
    agent.reset_vision_state()

    t0 = _time.perf_counter()

    # ---- main loop: step all active envs until enough episodes collected ----
    while episodes_done < EPISODES_PER_ITER:
        # 1. MuJoCo step + preprocess all active envs (merged - Opt #6)
        for i in range(n_envs):
            if not active[i]:
                continue
            obs, _, done_flags[i], trunc_flags[i], _ = envs[i].step(action_exec[i])
            obs_list[i] = obs
            steps[i] += 1
            raw_obs_buf[i].append(obs)
            proc_obs_buf[i].append(preprocess_fn(obs, dtype=np_dtype))

        # 2. Launch GPU inference ASYNC (Opt #5 - overlaps with teacher planning)
        #    Exclude done/trunc envs from GPU work; teacher-invalidated envs are
        #    handled later (tiny wasted GPU compute, avoids a sync point).
        gpu_ids = [i for i in range(n_envs)
                   if active[i] and not done_flags[i] and not trunc_flags[i]]
        n_gpu = len(gpu_ids)

        if need_student and n_gpu > 0:
            if use_cuda:
                # Opt #4: Fill pinned buffers via numpy views (no intermediate tensors)
                for idx, i in enumerate(gpu_ids):
                    if is_dual:
                        pin_imgs_l_np[idx] = proc_obs_buf[i][-1]["img_left"]
                        pin_imgs_r_np[idx] = proc_obs_buf[i][-1]["img_right"]
                    else:
                        pin_imgs_np[idx]  = proc_obs_buf[i][-1]["img"]
                    pin_winds_np[idx] = proc_obs_buf[i][-1]["wind_direction"]
                    pin_cols_np[idx]  = proc_obs_buf[i][-1]["collision"]
                if is_dual:
                    imgs_l_gpu = pin_imgs_l[:n_gpu].to(device, non_blocking=True)
                    imgs_r_gpu = pin_imgs_r[:n_gpu].to(device, non_blocking=True)
                else:
                    imgs_gpu  = pin_imgs[:n_gpu].to(device, non_blocking=True)
                winds_gpu = pin_winds[:n_gpu].to(device, dtype=dtype, non_blocking=True)
                cols_gpu  = pin_cols[:n_gpu].to(device, dtype=dtype, non_blocking=True)
            else:
                # Non-CUDA fallback (original style)
                if is_dual:
                    imgs_l_gpu = torch.from_numpy(
                        np.stack([proc_obs_buf[i][-1]["img_left"] for i in gpu_ids])
                    ).to(device)
                    imgs_r_gpu = torch.from_numpy(
                        np.stack([proc_obs_buf[i][-1]["img_right"] for i in gpu_ids])
                    ).to(device)
                else:
                    imgs_gpu = torch.from_numpy(
                        np.stack([proc_obs_buf[i][-1]["img"] for i in gpu_ids])
                    ).to(device)
                winds_gpu = torch.from_numpy(
                    np.stack([proc_obs_buf[i][-1]["wind_direction"] for i in gpu_ids])
                ).to(device, dtype=dtype)
                cols_gpu = torch.from_numpy(
                    np.stack([proc_obs_buf[i][-1]["collision"] for i in gpu_ids])
                ).to(device, dtype=dtype)

            if is_dual:
                xs_gpu = {"img_left": imgs_l_gpu, "img_right": imgs_r_gpu,
                          "wind_direction": winds_gpu, "collision": cols_gpu}
            else:
                xs_gpu = {"img": imgs_gpu, "wind_direction": winds_gpu, "collision": cols_gpu}

            # Opt #2: GPU index tensor for h gather/scatter
            gpu_idx_tensor = torch.tensor(gpu_ids, device=device, dtype=torch.long)
            h_active = h[gpu_idx_tensor]

            with torch.no_grad():
                h_new, student_actions = agent.step(h_active, None, x=xs_gpu)

            # Opt #1: Async GPU→CPU copy to pinned buffer (deferred sync)
            if use_cuda:
                pin_actions_out[:n_gpu].copy_(student_actions, non_blocking=True)

            # Opt #2: Write back hidden state via GPU index tensor
            h[gpu_idx_tensor] = h_new

        # 3. Teacher planning (CPU) — runs while GPU finishes (Opt #5)
        for i in range(n_envs):
            if not active[i]:
                continue
            teacher_action = teachers[i].act(envs[i])
            if teacher_action is None:
                # Invalid episode – discard & reset
                active[i] = False
            else:
                teacher_act_buf[i].append(teacher_action.copy())
            if done_flags[i] or trunc_flags[i] or not active[i]:
                active[i] = False

        # 4. Sync GPU results (Opt #1 - deferred synchronization)
        if need_student and n_gpu > 0:
            if use_cuda:
                torch.cuda.synchronize()
                student_actions_np = pin_actions_out_np[:n_gpu]
            else:
                student_actions_np = student_actions.numpy()

        # 5. Compute executed action per env
        for idx_a, i in enumerate(gpu_ids):
            if not active[i]:
                continue  # teacher invalidated this env
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
        reset_ids = []  # Opt #3: collect for batched h reset
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
                break  # enough episodes

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
            obs_list[i], _ = envs[i].reset()
            teachers[i].reset()
            reset_ids.append(i)
            active[i] = True

        # Opt #3: Batched hidden-state reset (single write instead of N scalar writes)
        if need_student and reset_ids:
            h[reset_ids] = 0.0

        # If no envs are active and we still need episodes, something is wrong
        if not any(active) and episodes_done < EPISODES_PER_ITER:
            print("\n[warn] All envs finished but target not reached. Resetting all.")
            all_reset_ids = list(range(n_envs))
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
                active[i] = True
            # Opt #3: Batched reset
            if need_student:
                h[all_reset_ids] = 0.0

    print()  # newline after progress
    return total_chunks


def apply_camera_dropout(xs, p_left=CAM_DROP_LEFT, p_right=CAM_DROP_RIGHT, p_both=CAM_DROP_BOTH):
    """
    Apply camera dropout to a batch of observations *in-place*.

    For each chunk (along the B dimension), randomly assign one of:
      - left camera blacked out  (probability p_left)
      - right camera blacked out (probability p_right)
      - both cameras blacked out (probability p_both)
      - no dropout               (remaining probability)

    Supports both single-backbone ('img' key, left/right halves concatenated)
    and dual-backbone ('img_left' / 'img_right' keys) layouts.
    """
    is_dual = "img_left" in xs

    if is_dual:
        B = xs["img_left"].shape[1]
    else:
        B = xs["img"].shape[1]

    # Draw one uniform random per chunk and assign dropout category
    rand_vals = np.random.rand(B)
    # [0, p_left) -> drop left | [p_left, p_left+p_right) -> drop right
    # [p_left+p_right, p_left+p_right+p_both) -> drop both | rest -> keep
    thresh_left  = p_left
    thresh_right = thresh_left + p_right
    thresh_both  = thresh_right + p_both

    drop_left_mask  = rand_vals < thresh_left
    drop_right_mask = (rand_vals >= thresh_left) & (rand_vals < thresh_right)
    drop_both_mask  = (rand_vals >= thresh_right) & (rand_vals < thresh_both)

    # Combine masks: left should be zeroed for drop_left OR drop_both
    zero_left  = drop_left_mask | drop_both_mask
    zero_right = drop_right_mask | drop_both_mask

    if is_dual:
        # xs["img_left"]  shape: (T, B, 1, 30, 30)  (torch tensor)
        # xs["img_right"] shape: (T, B, 1, 30, 30)
        for b in range(B):
            if zero_left[b]:
                xs["img_left"][:, b] = 0
            if zero_right[b]:
                xs["img_right"][:, b] = 0
    else:
        # xs["img"] shape: (T, B, 1, 30, 60)  — left half [:30], right half [30:]
        for b in range(B):
            if zero_left[b]:
                xs["img"][:, b, :, :, :30] = 0
            if zero_right[b]:
                xs["img"][:, b, :, :, 30:] = 0

    return xs


def train_step(agent, buffer, opt, accum_steps=GRAD_ACCUM_STEPS):
    """
    Single training step with gradient accumulation.
    """
    agent.train()  # Training mode
    
    # Zero gradients at the start of accumulation
    opt.zero_grad(set_to_none=True)
    
    total_loss = 0.0
    device = agent.device
    
    for accum_idx in range(accum_steps):
        # Sample a batch using balanced sampling
        xs, ys = buffer.sample_balanced_sequences(batch_size=BATCH_SIZE)
        
        # xs is dict {'img': (T, B, 1, H, W) uint8, 'wind_direction': (T, B, 2) float, ...}
        # ys is (T, B, 2) float
        
        # Apply camera dropout augmentation
        xs = apply_camera_dropout(xs)
        
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
    
    # Clip gradients and update weights (once per train_step call)
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    opt.step()
    
    # Return average loss over accumulation steps
    return total_loss / accum_steps
    



def main():
    agent_name = AGENT

    agent_cfg = _get_agent_config(agent_name)
    AgentClass = agent_cfg["class"]
    agent_label = agent_cfg["label"]
    checkpoint_prefix = agent_cfg["checkpoint_prefix"]
    loss_csv_path = os.path.join(LOSS_DIR, agent_cfg["loss_csv"])
    final_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_prefix}_final.pt")

    device = get_device()
    dtype = DTYPE
    print(f"[device] Using {device} ({dtype})")
    
    # Create N concurrent environments + teachers
    print(f"Creating {N_ENVS} concurrent environments...")
    envs = [_make_env() for _ in range(N_ENVS)]
    teachers = [_make_teacher() for _ in range(N_ENVS)]
    
    # Initialize Agent
    print(f"Initializing {agent_label} Agent...")
    agent_kwargs = dict(action_dim=2, hidden_size=256, dtype=dtype)
    if "backbone_type" in agent_cfg:
        agent_kwargs["backbone_type"] = agent_cfg["backbone_type"]
    agent = AgentClass(**agent_kwargs).to(device=device, dtype=dtype)
    
    # Resume from checkpoint if configured
    start_iter = 0
    if RESUME_CHECKPOINT_PATH and os.path.exists(RESUME_CHECKPOINT_PATH):
        print(f"Loading checkpoint from {RESUME_CHECKPOINT_PATH}")
        checkpoint = torch.load(RESUME_CHECKPOINT_PATH, map_location=device)
        agent.load_state_dict(checkpoint)
        # Infer iteration from filename if possible
        try:
            filename = os.path.basename(RESUME_CHECKPOINT_PATH)
            start_iter = int(filename.split('_')[-1].split('.')[0]) + 1
        except:
            print("Could not infer iteration from filename, starting at 0")
    
    # Configure Optimizer
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)
    
    # Buffer
    buffer = BalancedDAggerBuffer(capacity_per_category=MAX_CHUNKS_PER_CATEGORY)
    analyzer = PathAnalyzer()
    
    # DAgger Loop
    for i in range(start_iter, N_DAGGER_ITERS):
        beta = beta_schedule(i)
        beta_noise = noise_schedule(i)
        print(f"\n--- DAgger Iteration {i+1}/{N_DAGGER_ITERS} [Beta={beta:.2f}, Noise={beta_noise:.2f}] ---")
        
        # 1. Rollout & Collect (vectorized)
        t_start = _time.perf_counter()
        chunk_counts = rollout_and_collect_balanced(
            envs, teachers, agent, buffer, beta, device, dtype, beta_noise,
            analyzer, chunk_length=T_UNROLL + T_BURN, stride=T_BURN
        )
        t_rollout = _time.perf_counter() - t_start
        print(f"  Rollout took {t_rollout:.1f}s ({EPISODES_PER_ITER/t_rollout:.2f} ep/s)")
        
        print("Buffer Status:")
        for cat, count in buffer.get_counts().items():
            print(f"  {cat}: {count} chunks")
        
        if len(buffer) == 0:
            print("Buffer empty, skipping training this iteration.")
            continue
        
        # 2. Train
        print(f"Training for {TRAIN_STEPS_PER_ITER} steps...")
        running_loss = 0.0
        
        for step in range(TRAIN_STEPS_PER_ITER):
            loss_val = train_step(agent, buffer, optimizer)
            running_loss += loss_val
            
            if (step + 1) % 1 == 0:
                avg_loss = running_loss / 1
                print(f"\r  Step {step+1}: Loss = {avg_loss:.4f}", end="", flush=True)
                running_loss = 0.0
        
        print()
        
        # 3. Log & Checkpoint
        log_training_loss(loss_csv_path, i + 1, avg_loss, TRAIN_STEPS_PER_ITER, len(buffer), chunk_counts)
        save_checkpoint(agent, i + 1, checkpoint_prefix)
    
    print("Training Complete.")
    torch.save(agent.state_dict(), final_checkpoint_path)
    for env in envs:
        env.close()

if __name__ == "__main__":
    main()
