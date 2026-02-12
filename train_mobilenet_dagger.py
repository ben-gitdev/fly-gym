"""
DAgger imitation learning for the MobileNetV3-Large agent using vision-only inputs.
"""

from __future__ import annotations
        # 0. Import Any
from typing import Any
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
import os
import csv
import math

import numpy as np
import torch
import cv2


from environment.mujoco_two_cam_env_random_obstacles import MuJoCoTwoCamEnv
from agents.teacher_analytic_agent import PlannerAnalyticTeacher
from agents.mobilenet_agent import MobileNetAgent
from core.utils import get_device, obs_to_torch

# -----------------------------
# Paths (hardcoded)
# -----------------------------
from shared_config import (
    ENV_WIDTH,
    ENV_HEIGHT,
    MAX_EPISODE_STEPS,
    N_OBSTACLES,
    ARENA_HALF_EXTENT,
    RENDER_MODE,
    BATCH_CHUNK,
    DTYPE,
    CHECKPOINT_DIR,
    LOSS_DIR,
)

END_ON_COLLISION = False

# -----------------------------
# DAgger Hyperparameters
# -----------------------------
N_DAGGER_ITERS = 3
EPISODES_PER_ITER = 1000
TRAIN_STEPS_PER_ITER = 500

BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 4  # Number of mini-batches to accumulate before optimizer step

T_UNROLL = 80
T_BURN = 50
LR = 1e-4

BETA_START = 1.0 # 1.0: teacher drive, 0.0: agent drive
BETA_END = 0.0
BETA_DECAY = 0.5
BETA_WARMUP = 0  # Number of iterations to keep beta=1.0 at start

STEERING_LOSS_SCALE = 2.0  # Prioritize steering accuracy over velocity

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

LOSS_CSV_PATH = os.path.join(LOSS_DIR, "mobilenet_dagger_loss.csv")
FINAL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "mobilenet_dagger_final.pt")

# Path to checkpoint to resume from (set to None to train from scratch)
RESUME_CHECKPOINT_PATH = None  

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
        raw_obs_list: List[dict],
        teacher_actions: np.ndarray,
        keypoints: dict,
        chunk_length: int,
        stride: int,
        protected_start_steps: int,
    ) -> dict:
        """
        Segment observations into 4 category chunks with proper overlap handling.
        """
        n = len(raw_obs_list)
        chunks = {'straight': [], 'turn': [], 'collision': [], 'pre_collision': [], 'start': []}
        
        # 1. Extract START chunk (beginning of episode)
        if n >= chunk_length:
            obs_chunk = raw_obs_list[:chunk_length]
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
                    obs_chunk = raw_obs_list[chunk_start:chunk_end]
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
                    obs_chunk = raw_obs_list[actual_start:actual_start + chunk_length]
                    act_chunk = teacher_actions[actual_start:actual_start + chunk_length]
                    chunks['turn'].append((obs_chunk, act_chunk))
                continue
            num_chunks = -(-((segment_length - chunk_length)) // stride) + 1  # Ceiling division
            for i in range(num_chunks):
                chunk_start = actual_start + i * stride
                chunk_end = chunk_start + chunk_length
                if chunk_end <= n:
                    obs_chunk = raw_obs_list[chunk_start:chunk_end]
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
                    obs_chunk = raw_obs_list[chunk_start:chunk_end]
                    act_chunk = teacher_actions[chunk_start:chunk_end]
                    chunks['collision'].append((obs_chunk, act_chunk))
        
        # 5. Extract PRE-COLLISION chunks
        for collision_point in keypoints['collision_points']:
            chunk_end = collision_point
            chunk_start = chunk_end - chunk_length
            if chunk_start >= protected_start_steps:
                if chunk_end <= n:
                    obs_chunk = raw_obs_list[chunk_start:chunk_end]
                    act_chunk = teacher_actions[chunk_start:chunk_end]
                    chunks['pre_collision'].append((obs_chunk, act_chunk))
        
        return chunks


class BalancedDAggerBuffer:
    """4-category buffer for balanced training data sampling."""
    
    def __init__(self, capacity_per_category: int = MAX_CHUNKS_PER_CATEGORY):
        self.capacity = capacity_per_category
        
        # Each category stores processed chunks as lists of (xs, actions) tuples
        # xs might be uint8 for EfficientNet
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
        # xs is dict {'img': ..., 'goal': ...}
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
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"mobilenet_dagger_iter_{iter_idx}.pt")
    torch.save(agent.state_dict(), ckpt_path)
    return ckpt_path


def forward_policy_sequence(agent, xs):
    # Use the unified efficient sequence processing
    _, y_seq, _ = agent.forward_sequence(xs, checkpoint_steps=False)
    return y_seq

def rollout_episode(
    env, teacher, agent, beta: float, device, dtype, beta_noise: float = 0.0
) -> Tuple[List[dict], np.ndarray, bool]:
    """
    Run one episode and return raw observations + teacher actions.
    """
    raw_obs_list = []
    teacher_actions_list = []
    
    obs, _ = env.reset()
    agent.reset_vision_state()
    teacher.reset()
    h = torch.zeros(1, agent.hidden_size, device=device, dtype=dtype)
    steps = 0
    
    noise_disturbing = False
    noise_steps = 0
    noise = np.zeros(2, dtype=np.float32)
    action_exec = np.zeros(2, dtype=np.float32)
    
    while steps < MAX_EPISODE_STEPS:

        obs, _, done, trunc, _ = env.step(action_exec)
        steps += 1

        maybe_show_cameras(obs)
        
        # Store raw observation BEFORE processing
        raw_obs_list.append(obs)
        
        obs_t = obs_to_torch(obs, device=device, dtype=dtype)
        # obs_to_x now returns uint8 tensor for memory saving, but step() needs float form?
        # EfficientNetAgent.step() handles uint8 -> float conversion.
        xs = agent.obs_to_x(obs_t)
        
        with torch.no_grad():
            h, student_action = agent.step(
                h,
                obs_t,
                x=xs,
            )
        student_action_np = student_action.squeeze(0).cpu().numpy()
        teacher_action = teacher.act(env)
        
        if teacher_action is None:
            # Teacher cannot provide action - invalid episode
            return [], np.array([]), False
        
        teacher_actions_list.append(teacher_action.copy())
        
        # Always let teacher take over when there's a collision
        if teacher._rec_phase is not None:
            action_exec = teacher_action
        else:
            action_exec = beta * teacher_action + (1.0 - beta) * student_action_np
        
        # Noise Injection Logic (same as before)
        if not noise_disturbing and teacher._rec_phase is None and steps % NOISE_INTERVAL == 1:
            noise_disturbing = True
            noise_steps = 0
            noise = np.array([0.0, np.random.uniform(-1.0, 1.0)], dtype=np.float32) * beta_noise
        if noise_disturbing:
            action_exec += noise
            noise_steps += 1
            if noise_steps >= 50:
                noise_disturbing = False
        
        if done or trunc:
            break
    
    return raw_obs_list, np.array(teacher_actions_list, dtype=np.float32), True


def process_episode_to_chunks(
    agent,
    raw_obs_list: List[dict],
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
    obs_chunks_dict = analyzer.segment_into_chunks(
        raw_obs_list, teacher_actions, keypoints,
        chunk_length, stride, protected_start
    )
    
    # 4. Process each chunk through agent.obs_to_x()
    processed_chunks = {'straight': [], 'turn': [], 'collision': [], 'pre_collision': [], 'start': []}
    
    for category, chunks in obs_chunks_dict.items():
        for (obs_chunk, action_chunk) in chunks:
            if len(obs_chunk) != chunk_length:
                continue  # Skip invalid chunks
            
            xs_list = []
            for obs in obs_chunk:
                obs_t = obs_to_torch(obs, device=device, dtype=dtype)
                with torch.no_grad():
                    xs = agent.obs_to_x(obs_t)
                    # xs is dict {'img': uint8, 'goal': float}
                
                # Convert dict of tensors to dict of numpy
                xs_np = {k: v.squeeze(0).cpu().numpy() for k, v in xs.items()}
                xs_list.append(xs_np)
            
            # Stack: xs_arr becomes dict of arrays
            # {'img': (T, 3, H, W), 'goal': (T, 2)}
            xs_arr = {}
            for k in xs_list[0].keys():
                xs_arr[k] = np.stack([x[k] for x in xs_list], axis=0)
            
            processed_chunks[category].append((xs_arr, action_chunk))
    
    return processed_chunks


def rollout_and_collect_balanced(
    env, teacher, agent, buffer: BalancedDAggerBuffer,
    beta: float, device, dtype, beta_noise: float,
    analyzer: PathAnalyzer, chunk_length: int, stride: int
):
    """
    Collect episodes and add categorized chunks to balanced buffer.
    """
    agent.eval()  # Inference mode during rollout
    
    total_chunks = {'straight': 0, 'turn': 0, 'collision': 0, 'pre_collision': 0, 'start': 0}
    
    for ep in range(EPISODES_PER_ITER):
        # Phase 1: Collect raw episode
        print(f"\r Collecting episode {ep+1}/{EPISODES_PER_ITER}", end="", flush=True)
        raw_obs, teacher_actions, valid = rollout_episode(
            env, teacher, agent, beta, device, dtype, beta_noise
        )
        
        if not valid or len(raw_obs) < chunk_length:
            print(f"  [ep {ep+1}] Skipped (invalid or too short)")
            continue
        
        # Phase 2: Process into chunks (per-episode to save memory)
        processed = process_episode_to_chunks(
            agent, raw_obs, teacher_actions, analyzer,
            chunk_length=chunk_length,
            stride=stride,
            device=device, dtype=dtype
        )
        
        # Phase 3: Add to buffer
        for category, chunks in processed.items():
            for (xs, actions) in chunks:
                buffer.add_chunk(category, xs, actions)
                total_chunks[category] += 1
        
        # Raw observations are garbage collected after this iteration
        del raw_obs, teacher_actions, processed
    
    return total_chunks


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
        
        # xs is dict {'img': (T, B, 3, H, W) uint8, 'goal': (T, B, 2) float}
        # ys is (T, B, 2) float
        
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
    device = get_device()
    dtype = DTYPE
    print(f"[device] Using {device} ({dtype})")
    
    env = MuJoCoTwoCamEnv(
        width=ENV_WIDTH,
        height=ENV_HEIGHT,
        max_episode_steps=MAX_EPISODE_STEPS,
        n_obstacles=N_OBSTACLES,
        arena_half_extent=ARENA_HALF_EXTENT,
        render_mode=RENDER_MODE,
    )
    
    teacher = PlannerAnalyticTeacher(
        arrival_threshold=1.0,
        target_noise_prob=0.0
    )
    
    # Initialize MobileNet Agent
    print("Initializing MobileNetV3-Large Agent...")
    agent = MobileNetAgent(
        action_dim=2,
        hidden_size=256,
        dtype=dtype
    ).to(device=device, dtype=dtype)
    
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
    
    # Configure Optimizer: All params are trainable for EfficientNet + GRU
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)
    
    # Buffer
    buffer = BalancedDAggerBuffer(capacity_per_category=MAX_CHUNKS_PER_CATEGORY)
    analyzer = PathAnalyzer()
    
    # DAgger Loop
    for i in range(start_iter, N_DAGGER_ITERS):
        beta = beta_schedule(i)
        beta_noise = noise_schedule(i)
        print(f"\n--- DAgger Iteration {i+1}/{N_DAGGER_ITERS} [Beta={beta:.2f}, Noise={beta_noise:.2f}] ---")
        
        # 1. Rollout & Collect
        chunk_counts = rollout_and_collect_balanced(
            env, teacher, agent, buffer, beta, device, dtype, beta_noise,
            analyzer, chunk_length=T_UNROLL + T_BURN, stride=T_BURN
        )
        
        print("\nBuffer Status:")
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
            
            if (step + 1) % 10 == 0:
                avg_loss = running_loss / 10
                print(f"\r  Step {step+1}: Loss = {avg_loss:.4f}", end="", flush=True)
                running_loss = 0.0
        
        print()
        
        # 3. Log & Checkpoint
        # Calculate mean loss over the last epoch roughly
        # Actually we just logged running loss. We can log the last avg_loss.
        log_training_loss(i + 1, avg_loss, TRAIN_STEPS_PER_ITER, len(buffer), chunk_counts)
        save_checkpoint(agent, i + 1)
    
    print("Training Complete.")
    torch.save(agent.state_dict(), FINAL_CHECKPOINT_PATH)
    print(f"Saved final model to {FINAL_CHECKPOINT_PATH}")

if __name__ == "__main__":
    main()
