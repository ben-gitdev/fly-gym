"""
DAgger imitation learning for the connectome RNN agent using vision-only inputs.
Refactored to fix terminal sampling bias, action bounds, and precision issues.
"""

from __future__ import annotations
import csv
import os
from dataclasses import dataclass
from typing import Tuple, Dict
import matplotlib.pyplot as plt

import numpy as np
import torch
import cv2


from environment.mujoco_two_cam_env_random_obstacles import MuJoCoTwoCamEnv
from agents.teacher_analytic_agent import PlannerAnalyticTeacher
from agents.connectome_rnn_agent import ConnectomeAgent
from models.connectome_rnn_model import LeakyConnectomeRNNCell
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
    OLFACTORY_LEFT_CSV,
    OLFACTORY_RIGHT_CSV,
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
N_DAGGER_ITERS = 100
EPISODES_PER_ITER = 10
TRAIN_STEPS_PER_ITER = 150

BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2  # Number of mini-batches to accumulate before optimizer step

T_UNROLL = 80
T_BURN = 50
LR = 2e-4

BETA_START = 1.0
BETA_END = 0.02
BETA_DECAY = 0.02
BETA_WARMUP = 2  # Number of iterations to keep beta=1.0 at start

TURN_BIAS_STRENGTH = 10.0  # Weight for turning samples
STEERING_LOSS_SCALE = 2.0  # Prioritize steering accuracy over velocity

NOISE_INTERVAL = 100
START_NOISE = 0.8
NOISE_DECAY = 0.02

MAX_BUFFER_SIZE = 500_000



LOSS_CSV_PATH = os.path.join(LOSS_DIR, "connectome_rnn_dagger_loss.csv")
FINAL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "connectome_rnn_dagger.pt")

# Path to checkpoint to resume from (set to None to train from scratch)
RESUME_CHECKPOINT_PATH = None  # e.g., os.path.join(CHECKPOINT_DIR, "connectome_rnn_dagger_iter_50.pt")


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


class DAggerBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.initialized = False
        
        self.pr: np.ndarray = None
        self.action: np.ndarray = None
        self.done: np.ndarray = None

    def __len__(self):
        return self.size
    
    def _init_storage(self, nin: int, action_dim: int):
        print(f"[Buffer] Allocating RAM for {self.capacity} steps (Nin={nin})...")
        # FIX 3: Use float32 instead of float16 to preserve reservoir dynamics
        self.pr = np.zeros((self.capacity, nin), dtype=np.float32)
        self.action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity,), dtype=bool)
        self.initialized = True

    def add(self, pr: np.ndarray, teacher_action: np.ndarray, done: bool):
        if not self.initialized:
            self._init_storage(pr.shape[0], teacher_action.shape[0])
        self.pr[self.ptr] = pr.astype(np.float32)
        self.action[self.ptr] = teacher_action.astype(np.float32)
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_sequences(self, batch_size: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            pr_arr: (T, B, Nin)
            act_arr: (T, B, Act)
            mask_arr: (T, B, 1) - 1.0 for valid steps, 0.0 for padding/post-done steps
        """
        if self.size < seq_len:
            raise ValueError("Not enough data to sample sequences.")

        pr_list = []
        act_list = []
        mask_list = []

        max_start_logical = self.size - seq_len
        
        # We try to find batch_size valid sequences
        for _ in range(batch_size):
            start_logical = np.random.randint(0, max_start_logical + 1)
            # Find physical start index (ring buffer logic)
            # Note: oldest index is self.ptr if full, else 0
            oldest = self.ptr if self.size == self.capacity else 0
            start_phys = (oldest + start_logical) % self.capacity

            # Gather sequence indices handling wrap-around
            idxs = (start_phys + np.arange(seq_len)) % self.capacity
            
            # Extract raw data
            raw_pr = self.pr[idxs]       # (T, Nin)
            raw_act = self.action[idxs]  # (T, Act)
            raw_done = self.done[idxs]   # (T,)

            # FIX 1 & 4: Terminal State Handling
            # Instead of rejecting sequences with 'done', we truncate/mask them.
            # If done occurs at index 'k', valid data is 0..k (inclusive). 
            # k+1 onwards is masked out.
            
            # Find first occurrence of done
            done_indices = np.where(raw_done)[0]
            
            # Default mask is all ones
            mask = np.ones((seq_len, 1), dtype=np.float32)
            
            if len(done_indices) > 0:
                first_done = done_indices[0]
                # If done happens at step k, we want to train on step k (the crash/success),
                # but NOT on step k+1 (which is start of next episode or junk).
                # So we mask from first_done + 1 onwards.
                if first_done + 1 < seq_len:
                    mask[first_done + 1:] = 0.0
                    # Optional: Zero out the inputs/actions after done to be clean
                    # raw_pr[first_done + 1:] = 0
                    # raw_act[first_done + 1:] = 0

            pr_list.append(raw_pr)
            act_list.append(raw_act)
            mask_list.append(mask)

        pr_arr = torch.from_numpy(np.stack(pr_list, axis=1))     # (T,B,Nin)
        act_arr = torch.from_numpy(np.stack(act_list, axis=1))   # (T,B,Act)
        mask_arr = torch.from_numpy(np.stack(mask_list, axis=1)) # (T,B,1)
        
        return pr_arr, act_arr, mask_arr


def log_training_loss(iter_idx, mean_loss, batches, buffer_size):
    os.makedirs(os.path.dirname(LOSS_CSV_PATH) or ".", exist_ok=True)
    write_header = not os.path.exists(LOSS_CSV_PATH)
    with open(LOSS_CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["dagger_iter", "mean_loss", "batches", "buffer_size"])
        writer.writerow([iter_idx, mean_loss, batches, buffer_size])

def save_checkpoint(agent, iter_idx):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"connectome_rnn_dagger_iter_{iter_idx}.pt")
    torch.save(agent.state_dict(), ckpt_path)
    return ckpt_path


def forward_policy_sequence(agent, xs):
    # Use the unified efficient sequence processing
    _, y_seq, _ = agent.forward_sequence(xs, checkpoint_steps=USE_GRADIENT_CHECKPOINT)
    return y_seq

def rollout_and_collect(env, teacher, agent, buffer, beta, device, dtype, beta_noise=0.0):
    agent.eval()  # Inference mode during rollout
    for ep in range(EPISODES_PER_ITER):
        obs, _ = env.reset()
        agent.reset_vision_state()
        teacher.reset()
        h = torch.zeros(1, agent.cell.N, device=device, dtype=dtype)
        steps = 0
        
        noise_disturbing = False
        noise_steps = 0
        noise = np.zeros(2, dtype=np.float32)

        while steps < MAX_EPISODE_STEPS:
            maybe_show_cameras(obs)
            obs_t = obs_to_torch(obs, device=device, dtype=dtype)
            xs = agent.obs_to_x(obs_t)

            with torch.no_grad():
                h, student_action = agent.step(
                    h,
                    {"cam_left": obs_t["cam_left"], "cam_right": obs_t["cam_right"], "sensors": obs_t["sensors"]},
                    x=xs,
                )
            student_action_np = student_action.squeeze(0).cpu().numpy()
            teacher_action = teacher.act(env)
            if teacher_action is None: # teacher cannot provide action due to no path from bad obstacles placement, skip this episode
                break
            # always let teacher take over when there's a collision
            if teacher._rec_phase is not None:
                action_exec = teacher_action
                # print(teacher._rec_phase)
            else:
                action_exec = beta * teacher_action + (1.0 - beta) * student_action_np

            # Noise Injection Logic (Preserved)
            if not noise_disturbing and teacher._rec_phase is None and steps % NOISE_INTERVAL == 1:
                noise_disturbing = True
                noise_steps = 0
                noise = np.array([0.0, np.random.uniform(-1.0, 1.0)], dtype=np.float32) * beta_noise # noise disabled for now
            if noise_disturbing:
                action_exec += noise
                noise_steps += 1
                if noise_steps >= 50:
                    noise_disturbing = False

            # action_exec = np.clip(action_exec, -1.0, 1.0)
            obs, _, done, trunc, _ = env.step(action_exec)
            terminal = bool(done or trunc)

            buffer.add(xs.squeeze(0).detach().cpu().numpy(), teacher_action, terminal)
            steps += 1

            if terminal:
                break


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
    total_len = T_BURN + T_UNROLL
    
    # Zero gradients at the start of accumulation
    opt.zero_grad(set_to_none=True)
    
    total_loss = 0.0
    
    for accum_idx in range(accum_steps):
        # Sample a batch
        xs, ys, mask = buffer.sample_sequences(batch_size=BATCH_SIZE, seq_len=total_len)
        
        mu = forward_policy_sequence(agent, xs)
        
        ys = ys.to(device=mu.device, dtype=mu.dtype, non_blocking=True)
        mask = mask.to(device=mu.device, dtype=mu.dtype, non_blocking=True)

        # Slice for loss (ignore burn-in period)
        mu_tail = mu[T_BURN:]
        ys_tail = ys[T_BURN:]
        mask_tail = mask[T_BURN:]
        mask_tail_2d = mask_tail.squeeze(-1)

        # Compute masked MSE
        squared_error_vel = (mu_tail[..., 0] - ys_tail[..., 0]) ** 2
        squared_error_angle = STEERING_LOSS_SCALE * (1 - torch.cos(mu_tail[..., 1] - ys_tail[..., 1]))
        squared_error = squared_error_vel + squared_error_angle

        # Weighted Loss for turning samples
        turn_magnitude = torch.abs(ys_tail[..., 1])
        sample_weight = 1.0 + TURN_BIAS_STRENGTH * turn_magnitude
        weighted_squared_error = squared_error * sample_weight
        masked_error = weighted_squared_error * mask_tail_2d
        
        valid_elements = mask_tail_2d.sum()  # squared_error already sums over action dims
        # DEBUG: Check for empty batches or alpha saturation
        if valid_elements < 1.0:
            print(f"[train] Warning: Batch has {valid_elements.item()} valid elements!")
            
        loss = masked_error.sum() / (valid_elements + 1e-6)
        
        # Scale loss for gradient accumulation (average over accum steps)
        scaled_loss = loss / accum_steps
        
        # Backward pass (accumulates gradients)
        scaled_loss.backward()
        
        # Track unscaled loss for logging
        total_loss += loss.item()
        
        # Memory cleanup for this accumulation step
        del mu, xs, ys, mask, mu_tail, ys_tail, mask_tail, mask_tail_2d
        del squared_error, squared_error_vel, squared_error_angle
        del weighted_squared_error, masked_error, loss, scaled_loss
    
    # Gradient checks (after accumulation)
    if agent.cell.W_values.grad is None and agent.cell.W_values.requires_grad: 
        print("Warning: RNN weights have no gradients!")
        alphas = agent.cell.get_alphas().detach().cpu().numpy()
        print(f"  Debug: Alphas min/max/mean: {alphas.min():.4f}/{alphas.max():.4f}/{alphas.mean():.4f}")
    if agent.cell.bias.grad is None and agent.cell.bias.requires_grad: 
        print("Warning: RNN bias has no gradients!")
    
    # Clip gradients and update weights (once per train_step call)
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 10.0)
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
        end_on_collision=END_ON_COLLISION,
    )

    cell, pr_positions, input_splits = build_connectome_cell(
        edge_path=EDGE_PATH,
        device=device,
        dtype=dtype,
        photoreceptor_left_csv=PHOTORECEPTOR_LEFT_CSV,
        photoreceptor_right_csv=PHOTORECEPTOR_RIGHT_CSV,
        olfactory_left_csv=OLFACTORY_LEFT_CSV,
        olfactory_right_csv=OLFACTORY_RIGHT_CSV,
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

    teacher = PlannerAnalyticTeacher(
        arena_half_extent=env.arena,
        cell_size=0.1,
        robot_radius=0.2,
        safety_margin=0.01,
        obstacle_box_half=(0.4, 0.4),
        k_nearest_obs=5,
        device="cpu",
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    buffer = DAggerBuffer(capacity=MAX_BUFFER_SIZE)

    opt = configure_optimizer(agent)

    try:
        for it in range(N_DAGGER_ITERS):
            beta = beta_schedule(it)
            beta_noise = noise_schedule(it)
            print(f"\n[DAgger] Iteration {it+1}/{N_DAGGER_ITERS} | beta={beta:.3f} | noise={beta_noise:.3f} | buffer={len(buffer)}")

            rollout_and_collect(env, teacher, agent, buffer, beta, device, dtype, beta_noise=beta_noise)
            print(f"[data] Collected data; buffer size now {len(buffer)}")

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
            log_training_loss(it + 1, mean_loss, batches, len(buffer))

            if losses:
                print(f"[train] mean loss={mean_loss:.5f} | batches={batches}")
            else:
                print("[train] skipped (insufficient buffer)")

            if (it + 1) % 10 == 0:
                ckpt_path = save_checkpoint(agent, it + 1)
                print(f"[ckpt] Saved checkpoint to {ckpt_path}")

    finally:
        env.close()
        if RENDER_MODE == "human" and cv2 is not None:
            cv2.destroyAllWindows()
        torch.save(agent.state_dict(), FINAL_CHECKPOINT_PATH)
        print(f"Saved trained model to {FINAL_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
