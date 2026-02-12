"""
On-policy reinforcement learning for the connectome RNN agent (PPO-style).
The preprocessing, connectome wiring, and environment setup mirror the DAgger
implementation so checkpoints remain compatible.
"""

from __future__ import annotations

import os
import csv
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

from agents.connectome_rnn_agent import ConnectomeAgent
from core.utils import get_device
from environment.mujoco_two_cam_env_random_obstacles import MuJoCoTwoCamEnv
from shared_config import (
    ARENA_HALF_EXTENT,
    DTYPE,
    EDGE_PATH,
    ENV_HEIGHT,
    ENV_WIDTH,
    INPUT_SCALE_INIT,
    MAX_EPISODE_STEPS,
    N_OBSTACLES,
    PHOTORECEPTOR_LEFT_CSV,
    PHOTORECEPTOR_RIGHT_CSV,
    OLFACTORY_LEFT_CSV,
    OLFACTORY_RIGHT_CSV,
    TACTILE_LEFT_CSV,
    TACTILE_RIGHT_CSV,
    DESCENDING_NEURONS_CSV,
    CELL_TYPES_CSV,
    WIND_SENSING_CSV,
    TARGET_RHO,
    LEAK_ALPHA,
    ACTIVATION,
    BATCH_CHUNK,
    ROW_TILE_SIZE,
    USE_GRADIENT_CHECKPOINT,
    CHECKPOINT_DIR,
    LOSS_DIR,
)
from core.utils import build_connectome_cell, obs_to_torch

# -----------------------------
# RL Hyperparameters
# -----------------------------
ROLLOUT_STEPS = 20000
PPO_EPOCHS = 150
TOTAL_UPDATES = 2000
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
LR = 1e-4
VF_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 1.0
BPTT_HORIZON = 80  # Truncated BPTT horizon

# -----------------------------
# Reward Shaping
# -----------------------------
CTRL_PENALTY = 0.001   # Reduced from 0.002 to encourage movement
TIME_PENALTY = 0.01   # Increased from 0.005 to discourage stalling
PROG_SCALE = 10.0      # Increase progress weight
GOAL_BONUS = 40.0      # Keeping default
CONTACT_PENALTY = 5 


    
# -----------------------------
# Training Control Switches
# -----------------------------
TRAIN_INPUT_SCALE = True  
TRAIN_RNN_WEIGHTS = True      # Train the connections?
TRAIN_RNN_BIAS = True         # Train the biases?
TRAIN_VALUE_HEAD = True       # Always True for RL
TRAIN_POLICY_HEAD = True      # Always True for RL
END_ON_COLLISION = True

CHECKPOINT_INTERVAL = 10
FINAL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "connectome_rnn_rl.pt")
WARM_START_PATH = os.path.join(CHECKPOINT_DIR, "connectome_rnn_rl_warmup.pt")
RENDER_MODE = "human"  # Set to "human" to visualize during training


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.xs: List[torch.Tensor] = []
        self.priv_obs: List[torch.Tensor] = [] # New buffer
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.episode_starts: List[bool] = []
        self.timeouts: List[bool] = []
        self.bootstrap_values: List[torch.Tensor] = []

    def __len__(self):
        return len(self.rewards)

    def add(
        self,
        x: torch.Tensor,
        priv: torch.Tensor, 
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,
        episode_start: bool,
        timeout: bool,
        bootstrap_value: torch.Tensor,
    ) -> None:
        self.xs.append(x.detach().clone())
        self.priv_obs.append(priv.detach().clone())
        self.actions.append(action.detach().clone())
        self.log_probs.append(log_prob.detach().clone())
        self.values.append(value.detach().clone())
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.episode_starts.append(bool(episode_start))
        self.timeouts.append(bool(timeout))
        self.bootstrap_values.append(bootstrap_value)

    def to_tensors(
        self, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, ...]:
        xs = torch.stack(self.xs, dim=0).to(device=device, dtype=dtype)
        priv = torch.stack(self.priv_obs, dim=0).to(device=device, dtype=dtype)
        actions = torch.stack(self.actions, dim=0).to(device=device, dtype=dtype)
        log_probs = torch.stack(self.log_probs, dim=0).to(device=device, dtype=dtype).view(-1)
        values = torch.stack(self.values, dim=0).to(device=device, dtype=dtype).view(-1)
        rewards = torch.tensor(self.rewards, device=device, dtype=dtype)
        dones = torch.tensor(self.dones, device=device, dtype=dtype)
        episode_starts = torch.tensor(self.episode_starts, device=device, dtype=torch.bool)
        timeouts = torch.tensor(self.timeouts, device=device, dtype=torch.bool)
        bootstrap_values = torch.cat(self.bootstrap_values, dim=0).to(device=device, dtype=dtype)
        return xs, priv, actions, log_probs, values, rewards, dones, episode_starts, timeouts, bootstrap_values

    def episode_returns(self) -> List[float]:
        rets: List[float] = []
        acc = 0.0
        for r, d in zip(self.rewards, self.dones):
            acc += float(r)
            if d:
                rets.append(acc)
                acc = 0.0
        if acc:
            rets.append(acc)
        return rets


def make_env(render_mode: str | None) -> MuJoCoTwoCamEnv:
    return MuJoCoTwoCamEnv(
        width=ENV_WIDTH,
        height=ENV_HEIGHT,
        max_episode_steps=MAX_EPISODE_STEPS,
        n_obstacles=N_OBSTACLES,
        arena_half_extent=ARENA_HALF_EXTENT,
        ctrl_penalty=CTRL_PENALTY,
        goal_bonus=GOAL_BONUS,
        contact_penalty=CONTACT_PENALTY,
        time_penalty=TIME_PENALTY,
        prog_scale=PROG_SCALE,
        render_mode=render_mode,
        end_on_collision=END_ON_COLLISION,
    )


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lam: float,
    timeouts: torch.Tensor,
    bootstrap_values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized GAE computation with timeout/truncation handling."""
    T = rewards.size(0)
    device = rewards.device
    dtype = rewards.dtype
    
    # Compute masks: 1.0 if not done (continue), 0.0 if done (reset)
    masks = 1.0 - dones  # (T,)
    
    # Compute next_values: shift values by 1, pad with 0 at the end
    next_values = torch.cat([values[1:], torch.zeros(1, device=device, dtype=dtype)])
    
    # Compute deltas (TD residuals)
    # For normal transitions: delta = r + gamma * V(s') * mask - V(s)
    # For timeouts: delta = r + gamma * bootstrap_value - V(s)
    deltas = torch.zeros_like(rewards)
    
    # Normal case
    deltas = rewards + gamma * next_values * masks - values
    
    # Override for timeout steps: use bootstrap value instead of next_values
    timeout_mask = timeouts.bool()
    if timeout_mask.any():
        deltas[timeout_mask] = rewards[timeout_mask] + gamma * bootstrap_values[timeout_mask] - values[timeout_mask]
    
    # Vectorized GAE: advantages[t] = sum_{l=0}^{T-t-1} (gamma*lam)^l * delta[t+l]
    # We compute this via reverse cumsum with decay
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(1, device=device, dtype=dtype)
    
    # Process in reverse (still need loop for recursive structure, but operations are scalar)
    # This is the standard approach - true vectorization of GAE requires matrix operations
    # which are memory-intensive for large T. This loop is fast since it's just scalar ops.
    for t in reversed(range(T)):
        if timeouts[t]:
            # Timeout: bootstrap but reset GAE chain
            gae = deltas[t]
        else:
            # Normal: recursive GAE
            gae = deltas[t] + gamma * lam * masks[t] * gae
        advantages[t] = gae
    
    returns = advantages + values
    return returns, advantages

def evaluate_sequence(
    agent: ConnectomeAgent,
    xs: torch.Tensor,
    priv_obs: torch.Tensor,
    actions: torch.Tensor,
    episode_starts: torch.Tensor,
    dtype: torch.dtype,
    h_init: Optional[torch.Tensor] = None,
    use_gradient_checkpoint: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Efficiently evaluate a sequence of observations using batch processing.
    """
    device = xs.device
    
    # 1. Forward pass through RNN using efficient sequence processing
    # Note: ConnectomeRNN assumes continuous flow. We ignore resets within the chunk for speed.
    h_final, mean_seq, dn_seq = agent.forward_sequence(
        xs, h_init=h_init, checkpoint_steps=use_gradient_checkpoint
    )
    
    # 2. Re-evaluate Policy Distributions (Batch Mode)
    # mean_seq: (T, Nout) if B=1, or (T, B, Nout).
    if mean_seq.dim() == 2: # (T, Nout) -> (T, 1, Nout)
        mean_seq = mean_seq.unsqueeze(1)
        
    flat_mean = mean_seq.view(-1, 2)
    dist = agent._policy_dist(flat_mean)
    
    # actions: (T, Act) -> (T*B, Act)
    if actions.dim() == 2:
        actions = actions.unsqueeze(1)
    flat_actions = actions.view(-1, 2)
    
    # Compute log_probs and entropy in batch
    log_probs = dist.log_prob(flat_actions).sum(dim=-1) # (T*B,)
    entropies = dist.entropy().sum(dim=-1)              # (T*B,)
    
    # 3. Re-evaluate Value Head (Batch Mode)
    values = torch.zeros_like(log_probs)
    if agent.value_head is not None:
        # OPTIMIZATION: dn_seq already contains only output neurons (T, B, Nout)
        # No need to index_select - DN_seq is exactly what _value_from_hidden needs
        T, B, Nout = dn_seq.shape
        flat_dn = dn_seq.view(-1, Nout)
        
        # priv_obs: (T, B, PrivDim) or (T, PrivDim)
        if priv_obs.dim() == 2:
            priv_obs = priv_obs.unsqueeze(1) # (T, 1, PrivDim)
        flat_priv = priv_obs.view(-1, priv_obs.size(-1))
        
        # Calculate values by directly concatenating DN and priv_obs
        if agent.priv_obs_dim > 0:
            value_input = torch.cat([flat_dn, flat_priv], dim=1)
        else:
            value_input = flat_dn
        values = agent.value_head(value_input).view(-1)
        
    # Return flattened tensors for PPO
    return log_probs, entropies, values, h_final



def collect_rollout(
    env: MuJoCoTwoCamEnv,
    agent: ConnectomeAgent,
    device: torch.device,
    dtype: torch.dtype,
    steps: int,
    last_obs: Dict,
    last_episode_start: bool,
    initial_h: torch.Tensor,
) -> Tuple[RolloutBuffer, Dict, bool, torch.Tensor]:
    buffer = RolloutBuffer()
    obs = last_obs
    episode_start = last_episode_start
    h = initial_h.clone().to(device=device, dtype=dtype)

    for _ in range(steps):
        obs_t = obs_to_torch(obs, device=device, dtype=dtype)
        x = agent.obs_to_x(obs_t)
        
        priv = obs["privileged"]
        priv_t = torch.tensor(priv, device=device, dtype=dtype).unsqueeze(0) # (1, 17)

        with torch.no_grad():
            h, action, log_prob, value = agent.act(
                h,
                {"cam_left": obs_t["cam_left"], "cam_right": obs_t["cam_right"], "sensors": obs_t["sensors"]},
                x=x,
                deterministic=False,
                priv_obs=priv_t, 
            )

        # Actions are already properly bounded by agent:
        # - velocity in [-1, 1]
        # - angle in [-π, π]
        action_np = action.squeeze(0).cpu().numpy()
        next_obs, reward, done, trunc, _ = env.step(action_np)
        if trunc:
            # Bootstrap value for truncation
            with torch.no_grad():
                obs_next_t = obs_to_torch(next_obs, device=device, dtype=dtype)
                priv_next_t = torch.tensor(next_obs["privileged"], device=device, dtype=dtype).unsqueeze(0)
                # Pass update_state=False to avoid corrupting vision state during lookahead
                x_next = agent.obs_to_x(obs_next_t, update_state=False)
                # Explicitly detach h to ensure no gradient leakage
                h_next, _ = agent.step(h.detach(), obs_next_t, x=x_next)
                bootstrap_value = agent._value_from_hidden(h_next, priv_obs=priv_next_t)
        else:
            bootstrap_value = torch.zeros(1, device=device, dtype=dtype)

        buffer.add(
            x.squeeze(0).cpu(),
            priv_t.squeeze(0).cpu(),
            action.squeeze(0).cpu(),
            log_prob.cpu(),
            (value if value is not None else torch.zeros(1)).cpu(),
            float(reward),
            bool(done or trunc),
            episode_start,
            bool(trunc),
            bootstrap_value.view(1).cpu(),
        )

        if done or trunc:
            obs, _ = env.reset()
            # Reset vision state at the start of new episode
            agent.reset_vision_state()
            
            # Valid reasoning: if episode ends, h should be reset for the NEXT step's forward pass.
            # However, h is returned from agent.act() which was the result of the CURRENT step.
            # So next step (start of next iteration) should start with h=0.
            h = torch.zeros(1, agent.cell.N, device=device, dtype=dtype)
            episode_start = True
        else:
            obs = next_obs
            episode_start = False
            
    # Return the state for the next rollout
    return buffer, obs, episode_start, h


def ppo_update(
    agent: ConnectomeAgent,
    optimizer: torch.optim.Optimizer,
    xs: torch.Tensor,
    priv_obs: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    episode_starts: torch.Tensor,
    dtype: torch.dtype,
    initial_h: Optional[torch.Tensor] = None,
) -> Tuple[float, float, float, float, float]:
    # Initialize hidden state for the beginning of the rollout
    # If initial_h is provided, use it. Otherwise zero.
    if initial_h is not None:
        h_state = initial_h.clone().to(device=xs.device, dtype=dtype)
    else:
        h_state = torch.zeros(1, agent.cell.N, device=xs.device, dtype=dtype)
    
    total_samples = xs.size(0)
    assert total_samples == ROLLOUT_STEPS, f"Tensor size {total_samples} != ROLLOUT_STEPS {ROLLOUT_STEPS}"

    # Verify we have integers
    num_chunks = (total_samples + BPTT_HORIZON - 1) // BPTT_HORIZON

    accum_loss = 0.0
    accum_pol = 0.0
    accum_val = 0.0
    accum_ent = 0.0
    accum_kl = 0.0
    num_updates = 0
    
    # Zero gradients before the accumulation loop
    optimizer.zero_grad(set_to_none=True)

    # TBPTT Loop: iterate over chunks
    for start_t in range(0, total_samples, BPTT_HORIZON):
        end_t = min(start_t + BPTT_HORIZON, total_samples)
        
        # Slice mini-batch
        chunk_xs = xs[start_t:end_t]
        chunk_priv = priv_obs[start_t:end_t]
        chunk_actions = actions[start_t:end_t]
        chunk_old_log = old_log_probs[start_t:end_t]
        chunk_returns = returns[start_t:end_t]
        chunk_adv = advantages[start_t:end_t]
        chunk_starts = episode_starts[start_t:end_t]

        # Forward pass for this chunk given current h_state
        new_log_probs, entropies, values_pred, h_state = evaluate_sequence(
            agent, chunk_xs, chunk_priv, chunk_actions, chunk_starts, dtype, 
            h_init=h_state, use_gradient_checkpoint=USE_GRADIENT_CHECKPOINT
        )
        
        # Detach hidden state to stop gradient backprop to previous chunk
        h_state = h_state.detach()

        # Compute Loss
        ratio = torch.exp(new_log_probs - chunk_old_log)
        surr1 = ratio * chunk_adv
        surr2 = torch.clamp(ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF) * chunk_adv
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values_pred, chunk_returns)
        entropy_loss = entropies.mean()

        loss = policy_loss + VF_COEF * value_loss - ENTROPY_COEF * entropy_loss
        
        # Backward pass with scaled loss for gradient accumulation
        (loss / num_chunks).backward()

        approx_kl = (chunk_old_log - new_log_probs).mean().item()

        # Accumulate stats (use raw loss values)
        accum_loss += loss.item()
        accum_pol += policy_loss.item()
        accum_val += value_loss.item()
        accum_ent += entropy_loss.item()
        accum_kl += approx_kl
        num_updates += 1

    # Update parameters once after processing all chunks
    torch.nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
    optimizer.step()

    return (
        accum_loss / num_updates,
        accum_pol / num_updates,
        accum_val / num_updates,
        accum_ent / num_updates,
        accum_kl / num_updates,
    )


def configure_optimizer(agent: ConnectomeAgent) -> torch.optim.Optimizer:
    trainable_normal = []
    trainable_alpha = []
    print("[train] Configuring trainable parameters:")
    for name, p in agent.named_parameters():
        p.requires_grad = False
        train_this = False
        
        # Categorize parameter
        is_value_head = "value_head" in name
        is_policy_std = "policy_log_std" in name
        is_input_scale = "scale" in name
        is_retina = "r_" in name or "l1" in name or "l2" in name or "l3" in name or "amacrine" in name
        
        # Check if it belongs to the cell
        is_cell = "cell" in name
        
        # Determine specific cell parts
        is_readout = is_cell and ("readout_head" in name or "head" in name) # Part of policy
        is_bias = is_cell and "bias" in name
        is_alpha = is_cell and "alpha" in name
        is_weight = is_cell and not (is_readout or is_bias or is_alpha)
        is_wind_mlp = "wind_mlp" in name

        # Apply Logic
        if is_value_head and TRAIN_VALUE_HEAD:
            train_this = True
        elif is_policy_std and TRAIN_POLICY_HEAD:
            train_this = True
        elif is_readout and TRAIN_POLICY_HEAD:
            train_this = True
        elif is_input_scale and TRAIN_INPUT_SCALE:
            train_this = True
        elif is_wind_mlp:# and TRAIN_WIND_MLP:
            train_this = True
        elif is_retina:
            train_this = True
        elif is_cell:
            if is_bias and TRAIN_RNN_BIAS:
                train_this = True
            elif is_alpha and (TRAIN_RNN_WEIGHTS or TRAIN_RNN_BIAS): 
                 # Train alpha when either RNN weights or biases are being trained
                 train_this = True
            elif is_weight and TRAIN_RNN_WEIGHTS:
                train_this = True
            
        # Hard override for alpha if we want it to be separate? 
        # For now, let's link alpha to TRAIN_RNN_WEIGHTS or just enable it if bias is enabled?
        # A clearer approach is often to train alpha if we train ANY RNN dynamics.
        # But let's stick to TRAIN_RNN_WEIGHTS for alpha.
        
        if train_this:
            p.requires_grad = True
            if "alpha" in name:
                trainable_alpha.append(p)
                print(f"  [+] {name} (lr={LR*0.1:.2e})")
            else:
                trainable_normal.append(p)
                print(f"  [+] {name}")
        else:
            print(f"  [ ] {name}")
    
    # Build optimizer with conditional parameter groups
    param_groups = [{'params': trainable_normal, 'lr': LR}]
    if trainable_alpha:
        param_groups.append({'params': trainable_alpha, 'lr': LR * 0.1})
    
    optimizer = torch.optim.Adam(param_groups)
    print("[train] trainable normal params:", sum(p.numel() for p in trainable_normal))
    print("[train] trainable alpha params:", sum(p.numel() for p in trainable_alpha))
    
    return optimizer

def main():
    device = get_device()
    dtype = DTYPE
    print(f"[main] using device: {device}")

    # 1. Setup Environment
    env = make_env(render_mode=RENDER_MODE)
    
    # 2. Setup Agent
    # We need the cell first
    cell, pr_positions, input_splits = build_connectome_cell(
        edge_path=EDGE_PATH,
        device=device,
        dtype=DTYPE,
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
        train_readout_head=TRAIN_POLICY_HEAD,
        batch_chunk=BATCH_CHUNK,
        row_tile_size=ROW_TILE_SIZE,
    )
    agent = ConnectomeAgent(
        cell=cell,
        photoreceptor_positions=pr_positions,
        input_splits=input_splits,
        dtype=DTYPE,
        input_scale_init=INPUT_SCALE_INIT,
        use_value_head=True,      # Enable value head for RL
        learn_policy_std=True,    # Learn action noise
        policy_std_init=0.5,
    )
    agent.to(device)


    # Load starting checkpoint (DAgger or Warmup) with strict=False
    if TRAIN_VALUE_HEAD:
        print("[main] Initializing Value Head with priv_obs_dim=17")
        agent.init_value_head(priv_obs_dim=17)

    if os.path.exists(WARM_START_PATH):
        print(f"[main] Loading warm-start checkpoint: {WARM_START_PATH}")
        ckpt = torch.load(WARM_START_PATH, map_location=device)
        agent.load_state_dict(ckpt, strict=False)
    
    agent.to(device)

    optimizer = configure_optimizer(agent)

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOSS_DIR, exist_ok=True)

    # --- Initialize CSV Logging ---
    csv_log_path = os.path.join(LOSS_DIR, "training_log_rl.csv")
    csv_file = open(csv_log_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        'update', 'loss', 'policy_loss', 'value_loss', 
        'entropy', 'kl', 'mean_episode_return', 'steps'
    ])
    csv_writer.writeheader()
    print(f"[log] Logging training metrics to {csv_log_path}")

    # --- Initialize Training State ---
    obs, _ = env.reset()
    agent.reset_vision_state() # Initial reset
    episode_start = True
    agent_h = torch.zeros(1, agent.cell.N, device=device, dtype=dtype)

    try:
        for update in range(1, TOTAL_UPDATES + 1):
            
            # Save the hidden state AT THE START of the rollout for PPO updates
            h_for_update = agent_h.detach().clone()
            
            # Collect rollout
            buffer, obs, episode_start, agent_h = collect_rollout(
                env, agent, device, dtype, 
                steps=ROLLOUT_STEPS,
                last_obs=obs,
                last_episode_start=episode_start,
                initial_h=agent_h
            )
            
            xs, priv_obs, actions, old_log_probs, values, rewards, dones, episode_starts, timeouts, bootstrap_values = buffer.to_tensors(device, dtype)

            returns, advantages = compute_gae(
                rewards,
                dones,
                values,
                gamma=GAMMA,
                lam=GAE_LAMBDA,
                timeouts=timeouts,
                bootstrap_values=bootstrap_values,
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            losses = []
            pol_losses = []
            val_losses = []
            ents = []
            kls = []
            for _ in range(PPO_EPOCHS):
                loss, pol_loss, val_loss, ent, approx_kl = ppo_update(
                    agent,
                    optimizer,
                    xs,
                    priv_obs,
                    actions,
                    old_log_probs,
                    returns.detach(),
                    advantages.detach(),
                    episode_starts,
                    dtype=dtype,
                    initial_h=h_for_update, # Pass the start-of-rollout hidden state
                )
                losses.append(loss)
                pol_losses.append(pol_loss)
                val_losses.append(val_loss)
                ents.append(ent)
                kls.append(approx_kl)

            ep_returns = buffer.episode_returns()
            mean_return = float(np.mean(ep_returns)) if ep_returns else 0.0

            # Log metrics
            metrics = {
                'update': update,
                'loss': np.mean(losses),
                'policy_loss': np.mean(pol_losses),
                'value_loss': np.mean(val_losses),
                'entropy': np.mean(ents),
                'kl': np.mean(kls),
                'mean_episode_return': mean_return,
                'steps': len(buffer)
            }
            
            print(
                f"[update {update}/{TOTAL_UPDATES}] "
                f"loss={metrics['loss']:.4f} | policy={metrics['policy_loss']:.4f} | value={metrics['value_loss']:.4f} "
                f"| entropy={metrics['entropy']:.4f} | kl={metrics['kl']:.5f} | mean_ep_ret={metrics['mean_episode_return']:.2f} | steps={metrics['steps']}"
            )
            
            # Write to CSV
            csv_writer.writerow(metrics)
            csv_file.flush()  # Ensure data is written immediately

            if update % CHECKPOINT_INTERVAL == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"connectome_rnn_rl_{update}.pt")
                torch.save(agent.state_dict(), ckpt_path)
                print(f"[ckpt] Saved checkpoint to {ckpt_path}")
    finally:
        env.close()
        csv_file.close()
        torch.save(agent.state_dict(), FINAL_CHECKPOINT_PATH)
        print(f"[done] Saved final model to {FINAL_CHECKPOINT_PATH}")
        print(f"[done] Training log saved to {csv_log_path}")


if __name__ == "__main__":
    main()
