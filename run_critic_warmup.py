
import os
import numpy as np
import torch
import torch.nn.functional as F

from agents.connectome_rnn_agent import ConnectomeAgent
from agents.teacher_analytic_agent import PlannerAnalyticTeacher
from core.utils import get_device
from environment.mujoco_two_cam_env_random_obstacles import MuJoCoTwoCamEnv
from train_connectome_rnn_dagger import (
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
    TACTILE_LEFT_CSV,
    TACTILE_RIGHT_CSV,
    DESCENDING_NEURONS_CSV,
    CELL_TYPES_CSV,
    WIND_SENSING_CSV, # NEW
    TARGET_RHO,
    LEAK_ALPHA,
    ACTIVATION,
    BATCH_CHUNK,
    ROW_TILE_SIZE,
)
from train_connectome_rnn_rl import CTRL_PENALTY, TIME_PENALTY, PROG_SCALE, GOAL_BONUS, CONTACT_PENALTY, END_ON_COLLISION

from core.utils import build_connectome_cell, obs_to_torch

# Configuration
CHECKPOINT_DIR = "checkpoints"
DAGGER_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "connectome_rnn_dagger.pt")
WARMUP_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "connectome_rnn_rl_warmup.pt")
CRITIC_WARMUP_STEPS = 20000
GAMMA = 0.99
NOISE_STD = 1.0

EPOCHS = 300


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
        seed=1337,# Optional seed
        end_on_collision=END_ON_COLLISION,
    )

def compute_returns_to_go(rewards, dones, gamma=GAMMA):
    """Simple Monte Carlo returns for warmup (unbiased target)."""
    R = 0
    returns = []
    for r, d in zip(reversed(rewards), reversed(dones)):
        if d:
            R = 0
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns)

def warmup_critic(agent, env, device, dtype, steps=CRITIC_WARMUP_STEPS, noise_std=NOISE_STD):
    print(f"\n[warmup] Warming up critic for {steps} steps with Analytic Teacher (noise_std={noise_std})...")
    
    # 1. Setup Teacher
    teacher = PlannerAnalyticTeacher(
        arena_half_extent=env.arena,
        cell_size=0.2,
        robot_radius=0.2,
        safety_margin=0.10,
        obstacle_box_half=(0.4, 0.4),
        k_nearest_obs=5,
        device="cpu", # Teacher runs on CPU usually
    )

    # 2. Collect Data
    obs, _ = env.reset()
    agent.reset_vision_state()
    teacher.reset()
    
    xs_list = []
    priv_obs_list = []
    rewards_list = []
    dones_list = []
    episode_starts_list = []
    
    # Noise State
    noise_disturbing = False
    noise_timer = 0
    current_noise = np.zeros(2, dtype=np.float32)
    NOISE_DURATION = 50
    NOISE_INTERVAL = 100

    print("[warmup] Collecting trajectories...")
    episode_start = True
    for t in range(steps):
        # Teacher Act
        teacher_action = teacher.act(env)
        if teacher_action is None: 
            # Recover if teacher fails (rare)
            teacher_action = np.zeros(2, dtype=np.float32)

        # Agent Act (to get x pre-processing only, we don't use action)
        obs_t = obs_to_torch(obs, device=device, dtype=dtype)
        x = agent.obs_to_x(obs_t)
        
        priv = obs["privileged"] # numpy array (17,)
        priv_t = torch.tensor(priv, device=device, dtype=dtype).unsqueeze(0)  # (1, 17) for consistency
        
        # --- Noise Logic ---
        # 1. Safety: If teacher is recovering, NO NOISE
        if teacher._rec_phase is not None:
             noise_disturbing = False
             current_noise[:] = 0.0
        
        # 2. Trigger: If not disturbing and interval hit (and noise enabled)
        elif not noise_disturbing and noise_std > 0.0 and (t % NOISE_INTERVAL == 0):
            noise_disturbing = True
            noise_timer = NOISE_DURATION
            # Sample new noise vector
            current_noise = np.random.normal(0, noise_std, size=teacher_action.shape).astype(np.float32)
            
        # 3. Apply: If disturbing, apply noise and decrement timer
        if noise_disturbing:
            action_exec = np.clip(teacher_action + current_noise, -1.0, 1.0)
            noise_timer -= 1
            if noise_timer <= 0:
                noise_disturbing = False
        else:
            action_exec = teacher_action

        # Step Env
        obs, reward, done, trunc, _ = env.step(action_exec)
        terminal = done or trunc
        
        xs_list.append(x.detach().cpu())
        priv_obs_list.append(priv_t.detach().cpu())
        rewards_list.append(float(reward))
        dones_list.append(float(terminal))
        episode_starts_list.append(episode_start)
        
        if terminal:
            obs, _ = env.reset()
            agent.reset_vision_state()
            teacher.reset()
            episode_start = True
        else:
            episode_start = False

    print(f"[warmup] Collected {len(rewards_list)} steps. Computing returns...")
    
    # 3. Prepare Training Data
    # Compute Returns-to-Go
    returns = compute_returns_to_go(rewards_list, dones_list, GAMMA).to(device=device, dtype=dtype)
    xs = torch.stack(xs_list, dim=0).to(device=device, dtype=dtype) # (T, 1, Nin) because obs_to_x returns (1, Nin)
    xs = xs.squeeze(1) # (T, Nin)
    # Note: priv_t is now (1, 17) due to unsqueeze(0), so squeeze back to (17,) before stacking
    priv_obs = torch.stack([p.squeeze(0) for p in priv_obs_list], dim=0).to(device=device, dtype=dtype) # (T, 17)
    
    episode_starts = torch.tensor(episode_starts_list, device=device, dtype=torch.bool)
    
    # 4. Train Value Head
    # Re-init value head to accept privileged info
    agent.init_value_head(priv_obs_dim=17)
    agent.to(device)

    # Freeze RNN and input scale, only train value head
    for p in agent.parameters(): p.requires_grad = False
    for p in agent.value_head.parameters(): p.requires_grad = True
    
    opt = torch.optim.Adam(agent.value_head.parameters(), lr=1e-3) # Higher LR for warmup
    
    batch_size = 64
    epochs = EPOCHS
    
    print("[warmup] Training Value Head...")
    agent.train()
    
    # We need to re-run RNN to get 'h' for value head. 
    # Since we have full trajectories, we can process them.
    
    with torch.no_grad():
        h = torch.zeros(1, agent.cell.N, device=device, dtype=dtype)
        all_hs = []
        for t in range(len(xs)):
            if episode_starts[t]:
                h = torch.zeros(1, agent.cell.N, device=device, dtype=dtype)
            h, _ = agent.step(h, {}, x=xs[t].unsqueeze(0))
            all_hs.append(h.clone())
        all_hs = torch.cat(all_hs, dim=0) # (T, N)
        
    dataset = torch.utils.data.TensorDataset(all_hs, priv_obs, returns)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for ep in range(epochs):
        ep_loss = 0
        for b_h, b_priv, b_ret in loader:
            pred_val = agent._value_from_hidden(b_h, priv_obs=b_priv)
            loss = F.mse_loss(pred_val, b_ret)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        print(f"  Epoch {ep+1}/{epochs} | Loss: {ep_loss/len(loader):.4f}")

    # Unfreeze everything
    for p in agent.parameters(): p.requires_grad = True
    print("[warmup] Done. Agent parameters unfrozen.\n")


def main():
    device = get_device()
    dtype = DTYPE
    print(f"[device] Using {device} ({dtype})")
    
    if not os.path.exists(DAGGER_CHECKPOINT_PATH):
        print(f"Error: DAgger checkpoint not found at {DAGGER_CHECKPOINT_PATH}")
        return

    # Create Env
    env = make_env(render_mode=None)
    
    # Create Agent
    cell, pr_positions, input_splits = build_connectome_cell(
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
        train_rnn_weights=False,
        train_readout_head=True,
        batch_chunk=BATCH_CHUNK,
        row_tile_size=ROW_TILE_SIZE,
    )
    agent = ConnectomeAgent(
        cell,
        photoreceptor_positions=pr_positions,
        input_splits=input_splits,
        dtype=dtype,
        input_scale_init=INPUT_SCALE_INIT,
        use_value_head=True,
        learn_policy_std=True,
    ).to(device)

    # Load DAgger Weights
    print(f"[load] Loading DAgger weights from {DAGGER_CHECKPOINT_PATH}...")
    state_dict = torch.load(DAGGER_CHECKPOINT_PATH, map_location=device)
    keys = agent.load_state_dict(state_dict, strict=False)
    print(f"[load] Loaded with missing keys: {len(keys.missing_keys)} (expected for RL heads)")

    # Run Warmup
    try:
        warmup_critic(agent, env, device, dtype, steps=CRITIC_WARMUP_STEPS)
    finally:
        env.close()

    # Save
    print(f"[save] Saving warmed-up model to {WARMUP_CHECKPOINT_PATH}...")
    torch.save(agent.state_dict(), WARMUP_CHECKPOINT_PATH)
    print("[done] Saved.")

if __name__ == "__main__":
    main()
