"""
Shared training configuration module.
Used by both train_connectome_rnn_dagger.py and train_connectome_rnn_rl.py 
to ensure consistent optimizer configuration.
"""

import torch

# -----------------------------
# Training Flags
# -----------------------------
TRAIN_RNN_WEIGHTS = True
TRAIN_RNN_BIAS = True
TRAIN_READOUT_HEAD = True
TRAIN_VALUE_HEAD = True
TRAIN_INPUT_SCALE = True
TRAIN_POLICY_HEAD = True
TRAIN_WIND_MLP = True
TRAIN_RETINA = True

# -----------------------------
# Learning Rate
# -----------------------------
LR = 1e-4

# BASE_PATH = "connectomes/ws_small_world/"
BASE_PATH = "connectomes/drosophila adult connectome/"
# -----------------------------
# Paths (hardcoded)
# -----------------------------
EDGE_PATH = "connectomes/drosophila adult connectome/connections_princeton_random.csv"
# EDGE_PATH = BASE_PATH + "connections_ws_small_world.csv"
PHOTORECEPTOR_LEFT_CSV = BASE_PATH + "visual_column_L1_L2_L3_rear_view_left.csv"
PHOTORECEPTOR_RIGHT_CSV = BASE_PATH + "visual_column_L1_L2_L3_rear_view_right.csv"
TACTILE_LEFT_CSV = BASE_PATH + "head_bristles_left.csv"
TACTILE_RIGHT_CSV = BASE_PATH + "head_bristles_right.csv"
DESCENDING_NEURONS_CSV = BASE_PATH + "descending_neurons.csv"
CELL_TYPES_CSV = BASE_PATH + "consolidated_cell_types.csv"
WIND_SENSING_CSV = BASE_PATH + "JO-C_and_JO-E.csv"

# -----------------------------
# Environment & Model Settings
# -----------------------------
ENV_WIDTH = 128
ENV_HEIGHT = 128
MAX_EPISODE_STEPS = 600
N_OBSTACLES = 20
ARENA_HALF_EXTENT = 7.0
RENDER_MODE = None
END_ON_COLLISION = False

LEAK_ALPHA = 0.2
ACTIVATION = "tanh"
TARGET_RHO = 0.9
BATCH_CHUNK = 8
ROW_TILE_SIZE = 69320
INPUT_SCALE_INIT = 1.0
DTYPE = torch.float32
USE_GRADIENT_CHECKPOINT = False

CHECKPOINT_DIR = "checkpoints"
LOSS_DIR = "loss"


def configure_optimizer(
    agent,
    lr: float = LR,
    train_rnn_weights: bool = TRAIN_RNN_WEIGHTS,
    train_rnn_bias: bool = TRAIN_RNN_BIAS,
    train_readout_head: bool = TRAIN_READOUT_HEAD,
    train_input_scale: bool = TRAIN_INPUT_SCALE,
    train_policy_head: bool = TRAIN_POLICY_HEAD,
    train_value_head: bool = TRAIN_VALUE_HEAD,
    train_wind_mlp: bool = TRAIN_WIND_MLP,
    train_retina: bool = TRAIN_RETINA,
) -> torch.optim.Optimizer:
    """
    Configure optimizer with explicit control over which parameters to train.
    
    Args:
        agent: ConnectomeAgent instance
        lr: Base learning rate
        train_rnn_weights: Train W_values (sparse RNN connectivity)
        train_rnn_bias: Train RNN bias
        train_readout_head: Train the readout head (policy output)
        train_input_scale: Train input scaling parameters
        train_policy_head: Train policy head (same as readout in many cases)
        train_value_head: Train value head (for RL)
        train_wind_mlp: Train wind sensing MLP
        train_retina: Train virtual retina parameters (r_scale, l1_alpha, amacrine, etc.)
        
    Returns:
        Configured Adam optimizer
    """
    trainable_normal = []
    trainable_alpha = []
    
    print("[train] Configuring trainable parameters:")
    for name, p in agent.named_parameters():
        p.requires_grad = False
        train_this = False
        
        # Categorize parameter
        is_value_head = "value_head" in name
        is_policy_std = "policy_log_std" in name
        is_input_scale = "scale" in name and "input_scale" in name
        is_retina = "r_" in name or "l1" in name or "l2" in name or "l3" in name or "amacrine" in name
        
        # Check if it belongs to the cell
        is_cell = "cell" in name
        
        # Determine specific cell parts
        is_readout = is_cell and ("readout_head" in name or "head" in name)
        is_bias = is_cell and "bias" in name
        is_alpha = is_cell and "alpha" in name
        is_weight = is_cell and not (is_readout or is_bias or is_alpha)
        is_wind_mlp = "wind_mlp" in name

        # Apply Logic
        if is_value_head and train_value_head:
            train_this = True
        elif is_policy_std and train_policy_head:
            train_this = True
        elif is_readout and train_readout_head:
            train_this = True
        elif is_input_scale and train_input_scale:
            train_this = True
        elif is_wind_mlp and train_wind_mlp:
            train_this = True
        elif is_retina and train_retina:
            train_this = True
        elif is_cell:
            if is_bias and train_rnn_bias:
                train_this = True
            elif is_alpha and (train_rnn_weights or train_rnn_bias): 
                # Train alpha when either RNN weights or biases are being trained
                train_this = True
            elif is_weight and train_rnn_weights:
                train_this = True
            
        if train_this:
            p.requires_grad = True
            if "alpha" in name:
                trainable_alpha.append(p)
                print(f"  [+] {name} (lr={lr*0.1:.2e})")
            else:
                trainable_normal.append(p)
                print(f"  [+] {name}")
        else:
            print(f"  [ ] {name}")
    
    # Build optimizer with conditional parameter groups
    param_groups = [{'params': trainable_normal, 'lr': lr}]
    if trainable_alpha:
        param_groups.append({'params': trainable_alpha, 'lr': lr * 0.1})
    
    optimizer = torch.optim.Adam(param_groups)
    print("[train] trainable normal params:", sum(p.numel() for p in trainable_normal))
    print("[train] trainable alpha params:", sum(p.numel() for p in trainable_alpha))
    
    return optimizer
