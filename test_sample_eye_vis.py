
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Sequence
import cv2

# Add project root to path
sys.path.append(os.getcwd())

from agents.connectome_rnn_agent import ConnectomeAgent
from core.utils import build_connectome_cell, get_device

# Reuse constants from train_connectome_rnn_dagger.py
EDGE_PATH = "connectomes/drosophila adult connectome/Connectivity_783.parquet"
PHOTORECEPTOR_LEFT_CSV = "connectomes/drosophila adult connectome/visual_column_L1_L2_L3_rear_view_left.csv"
PHOTORECEPTOR_RIGHT_CSV = "connectomes/drosophila adult connectome/visual_column_L1_L2_L3_rear_view_right.csv"
OLFACTORY_LEFT_CSV = "connectomes/drosophila adult connectome/olfactory_ORN_DM1_left.csv"
OLFACTORY_RIGHT_CSV = "connectomes/drosophila adult connectome/olfactory_ORN_DM1_right.csv"
TACTILE_LEFT_CSV = "connectomes/drosophila adult connectome/head_bristles_left.csv"
TACTILE_RIGHT_CSV = "connectomes/drosophila adult connectome/head_bristles_right.csv"
DESCENDING_NEURONS_CSV = "connectomes/drosophila adult connectome/descending_neurons.csv"
CELL_TYPES_CSV = "connectomes/drosophila adult connectome/consolidated_cell_types.csv"
WIND_SENSING_CSV = "connectomes/drosophila adult connectome/JO-C_and_JO-E.csv"

# Dummy cell for initializing agent
class DummyCell(nn.Module):
    def __init__(self, Nin, N, Nout, dtype):
        super().__init__()
        self.Nin = Nin
        self.N = N
        self.Nout = Nout
        self.W_values = torch.zeros(1, dtype=dtype) # Mock
        self.output_nodes = list(range(Nout))
        # self.readout_head = nn.Linear(N, 2) # Mock

def main():
    device = torch.device("cpu") # Test on CPU
    dtype = torch.float32

    print("Building connectome to get positions (this might take a moment)...")
    # We use the utility to get positions, but we can't easily decouple it without copying code.
    # So we'll just run build_connectome_cell to get everything correctly.
    # It might load the big matrix but we just need the positions.
    try:
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
            batch_chunk=8,
            row_tile_size=69320
        )
    except Exception as e:
        print(f"Error loading connectome: {e}")
        return

    print("Creating Agent...")
    agent = ConnectomeAgent(
        cell,
        photoreceptor_positions=pr_positions,
        input_splits=input_splits,
        dtype=dtype,
        input_scale_init=1.0
    ).to(device)

    # CREATE TEST IMAGE
    # Use a gradient and some shapes to see sampling clearly
    H, W = 256, 256
    img_np = np.zeros((H, W), dtype=np.float32)
    
    # 1. Vertical gradient
    for y in range(H):
        img_np[y, :] = y / H

    # 2. Add grid lines
    img_np[::32, :] = 1.0
    img_np[:, ::32] = 1.0

    # 3. Add a circle
    cv2.circle(img_np, (128, 128), 50, 0.5, -1)

    # 4. Normalize to [0, 1] for display (it's already mostly there)
    plt.imsave("test_input_image.png", img_np, cmap='gray')
    
    # Prepare tensor
    # Agent expects (B, 1, H, W) for _sample_eye, wait.
    # update: obs_to_x calls _to_grayscale which calls weights.sum... 
    # _sample_eye expects grayscale img (B, 1, H, W)?
    # in `obs_to_x`: cam_left = self._to_grayscale(obs["cam_left"])
    # obs["cam_left"] is usually (H, W, 3). 
    # _to_grayscale converts to (B, 1, H, W) if batching is handled...
    # obs_to_torch does: cam_left = torch.from_numpy(obs["cam_left"]).permute(2, 0, 1).unsqueeze(0)
    # So (B, C, H, W).
    
    img_t = torch.tensor(img_np, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    
    # VISUALIZE
    # We want to show where each photoreceptor sampled from.
    # The grid contains sample coordinates in [-1, 1].
    
    # Let's verify L1 Left
    grid_L1 = agent.grid_L1_left # (1, N, 1, 2)
    # N is number of points.
    
    if grid_L1 is None:
        print("grid_L1_left is None!")
        return

    # Sample
    sampled_vals = agent._sample_eye(img_t, grid_L1) # (1, N)
    sampled_vals_np = sampled_vals.detach().cpu().numpy().flatten()
    
    # GRID COORDS to ENVS COORDS (0, W), (0, H)
    # Grid is (x, y) in [-1, 1].
    # -1 is left/top? F.grid_sample(align_corners=True) matches -1 to pixel 0, +1 to pixel W-1.
    
    print(f"Sampled {len(sampled_vals_np)} L1 descriptors.")
    
    grid_np = grid_L1.detach().cpu().numpy().squeeze() # (N, 2)
    
    # Convert [-1, 1] to pixel coords
    # x_pix = (x + 1) * (W - 1) / 2
    # y_pix = (y + 1) * (H - 1) / 2
    
    x_grid = grid_np[:, 0]
    y_grid = grid_np[:, 1]
    
    x_pix = (x_grid + 1.0) * (W - 1.0) / 2.0
    y_pix = (y_grid + 1.0) * (H - 1.0) / 2.0
    
    # Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(img_np, cmap='gray', origin='upper', extent=[0, W, H, 0]) # Check extent convention
    # Note: imshow origin='upper' puts (0,0) at top-left.
    # grid_sample y=-1 is top? let's verify.
    # PyTorch grid_sample: (-1, -1) is TOP-LEFT.
    
    # SCATTER POINTS
    # Color them by sampled value to verify correctness (should match underlying image pixel)
    sc = plt.scatter(x_pix, y_pix, c=sampled_vals_np, cmap='viridis', s=100, edgecolors='none')
    plt.colorbar(sc, label='Sampled Value')
    plt.title("L1 Left Sampled Points (Colored by Activation)")
    # imshow with origin='upper' (default) puts index 0 at top. 
    # Our y_pix is 0 at top (-1 -> 0).
    # So we don't need invert_yaxis unless extent messed it up. 
    # imshow's default extent is usually (-0.5, W-0.5, H-0.5, -0.5).
    # Let's just use default and map coords correctly.
    
    plt.savefig("sample_eye_vis_L1_left.png")
    print("Saved sample_eye_vis_L1_left.png")
    plt.close()

    # DO THE SAME FOR L2, L3 check if they cover different areas?
    # Actually, let's plot all grids overlaid to see relative coverage.
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img_np, cmap='gray', origin='upper')
    
    colors = ['r', 'g', 'b']
    labels = ['L1', 'L2', 'L3']
    grids = [agent.grid_L1_left, agent.grid_L2_left, agent.grid_L3_left]
    
    for i, g in enumerate(grids):
        if g is None: continue
        gnp = g.detach().cpu().numpy().squeeze()
        x = (gnp[:, 0] + 1) * (W - 1) / 2
        y = (gnp[:, 1] + 1) * (H - 1) / 2
        plt.scatter(x, y, s=50, c=colors[i], label=labels[i], alpha=0.5)
        
    plt.legend()
    plt.title("L1, L2, L3 Left Coverage Overlay")
    plt.savefig("sample_eye_vis_all_left.png")
    print("Saved sample_eye_vis_all_left.png")

if __name__ == "__main__":
    main()
