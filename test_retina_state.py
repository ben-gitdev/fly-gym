
import torch
import torch.nn as nn
from agents.connectome_rnn_agent import ConnectomeAgent
import numpy as np

# Mock Cell
class MockCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.Nin = 100 # Arbitrary
        self.N = 10
        self.Nout = 2
        self.W_values = torch.zeros(1)
        self.readout_head = nn.Linear(10, 2)
        
    def forward(self, h, x, checkpoint_steps=False, store_sequence=False):
        return h, torch.zeros(x.size(1), 2)

def test_retina_state():
    # Setup
    cell = MockCell()
    pr_pos = [[0.0, 0.0]] * 10
    # Create fake inputs
    # We need specific names for splits to trigger retina paths
    input_splits = {
        "pr_L1_left": (0, 1), "pr_L2_left": (1, 2), "pr_L3_left": (2, 3),
        "pr_L1_right": (3, 4), "pr_L2_right": (4, 5), "pr_L3_right": (5, 6),
        "olf_left": (6, 7), "olf_right": (7, 8),
        "tactile_left": (8, 9), "tactile_right": (9, 10),
        "wind": (10, 12)
    }
    
    agent = ConnectomeAgent(
        cell, 
        photoreceptor_positions=pr_pos, 
        input_splits=input_splits
    )
    
    # Mock Obs
    # Create a random image (1, H, W)
    H, W = 32, 32
    obs = {
        "cam_left": np.random.rand(H, W, 3).astype(np.float32),
        "cam_right": np.random.rand(H, W, 3).astype(np.float32),
        "sensors": {
            "vec_to_goal": torch.tensor([[1.0, 0.0]]),
            "vec_left_to_goal": torch.tensor([[1.0, 0.0]]),
            "vec_right_to_goal": torch.tensor([[1.0, 0.0]]),
            "collision": torch.tensor([[0.0]]),
            "collision_angle": torch.tensor([[0.0]]),
            "wind_direction": torch.tensor([[0.0, 1.0]])
        }
    }
    
    print("Initial State Left:", agent.state_vision_left)
    assert agent.state_vision_left is None
    
    # 1. First Step (update_state=True)
    print("Step 1: Normal Update")
    x1 = agent.obs_to_x(obs, update_state=True)
    
    assert agent.state_vision_left is not None
    assert "all" in agent.state_vision_left
    state_v1 = agent.state_vision_left["all"]
    # Provide deep check - clone it
    saved_state_v1 = [t.clone() for t in state_v1]
    
    print("State initialized. L1 norm:", state_v1[0].norm().item())

    # 2. Second Step (update_state=False) - Simulate lookahead
    # Change input slightly to ensure output would be different if processed
    obs["cam_left"] = np.random.rand(H, W, 3).astype(np.float32)
    print("Step 2: Lookahead (update_state=False)")
    x2 = agent.obs_to_x(obs, update_state=False)
    
    # State should be UNCHANGED
    state_v2 = agent.state_vision_left["all"]
    for i in range(3):
        diff = (state_v2[i] - saved_state_v1[i]).abs().sum().item()
        if diff != 0:
            print(f"FAILURE: State {i} changed! Diff: {diff}")
        else:
            print(f"Success: State {i} unchanged.")
            
    # 3. Third Step (Normal Update)
    print("Step 3: Normal Update")
    x3 = agent.obs_to_x(obs, update_state=True)
    
    state_v3 = agent.state_vision_left["all"]
    # Now it SHOULD change
    diff = 0
    for i in range(3):
        diff += (state_v3[i] - saved_state_v1[i]).abs().sum().item()
    
    if diff > 0:
        print(f"Success: State updated. Diff: {diff}")
    else:
        print("Warning: State did not change (input might be too similar or logic static?)")
        
    # 4. Reset
    print("Step 4: Reset")
    agent.reset_vision_state()
    assert agent.state_vision_left is None
    print("Success: State reset.")

if __name__ == "__main__":
    test_retina_state()
