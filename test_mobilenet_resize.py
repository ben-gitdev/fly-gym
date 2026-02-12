
import torch
from agents.mobilenet_agent import MobileNetAgent
import numpy as np

def test_resize():
    agent = MobileNetAgent()
    device = agent.device
    
    # Create fake obs
    B, H, W = 2, 128, 128
    # Test CHW input
    cam_left = torch.randint(0, 255, (B, 3, H, W), dtype=torch.uint8).to(device)
    cam_right = torch.randint(0, 255, (B, 3, H, W), dtype=torch.uint8).to(device)
    goal = torch.randn(B, 2).to(device)
    collision = torch.zeros(B, 1).to(device)
    
    obs = {
        "cam_left": cam_left,
        "cam_right": cam_right,
        "sensors": {
            "vec_to_goal": goal,
            "collision": collision
        }
    }
    
    print("\n--- Testing obs_to_x (CHW) ---")
    x = agent.obs_to_x(obs)
    print("img shape:", x['img'].shape)
    print("img dtype:", x['img'].dtype)
    
    assert x['img'].shape == (B, 3, 30, 60), f"Expected (B, 3, 30, 60), got {x['img'].shape}"
    assert x['img'].dtype == torch.uint8, "Expected uint8"
    
    print("\n--- Testing forward ---")
    # Forward expects dict as is (it handles normalization)
    # The output of obs_to_x is usually passed to forward during training step (via buffer sampling)
    # But during rollout step, obs_to_x result is used directly.
    
    # Let's test forward pass
    h = torch.zeros(B, agent.hidden_size).to(device)
    
    # Convert uint8 back to float inside forward? Checks logic inside forward.
    # Forward handles uint8 normalization.
    action, h_next = agent(x, h)
    
    print("Action shape:", action.shape)
    print("Hidden shape:", h_next.shape)
    print("Forward pass successful!")

    print("\n--- Testing obs_to_x (HWC) ---")
    # Test HWC
    cam_left_hwc = cam_left.permute(0, 2, 3, 1) # (B, H, W, 3)
    cam_right_hwc = cam_right.permute(0, 2, 3, 1)
    obs_hwc = {
        "cam_left": cam_left_hwc,
        "cam_right": cam_right_hwc,
        "sensors": {
            "vec_to_goal": goal,
            "collision": collision
        }
    }
    
    x_hwc = agent.obs_to_x(obs_hwc)
    print("img shape (HWC input):", x_hwc['img'].shape)
    assert x_hwc['img'].shape == (B, 3, 30, 60), f"Expected (B, 3, 30, 60) for HWC input, got {x_hwc['img'].shape}"

if __name__ == "__main__":
    test_resize()
