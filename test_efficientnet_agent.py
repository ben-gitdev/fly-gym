
import torch
from agents.efficientnet_agent import EfficientNetAgent

print("Starting test...")
try:
    print("Initializing Agent...")
    agent = EfficientNetAgent(action_dim=2, hidden_size=256)
    print("Agent Initialized.")
    
    print("Creating dummy input...")
    # (B, 3, 128, 256) uint8
    x = torch.randint(0, 255, (1, 3, 128, 256), dtype=torch.uint8)
    h = torch.zeros(1, 256)
    
    print("Running forward pass...")
    # step() handles normalization
    h_next, action = agent.step(h, obs=None, x=x)
    print("Forward pass successful.")
    print(f"Action shape: {action.shape}")
    print(f"H shape: {h_next.shape}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("Test Complete.")
