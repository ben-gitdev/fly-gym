"""Quick shape test for DualBackboneAgent."""
import sys
import torch
print("Importing DualBackboneAgent...", flush=True)
from agents.dual_backbone_agent import DualBackboneAgent

B = 2

try:
    print("Creating EfficientNet dual agent...", flush=True)
    agent = DualBackboneAgent(backbone_type="efficientnet", action_dim=2, hidden_size=256)
    x = {
        "img_left": torch.randint(0, 255, (B, 1, 30, 30), dtype=torch.uint8),
        "img_right": torch.randint(0, 255, (B, 1, 30, 30), dtype=torch.uint8),
        "wind_direction": torch.randn(B, 2),
        "collision": torch.randn(B, 1),
    }
    h = torch.zeros(B, 256)
    print("Running forward...", flush=True)
    action, h_new = agent.forward(x, h)
    print(f"  action={action.shape}, h={h_new.shape}", flush=True)
    assert action.shape == (B, 2)
    assert h_new.shape == (B, 256)

    print("Running forward_sequence...", flush=True)
    T = 5
    xs = {
        "img_left": torch.randint(0, 255, (T, B, 1, 30, 30), dtype=torch.uint8),
        "img_right": torch.randint(0, 255, (T, B, 1, 30, 30), dtype=torch.uint8),
        "wind_direction": torch.randn(T, B, 2),
        "collision": torch.randn(T, B, 1),
    }
    h_out, y_seq, _ = agent.forward_sequence(xs)
    print(f"  y_seq={y_seq.shape}, h_out={h_out.shape}", flush=True)
    assert y_seq.shape == (T, B, 2)
    assert h_out.shape == (B, 256)

    print("All tests PASSED!", flush=True)
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
