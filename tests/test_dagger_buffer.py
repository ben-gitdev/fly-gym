
import sys
import os
import torch
import numpy as np
import pytest

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_efficientnet_dagger import BalancedDAggerBuffer

def test_balanced_buffer_sampling():
    buffer = BalancedDAggerBuffer(capacity_per_category=10)
    
    # Create dummy data
    img = np.zeros((3, 128, 128), dtype=np.uint8)
    goal = np.zeros(2, dtype=np.float32)
    collision = np.zeros(1, dtype=np.float32)
    xs = {'img': img, 'goal': goal, 'collision': collision}
    action = np.zeros((1, 2), dtype=np.float32)
    
    # 1. Test error on completely empty buffer
    try:
        buffer.sample_balanced_sequences(batch_size=4)
    except ValueError as e:
        assert str(e) == "All buffers are empty!"
        print("Caught expected error for empty buffer.")

    # 2. Add data to ONLY ONE category ('straight')
    buffer.add_chunk('straight', xs, action)
    print("Added 1 chunk to straight.")
    
    # Sample batch > 1
    xs_batch, act_batch = buffer.sample_balanced_sequences(batch_size=4)
    print(f"Sampled batch size: {act_batch.shape[1]}")
    
    assert act_batch.shape[1] == 4, f"Expected batch size 4, got {act_batch.shape[1]}"
    
    # 3. Add data to TWO categories
    buffer.add_chunk('turn', xs, action)
    print("Added 1 chunk to turn.")
    
    xs_batch, act_batch = buffer.sample_balanced_sequences(batch_size=4)
    print(f"Sampled batch size with 2 categories: {act_batch.shape[1]}")
    assert act_batch.shape[1] == 4

    print("Test Validated!")

if __name__ == "__main__":
    test_balanced_buffer_sampling()
