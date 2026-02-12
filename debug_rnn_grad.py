
print("Start script")
import torch
import torch.nn as nn
import sys
import os

# Add current dir to path just in case
sys.path.append(os.getcwd())

try:
    from models.connectome_rnn_model import MemoryEfficientSparseMM, LeakyConnectomeRNNCell
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

import torch.nn.functional as F

print("Imports done")

def test_sparse_grad():
    print("Testing Sparse MM Gradients...")
    
    N = 10
    Nin = 5
    Nout = 2
    
    # Random sparse W
    W = torch.randn(N, N).to_sparse_csr()
    row_indices = W.crow_indices()
    col_indices = W.col_indices()
    values = W.values().clone().detach().requires_grad_(True)
    
    # Use the custom function directly
    shape = (N, N)
    
    # Fake coo rows
    row_counts = row_indices[1:] - row_indices[:-1]
    coo_rows = torch.repeat_interleave(torch.arange(N), row_counts)
    
    h = torch.randn(2, N, requires_grad=True) # Batch 2
    
    # Forward
    print("Calling apply...")
    Wh = MemoryEfficientSparseMM.apply(values, row_indices, col_indices, coo_rows, tuple(shape), h)
    print("Forward done")
    
    loss = Wh.sum()
    print("Backward...")
    loss.backward()
    
    if values.grad is None:
        print("FAIL: values.grad is None")
    else:
        print(f"PASS: values.grad norm = {values.grad.norm().item()}")
        
def test_agent_loop():
    print("\nTesting Agent Loop Simulation...")
    
    # Mock parameters
    N = 100
    Nin = 10
    Nout = 2
    
    # Create fake sparse matrix
    W_dense = torch.randn(N, N)
    mask = torch.rand(N, N) > 0.8
    W_sparse = (W_dense * mask).to_sparse_coo()
    
    input_nodes = list(range(Nin))
    output_nodes = list(range(Nout))
    
    print("Building Cell...")
    cell = LeakyConnectomeRNNCell(
        W=W_sparse,
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        neuron_type_ids=torch.zeros(N),
        num_cell_types=1,
        train_rnn_weights=True
    )
    
    opt = torch.optim.Adam(cell.parameters(), lr=1e-3)
    
    for i in range(5):
        print(f"Iter {i}")
        xs = torch.randn(20, 4, Nin) # T=20, B=4, Nin
        h0 = torch.zeros(4, N)
        
        # Forward
        opt.zero_grad()
        h_final, y_seq = cell(h0, xs, store_sequence=False)
        
        loss = h_final.sum()
        loss.backward()
        
        if cell.W_values.grad is None:
            print("  WARNING: W_values.grad is None!")
        else:
            print(f"  Grad Norm: {cell.W_values.grad.norm().item()}")
            
        opt.step()

if __name__ == "__main__":
    test_sparse_grad()
    test_agent_loop()
