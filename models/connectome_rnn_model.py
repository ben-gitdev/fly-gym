from __future__ import annotations
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# -----------------------------------------------------------------------------
# CUSTOM AUTOGRAD FUNCTION (Hybrid CSR/COO)
# -----------------------------------------------------------------------------
class MemoryEfficientSparseMM(torch.autograd.Function):
    """
    Performs Sparse x Dense matrix multiplication using a Hybrid strategy:
    - Forward: Uses CSR format for maximum speed (cuSPARSE optimized).
    - Backward: Uses COO indices for O(NNZ) gradient calculation (memory efficient).
    """
    @staticmethod
    def forward(ctx, values, crow_indices, col_indices, coo_rows, shape, x):
        # x shape: (Batch, N_in)
        # values shape: (NNZ,)
        
        # Save tensors needed for backward
        # We need coo_rows (uncompressed) for the gather operation in backward
        # We reuse col_indices (which are the same for CSR and sorted COO)
        ctx.save_for_backward(values, coo_rows, col_indices, x)
        ctx.shape = shape
        
        # Construct CSR tensor for Fast Forward Pass
        # We use the CSR indices (crow_indices) for speed here
        w_sparse = torch.sparse_csr_tensor(
            crow_indices, col_indices, values, size=shape, check_invariants=False
        )
        
        # Operation: Wh = (W @ h.T).T
        return torch.sparse.mm(w_sparse, x.t()).t()

    @staticmethod
    def backward(ctx, grad_output):
        # print("DEBUG: MemoryEfficientSparseMM.backward called!")
        values, coo_rows, col_indices, x = ctx.saved_tensors
        rows, cols = ctx.shape
        
        # 1. Gradient w.r.t Input (dL/dX)
        # dX = (dL/dY) @ W
        # We reconstruct W using COO indices for the transpose operation
        # (It's often cheap to treat the transpose of CSR as CSC, or just use COO)
        w_sparse = torch.sparse_coo_tensor(
            torch.stack([coo_rows, col_indices]), values, (rows, cols),
            is_coalesced=True
        )
        
        # dX = (W.t() @ grad_output.t()).t()
        grad_input = torch.sparse.mm(w_sparse.t(), grad_output.t()).t()
        
        # 2. Gradient w.r.t Weight Values (dL/dW_values)
        # We use the explicit coo_rows to "gather" the gradients directly.
        
        # Gather specific rows from grad_output and columns from input x
        # coo_rows and col_indices are shape (NNZ,)
        grad_out_gathered = grad_output[:, coo_rows] 
        x_gathered = x[:, col_indices]
        
        # Element-wise multiply and sum over batch dimension
        # Result: (NNZ,) - Same size as 'values'
        grad_values = torch.sum(grad_out_gathered * x_gathered, dim=0)
        
        # Return gradients matching forward signature:
        # (values, crow_indices, col_indices, coo_rows, shape, x)
        return grad_values, None, None, None, None, grad_input

# -----------------------------------------------------------------------------
# MODEL CLASS
# -----------------------------------------------------------------------------

class LeakyConnectomeRNNCell(nn.Module):
    def __init__(
        self,
        W: torch.Tensor,
        input_nodes: List[int],
        output_nodes: List[int],
        neuron_type_ids: torch.Tensor, 
        num_cell_types: int,
        leak_alpha: float = 0.2, 
        activation: str = "tanh",
        train_rnn_weights: bool = True,
        train_readout_head: bool = True,
        dtype: torch.dtype = torch.float32,
        batch_chunk: int = 8,
        row_tile_size: int = 0,
        compile_step_fn: bool = False,
    ):
        super().__init__()

        # --- Sparse Matrix Setup ---
        assert W.layout == torch.sparse_coo, "W must be a sparse_coo_tensor"
        Wc = W.coalesce()
        Wcsr = Wc.to_sparse_csr()
        self.N = Wc.shape[0]
        self.Nin = len(input_nodes)
        self.Nout = len(output_nodes)
        self.activation = activation
        
        # CSR Indices (For Fast Forward Pass)
        self.register_buffer("W_crow_indices", Wcsr.crow_indices())
        self.register_buffer("W_col_indices", Wcsr.col_indices())
        self.register_buffer("W_size", torch.tensor(Wcsr.shape, dtype=torch.int64))
        
        # COO Indices (For Fast Backward Pass)
        # We need the uncompressed row indices.
        # CRITICAL: Derive coo_rows from CSR crow_indices to ensure ordering matches CSR values
        # This "uncompresses" the CSR row pointers back to explicit row indices
        row_counts = Wcsr.crow_indices()[1:] - Wcsr.crow_indices()[:-1]
        coo_rows = torch.repeat_interleave(torch.arange(self.N, device=W.device), row_counts)
        self.register_buffer("W_coo_rows", coo_rows)
        
        # Values from CSR (matches CSR col_indices ordering)
        self.W_values = nn.Parameter(Wcsr.values().to(dtype=dtype), requires_grad=train_rnn_weights)
        
        self.register_buffer("input_nodes", torch.tensor(input_nodes, dtype=torch.int64))
        self.register_buffer("output_nodes", torch.tensor(output_nodes, dtype=torch.int64))
        
        # --- Cell Type Alpha Logic ---
        self.register_buffer("neuron_type_ids", neuron_type_ids.to(dtype=torch.long))
        
        import math
        # Prevent log(0)
        leak_alpha = max(min(leak_alpha, 0.99), 0.01)
        init_logit = math.log(leak_alpha / (1.0 - leak_alpha))
        self.alpha_logits = nn.Parameter(
            torch.full((num_cell_types,), init_logit, dtype=dtype)
        )
        
        # --- Readout Head ---
        head = nn.Sequential(
            nn.Linear(len(output_nodes), 128, dtype=dtype),
            nn.ReLU(),
            nn.Linear(128, 64, dtype=dtype),
            nn.ReLU(),
            nn.Linear(64, 2, dtype=dtype),
        )
        if not train_readout_head:
            for p in head.parameters(): p.requires_grad = False
        self.readout_head = head

        self.bias = nn.Parameter(torch.zeros(self.N, dtype=dtype))
        
        # --- Caching ---
        self._cached_alphas = None
        self._alphas_dirty = True
        
        # --- Compilation ---
        if compile_step_fn:
            import sys
            try:
                if sys.platform == "win32":
                    self.step_fn = torch.compile(self.step_fn, backend="aot_eager")
                else:
                    self.step_fn = torch.compile(self.step_fn, mode="reduce-overhead")
            except Exception as e:
                import warnings
                warnings.warn(f"torch.compile failed, falling back to eager mode: {e}")

    def invalidate_caches(self):
        """Invalidate all caches."""
        self._alphas_dirty = True

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "tanh": return torch.tanh(x)
        elif self.activation == "relu": return F.relu(x)
        return x

    def get_alphas(self) -> torch.Tensor:
        if self._cached_alphas is None or self._alphas_dirty:
            alphas_per_type = torch.sigmoid(self.alpha_logits) * 0.99 + 0.01
            self._cached_alphas = alphas_per_type[self.neuron_type_ids]
            self._alphas_dirty = False
        return self._cached_alphas

    def step_fn(
        self, h: torch.Tensor, x: Optional[torch.Tensor], 
        W_values: torch.Tensor, bias: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        """
        Single RNN step using Memory Efficient Sparse MM.
        """
        # CUSTOM AUTOGRAD CALL
        # Passes CSR indices for forward, COO rows for backward
        Wh = MemoryEfficientSparseMM.apply(
            W_values, 
            self.W_crow_indices,
            self.W_col_indices,
            self.W_coo_rows,
            tuple(self.W_size.tolist()),
            h
        )
        
        pre = Wh + bias
        if x is not None:
            pre = pre.index_add(1, self.input_nodes, x)
        
        # h_new = (1 - alpha) * h + alpha * phi(pre)
        return torch.lerp(h, self.phi(pre), alpha)

    def run_block(
        self,
        h_init: torch.Tensor,
        xs_chunk: Optional[torch.Tensor],
        W_values: torch.Tensor,
        bias: torch.Tensor,
        alpha: torch.Tensor,
        store_sequence: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        h = h_init
        curr_outputs = []
        T_chunk = xs_chunk.size(0) if xs_chunk is not None else 1
        
        for t in range(T_chunk):
            xt = xs_chunk[t] if xs_chunk is not None else None
            h = self.step_fn(h, xt, W_values, bias, alpha)
            
            if store_sequence:
                dn_act = h.index_select(1, self.output_nodes)
                curr_outputs.append(dn_act)
        
        if store_sequence:
            chunk_stack = torch.stack(curr_outputs, dim=0)
            return h, chunk_stack
        else:
            return h, None

    def forward(
        self,
        h0: torch.Tensor,
        xs: Optional[torch.Tensor],
        checkpoint_steps: bool = False,
        store_sequence: bool = False,
        chunk_size: int = 100, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        T = xs.size(0) if xs is not None else 1
        
        # Optimization: Compute alpha once
        alpha = self.get_alphas()
        
        if self.training:
            self._alphas_dirty = True
        
        do_checkpoint = checkpoint_steps and (
            self.W_values.requires_grad or self.bias.requires_grad or 
            h0.requires_grad or self.alpha_logits.requires_grad
        )
        
        h = h0
        last_y = None
        all_seq_outputs = [] if store_sequence else None

        num_chunks = (T + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            start_t = i * chunk_size
            end_t = min((i + 1) * chunk_size, T)
            
            if xs is not None:
                xs_slice = xs[start_t:end_t]
            else:
                xs_slice = None 

            if do_checkpoint:
                h, chunk_out = checkpoint(
                    self.run_block,
                    h, xs_slice, self.W_values, self.bias, alpha, store_sequence,
                    use_reentrant=False
                )
            else:
                h, chunk_out = self.run_block(
                    h, xs_slice, self.W_values, self.bias, alpha, store_sequence
                )
            
            if store_sequence and chunk_out is not None:
                all_seq_outputs.append(chunk_out)

        dn_act = h.index_select(1, self.output_nodes)
        last_y = self.readout_head(dn_act) if self.readout_head else dn_act
            
        if store_sequence: 
            DN_seq = torch.cat(all_seq_outputs, dim=0)
            if self.readout_head:
                Y_seq = self.readout_head(DN_seq)
            else:
                Y_seq = DN_seq
            return h, Y_seq, DN_seq
        else: 
            return h, last_y