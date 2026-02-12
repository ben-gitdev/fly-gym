from __future__ import annotations
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class LeakyConnectomeRNNCell(nn.Module):
    def __init__(
        self,
        W: torch.Tensor,
        input_nodes: List[int],
        output_nodes: List[int],
        # NEW Arguments
        neuron_type_ids: torch.Tensor, 
        num_cell_types: int,
        # ...
        leak_alpha: float = 0.2, # Used as initialization mean
        activation: str = "tanh",
        train_rnn_weights: bool = True,
        train_readout_head: bool = True,
        dtype: torch.dtype = torch.float32,
        batch_chunk: int = 8,
        row_tile_size: int = 0,
        compile_step_fn: bool = False,  # OPTIMIZATION #1: torch.compile
    ):
        super().__init__()

        # ... (Standard W setup code preserved) ...
        assert W.layout == torch.sparse_coo, "W must be a sparse_coo_tensor"
        Wc = W.coalesce()
        Wcsr = Wc.to_sparse_csr()
        self.N = Wc.shape[0]
        self.Nin = len(input_nodes)
        self.Nout = len(output_nodes)
        self.activation = activation
        
        # --- Standard Register Buffers ---
        self.register_buffer("W_crow_indices", Wcsr.crow_indices())
        self.register_buffer("W_col_indices", Wcsr.col_indices())
        self.register_buffer("W_size", torch.tensor(Wcsr.shape, dtype=torch.int64))
        self.W_values = nn.Parameter(Wcsr.values().to(dtype=dtype), requires_grad=train_rnn_weights)
        self.register_buffer("input_nodes", torch.tensor(input_nodes, dtype=torch.int64))
        self.register_buffer("output_nodes", torch.tensor(output_nodes, dtype=torch.int64))
        
        # --- NEW: Cell Type Alpha Logic ---
        # 1. Store the map (N,) tells us which type each neuron is
        self.register_buffer("neuron_type_ids", neuron_type_ids.to(dtype=torch.long))
        
        # 2. Trainable Parameters for Alphas (One per cell type)
        # Initialize logits so sigmoid(logits) approx equals leak_alpha (e.g. 0.2)
        # x = ln(p / (1-p)) -> ln(0.2/0.8) = -1.386
        import math
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
        
        # --- OPTIMIZATION: Caching for performance ---
        # Cache alpha values to avoid redundant sigmoid+gather per timestep
        self._cached_alphas = None
        self._alphas_dirty = True
        
        # OPTIMIZATION #2: Cache the sparse W matrix
        self._W_cache = None
        self._W_dirty = True
        
        # OPTIMIZATION #1: Optionally compile step_fn with torch.compile
        # Use 'aot_eager' backend for Windows compatibility (Triton only works on Linux)
        if compile_step_fn:
            import sys
            try:
                if sys.platform == "win32":
                    # Windows: use aot_eager (no Triton required)
                    self.step_fn = torch.compile(self.step_fn, backend="aot_eager")
                else:
                    # Linux: use inductor with reduce-overhead for best performance
                    self.step_fn = torch.compile(self.step_fn, mode="reduce-overhead")
            except Exception as e:
                import warnings
                warnings.warn(f"torch.compile failed, falling back to eager mode: {e}")

    def _build_sparse_W(self) -> torch.Tensor:
        """Build the sparse weight matrix. Uses caching for performance.
        """
        if self._W_cache is None or self._W_dirty:
            self._W_cache = torch.sparse_csr_tensor(
                self.W_crow_indices, self.W_col_indices, self.W_values,
                size=tuple(self.W_size.tolist()), check_invariants=False,
            )
            self._W_dirty = False
        return self._W_cache
    
    def invalidate_caches(self):
        """Invalidate all caches. Call after optimizer.step() when training RNN weights."""
        self._W_dirty = True
        self._alphas_dirty = True

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "tanh": return torch.tanh(x)
        elif self.activation == "relu": return F.relu(x)
        return x

    def get_alphas(self) -> torch.Tensor:
        """
        Returns a (N,) tensor of alpha values for every neuron.
        Cached to avoid redundant sigmoid+gather operations per timestep.
        """
        if self._cached_alphas is None or self._alphas_dirty:
            # 1. Sigmoid to enforce (0, 1) range
            # 2. Scale slightly to avoid exactly 0 or 1 (stability)
            # Range: [0.01, 1.0]
            alphas_per_type = torch.sigmoid(self.alpha_logits) * 0.99 + 0.01
            
            # 3. Gather: Map type_idx -> neuron_idx
            # neuron_type_ids is (N,), alphas_per_type is (NumTypes,)
            self._cached_alphas = alphas_per_type[self.neuron_type_ids]
            self._alphas_dirty = False
        return self._cached_alphas

    
    def step_fn(
        self, h: torch.Tensor, x: Optional[torch.Tensor], W: torch.Tensor, 
        bias: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        """Single RNN step. Alpha is passed in to avoid redundant computation."""
        Wh = torch.sparse.mm(W, h.T).T
        pre = Wh + bias
        if x is not None:
            pre = pre.index_add(1, self.input_nodes, x)
        
        # OPTIMIZATION: Use torch.lerp for faster interpolation
        # h_new = (1 - alpha) * h + alpha * phi(pre)
        # lerp(start, end, weight) -> start + weight * (end - start)
        # equivalent to (1 - weight) * start + weight * end
        return torch.lerp(h, self.phi(pre), alpha)

    def run_block(
        self,
        h_init: torch.Tensor,
        xs_chunk: Optional[torch.Tensor],
        W: torch.Tensor,
        bias: torch.Tensor,
        alpha: torch.Tensor,
        store_sequence: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Runs the RNN for a block of steps (chunk).
        Used for block gradient checkpointing.
        """
        h = h_init
        curr_outputs = []
        
        # xs_chunk is (T_chunk, B, Nin) or None
        T_chunk = xs_chunk.size(0) if xs_chunk is not None else 1
        
        for t in range(T_chunk):
            xt = xs_chunk[t] if xs_chunk is not None else None
            h = self.step_fn(h, xt, W, bias, alpha)
            
            if store_sequence:
                dn_act = h.index_select(1, self.output_nodes)
                curr_outputs.append(dn_act)
        
        if store_sequence:
            # Stack outputs for this chunk: (T_chunk, B, Nout)
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
        chunk_size: int = 2, # OPTIMIZATION: Block Checkpointing
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        T = xs.size(0) if xs is not None else 1
        W = self._build_sparse_W()  # Uses cache
        
        # OPTIMIZATION: Compute alpha once, reuse across all timesteps
        alpha = self.get_alphas()  # Uses cache, shape (N,)
        
        if self.training:
            # Mark caches dirty so they rebuild on next forward pass
            self._alphas_dirty = True
            self._W_dirty = True
        
        # Determine if we should checkpoint
        # Checkpointing only useful if we need gradients
        do_checkpoint = checkpoint_steps and (
            self.W_values.requires_grad or self.bias.requires_grad or 
            h0.requires_grad or self.alpha_logits.requires_grad
        )
        
        h = h0
        last_y = torch.zeros(h.size(0), self.Nout, device=h.device, dtype=h.dtype)
        
        # Store outputs as a list of chunks to avoid huge concat if possible
        # or just append to list
        all_seq_outputs = [] if store_sequence else None

        # Main Loop: Iterate by chunks
        # If xs is None (e.g. generative mode), we rely on T being passed or inferred?
        # The original code inferred T=1 if xs is None.
        # But if we want to generate > 1 steps without input, original code didn't support that 
        # (original loop was `for t in range(T)` where T=1). 
        # So we assume T is correct.
        
        # NOTE: If T < chunk_size, it will just be one chunk.
        
        # Pre-calculate number of chunks
        num_chunks = (T + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            start_t = i * chunk_size
            end_t = min((i + 1) * chunk_size, T)
            
            # Slice input for this chunk
            if xs is not None:
                xs_slice = xs[start_t:end_t]
            else:
                # If T > 1 but xs is None, we need to pass None or handle logic.
                # However, typically xs is None implies T=1 in this codebase context.
                # Just in case T > 1 and xs is None (autoregressive?), we pass None.
                # But slice logic on None fails.
                # If xs is None, we just pass None to run_block, and it handles loop range.
                xs_slice = None
                # If T > 1 but xs is None, we need a way to tell run_block how many steps.
                # In current codebase T=1 if xs is None.
                # If we expanded this later, we'd need to change run_block signature to take T_chunk explicitly.
                # For now, xs_slice being None implies T_chunk=1 because of how T is calc line 141.
                # Actually, check line 141: T = xs.size(0) if xs is not None else 1.
                # So if xs is None, T=1, chunk_size>=1, so num_chunks=1.
                pass

            if do_checkpoint:
                # Checkpoint the block
                # NOTE: run_block returns (h, chunk_out)
                h, chunk_out = checkpoint(
                    self.run_block,
                    h, xs_slice, W, self.bias, alpha, store_sequence,
                    use_reentrant=False
                )
            else:
                h, chunk_out = self.run_block(
                    h, xs_slice, W, self.bias, alpha, store_sequence
                )
            
            if store_sequence and chunk_out is not None:
                all_seq_outputs.append(chunk_out)

        # Final cleanup / formatting
        # Last outputs
        dn_act = h.index_select(1, self.output_nodes)
        last_y = self.readout_head(dn_act) if self.readout_head else dn_act
            
        if store_sequence: 
            # Concatenate all chunks: List[(T1, B, Nout), (T2, B, Nout)...]
            # Result: (T, B, Nout)
            DN_seq = torch.cat(all_seq_outputs, dim=0)
            
            # Compute Y_seq from DN_seq efficiently in batch
            if self.readout_head:
                Y_seq = self.readout_head(DN_seq)
            else:
                Y_seq = DN_seq
            
            return h, Y_seq, DN_seq
        else: 
            return h, last_y
