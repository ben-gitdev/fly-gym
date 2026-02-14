import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Any, Dict, Optional, Sequence, Tuple
# from environment.mujoco_two_cam_env_random_obstacles import MuJoCoTwoCamEnv


class ConnectomeAgent(nn.Module):
    """
    Wrap the connectome RNN cell and expose a policy/value head that
    operates on photoreceptor activations instead of learned image encoders.
    """

    def __init__(
        self,
        cell,
        photoreceptor_positions: Sequence[Sequence[float]],
        input_splits: Dict[str, Tuple[int, int]],
        dtype: torch.dtype = torch.float32,
        input_scale_init: float = 1.0,
        use_value_head: bool = False,
        learn_policy_std: bool = False,
        policy_std_init: float = 0.3,
        # env: MuJoCoTwoCamEnv = None
    ):
        super().__init__()
        # self.env = env
        self.cell = cell
        self.input_splits = input_splits  # name -> (start, end) indices into flat input vector
        self.action_dim = cell.readout_head[-1].out_features if hasattr(cell, "readout_head") else 2
        
        # Per-type input scaling
        # Initialize all to the same value first
        init_val = float(input_scale_init)
        self.input_scale_vision_L1 = nn.Parameter(torch.tensor(init_val, dtype=dtype))
        self.input_scale_vision_L2 = nn.Parameter(torch.tensor(init_val, dtype=dtype))
        self.input_scale_vision_L3 = nn.Parameter(torch.tensor(init_val, dtype=dtype))
        self.input_scale_olfactory = nn.Parameter(torch.tensor(init_val, dtype=dtype))
        self.input_scale_tactile = nn.Parameter(torch.tensor(init_val, dtype=dtype))
        self.input_scale_wind = nn.Parameter(torch.tensor(init_val, dtype=dtype))
        self.input_scale_other = nn.Parameter(torch.tensor(init_val, dtype=dtype))

        pr_L1_left_start, pr_L1_left_end = self.input_splits["pr_L1_left"]
        pr_L2_left_start, pr_L2_left_end = self.input_splits["pr_L2_left"]
        pr_L3_left_start, pr_L3_left_end = self.input_splits["pr_L3_left"]
        pr_L1_right_start, pr_L1_right_end = self.input_splits["pr_L1_right"]
        pr_L2_right_start, pr_L2_right_end = self.input_splits["pr_L2_right"]
        pr_L3_right_start, pr_L3_right_end = self.input_splits["pr_L3_right"]
        self.register_buffer(
            "grid_L1_left", self._make_grid(photoreceptor_positions[pr_L1_left_start:pr_L1_left_end])
        )
        self.register_buffer(
            "grid_L2_left", self._make_grid(photoreceptor_positions[pr_L2_left_start:pr_L2_left_end])
        )
        self.register_buffer(
            "grid_L3_left", self._make_grid(photoreceptor_positions[pr_L3_left_start:pr_L3_left_end])
        )
        self.register_buffer(
            "grid_L1_right", self._make_grid(photoreceptor_positions[pr_L1_right_start:pr_L1_right_end])
        )
        self.register_buffer(
            "grid_L2_right", self._make_grid(photoreceptor_positions[pr_L2_right_start:pr_L2_right_end])
        )
        self.register_buffer(
            "grid_L3_right", self._make_grid(photoreceptor_positions[pr_L3_right_start:pr_L3_right_end])
        )
        self.register_buffer(
            "gray_weights",
            torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32)
            .view(1, 3, 1, 1),
            persistent=False,
        )

        # Optional RL heads
        self.value_head: Optional[nn.Module] = None
        self.priv_obs_dim = 0
        if use_value_head:
            # If privileged obs dim > 0, we concat it to the RNN hidden state
            # If passed as kwarg (we can check kwarg in init but let's assume standard use)
            pass 

        if learn_policy_std:
            init_log_std = math.log(max(policy_std_init, 1e-6))
            self.policy_log_std = nn.Parameter(
                torch.full((self.action_dim,), float(init_log_std), dtype=dtype)
            )
        else:
            self.register_parameter("policy_log_std", None)

        # NEW: Wind MLP
        # Check if "wind" is in input_splits and has length > 0
        if "wind" in self.input_splits and self._segment_length("wind") > 0:
            n_wind_neurons = self._segment_length("wind")
            self.wind_mlp = nn.Sequential(
                nn.Linear(2, 128, dtype=dtype),
                nn.ReLU(),
                nn.Linear(128, n_wind_neurons, dtype=dtype),
            )
        else:
            self.wind_mlp = None

        # Vision State
        self.state_vision_left: Optional[Dict[str, torch.Tensor]] = None
        self.state_vision_right: Optional[Dict[str, torch.Tensor]] = None

        # 1. VIRTUAL R-CELLS (Parameters)
        # Learnable scale/bias for the log transform (handling camera dynamic range)
        self.r_scale = nn.Parameter(torch.tensor(1.0))
        self.r_bias = nn.Parameter(torch.tensor(0.01)) # Avoid log(0)

        # 2. VIRTUAL SYNAPSES (The Temporal Filter)
        # We use a simplified RNN cell logic for L1/L2 (High Pass)
        # 'alpha' is the decay rate (1/tau). We make it learnable.
        # Initializing near 0.5 mimics a balanced biological filter.
        self.l1_alpha = nn.Parameter(torch.tensor(0.5))
        self.l2_alpha = nn.Parameter(torch.tensor(0.5))
        
        # L3 is Low Pass (Sustained), so it needs a different alpha (slower decay)
        self.l3_alpha = nn.Parameter(torch.tensor(0.1))

    def init_value_head(self, priv_obs_dim=0):
        # ... (rest of method unchanged) ...
        # Helper to init value head after construction if needed, or re-init
        self.priv_obs_dim = priv_obs_dim
        input_dim = self.cell.Nout + priv_obs_dim
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, 128, dtype=self.input_scale_vision_L1.dtype),
            nn.ReLU(),
            nn.Linear(128, 1, dtype=self.input_scale_vision_L1.dtype),
        )

    def set_value_head_config(self, use_value_head: bool, priv_obs_dim: int = 0):
         if not use_value_head:
             self.value_head = None
             return
         self.init_value_head(priv_obs_dim)

    def reset_vision_state(self):
        """Reset internal vision states (e.g. for motion detection) at episode start."""
        self.state_vision_left = None
        self.state_vision_right = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _make_grid(self, positions: Sequence[Sequence[float]]) -> Optional[torch.Tensor]:
        """
        Build a sampling grid for F.grid_sample from photoreceptor positions.

        Parameters
        ----------
        photoreceptor_positions : list[(x, y)]
            Normalized coordinates in [-1, 1].

        Returns
        -------
        grid : torch.Tensor
            Shape (1, N_pr, 1, 2) in normalized [-1, 1] coordinates (x, y),
            suitable for F.grid_sample with align_corners=True.
        """
        if not positions:
            return None

        pos = torch.as_tensor(positions, device=torch.device("cpu"), dtype=torch.float32)
        assert pos.dim() == 2 and pos.size(1) == 2, "positions must be (N_pr, 2)"

        # Input positions are already normalized to [-1, 1]
        grid = pos.clone()
        grid[:, 1] = -grid[:, 1]         # Flip Y from Cartesian (up) to Image (down)
        grid = grid.view(1, -1, 1, 2)    # (1, N_pr, 1, 2)

        return grid

    def _ensure_grid_device(
        self, grid: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        if grid is None:
            return None
        if grid.device != device or grid.dtype != dtype:
            return grid.to(device=device, dtype=dtype)
        return grid

    def _warn_if_nan(self, tensor: torch.Tensor, context: str) -> None:
        if torch.isnan(tensor).any():
            print(f"Warning: NaN encountered in {context}; applying torch.nan_to_num().")

    @property
    def input_scale(self) -> torch.Tensor:
        """
        Dynamic property that builds the full input_scale vector (Nin,)
        from the per-type scalar parameters.
        
        Uses gradient-safe operations (torch.where) instead of in-place indexing.
        """
        Nin = self.cell.Nin
        dtype = self.input_scale_vision_L1.dtype
        device = self.device
        
        # Start with base scale (input_scale_other)
        scale_vec = self.input_scale_other.expand(Nin)
        
        # Create index tensor for masking
        indices = torch.arange(Nin, device=device)
        
        # Helper to apply scale using torch.where (gradient-safe)
        def apply_scale(current_vec, name, val):
            if name in self.input_splits:
                start, end = self.input_splits[name]
                if end > start:
                    mask = (indices >= start) & (indices < end)
                    return torch.where(mask, val.expand(Nin), current_vec)
            return current_vec

        scale_vec = apply_scale(scale_vec, "pr_L1_left", self.input_scale_vision_L1)
        scale_vec = apply_scale(scale_vec, "pr_L2_left", self.input_scale_vision_L2)
        scale_vec = apply_scale(scale_vec, "pr_L3_left", self.input_scale_vision_L3)
        scale_vec = apply_scale(scale_vec, "pr_L1_right", self.input_scale_vision_L1)
        scale_vec = apply_scale(scale_vec, "pr_L2_right", self.input_scale_vision_L2)
        scale_vec = apply_scale(scale_vec, "pr_L3_right", self.input_scale_vision_L3)
        scale_vec = apply_scale(scale_vec, "olf_left", self.input_scale_olfactory)
        scale_vec = apply_scale(scale_vec, "olf_right", self.input_scale_olfactory)
        scale_vec = apply_scale(scale_vec, "tactile_left", self.input_scale_tactile)
        scale_vec = apply_scale(scale_vec, "tactile_right", self.input_scale_tactile)
        scale_vec = apply_scale(scale_vec, "wind", self.input_scale_wind)
        
        return scale_vec

    def _segment_length(self, name: str) -> int:
        try:
            start, end = self.input_splits[name]
        except KeyError as exc:
            # For backward compatibility, return 0 if not found
            return 0
        return max(end - start, 0)

    def _to_grayscale(self, img: torch.Tensor) -> torch.Tensor:
        # Ensure grayscale intensities are normalized to [0, 1] (positive)
        if img.size(1) == 1:
            return img / 255.0 
        weights = self.gray_weights.to(device=img.device, dtype=img.dtype)
        gray = (img * weights).sum(dim=1, keepdim=True)
        return gray / 255.0 # Positive intensity for R-cell model

    def _sample_eye(
        self, img: torch.Tensor, grid: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if grid is None:
            return torch.zeros(img.size(0), 0, device=img.device, dtype=img.dtype)
        
        # Apply smoothing (local averaging) to prevent aliasing before sampling
        # We assume img is (B, 1, H, W)
        img = F.avg_pool2d(img, kernel_size=5, stride=1, padding=2, count_include_pad=False)

        aligned_grid = self._ensure_grid_device(grid, img.device, img.dtype)
        assert aligned_grid is not None
        # grid: (1, N, 1, 2) -> (B, N, 1, 2)
        grid_batch = aligned_grid.expand(img.size(0), -1, -1, -1).contiguous()
        sampled = F.grid_sample(
            img,
            grid_batch,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        # (B, 1, N, 1) -> (B, N)
        return sampled.squeeze(1).squeeze(-1)




    def _slice_input(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """Helper to slice x along the last dimension using input_splits."""
        if name not in self.input_splits:
            # Return empty tensor for missing keys (consistent with _segment_length)
            return x[..., :0]
        start, end = self.input_splits[name]
        return x[..., start:end]

    def _virtual_retina(
        self, x_L: torch.Tensor, state_prev=None, update_state=True, type = "L1"
    ) -> torch.Tensor:
        """
        Process a single visual layer (L1, L2, or L3) using the fly retina model.
        """
        # --- STAGE 0: PHOTORECEPTOR LOG TRANSFORM ---
        # Logarithmic encoding of light intensity (Weber-Fechner Law)
        # Use softplus to ensure bias is always positive, preventing log(<=0) -> NaN
        # x_L coming in is normalized [0, ~1] from _sample_eye -> intensity
        safe_bias = F.softplus(self.r_bias)  # Always positive, smooth gradient
        x_L = self.r_scale * torch.log(x_L + safe_bias)

        # --- STAGE 3: L-CELL TEMPORAL FILTERING ---
        if state_prev is None:
            # Initialize states with current input (assumption: start at steady state)
            L_hidden = x_L.detach()
        else:
            L_hidden = state_prev

        if not update_state:
             # Just return processed features based on EXISTING state, 
             # effectively peeking at what the output WOULD be without advancing time.
             # Wait, if we don't update state, do we use the OLD state? Yes.
             # So we calculate output using (current_input, old_state).
             pass

        # L1 & L2 Computation (Transient / High Pass)
        # Signal = Input - Smoothed_History
        # We constrain alpha to 0-1 using sigmoid to keep stability
        if type == "L1":
            alpha_l = torch.sigmoid(self.l1_alpha)
        elif type == "L2":
            alpha_l = torch.sigmoid(self.l2_alpha)
        elif type == "L3":
            alpha_l = torch.sigmoid(self.l3_alpha)
        
        if  type == "L1" or type == "L2":
            # Update hidden state (Low pass filter of history)
            L_hidden_new = (1 - alpha_l) * L_hidden + alpha_l * x_L
        
            # Output is the difference (The "Change")
            # In biology, the synapse inverts sign. Here we just take diff.
            # This matches the L1/L2 "Edge Detector" role.
            L_out = x_L - L_hidden_new

        elif type == "L3":
            # L3 Computation (Sustained / Low Pass)
            # L3 is basically just the smoothed history itself (tonic)
            # alpha_l was already computed above at line 296
            L_hidden_new = (1 - alpha_l) * L_hidden + alpha_l * x_L
            L_out = L_hidden_new

        # Pack states for next timestep
        # NOTE: We do NOT detach here to allow gradient flow for BPTT.
        # This enables training of l1_alpha, l2_alpha, l3_alpha through temporal dynamics.
        # Gradient explosion is mitigated by the sigmoid constraint on alpha (keeps in [0,1]).
        if update_state:
            next_state = L_hidden_new
        else:
            next_state = state_prev
        
        return L_out, next_state
    
    def _sample_olfactory(
        self, sensors: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        n_olf_left = self._segment_length("olf_left")
        n_olf_right = self._segment_length("olf_right")
        base = sensors.get("vec_to_goal")
        if base is None:
            raise KeyError("sensors missing required key 'vec_to_goal'")
        batch = base.size(0)
        device = base.device
        dtype = base.dtype

        if n_olf_left == 0 and n_olf_right == 0:
            # print("Warning: No olfactory receptors found in input_splits, no olfactory input to network.")
            return torch.zeros(batch, 0, device=device, dtype=dtype)

        vec_left = sensors.get("vec_left_to_goal", base)
        vec_right = sensors.get("vec_right_to_goal", base)

        dist_left = torch.linalg.norm(vec_left, dim=1)
        dist_right = torch.linalg.norm(vec_right, dim=1)

        acts_left = 0.3 / (dist_left + 0.1)
        acts_right = 0.3 / (dist_right + 0.1)

        feats = []
        if n_olf_left > 0:
            feats.append(acts_left.unsqueeze(1).expand(-1, n_olf_left))
        if n_olf_right > 0:
            feats.append(acts_right.unsqueeze(1).expand(-1, n_olf_right))

        return torch.cat(feats, dim=1) if feats else torch.zeros(batch, 0, device=device, dtype=dtype)
    
    def _sample_tactile(
        self, sensors: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        n_tactile_left = self._segment_length("tactile_left")
        n_tactile_right = self._segment_length("tactile_right")
        n_tactile = n_tactile_left + n_tactile_right
        collision = sensors.get("collision")
        angle = sensors.get("collision_angle")
        if collision is None:
            raise KeyError("sensors missing required key 'collision'")
        if angle is None:
            raise KeyError("sensors missing required key 'collision_angle'")

        device = collision.device
        dtype = angle.dtype
        batch = collision.size(0)

        if n_tactile == 0:
            # print("Warning: No tactile receptors found in input_splits, no tactile input to network.")
            return torch.zeros(batch, 0, device=device, dtype=dtype)

        collision_mask = collision.bool().view(-1)
        angle = angle.view(-1)

        if not collision_mask.any():
            return torch.zeros(batch, n_tactile, device=device, dtype=dtype)

        acts_left = torch.zeros(batch, n_tactile_left, device=device, dtype=dtype)
        acts_right = torch.zeros(batch, n_tactile_right, device=device, dtype=dtype)

        small_angle = torch.abs(angle) < 0.1745  # ~10 deg
        left_side = angle > 0
        right_side = angle < 0

        if n_tactile_left > 0:
            acts_left[collision_mask & (small_angle | left_side)] = 1.0
        if n_tactile_right > 0:
            acts_right[collision_mask & (small_angle | right_side)] = 1.0

        if n_tactile_left == 0:
            return acts_right
        if n_tactile_right == 0:
            return acts_left
        return torch.cat([acts_left, acts_right], dim=1)

    # NEW: Wind Sampling
    def _sample_wind(self, sensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        wind = sensors.get("wind_direction")
        if wind is None:
            raise KeyError("sensors missing required key 'wind_direction'")
        
        batch = wind.size(0)
        n_wind = self._segment_length("wind")
        device = wind.device
        dtype = wind.dtype
        
        if n_wind == 0:
            # print("Warning: No wind receptors found in input_splits, no wind input to network.")
            return torch.zeros(batch, 0, device=device, dtype=dtype)
        x_wind = torch.zeros(batch, n_wind, device=device, dtype=dtype)
        x_wind[:, :2] = wind
        return x_wind

    def obs_to_x(self, obs: Dict[str, Any]) -> torch.Tensor:
        """
        Convert observation dict to input tensor x.
        
        Args:
            obs: Observation dictionary
         
        """
        cam_left = self._to_grayscale(obs["cam_left"])
        cam_right = self._to_grayscale(obs["cam_right"])
        sensors = obs["sensors"]

        x_left_L1 = self._sample_eye(cam_left, self.grid_L1_left)
        x_left_L2 = self._sample_eye(cam_left, self.grid_L2_left)
        x_left_L3 = self._sample_eye(cam_left, self.grid_L3_left)
        x_right_L1 = self._sample_eye(cam_right, self.grid_L1_right)
        x_right_L2 = self._sample_eye(cam_right, self.grid_L2_right)
        x_right_L3 = self._sample_eye(cam_right, self.grid_L3_right)
        
        x_olf = self._sample_olfactory(sensors)
        x_tactile = self._sample_tactile(sensors)
        x_wind = self._sample_wind(sensors) # NEW
        
        x = torch.cat([x_left_L1, x_left_L2, x_left_L3, x_right_L1, x_right_L2, x_right_L3, x_olf, x_tactile, x_wind], dim=1)
        return x.to(dtype=self.cell.W_values.dtype)

    def x_to_action(self, x: torch.Tensor, update_state: bool = True) -> torch.Tensor:
        # Check dimensions to decide between Sequence or Batch mode
        # x is either (B, Nin) or (T, B, Nin)
        is_sequence = (x.dim() == 3)
        
        # Ensure state dicts exist if we are updating (or just need consistency)
        if self.state_vision_left is None: self.state_vision_left = {}
        if self.state_vision_right is None: self.state_vision_right = {}

        # -------------------------------------------------------------
        # Helper for Virtual Retina Step (Single Time Step)
        # -------------------------------------------------------------
        def retina_step_batch(x_curr, states_L, states_R, update):
            # Slicing
            x_L1_L = self._slice_input(x_curr, "pr_L1_left")
            x_L2_L = self._slice_input(x_curr, "pr_L2_left")
            x_L3_L = self._slice_input(x_curr, "pr_L3_left")
            x_L1_R = self._slice_input(x_curr, "pr_L1_right")
            x_L2_R = self._slice_input(x_curr, "pr_L2_right")
            x_L3_R = self._slice_input(x_curr, "pr_L3_right")

            # Apply Dynamics
            # Left
            a_L1_L, s_L1_L = self._virtual_retina(x_L1_L, states_L.get("L1"), update, "L1")
            a_L2_L, s_L2_L = self._virtual_retina(x_L2_L, states_L.get("L2"), update, "L2")
            a_L3_L, s_L3_L = self._virtual_retina(x_L3_L, states_L.get("L3"), update, "L3")
            # Right
            a_L1_R, s_L1_R = self._virtual_retina(x_L1_R, states_R.get("L1"), update, "L1")
            a_L2_R, s_L2_R = self._virtual_retina(x_L2_R, states_R.get("L2"), update, "L2")
            a_L3_R, s_L3_R = self._virtual_retina(x_L3_R, states_R.get("L3"), update, "L3")

            new_states_L = {"L1": s_L1_L, "L2": s_L2_L, "L3": s_L3_L}
            new_states_R = {"L1": s_L1_R, "L2": s_L2_R, "L3": s_L3_R}
            
            # Other inputs pass through unchanged
            # We can re-assemble efficiently by just replacing the visual parts or re-catting all
            # Re-cat is safer to ensure order
            x_olf_L = self._slice_input(x_curr, "olf_left")*0.0+1.0
            x_olf_R = self._slice_input(x_curr, "olf_right")*0.0+1.0
            x_tac_L = self._slice_input(x_curr, "tactile_left")
            x_tac_R = self._slice_input(x_curr, "tactile_right")
            x_wind = self._slice_input(x_curr, "wind")
            # wind direction (first two elements of x_wind) goes through MLP
            if self.wind_mlp is None:
                # If no MLP, we assume the slice is either empty or we can't process it.
                # In most cases, if wind_mlp is None, the wind segment length is 0.
                pass 
            else:
                x_wind = self.wind_mlp(x_wind[:, 0:2])
            use_vision_left = True
            if not use_vision_left:
                a_L1_L*=0.0
                a_L2_L*=0.0
                a_L3_L*=0.0
            use_vision_right = True
            if not use_vision_right:
                a_L1_R*=0.0
                a_L2_R*=0.0
                a_L3_R*=0.0
            out = torch.cat([a_L1_L, a_L2_L, a_L3_L, a_L1_R, a_L2_R, a_L3_R, 
                             x_olf_L, x_olf_R, x_tac_L, x_tac_R, x_wind], dim=-1)
            
            return out, new_states_L, new_states_R

        # -------------------------------------------------------------
        # Execution
        # -------------------------------------------------------------
        prev_L = self.state_vision_left
        prev_R = self.state_vision_right
        
        if not is_sequence:
            # --- Single Step (Batch) ---
            out_x, ns_L, ns_R = retina_step_batch(x, prev_L, prev_R, update_state)
            if update_state:
                self.state_vision_left = ns_L
                self.state_vision_right = ns_R
        else:
            # --- Sequence (Time, Batch) ---
            # We must loop over time to correct propagate gradients through the filter state
            T = x.size(0)
            outs = []
            
            # For sequence processing, we usually want to carry hidden state through the sequence
            # But the 'update_state' flag in arguments usually refers to "updating the PERMANENT agent state".
            # For a forward pass on a buffer sequence (Lookahead/Training), we don't update permanent state,
            # but we DO update the transient state flowing through the loop.
            
            # Initialize loop state from permanent state (or zeros if None, handled in _virtual_retina)
            # Make copies of the logic_state to mutate during loop
            loop_state_L = prev_L
            loop_state_R = prev_R
            
            for t in range(T):
                x_t = x[t] # (B, Nin)
                # We always "update" the loop state, effectively flowing time forward
                out_t, loop_state_L, loop_state_R = retina_step_batch(x_t, loop_state_L, loop_state_R, True)
                outs.append(out_t)
            
            out_x = torch.stack(outs, dim=0)
            
            # We generally do NOT update the agent's permanent state after a training forward pass
            # so we ignore loop_state_L/R final values here unless update_state is True
            # (which is usually False for forward_sequence calls)
            if update_state:
                self.state_vision_left = loop_state_L
                self.state_vision_right = loop_state_R

        # Final Scaling
        out_x = out_x * self.input_scale
        return out_x

    def step(self, h: torch.Tensor, obs: Dict[str, Any], x: Optional[torch.Tensor] = None):
        # Allow passing precomputed photoreceptor activations to avoid redundant processing.
        if x is None:
            x = self.obs_to_x(obs)
        
        else:
            x = x.to(device=self.device, dtype=self.cell.W_values.dtype)

        x = self.x_to_action(x)
        xs = x.unsqueeze(0)
        hT, y = self.cell(h,xs,checkpoint_steps=False,store_sequence=False)
        
        # Clean NaN from RNN output
        self._warn_if_nan(y, "NAN in RNN output, replacing with zeros")
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # Bound outputs to expected ranges:
        # - Velocity (dim 0): [-1, 1] for environment compatibility
        # - Angle (dim 1): [-π, π] for heading control
        y = y.clone()  # Avoid in-place modification issues
        y[:, 0] = torch.tanh(y[:, 0]) *3  # Velocity in [-1, 1]
        y[:, 1] = math.pi * torch.tanh(y[:, 1])  # Angle in [-pi, pi]

        return hT, y

    def _value_from_hidden(self, h: torch.Tensor, priv_obs: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.value_head is None:
            raise RuntimeError("Value head is disabled; enable use_value_head to compute values.")
        
        dn_act = h.index_select(1, self.cell.output_nodes)
        
        if self.priv_obs_dim > 0:
            if priv_obs is None:
                raise ValueError(f"Value head expects privileged obs of dim {self.priv_obs_dim}, got None")
            # Concat
            inp = torch.cat([dn_act, priv_obs], dim=1)
        else:
            inp = dn_act
            
        return self.value_head(inp).view(-1)

    def _policy_dist(self, mean: torch.Tensor) -> Normal:
        if self.policy_log_std is None:
            log_std = torch.zeros_like(mean)
        else:
            log_std = self.policy_log_std.to(device=mean.device, dtype=mean.dtype)
            log_std = log_std.view(1, -1).expand_as(mean)
        std = torch.exp(log_std).clamp(min=1e-6)
        return Normal(mean, std)

    def forward_sequence(
        self,
        xs: torch.Tensor,
        h_init: Optional[torch.Tensor] = None,
        checkpoint_steps: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a sequence of observations efficiently.
        
        Args:
            xs: Tensor of shape (T, B, Nin) or (T, Nin) - precomputed photoreceptor activations
            h_init: Initial hidden state (B, N) or (1, N)
            checkpoint_steps: Whether to use gradient checkpointing
            
        Returns:
            h_final: Final full hidden state for RNN continuity
            y_seq: Sequence of policy outputs (T, B, Nout)
            dn_seq: Sequence of output neuron activations (T, B, Nout) for value computation
            
        Note:
            Vision states are reset at the start of each batch since xs typically
            comes from a replay buffer where temporal continuity is not guaranteed.
        """
        # Reset vision states - sequences may come from different episodes
        self.reset_vision_state()
        
        if xs.dim() == 2:
            xs = xs.unsqueeze(1) # (T, 1, Nin)
            
        device = self.device
        dtype = self.cell.W_values.dtype
        
        # Ensure correct type/device
        # update_state=False prevents the training batch sequence from overwriting 
        # the agent's persistent internal vision state (used for inference).
        xs = xs.to(device=device, dtype=dtype)
        xs = self.x_to_action(xs, update_state=False)
        
        if h_init is None:
            B = xs.size(1)
            h = torch.zeros(B, self.cell.N, device=device, dtype=dtype)
        else:
            h = h_init.to(device=device, dtype=dtype)
            
        # Call cell with store_sequence=True
        # Returns (h_final, Y_seq, DN_seq) after optimization
        h_final, y_seq, dn_seq = self.cell(h, xs, checkpoint_steps=checkpoint_steps, store_sequence=True)
        
        # Post-process outputs (Tanh, etc) same as step()
        # y_seq is (T, B, Nout)
        y_seq = torch.nan_to_num(y_seq, nan=0.0)
        
        # Avoid in-place modification
        # Velocity (dim 0): [-1, 1]
        # Angle (dim 1): [-pi, pi]
        y_out = y_seq.clone()
        y_out[..., 0] = torch.tanh(y_out[..., 0]) *3
        y_out[..., 1] = math.pi * torch.tanh(y_out[..., 1])
        
        # Return: h_final (for RNN continuity), y_out (policy), dn_seq (for value)
        return h_final, y_out, dn_seq

    def act(
        self,
        h: torch.Tensor,
        obs: Dict[str, Any],
        x: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        priv_obs: Optional[torch.Tensor] = None,
    ):
        """
        RL-friendly action helper that returns action, log prob, and value (if enabled).
        Keeps the original `step` API intact for DAgger users.
        
        Action dimensions:
        - action[:, 0]: velocity in [-1, 1]
        - action[:, 1]: heading angle in [-π, π]
        """
        # x=self.x_to_action(x) # Removed double call, step() handles it
        hT, mean = self.step(h, obs, x=x)
        dist = self._policy_dist(mean)
        
        if deterministic:
            action_out = mean
            # For deterministic, log_prob is technically undefined/inf, but we return prob of mode
            log_prob = dist.log_prob(mean).sum(dim=-1)
        else:
            raw_action = dist.rsample()
            
            # PPO CORRECTNESS FIX:
            # 1. Calculate log_prob on the RAW action (the latent Gaussian sample)
            #    This ensures the probability density matches the random variable.
            # 2. Return the CLAMPED action to the env.
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            
            # Apply per-dimension clamping for environment interaction
            clamped_vel = raw_action[:, 0].clamp(-1.0, 1.0)
            clamped_angle = raw_action[:, 1].clamp(-math.pi, math.pi)
            action_out = torch.stack([clamped_vel, clamped_angle], dim=-1)
        
        value = None
        if self.value_head is not None:
            value = self._value_from_hidden(hT, priv_obs=priv_obs)
             
        return hT, action_out, log_prob, value
