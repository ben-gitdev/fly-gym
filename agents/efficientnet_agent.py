
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Any, Dict, Optional, Tuple, Sequence
import math
import torchvision.models as models

class EfficientNetAgent(nn.Module):
    """
    Agent that uses EfficientNet-B0 to process visual observations and a GRU for memory.
    Input: Stereo camera images (stacked horizontally or processed separately).
    Output: Action (velocity, steering).
    """
    def __init__(
        self,
        action_dim: int = 2,
        hidden_size: int = 256,
        dtype: torch.dtype = torch.float32,
        learn_policy_std: bool = False,
        policy_std_init: float = 0.3,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.dtype = dtype

        # 1. Visual Encoder: EfficientNet-B0 (Pretrained)
        # We assume input is (B, 1, H, W).
        # EfficientNet-B0 expects 224x224 usually, but can handle other sizes.
        # Our images are 30x30 per eye. Stacked: 30x60.
        # We will use the features from the last conv layer before classification.
        
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Modify first layer to accept 1 channel
        first_conv_layer = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(1, first_conv_layer.out_channels, 
                                                 kernel_size=first_conv_layer.kernel_size, 
                                                 stride=first_conv_layer.stride, 
                                                 padding=first_conv_layer.padding, 
                                                 bias=False)
        
        # Remove the classifier head
        # The feature extractor output size for B0 is 1280 flat features after pooling.
        self.backbone.classifier = nn.Identity()
        self.feature_dim = 1280 
        
        # Wind direction is 2D vector
        self.wind_dir_dim = 2
        # Collision is 1D scalar
        self.collision_dim = 1

        # 2. Memory: GRU Cell
        # Input is Visual Features + Wind Direction + Collision
        self.gru = nn.GRUCell(input_size=self.feature_dim + self.wind_dir_dim + self.collision_dim, hidden_size=hidden_size)

        # 3. Policy Head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        # 4. Optional: Learnable Std
        if learn_policy_std:
            init_log_std = math.log(max(policy_std_init, 1e-6))
            self.policy_log_std = nn.Parameter(
                torch.full((self.action_dim,), float(init_log_std), dtype=dtype)
            )
        else:
            self.register_parameter("policy_log_std", None)

        # 5. Normalization stats (ImageNet - Grayscale)
        # RGB mean: [0.485, 0.456, 0.406] -> avg: 0.449
        # RGB std: [0.229, 0.224, 0.225] -> avg: 0.226
        self.register_buffer("mean", torch.tensor([0.449]).view(1, 1, 1, 1))
        self.register_buffer("std", torch.tensor([0.226]).view(1, 1, 1, 1))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _process_images(self, obs: Dict[str, Any]) -> torch.Tensor:
        """
        Process raw observation dict into tensor batch.
        Concat left and right cameras -> (B, 1, H, W_combined)
        Returns UINT8 tensor (0-255).
        """
        cam_left = obs["cam_left"]
        cam_right = obs["cam_right"]

        # If input is (H, W, 3) numpy, convert to (B, 3, H, W) torch
        # Check dim. If (B, H, W, 3), permute.
        if cam_left.ndim == 4 and cam_left.shape[-1] == 3:
            cam_left = cam_left.permute(0, 3, 1, 2)
            cam_right = cam_right.permute(0, 3, 1, 2)
            
        # Resize to 30x30
        if cam_left.shape[-2:] != (30, 30):
            cam_left = F.interpolate(cam_left.float(), size=(30, 30), mode='area')
            cam_right = F.interpolate(cam_right.float(), size=(30, 30), mode='area')

        # Convert to Grayscale: 0.299R + 0.587G + 0.114B
        if cam_left.shape[1] == 3:
             cam_left = 0.299 * cam_left[:, 0:1] + 0.587 * cam_left[:, 1:2] + 0.114 * cam_left[:, 2:3]
             cam_right = 0.299 * cam_right[:, 0:1] + 0.587 * cam_right[:, 1:2] + 0.114 * cam_right[:, 2:3]
        
        # Concat horizontally (W dimension is last)
        # Left | Right
        x = torch.cat([cam_left, cam_right], dim=3)
        
        # Ensure uint8
        if x.dtype != torch.uint8:
             # Assume it's float 0-1 if not uint8? Or float 0-255?
             # If float 0-1, scale to 0-255
             if x.max() <= 1.05 and x.dtype.is_floating_point:
                 x = (x * 255).to(torch.uint8)
             else:
                 x = x.to(torch.uint8)
        
        return x

    def _process_wind_direction(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Extract wind direction sensor."""
        sensors = obs["sensors"]
        
        vec = sensors.get("wind_direction", torch.zeros(1, 2, device=self.device))

        if vec.dim() == 1:
            vec = vec.unsqueeze(0)
            
        return vec.float()

    def _process_collision(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Extract collision sensor."""
        sensors = obs["sensors"]
        col = sensors.get("collision", torch.zeros(1, device=self.device))
        
        # Ensure (B, 1)
        if col.dim() == 1:
            col = col.unsqueeze(1)
            
        return col.float()

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, 3, H, W) float
        return (x - self.mean) / self.std

    def obs_to_x(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Wrapper to match ConnectomeAgent API. 
        Returns dict with 'img' (uint8), 'wind_direction' (float), 'collision' (float).
        """
        return {
            "img": self._process_images(obs),
            "wind_direction": self._process_wind_direction(obs),
            "collision": self._process_collision(obs)
        }

    def forward(self, x: Dict[str, torch.Tensor], h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step forward.
        x: Dict {'img': (B, 3, H, W), 'wind_direction': (B, 2), 'collision': (B, 1)}
        h: (B, hidden_size)
        """
        img = x["img"]
        wind_dir = x["wind_direction"]
        col = x["collision"]
        
        # Normalize Image
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
            img = self._norm(img)

        # EfficientNet Backbone
        vis_features = self.backbone(img) # (B, 1280)
        
        # Concatenate with Wind Direction and Collision
        wind_dir = wind_dir.to(vis_features.device, dtype=vis_features.dtype)
        col = col.to(vis_features.device, dtype=vis_features.dtype)
        
        combined_features = torch.cat([vis_features, wind_dir, col], dim=1) # (B, 1283)
        
        # GRU
        h_next = self.gru(combined_features, h)
        
        # Policy
        action = self.policy_head(h_next)
        
        return action, h_next

    def step(self, h: torch.Tensor, obs: Dict[str, Any], x: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference step (compatible with rollout_episode).
        """
        if x is None:
            x = self.obs_to_x(obs)
        
        # x is dict.
        # Ensure tensors are on device.
        x["img"] = x["img"].to(self.device)
        x["wind_direction"] = x["wind_direction"].to(self.device)
        x["collision"] = x["collision"].to(self.device)
        
        # Ensure h is on device
        h = h.to(self.device)

        action, h_next = self.forward(x, h)

        # Output bounds mapping (same as ConnectomeAgent)
        action_out = action.clone()
        action_out[:, 0] = torch.tanh(action[:, 0]) * 3.0 
        action_out[:, 1] = math.pi * torch.tanh(action[:, 1])

        return h_next, action_out

    def forward_sequence(
        self,
        xs: Dict[str, torch.Tensor], # {'img': ..., 'wind_direction': ..., 'collision': ...}
        h_init: Optional[torch.Tensor] = None,
        checkpoint_steps: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sequence processing for training.
        """
        imgs = xs["img"] # (T, B, 3, H, W)
        wind_dirs = xs["wind_direction"] # (T, B, 2)
        collisions = xs["collision"] # (T, B, 1)
        
        T, B, C, H, W = imgs.shape
        
        if h_init is None:
            h = torch.zeros(B, self.hidden_size, device=self.device, dtype=self.dtype)
        else:
            h = h_init

        outs = []
        
        for t in range(T):
            img_t = imgs[t].to(self.device)
            wind_dir_t = wind_dirs[t].to(self.device)
            col_t = collisions[t].to(self.device)
            
            # Form standard input dict for forward
            x_t = {"img": img_t, "wind_direction": wind_dir_t, "collision": col_t}
            
            # Forward (includes normalization inside)
            action, h = self.forward(x_t, h)
            outs.append(action)

        y_seq = torch.stack(outs, dim=0)
        
        y_out = y_seq.clone()
        y_out[..., 0] = torch.tanh(y_out[..., 0]) * 3.0
        y_out[..., 1] = math.pi * torch.tanh(y_out[..., 1])
        
        return h, y_out, None # dummy dn_seq

    def reset_vision_state(self):
        pass # No persistent vision state to reset (like alpha filters)
