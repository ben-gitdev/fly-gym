
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple
import math
import torchvision.models as models


class DualBackboneAgent(nn.Module):
    """
    Agent with two separate CNN backbones (EfficientNet-B0 or MobileNetV3-Large),
    one per camera. Their features are concatenated with sensor inputs and fed
    into a GRU for temporal memory, followed by a policy head for action output.

    Input:
        img_left:  (B, 1, 30, 30)  — left camera grayscale
        img_right: (B, 1, 30, 30)  — right camera grayscale
        wind_direction: (B, 2)
        collision: (B, 1)
    Output:
        action: (B, 2) — (velocity, steering)
    """

    BACKBONE_CONFIGS = {
        "efficientnet": {
            "factory": lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
            "feature_dim": 1280,
        },
        "mobilenet": {
            "factory": lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT),
            "feature_dim": 960,
        },
    }

    def __init__(
        self,
        backbone_type: str = "efficientnet",
        action_dim: int = 2,
        hidden_size: int = 256,
        dtype: torch.dtype = torch.float32,
        learn_policy_std: bool = False,
        policy_std_init: float = 0.3,
    ):
        super().__init__()
        if backbone_type not in self.BACKBONE_CONFIGS:
            raise ValueError(
                f"Unknown backbone_type '{backbone_type}'. "
                f"Choose from: {list(self.BACKBONE_CONFIGS.keys())}"
            )

        self.backbone_type = backbone_type
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.dtype = dtype

        cfg = self.BACKBONE_CONFIGS[backbone_type]
        self.feature_dim = cfg["feature_dim"]  # per backbone

        # ----- Two independent backbone instances -----
        self.backbone_left = self._build_backbone(cfg)
        self.backbone_right = self._build_backbone(cfg)

        # ----- Sensor dimensions -----
        self.wind_dir_dim = 2
        self.collision_dim = 1

        # GRU input = 2 * feature_dim + wind_dir + collision
        gru_input_size = 2 * self.feature_dim + self.wind_dir_dim + self.collision_dim

        # ----- Memory: GRU Cell -----
        self.gru = nn.GRUCell(input_size=gru_input_size, hidden_size=hidden_size)

        # ----- Policy Head -----
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        # ----- Optional: Learnable Std -----
        if learn_policy_std:
            init_log_std = math.log(max(policy_std_init, 1e-6))
            self.policy_log_std = nn.Parameter(
                torch.full((self.action_dim,), float(init_log_std), dtype=dtype)
            )
        else:
            self.register_parameter("policy_log_std", None)

        # ----- Normalization stats (ImageNet grayscale average) -----
        self.register_buffer("mean", torch.tensor([0.449]).view(1, 1, 1, 1))
        self.register_buffer("std", torch.tensor([0.226]).view(1, 1, 1, 1))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_backbone(cfg: dict) -> nn.Module:
        """Instantiate a backbone, modify first conv for 1-ch input, remove classifier."""
        backbone = cfg["factory"]()
        # Modify first conv to accept 1 channel
        first_conv = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            1,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
        # Remove classifier head
        backbone.classifier = nn.Identity()
        return backbone

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """ImageNet-style normalisation on float [0, 1] grayscale."""
        return (x - self.mean) / self.std

    # ------------------------------------------------------------------
    # Observation processing
    # ------------------------------------------------------------------
    def _process_single_cam(self, cam: torch.Tensor) -> torch.Tensor:
        """
        Process a single camera tensor.
        Input can be (B, H, W, 3) or (B, 3, H, W) or (B, 1, H, W).
        Returns UINT8 (B, 1, 30, 30).
        """
        if cam.ndim == 4 and cam.shape[-1] == 3:
            cam = cam.permute(0, 3, 1, 2)  # → (B, 3, H, W)

        # Resize to 30×30 if needed
        if cam.shape[-2:] != (30, 30):
            cam = F.interpolate(cam.float(), size=(30, 30), mode="area")

        # Convert RGB to grayscale
        if cam.shape[1] == 3:
            cam = 0.299 * cam[:, 0:1] + 0.587 * cam[:, 1:2] + 0.114 * cam[:, 2:3]

        # Ensure uint8
        if cam.dtype != torch.uint8:
            if cam.max() <= 1.05 and cam.dtype.is_floating_point:
                cam = (cam * 255).to(torch.uint8)
            else:
                cam = cam.to(torch.uint8)

        return cam  # (B, 1, 30, 30)

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
        if col.dim() == 1:
            col = col.unsqueeze(1)
        return col.float()

    def obs_to_x(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Convert raw observation to network input dict.
        Returns dict with 'img_left', 'img_right' (uint8), 'wind_direction', 'collision'.
        """
        return {
            "img_left": self._process_single_cam(obs["cam_left"]),
            "img_right": self._process_single_cam(obs["cam_right"]),
            "wind_direction": self._process_wind_direction(obs),
            "collision": self._process_collision(obs),
        }

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------
    def forward(
        self, x: Dict[str, torch.Tensor], h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step forward.
        x: {'img_left': (B,1,30,30), 'img_right': (B,1,30,30),
             'wind_direction': (B,2), 'collision': (B,1)}
        h: (B, hidden_size)
        """
        img_l = x["img_left"]
        img_r = x["img_right"]
        wind_dir = x["wind_direction"]
        col = x["collision"]

        # Normalise images
        if img_l.dtype == torch.uint8:
            img_l = img_l.float() / 255.0
            img_l = self._norm(img_l)
        if img_r.dtype == torch.uint8:
            img_r = img_r.float() / 255.0
            img_r = self._norm(img_r)

        # Two-backbone feature extraction
        feat_l = self.backbone_left(img_l)   # (B, feature_dim)
        feat_r = self.backbone_right(img_r)  # (B, feature_dim)

        # Cast sensors to same device/dtype
        wind_dir = wind_dir.to(feat_l.device, dtype=feat_l.dtype)
        col = col.to(feat_l.device, dtype=feat_l.dtype)

        # Concatenate all features
        combined = torch.cat([feat_l, feat_r, wind_dir, col], dim=1)

        # GRU → Policy
        h_next = self.gru(combined, h)
        action = self.policy_head(h_next)

        return action, h_next

    def step(
        self,
        h: torch.Tensor,
        obs: Dict[str, Any],
        x: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference step (compatible with rollout_episode).
        """
        if x is None:
            x = self.obs_to_x(obs)

        # Ensure tensors are on device
        x["img_left"] = x["img_left"].to(self.device)
        x["img_right"] = x["img_right"].to(self.device)
        x["wind_direction"] = x["wind_direction"].to(self.device)
        x["collision"] = x["collision"].to(self.device)
        h = h.to(self.device)

        action, h_next = self.forward(x, h)

        # Output bounds mapping (same as other agents)
        action_out = action.clone()
        action_out[:, 0] = torch.tanh(action[:, 0]) * 3.0
        action_out[:, 1] = math.pi * torch.tanh(action[:, 1])

        return h_next, action_out

    def forward_sequence(
        self,
        xs: Dict[str, torch.Tensor],
        h_init: Optional[torch.Tensor] = None,
        checkpoint_steps: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sequence processing for training.
        xs: {'img_left': (T,B,1,30,30), 'img_right': (T,B,1,30,30),
             'wind_direction': (T,B,2), 'collision': (T,B,1)}
        """
        imgs_l = xs["img_left"]       # (T, B, 1, 30, 30)
        imgs_r = xs["img_right"]      # (T, B, 1, 30, 30)
        wind_dirs = xs["wind_direction"]  # (T, B, 2)
        collisions = xs["collision"]      # (T, B, 1)

        T, B = imgs_l.shape[:2]

        if h_init is None:
            h = torch.zeros(B, self.hidden_size, device=self.device, dtype=self.dtype)
        else:
            h = h_init

        outs = []

        for t in range(T):
            x_t = {
                "img_left": imgs_l[t].to(self.device),
                "img_right": imgs_r[t].to(self.device),
                "wind_direction": wind_dirs[t].to(self.device),
                "collision": collisions[t].to(self.device),
            }
            action, h = self.forward(x_t, h)
            outs.append(action)

        y_seq = torch.stack(outs, dim=0)

        y_out = y_seq.clone()
        y_out[..., 0] = torch.tanh(y_out[..., 0]) * 3.0
        y_out[..., 1] = math.pi * torch.tanh(y_out[..., 1])

        return h, y_out, None  # dummy dn_seq for API compatibility

    def reset_vision_state(self):
        pass  # No persistent vision state to reset
