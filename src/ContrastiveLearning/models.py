from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tvm


# =============================================================================
# Contrastive 
# =============================================================================
@dataclass(frozen=True)
class EncoderConfig:
    name: str = "mobilenet_v2"  # "mobilenet_v2" | "small_cnn"
    pretrained: bool = True
    projection_dim: int = 128


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 3, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return self.out(h)


class SimCLR(nn.Module):
    """
    SimCLR-style encoder + projection head.
    Returns:
        z_i, z_j (projection-space features), and h_i, h_j (encoder features)
    """

    def __init__(self, cfg: EncoderConfig, in_channels: int = 3) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.name == "mobilenet_v2":
            weights = tvm.MobileNet_V2_Weights.DEFAULT if cfg.pretrained else None
            base = tvm.mobilenet_v2(weights=weights)
            self.encoder = base.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            enc_dim = 1280
        elif cfg.name == "small_cnn":
            self.encoder = SmallCNN(in_channels=in_channels, out_dim=256)
            self.pool = None
            enc_dim = 256
        else:
            raise ValueError(f"Unknown encoder name: {cfg.name}")

        self.projection = nn.Sequential(
            nn.Linear(enc_dim, enc_dim, bias=False),
            nn.BatchNorm1d(enc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(enc_dim, cfg.projection_dim, bias=False),
            nn.BatchNorm1d(cfg.projection_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.name == "mobilenet_v2":
            h = self.encoder(x)
            h = self.pool(h).flatten(1)
            return h
        return self.encoder(x)

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor):
        h_i = self.encode(x_i)
        h_j = self.encode(x_j)
        z_i = nn.functional.normalize(self.projection(h_i), dim=1)
        z_j = nn.functional.normalize(self.projection(h_j), dim=1)
        return z_i, z_j, h_i, h_j


# =============================================================================
# Two-step supervised classifier  (tumor type -> grade conditioned on type)
# =============================================================================
class Backbone(nn.Module):
    """
    Simple supervised backbone producing a single feature vector per image.
    """
    def __init__(self, name: str = "mobilenet_v2", pretrained: bool = True, in_channels: int = 3) -> None:
        super().__init__()
        self.name = name
        if name == "mobilenet_v2":
            weights = tvm.MobileNet_V2_Weights.DEFAULT if pretrained else None
            base = tvm.mobilenet_v2(weights=weights)
            self.features = base.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.out_dim = 1280
        elif name == "small_cnn":
            self.features = SmallCNN(in_channels=in_channels, out_dim=256)
            self.pool = None
            self.out_dim = 256
        else:
            raise ValueError(f"Unknown backbone: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.name == "mobilenet_v2":
            h = self.features(x)
            h = self.pool(h).flatten(1)
            return h
        return self.features(x)


class TwoStepClassifier(nn.Module):
    """
    Hierarchical model:
      - shared backbone
      - tumor type head: (T)
      - grade heads: one per tumor type, each predicts grade among (G)

    Forward returns:
      type_logits: (B, T)
      grade_logits: (B, T, G)
    """
    def __init__(
        self,
        num_types: int,
        num_grades: int = 4,
        *,
        backbone: str = "mobilenet_v2",
        pretrained: bool = True,
        in_channels: int = 3,
        hidden: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_types = int(num_types)
        self.num_grades = int(num_grades)

        self.backbone = Backbone(name=backbone, pretrained=pretrained, in_channels=in_channels)

        self.type_head = nn.Sequential(
            nn.Linear(self.backbone.out_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.num_types),
        )

        self.grade_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.backbone.out_dim, hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, self.num_grades),
                )
                for _ in range(self.num_types)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)  # (B, D)
        type_logits = self.type_head(feat)  # (B, T)
        grade_logits = torch.stack([h(feat) for h in self.grade_heads], dim=1)  # (B, T, G)
        return type_logits, grade_logits
