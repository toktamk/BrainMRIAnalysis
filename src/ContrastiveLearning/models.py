from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tvm


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
            # feature extractor
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
            nn.Linear(enc_dim, enc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(enc_dim, cfg.projection_dim),
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