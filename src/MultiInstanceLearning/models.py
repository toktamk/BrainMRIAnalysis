from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, emb_dim: int = 256) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
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
        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*N, C, H, W)
        h = self.cnn(x).flatten(1)
        return self.fc(h)


class AttentionMIL(nn.Module):
    """
    Ilse et al.-style attention MIL:
      h_i = encoder(x_i)
      a_i = softmax(w^T tanh(V h_i))
      z = sum_i a_i h_i
      logits = classifier(z)
    """

    def __init__(self, in_channels: int = 3, emb_dim: int = 256, attn_dim: int = 128, num_classes: int = 4):
        super().__init__()
        self.encoder = InstanceEncoder(in_channels=in_channels, emb_dim=emb_dim)
        self.attn = nn.Sequential(
            nn.Linear(emb_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, num_classes),
        )

    def forward(self, bag: torch.Tensor) -> torch.Tensor:
        # bag: (B, N, C, H, W)
        b, n, c, h, w = bag.shape
        x = bag.view(b * n, c, h, w)
        h_i = self.encoder(x).view(b, n, -1)  # (B,N,D)

        a = self.attn(h_i).squeeze(-1)         # (B,N)
        a = torch.softmax(a, dim=1).unsqueeze(-1)  # (B,N,1)

        z = (a * h_i).sum(dim=1)               # (B,D)
        logits = self.classifier(z)            # (B,K)
        return logits