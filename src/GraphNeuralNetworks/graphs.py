from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch_geometric.data import Data


@dataclass(frozen=True)
class GraphBuildConfig:
    chain_edges: bool = True


class SliceEncoder(nn.Module):
    """
    Encodes a slice image tensor (C,H,W) to a feature vector.
    Minimal CNN to avoid heavy dependencies; replace with a pretrained encoder if desired.
    """
    def __init__(self, in_channels: int = 3, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,C,H,W)
        h = self.net(x).flatten(1)
        return self.fc(h)


def build_chain_graph(node_features: torch.Tensor) -> torch.Tensor:
    """
    Build a simple chain adjacency for N nodes:
      0-1-2-...-(N-1)
    Returns edge_index shape (2, 2*(N-1)) for undirected chain.
    """
    n = node_features.shape[0]
    if n < 2:
        return torch.empty((2, 0), dtype=torch.long)

    src = torch.arange(0, n - 1, dtype=torch.long)
    dst = src + 1
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return edge_index


def slices_to_pyg_data(
    slices: torch.Tensor,
    *,
    encoder: Optional[nn.Module] = None,
    cfg: GraphBuildConfig = GraphBuildConfig(),
    y: Optional[int] = None,
) -> Data:
    """
    slices: (N, C, H, W) float tensor.
    Produces Data(x=(N,F), edge_index=(2,E), y=(1,))
    """
    if slices.ndim != 4:
        raise ValueError(f"Expected slices (N,C,H,W), got: {tuple(slices.shape)}")

    with torch.no_grad():
        if encoder is None:
            # default: flatten to features (not ideal, but deterministic)
            x = slices.flatten(1)
        else:
            x = encoder(slices)

    edge_index = build_chain_graph(x) if cfg.chain_edges else torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.long)
    return data