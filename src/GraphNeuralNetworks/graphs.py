
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

try:
    from torch_geometric.data import Data
except Exception as e:  # pragma: no cover
    Data = None
    _PYG_IMPORT_ERROR = e


@dataclass(frozen=True)
class GraphBuildConfig:
    chain_edges: bool = True


class SliceEncoder(nn.Module):
    """
    Encodes a slice image tensor (C,H,W) to a feature vector. 
    Minimal CNN; feel free to swap with a stronger backbone.
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


def build_chain_graph(num_nodes: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Build an undirected chain adjacency for N nodes :
      0-1-2-...-(N-1)
    Returns edge_index shape (2, 2*(N-1))
    """
    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    src = torch.arange(0, num_nodes - 1, dtype=torch.long, device=device)
    dst = src + 1
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return edge_index

def slices_to_pyg_data(slices=None, *, bag=None, y=None, y_type=None, y_grade=None, encoder=None, meta=None, **kwargs):
    """
    Build a PyG graph from a bag of slices (N,C,H,W).

    Accepts both parameter names for compatibility:
      - slices_to_pyg_data(slices=<tensor>, ...)
      - slices_to_pyg_data(bag=<tensor>, ...)   # alias
    """
    if slices is None:
        slices = bag
    if slices is None:
        raise TypeError("Provide 'slices' (positional) or 'bag=' tensor.")

    if Data is None:
        raise ImportError(f"torch_geometric is required: {_PYG_IMPORT_ERROR}")

    if slices.ndim != 4:
        raise ValueError(f"Expected slices (N,C,H,W), got: {tuple(slices.shape)}")

    # IMPORTANT: do NOT wrap in torch.no_grad() if encoder is trainable
    if encoder is None:
        x = slices.flatten(1)
    else:
        x = encoder(slices)

    chain_edges = bool(kwargs.pop("chain_edges", True))
    edge_index = (
        build_chain_graph(x.shape[0], device=x.device)
        if chain_edges
        else torch.empty((2, 0), dtype=torch.long, device=x.device)
    )
    data = Data(x=x, edge_index=edge_index)

    if y_type is not None:
        data.y_type = torch.tensor([int(y_type)], dtype=torch.long, device=x.device)
    if y_grade is not None:
        data.y_grade = torch.tensor([int(y_grade)], dtype=torch.long, device=x.device)
    if meta is not None:
        data.meta = meta
    return data
