# src/GraphNeuralNetworks/graph_builder.py
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

try:
    from torch_geometric.data import Data
except Exception as e:  # pragma: no cover
    Data = None
    _PYG_IMPORT_ERROR = e


def bag_to_graph(
    bag: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    meta: Optional[Dict[str, Any]] = None,
    *,
    edge_mode: str = "chain",
) -> "Data":
    """
    Convert a MIL bag of images into a PyG graph.

    Args:
        bag: Tensor of shape (N, C, H, W)
        y: Optional graph label (scalar tensor)
        meta: Optional metadata dict; stored on the Data object (as .meta)
        edge_mode: "chain" (default). Chain edges connect i <-> i+1.

    Returns:
        torch_geometric.data.Data with:
            x: (N, F) node features
            edge_index: (2, E)
            y: optional graph label
            meta: optional metadata
    """
    if Data is None:
        raise ImportError(f"torch_geometric is required for bag_to_graph: {_PYG_IMPORT_ERROR}")

    if not isinstance(bag, torch.Tensor):
        raise TypeError(f"bag must be torch.Tensor, got {type(bag)}")
    if bag.ndim != 4:
        raise ValueError(f"bag must have shape (N,C,H,W). Got {tuple(bag.shape)}")

    N, C, H, W = bag.shape
    if N < 1:
        raise ValueError("bag must contain at least 1 instance")

    # Node features: channel-wise mean and std over spatial dims
    # mean: (N, C), std: (N, C) -> x: (N, 2C)
    mean = bag.mean(dim=(2, 3))
    std = bag.std(dim=(2, 3), unbiased=False)
    x = torch.cat([mean, std], dim=1).contiguous()  # (N, 2C)

    # Edges
    if edge_mode == "chain":
        if N == 1:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=bag.device)
        else:
            src = torch.arange(0, N - 1, device=bag.device, dtype=torch.long)
            dst = src + 1
            # undirected: i->i+1 and i+1->i
            edge_index = torch.stack(
                [torch.cat([src, dst]), torch.cat([dst, src])],
                dim=0,
            )
    else:
        raise ValueError(f"Unknown edge_mode='{edge_mode}'. Supported: 'chain'")

    data = Data(x=x, edge_index=edge_index)

    if y is not None:
        # store as graph label; ensure tensor
        data.y = y if isinstance(y, torch.Tensor) else torch.tensor(y)

    if meta is not None:
        data.meta = meta

    return data