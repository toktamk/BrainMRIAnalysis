
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

try:
    from torch_geometric.data import Data
except Exception as e:  # pragma: no cover
    Data = None
    _PYG_IMPORT_ERROR = e


def bag_to_graph_basic_stats(
    bag: torch.Tensor,
    *,
    y_type: Optional[int] = None,
    y_grade: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
    edge_mode: str = "chain",
) -> "Data":
    """
    Convert a bag of slice images into a simple PyG graph using basic image statistics.

    This is a lightweight alternative to CNN-based node encoding. It is deterministic and
    can be used for debugging / sanity checks.
    Args:
        bag: (N, C, H, W)
        y_type: tumor type label index (optional)
        y_grade: grade label index 0..3 (optional)
        meta: metadata dict stored as data.meta
        edge_mode: "chain" builds i<->i+1 edges.

    Returns:
        Data(x=(N, 2C), edge_index=(2, E), y_type=(1,), y_grade=(1,))
    """
    if Data is None:
        raise ImportError(f"torch_geometric is required: {_PYG_IMPORT_ERROR}")

    if not isinstance(bag, torch.Tensor):
        raise TypeError(f"bag must be torch.Tensor, got {type(bag)}")
    if bag.ndim != 4:
        raise ValueError(f"bag must have shape (N,C,H,W). Got {tuple(bag.shape)}")
    n, c, _, _ = bag.shape
    if n < 1:
        raise ValueError("bag must contain at least 1 instance")

    mean = bag.mean(dim=(2, 3))
    std = bag.std(dim=(2, 3), unbiased=False)
    x = torch.cat([mean, std], dim=1).contiguous()  # (N, 2C)

    if edge_mode == "chain":
        if n == 1:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=bag.device)
        else:
            src = torch.arange(0, n - 1, device=bag.device, dtype=torch.long)
            dst = src + 1
            edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    else:
        raise ValueError(f"Unknown edge_mode='{edge_mode}'. Supported: 'chain'")

    data = Data(x=x, edge_index=edge_index)

    if y_type is not None:
        data.y_type = torch.tensor([int(y_type)], dtype=torch.long, device=bag.device)
    if y_grade is not None:
        data.y_grade = torch.tensor([int(y_grade)], dtype=torch.long, device=bag.device)
    if meta is not None:
        data.meta = meta
    return data

# ---------------------------------------------------------------------
# Backward-compatible API (kept because __init__.py imports bag_to_graph)
# ---------------------------------------------------------------------
def bag_to_graph(bag, y=None, *, y_type=None, y_grade=None, encoder=None, meta=None, **kwargs):
    """
    Compatibility wrapper expected by older code/tests.
    Delegates to graphs.slices_to_pyg_data.

    Supports both:
      - single-head: bag_to_graph(bag, y=...)
      - two-step:    bag_to_graph(bag, y_type=..., y_grade=...)
    """
    from .graphs import slices_to_pyg_data

    # If caller uses two-step labels
    if y_type is not None or y_grade is not None:
        if y_type is None or y_grade is None:
            raise ValueError("Provide both y_type and y_grade for two-step graphs.")
        return slices_to_pyg_data(bag, y_type=int(y_type), y_grade=int(y_grade), encoder=encoder, meta=meta)

    # Fallback: single-head
    if y is None:
        raise ValueError("Provide y (single-head) or (y_type,y_grade) (two-step).")
    return slices_to_pyg_data(bag, y=int(y), encoder=encoder, meta=meta)