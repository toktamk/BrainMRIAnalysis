
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
except Exception as e:  # pragma: no cover
    GCNConv = None
    global_mean_pool = None
    _PYG_IMPORT_ERROR = e


class TwoStepGNNClassifier(nn.Module):
    """
    Hierarchical /  two-step classifier:
      1) tumor type prediction (T classes)
      2) grade prediction conditioned on tumor type via per-type grade heads (G classes each)

    Forward returns:
      type_logits: (B, T)
      grade_logits: (B, T, G)
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_types: int, num_grades: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        if GCNConv is None:
            raise ImportError(f"torch_geometric is required: {_PYG_IMPORT_ERROR}")

        self.num_types = int(num_types)
        self.num_grades = int(num_grades)
        self.dropout = float(dropout)

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.num_types),
        )

        self.grade_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, self.num_grades),
                )
                for _ in range(self.num_types)
            ]
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros((x.size(0),), dtype=torch.long)

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))

        g = global_mean_pool(x, batch)  # (B, hidden_dim)

        type_logits = self.type_head(g)  # (B, T)

        # (B, T, G)
        grade_logits = torch.stack([head(g) for head in self.grade_heads], dim=1)
        return type_logits, grade_logits


def two_step_loss(
    type_logits: torch.Tensor,
    grade_logits: torch.Tensor,
    y_type: torch.Tensor,
    y_grade: torch.Tensor,
    *,
    lambda_type: float = 1.0,
    lambda_grade: float = 1.0,
) -> torch.Tensor:
    """
    Teacher-forced grade loss: choose grade head by *true* tumor type.

    Args:
        type_logits: (B, T)
        grade_logits: (B, T, G)
        y_type: (B,)
        y_grade: (B,)  (0..G-1)
    """
    y_type = y_type.view(-1).long()
    y_grade = y_grade.view(-1).long()

    loss_type = F.cross_entropy(type_logits, y_type)

    # Select correct type head per sample
    b = y_type.shape[0]
    chosen = grade_logits[torch.arange(b, device=grade_logits.device), y_type]  # (B, G)
    loss_grade = F.cross_entropy(chosen, y_grade)

    return lambda_type * loss_type + lambda_grade * loss_grade


@torch.no_grad()
def two_step_metrics(
    type_logits: torch.Tensor,
    grade_logits: torch.Tensor,
    y_type: torch.Tensor,
    y_grade: torch.Tensor,
) -> dict:
    """
    Returns:
      type_acc
      grade_acc_given_true_type
      end2end_grade_acc (using predicted type then grade head)
      end2end_all_correct
    """
    y_type = y_type.view(-1).long()
    y_grade = y_grade.view(-1).long()

    type_pred = type_logits.argmax(dim=1)
    type_acc = (type_pred == y_type).float().mean().item()

    b = y_type.shape[0]
    grade_pred_tf = grade_logits[torch.arange(b, device=grade_logits.device), y_type].argmax(dim=1)
    grade_acc_tf = (grade_pred_tf == y_grade).float().mean().item()

    grade_pred_e2e = grade_logits[torch.arange(b, device=grade_logits.device), type_pred].argmax(dim=1)
    grade_acc_e2e = (grade_pred_e2e == y_grade).float().mean().item()

    all_correct = ((type_pred == y_type) & (grade_pred_e2e == y_grade)).float().mean().item()

    return {
        "type_acc": float(type_acc),
        "grade_acc_given_true_type": float(grade_acc_tf),
        "end2end_grade_acc": float(grade_acc_e2e),
        "end2end_all_correct": float(all_correct),
    }

# ---------------------------------------------------------------------
# Backward-compatible API (kept because __init__.py imports GNNClassifier)
# ---------------------------------------------------------------------
class GNNClassifier(TwoStepGNNClassifier):
    """
    Compatibility wrapper for older code expecting a single-head classifier.

    Behavior:
      - forward(data) returns logits for a single label space.
      - We implement it by returning ONLY the tumor-type logits from the two-step model.
        (If you used GNNClassifier for grade before, update callers to TwoStepGNNClassifier.)
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int = 4, **kwargs):
        # Interpret num_classes as "num_types" in the new world, unless overridden
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_types=int(kwargs.pop("num_types", num_classes)),
            num_grades=int(kwargs.pop("num_grades", 4)),
            **kwargs,
        )

    def forward(self, data):
        type_logits, _grade_logits = super().forward(data)
        return type_logits