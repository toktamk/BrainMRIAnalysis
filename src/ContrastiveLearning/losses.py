from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Standard SimCLR NT-Xent for a  batch of positive pairs (z_i, z_j).
    Correct positive indexing: sample k in z_i is positive with sample k in z_j.
    """

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        if z_i.shape != z_j.shape:
            raise ValueError(f"Shape mismatch: {z_i.shape} vs {z_j.shape}")

        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D)

        # cosine similarity based on normalized dot product
        sim = (z @ z.T) / self.temperature  # (2B, 2B)

        # mask self-similarity
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float("-inf"))

        # positives: i<->i+B
        targets = torch.arange(batch_size, device=z.device)
        targets = torch.cat([targets + batch_size, targets], dim=0)  # (2B,)

        loss = F.cross_entropy(sim, targets)
        return loss


def two_step_loss(
    type_logits: torch.Tensor,
    grade_logits: torch.Tensor,
    y_type: torch.Tensor,
    y_grade: torch.Tensor,
    *,
    type_weight: float = 1.0,
    grade_weight: float = 1.0,
) -> torch.Tensor:
    """
    Hierarchical two-step loss.

    Args:
      type_logits: (B, T)
      grade_logits: (B, T, G) where grade_logits[b, t] is the grade head for type t.
      y_type: (B,)
      y_grade: (B,)
    """
    if type_logits.ndim != 2:
        raise ValueError(f"type_logits must be (B,T), got {tuple(type_logits.shape)}")
    if grade_logits.ndim != 3:
        raise ValueError(f"grade_logits must be (B,T,G), got {tuple(grade_logits.shape)}")

    y_type = y_type.long().view(-1)
    y_grade = y_grade.long().view(-1)
    if y_type.numel() != type_logits.shape[0] or y_grade.numel() != type_logits.shape[0]:
        raise ValueError("Batch size mismatch among logits and targets.")

    # Type loss
    loss_type = F.cross_entropy(type_logits, y_type)

    # Select the grade head corresponding to the TRUE type (teacher forcing)
    b = torch.arange(type_logits.shape[0], device=type_logits.device)
    sel_grade_logits = grade_logits[b, y_type]  # (B, G)
    loss_grade = F.cross_entropy(sel_grade_logits, y_grade)

    return type_weight * loss_type + grade_weight * loss_grade


@torch.no_grad()
def two_step_metrics(
    type_logits: torch.Tensor,
    grade_logits: torch.Tensor,
    y_type: torch.Tensor,
    y_grade: torch.Tensor,
) -> dict:
    """
    Returns:
      type_acc: tumor type accuracy
      grade_acc_given_true_type: grade accuracy when conditioning on true type head (teacher forcing)
      end2end_grade_acc: grade accuracy when using predicted type head (true inference path)
      end2end_all_correct: fraction where both type and grade are correct (pred-type path)
    """
    y_type = y_type.long().view(-1)
    y_grade = y_grade.long().view(-1)

    type_pred = type_logits.argmax(dim=1)

    # Teacher-forced grade (debugging; isolates grade head quality)
    b = torch.arange(type_logits.shape[0], device=type_logits.device)
    grade_pred_tf = grade_logits[b, y_type].argmax(dim=1)

    # End-to-end grade (real inference): pick grade head by predicted type
    grade_pred_e2e = grade_logits[b, type_pred].argmax(dim=1)

    type_acc = (type_pred == y_type).float().mean().item()
    grade_acc_tf = (grade_pred_tf == y_grade).float().mean().item()
    grade_acc_e2e = (grade_pred_e2e == y_grade).float().mean().item()
    all_correct = ((type_pred == y_type) & (grade_pred_e2e == y_grade)).float().mean().item()

    return {
        "type_acc": float(type_acc),
        "grade_acc_given_true_type": float(grade_acc_tf),
        "end2end_grade_acc": float(grade_acc_e2e),
        "end2end_all_correct": float(all_correct),
    }
