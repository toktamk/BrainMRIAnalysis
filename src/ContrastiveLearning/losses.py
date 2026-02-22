from __future__ import annotations

import torch
import torch.nn as nn


class NTXentLoss(nn.Module):
    """
    Standard SimCLR NT-Xent for a batch of positive pairs (z_i, z_j).
    Correct positive indexing: sample k in z_i is positive with sample k in z_j.
    """

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        if z_i.shape != z_j.shape:
            raise ValueError(f"Shape mismatch: {z_i.shape} vs {z_j.shape}")

        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D)

        # cosine similarity via normalized dot product
        sim = (z @ z.T) / self.temperature  # (2B, 2B)

        # mask self-similarity
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float("-inf"))

        # positives: i<->i+B
        targets = torch.arange(batch_size, device=z.device)
        targets = torch.cat([targets + batch_size, targets], dim=0)  # (2B,)

        loss = nn.functional.cross_entropy(sim, targets)
        return loss