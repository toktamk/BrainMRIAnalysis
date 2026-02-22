from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def first_batch_loss_contrastive(data_root: Path, seed: int) -> float:
    seed_everything(seed)

    from torch.utils.data import DataLoader
    from src.ContrastiveLearning.data import ContrastiveDataConfig, MRIPairDataset
    from src.ContrastiveLearning.losses import NTXentLoss
    from src.ContrastiveLearning.models import EncoderConfig, SimCLR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = MRIPairDataset(ContrastiveDataConfig(root=data_root))
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

    model = SimCLR(EncoderConfig(name="small_cnn", pretrained=False, projection_dim=64)).to(device)
    loss_fn = NTXentLoss()

    x_i, x_j = next(iter(dl))
    x_i, x_j = x_i.to(device), x_j.to(device)

    with torch.no_grad():
        z_i, z_j, _, _ = model(x_i, x_j)
        loss = loss_fn(z_i, z_j).item()
    return float(loss)


def main() -> None:
    ap = argparse.ArgumentParser(description="Determinism smoke test: same seed -> same first-batch loss.")
    ap.add_argument("--data_root", required=True, type=str)
    ap.add_argument("--method", choices=["contrastive"], default="contrastive")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tol", type=float, default=1e-7)
    args = ap.parse_args()

    data_root = Path(args.data_root)

    if args.method == "contrastive":
        a = first_batch_loss_contrastive(data_root, args.seed)
        b = first_batch_loss_contrastive(data_root, args.seed)

    diff = abs(a - b)
    print(f"loss_a={a:.10f}")
    print(f"loss_b={b:.10f}")
    print(f"abs_diff={diff:.10f}  tol={args.tol:.10f}")

    if diff > args.tol:
        raise RuntimeError(
            "Reproducibility check failed: losses differ beyond tolerance. "
            "Consider setting num_workers=0, controlling augmentations, and checking CUDA determinism."
        )

    print("Reproducibility check passed.")


if __name__ == "__main__":
    main()