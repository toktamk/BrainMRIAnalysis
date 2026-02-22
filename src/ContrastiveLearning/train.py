from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ContrastiveDataConfig, MRIPairDataset
from losses import NTXentLoss
from models import EncoderConfig, SimCLR


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--encoder", type=str, default="mobilenet_v2")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--outdir", type=str, default="runs/contrastive")
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data_cfg = ContrastiveDataConfig(root=Path(args.data_root), image_size=224, grayscale=False)
    ds = MRIPairDataset(data_cfg)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = SimCLR(EncoderConfig(name=args.encoder, pretrained=args.pretrained), in_channels=3).to(device)
    loss_fn = NTXentLoss(temperature=args.temperature)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Save config for reproducibility
    with (outdir / "config.json").open("w") as f:
        json.dump(
            {"args": vars(args), "data_cfg": asdict(data_cfg)},
            f,
            indent=2,
        )

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for x_i, x_j in dl:
            x_i = x_i.to(device)
            x_j = x_j.to(device)

            z_i, z_j, _, _ = model(x_i, x_j)
            loss = loss_fn(z_i, z_j)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())

        avg = running / max(1, len(dl))
        print(f"[Contrastive] epoch={epoch:03d} loss={avg:.4f}")

        if avg < best:
            best = avg
            ckpt = {"model": model.state_dict(), "epoch": epoch, "loss": avg}
            torch.save(ckpt, outdir / "best.pt")

    print(f"Done. Best loss={best:.4f}. Checkpoint: {outdir/'best.pt'}")


if __name__ == "__main__":
    main()