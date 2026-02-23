from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from MRIData import MRIDataConfig, MRIContrastivePairDataset, default_simclr_transform
from models import EncoderConfig, SimCLR
from losses import NTXentLoss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model: SimCLR, loader: DataLoader, opt, loss_fn, device) -> float:
    model.train()
    total = 0.0
    steps = 0
    for x_i, x_j in loader:
        x_i = x_i.to(device, non_blocking=True)
        x_j = x_j.to(device, non_blocking=True)

        z_i, z_j, _, _ = model(x_i, x_j)
        loss = loss_fn(z_i, z_j)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total += float(loss.item())
        steps += 1
    return total / max(1, steps)


@torch.no_grad()
def eval_one_epoch(model: SimCLR, loader: DataLoader, loss_fn, device) -> float:
    model.eval()
    total = 0.0
    steps = 0
    for x_i, x_j in loader:
        x_i = x_i.to(device, non_blocking=True)
        x_j = x_j.to(device, non_blocking=True)
        z_i, z_j, _, _ = model(x_i, x_j)
        loss = loss_fn(z_i, z_j)
        total += float(loss.item())
        steps += 1
    return total / max(1, steps)


def main() -> None:
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--grayscale", action="store_true")
    p.add_argument("--pair_stride", type=int, default=1)

    # simclr
    p.add_argument("--encoder", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "small_cnn"])
    p.add_argument("--pretrained", action="store_true", help="If set, initializes encoder from ImageNet weights (for mobilenet_v2).")
    p.add_argument("--projection_dim", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.5)

    # training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--val_frac", type=float, default=0.05)

    # output
    p.add_argument("--outdir", type=str, default="runs/simclr_pretrain")
    p.add_argument("--save_name", type=str, default="simclr_pretrain.pt")

    args = p.parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # dataset
    cfg = MRIDataConfig(
        root=args.data_root,
        mode="contrastive",
        target="subtype",
        image_size=args.image_size,
        grayscale=args.grayscale,
        pair_stride=args.pair_stride,
        seed=args.seed,
    )
    transform = default_simclr_transform(args.image_size, args.grayscale)
    ds = MRIContrastivePairDataset(cfg, transform=transform)

    # small validation split for early monitoring (no labels needed)
    val_len = max(1, int(round(len(ds) * args.val_frac)))
    train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False
    )

    # model
    enc_cfg = EncoderConfig(name=args.encoder, pretrained=args.pretrained, projection_dim=args.projection_dim)
    model = SimCLR(enc_cfg, in_channels=1 if args.grayscale else 3).to(device)

    loss_fn = NTXentLoss(temperature=args.temperature).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # save run config
    with (outdir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "data_cfg": asdict(cfg), "n_pairs": len(ds)}, f, indent=2)

    best_val = float("inf")
    best_path = outdir / args.save_name

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, loss_fn, device)
        va = eval_one_epoch(model, val_loader, loss_fn, device)

        print(f"[simclr] epoch {epoch:03d}/{args.epochs} train_loss={tr:.4f} val_loss={va:.4f}")

        # save best by val loss
        if va < best_val:
            best_val = va
            torch.save(
                {
                    "simclr_state": model.state_dict(),   # contains encoder.* and projection.* keys
                    "encoder_cfg": asdict(enc_cfg),
                    "data_cfg": asdict(cfg),
                    "epoch": epoch,
                    "val_loss": best_val,
                },
                best_path,
            )

    print(f"Saved SimCLR checkpoint -> {best_path.resolve()}")


if __name__ == "__main__":
    # Helps avoid OpenMP duplicate warnings on some Windows setups
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
