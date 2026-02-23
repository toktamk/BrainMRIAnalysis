from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from MRIData import MRIDataConfig, MRIBagDataset
from losses import two_step_loss, two_step_metrics
from models import TwoStepClassifier


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def patient_pool_logits(type_logits_slices: torch.Tensor, grade_logits_slices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pools  slice-level logits to patient-level logits by mean over slices.

    Inputs:
      type_logits_slices:  (B, N, T)
      grade_logits_slices: (B, N, T, G)

    Returns:
      type_logits:  (B, T)
      grade_logits: (B, T, G)
    """
    type_logits = type_logits_slices.mean(dim=1)
    grade_logits = grade_logits_slices.mean(dim=1)
    return type_logits, grade_logits


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--outdir", type=str, default="runs/two_step")
    p.add_argument("--seed", type=int, default=7)

    # data
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--bag_size", type=int, default=8)
    p.add_argument("--bag_policy", type=str, default="uniform", choices=["uniform", "first", "center"])
    p.add_argument("--grayscale", action="store_true")

    # training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--type_weight", type=float, default=1.0)
    p.add_argument("--grade_weight", type=float, default=1.0)

    # model
    p.add_argument("--backbone", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "small_cnn"])
    p.add_argument("--pretrained", action="store_true")

    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = MRIDataConfig(
        root=args.data_root,
        image_size=args.image_size,
        grayscale=args.grayscale,
        mode="bag",
        target="two_step",
        bag_size=args.bag_size,
        bag_policy=args.bag_policy,
        seed=args.seed,
    )

    ds = MRIBagDataset(cfg)
    # Simple splitting by patient index (deterministic). Replace with patient-level stratified split later.
    n = len(ds)
    idx = np.arange(n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    n_train = int(round(0.7 * n))
    n_val = int(round(0.15 * n))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train_ds = torch.utils.data.Subset(ds, train_idx.tolist())
    val_ds = torch.utils.data.Subset(ds, val_idx.tolist())
    test_ds = torch.utils.data.Subset(ds, test_idx.tolist())

    # get label spaces
    num_types = len(ds.type_encoder)
    num_grades = len(ds.grade_encoder)

    model = TwoStepClassifier(
        num_types=num_types,
        num_grades=num_grades,
        backbone=args.backbone,
        pretrained=args.pretrained,
        in_channels=1 if args.grayscale else 3,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Save config
    with (outdir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "data_cfg": asdict(cfg), "num_types": num_types, "num_grades": num_grades}, f, indent=2)

    best_val = -1.0
    best_path = outdir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for bag, y_type, y_grade, _meta in train_loader:
            # bag: (B, N, C, H, W) -> slice batch: (B*N, C, H, W)
            B, N = bag.shape[0], bag.shape[1]
            bag = bag.to(device)
            y_type = y_type.to(device)
            y_grade = y_grade.to(device)

            x = bag.view(B * N, *bag.shape[2:])  # (B*N, C,H,W)
            type_logits_s, grade_logits_s = model(x)  # (B*N,T), (B*N,T,G)
            type_logits_s = type_logits_s.view(B, N, -1)
            grade_logits_s = grade_logits_s.view(B, N, num_types, num_grades)

            type_logits, grade_logits = patient_pool_logits(type_logits_s, grade_logits_s)

            loss = two_step_loss(
                type_logits,
                grade_logits,
                y_type,
                y_grade,
                type_weight=args.type_weight,
                grade_weight=args.grade_weight,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())

        # val
        model.eval()
        metrics_accum = {"type_acc": 0.0, "grade_acc_given_true_type": 0.0, "end2end_grade_acc": 0.0, "end2end_all_correct": 0.0}
        n_batches = 0

        with torch.no_grad():
            for bag, y_type, y_grade, _meta in val_loader:
                B, N = bag.shape[0], bag.shape[1]
                bag = bag.to(device)
                y_type = y_type.to(device)
                y_grade = y_grade.to(device)

                x = bag.view(B * N, *bag.shape[2:])
                type_logits_s, grade_logits_s = model(x)
                type_logits_s = type_logits_s.view(B, N, -1)
                grade_logits_s = grade_logits_s.view(B, N, num_types, num_grades)
                type_logits, grade_logits = patient_pool_logits(type_logits_s, grade_logits_s)

                m = two_step_metrics(type_logits, grade_logits, y_type, y_grade)
                for k in metrics_accum:
                    metrics_accum[k] += m[k]
                n_batches += 1

        for k in metrics_accum:
            metrics_accum[k] /= max(1, n_batches)

        val_score = metrics_accum["end2end_all_correct"]  # strict metric: both correct
        avg_loss = running / max(1, len(train_loader))
        print(f"[epoch {epoch:03d}] train_loss={avg_loss:.4f} val={metrics_accum}")

        if val_score > best_val:
            best_val = val_score
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": metrics_accum}, best_path)

    print(f"Best checkpoint: {best_path} (best end2end_all_correct={best_val:.4f})")

    # test
    ck = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    metrics_accum = {"type_acc": 0.0, "grade_acc_given_true_type": 0.0, "end2end_grade_acc": 0.0, "end2end_all_correct": 0.0}
    n_batches = 0
    with torch.no_grad():
        for bag, y_type, y_grade, _meta in test_loader:
            B, N = bag.shape[0], bag.shape[1]
            bag = bag.to(device)
            y_type = y_type.to(device)
            y_grade = y_grade.to(device)

            x = bag.view(B * N, *bag.shape[2:])
            type_logits_s, grade_logits_s = model(x)
            type_logits_s = type_logits_s.view(B, N, -1)
            grade_logits_s = grade_logits_s.view(B, N, num_types, num_grades)
            type_logits, grade_logits = patient_pool_logits(type_logits_s, grade_logits_s)

            m = two_step_metrics(type_logits, grade_logits, y_type, y_grade)
            for k in metrics_accum:
                metrics_accum[k] += m[k]
            n_batches += 1

    for k in metrics_accum:
        metrics_accum[k] /= max(1, n_batches)

    print(f"[TEST] {metrics_accum}")
    with (outdir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_accum, f, indent=2)


if __name__ == "__main__":
    main()
