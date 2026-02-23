from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from MRIData import MRIDataConfig, MRIBagDataset, default_transform
from models import TwoStepClassifier
from losses import two_step_loss, two_step_metrics


# -------------------------
# Reproducibility.
# -------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Robust patient-level split
# -------------------------
def _min_count(y) -> int:
    c = Counter(y)
    return min(c.values()) if len(c) else 0


def make_patient_splits(
    ds: MRIBagDataset,
    seed: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(ds)
    idx = np.arange(n)

    # ds.patients: (subtype_name, patient_dir, tumor_type, tumor_grade)
    y_subtype = np.array([ds.patients[i][0] for i in range(n)], dtype=object)
    y_type = np.array([ds.patients[i][2] for i in range(n)], dtype=object)   # can be string
    y_grade = np.array([ds.patients[i][3] for i in range(n)], dtype=object)  # can be int or string
    y_typegrade = np.array([f"{t}_{g}" for t, g in zip(y_type, y_grade)], dtype=object)

    def _split_once(X_idx, y, test_size):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        a, b = next(sss.split(X_idx, y))
        return X_idx[a], X_idx[b]

    def _try_two_stage(y_full):
        train_idx, temp_idx = _split_once(idx, y_full, test_size=(1.0 - train_frac))
        temp_y = y_full[temp_idx]
        val_rel = val_frac / max(1e-8, (1.0 - train_frac))
        val_idx, test_idx = _split_once(temp_idx, temp_y, test_size=(1.0 - val_rel))
        return train_idx, val_idx, test_idx

    candidates = [("subtype", y_subtype), ("type+grade", y_typegrade), ("type", y_type)]
    for _name, y in candidates:
        if _min_count(y) < 2:
            continue
        try:
            return _try_two_stage(y)
        except ValueError:
            continue

    # random fallback
    rng = np.random.default_rng(seed)
    perm = rng.permutation(idx)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx


# -------------------------
# Pooling and probabilities
# -------------------------
def patient_pool_logits(
    type_logits_slices: torch.Tensor,   # (B, N, T)
    grade_logits_slices: torch.Tensor,  # (B, N, T, G)
) -> Tuple[torch.Tensor, torch.Tensor]:
    return type_logits_slices.mean(dim=1), grade_logits_slices.mean(dim=1)


@torch.no_grad()
def end2end_grade_probs(
    type_logits: torch.Tensor,    # (B, T)
    grade_logits: torch.Tensor,   # (B, T, G)
) -> torch.Tensor:
    """
    p(g) = sum_t p(t) * p(g|t)
    """
    p_type = F.softmax(type_logits, dim=1)                 # (B, T)
    p_grade_given_type = F.softmax(grade_logits, dim=2)    # (B, T, G)
    return torch.einsum("bt,btg->bg", p_type, p_grade_given_type)  # (B, G)


# -------------------------
# SSL -> backbone loader
# -------------------------
def load_simclr_encoder_into_twostep(model: TwoStepClassifier, ssl_ckpt_path: str, strict: bool = False) -> None:
    """
    Loads encoder weights from SimCLR checkpoint into model.backbone.features.
    Expects checkpoint saved by pretrain_simclr.py:
        torch.save({"simclr_state": simclr.state_dict(), ...}, path)
    And simclr.state_dict contains keys like "encoder.0.weight", "projection.0.weight", ...
    """
    ckpt = torch.load(ssl_ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "simclr_state" in ckpt:
        sd = ckpt["simclr_state"]
    elif isinstance(ckpt, dict):
        # maybe user saved raw state_dict directly
        sd = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    enc = {}
    for k, v in sd.items():
        if k.startswith("encoder."):
            enc[k[len("encoder."):]] = v

    if not enc:
        raise ValueError("No keys starting with 'encoder.' found in SSL checkpoint.")

    missing, unexpected = model.backbone.features.load_state_dict(enc, strict=strict)
    print(f"[SSL] Loaded encoder into TwoStep backbone from: {ssl_ckpt_path}")
    if (missing or unexpected) and not strict:
        print(f"[SSL] missing keys (showing up to 10): {missing[:10]}")
        print(f"[SSL] unexpected keys (showing up to 10): {unexpected[:10]}")


# -------------------------
# Train / Eval loops
# -------------------------
def train_one_epoch(
    model: TwoStepClassifier,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    num_types: int,
    num_grades: int,
    type_weight: float,
    grade_weight: float,
) -> float:
    model.train()
    running = 0.0
    n_steps = 0

    for bag, y_type, y_grade, _meta in loader:
        B, N = bag.shape[0], bag.shape[1]
        bag = bag.to(device, non_blocking=True)
        y_type = y_type.to(device, non_blocking=True)
        y_grade = y_grade.to(device, non_blocking=True)

        x = bag.view(B * N, *bag.shape[2:])
        type_logits_s, grade_logits_s = model(x)

        type_logits_s = type_logits_s.view(B, N, num_types)
        grade_logits_s = grade_logits_s.view(B, N, num_types, num_grades)

        type_logits, grade_logits = patient_pool_logits(type_logits_s, grade_logits_s)

        loss = two_step_loss(
            type_logits, grade_logits, y_type, y_grade,
            type_weight=type_weight, grade_weight=grade_weight
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        running += float(loss.item())
        n_steps += 1

    return running / max(1, n_steps)


@torch.no_grad()
def eval_split(
    model: TwoStepClassifier,
    loader: DataLoader,
    device: torch.device,
    num_types: int,
    num_grades: int,
) -> dict:
    model.eval()
    metrics_accum = {"type_acc": 0.0, "grade_acc_given_true_type": 0.0, "end2end_grade_acc": 0.0, "end2end_all_correct": 0.0}
    n_batches = 0

    for bag, y_type, y_grade, _meta in loader:
        B, N = bag.shape[0], bag.shape[1]
        bag = bag.to(device, non_blocking=True)
        y_type = y_type.to(device, non_blocking=True)
        y_grade = y_grade.to(device, non_blocking=True)

        x = bag.view(B * N, *bag.shape[2:])
        type_logits_s, grade_logits_s = model(x)

        type_logits_s = type_logits_s.view(B, N, num_types)
        grade_logits_s = grade_logits_s.view(B, N, num_types, num_grades)
        type_logits, grade_logits = patient_pool_logits(type_logits_s, grade_logits_s)

        m = two_step_metrics(type_logits, grade_logits, y_type, y_grade)
        for k in metrics_accum:
            metrics_accum[k] += m[k]
        n_batches += 1

    for k in metrics_accum:
        metrics_accum[k] /= max(1, n_batches)

    return metrics_accum


@torch.no_grad()
def infer_test(
    model: TwoStepClassifier,
    loader: DataLoader,
    device: torch.device,
    num_types: int,
    num_grades: int,
) -> dict:
    model.eval()

    y_true_type, y_true_grade = [], []
    p_type_all, p_grade_all = [], []

    for bag, y_type, y_grade, _meta in loader:
        B, N = bag.shape[0], bag.shape[1]
        bag = bag.to(device, non_blocking=True)
        y_type = y_type.to(device, non_blocking=True)
        y_grade = y_grade.to(device, non_blocking=True)

        x = bag.view(B * N, *bag.shape[2:])
        type_logits_s, grade_logits_s = model(x)

        type_logits_s = type_logits_s.view(B, N, num_types)
        grade_logits_s = grade_logits_s.view(B, N, num_types, num_grades)
        type_logits, grade_logits = patient_pool_logits(type_logits_s, grade_logits_s)

        p_type = F.softmax(type_logits, dim=1)
        p_grade = end2end_grade_probs(type_logits, grade_logits)

        y_true_type.extend(y_type.cpu().tolist())
        y_true_grade.extend(y_grade.cpu().tolist())
        p_type_all.append(p_type.cpu().numpy())
        p_grade_all.append(p_grade.cpu().numpy())

    y_true_type = np.asarray(y_true_type, dtype=int)
    y_true_grade = np.asarray(y_true_grade, dtype=int)
    p_type = np.concatenate(p_type_all, axis=0)
    p_grade = np.concatenate(p_grade_all, axis=0)

    return {
        "y_true_type": y_true_type,
        "y_true_grade": y_true_grade,
        "p_type": p_type,
        "p_grade": p_grade,
        "type_pred": p_type.argmax(axis=1),
        "grade_pred": p_grade.argmax(axis=1),
    }


# -------------------------
# Plotting helpers
# -------------------------
def plot_multiclass_ovr_roc(y_true: np.ndarray, y_score: np.ndarray, class_names: List[str], out_png: Path, title: str) -> Dict[str, float]:
    C = y_score.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(C)))

    aucs: Dict[str, float] = {}
    plt.figure()
    for c in range(C):
        if y_bin[:, c].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_score[:, c])
        roc_auc = auc(fpr, tpr)
        aucs[class_names[c]] = float(roc_auc)
        plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return aucs


def plot_multiclass_ovr_pr(y_true: np.ndarray, y_score: np.ndarray, class_names: List[str], out_png: Path, title: str) -> Dict[str, float]:
    C = y_score.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(C)))

    aps: Dict[str, float] = {}
    plt.figure()
    for c in range(C):
        if y_bin[:, c].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin[:, c], y_score[:, c])
        ap = average_precision_score(y_bin[:, c], y_score[:, c])
        aps[class_names[c]] = float(ap)
        plt.plot(rec, prec, label=f"{class_names[c]} (AP={ap:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return aps


# -------------------------
# Main
# -------------------------
def main() -> None:
    p = argparse.ArgumentParser()

    # paths
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--outdir", type=str, default="runs/e2e_two_step_ssl")

    # data
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--bag_size", type=int, default=8)
    p.add_argument("--bag_policy", type=str, default="uniform", choices=["uniform", "first", "center"])
    p.add_argument("--grayscale", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)

    # split
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--train_frac", type=float, default=0.70)
    p.add_argument("--val_frac", type=float, default=0.15)

    # training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--type_weight", type=float, default=1.0)
    p.add_argument("--grade_weight", type=float, default=1.0)
    p.add_argument("--early_stop_patience", type=int, default=8)

    # model
    p.add_argument("--backbone", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "small_cnn"])
    p.add_argument("--pretrained", action="store_true", help="ImageNet pretrained (not SSL).")

    # SSL loading
    p.add_argument("--ssl_ckpt", type=str, default=None, help="Path to SimCLR checkpoint (.pt) from pretrain_simclr.py")
    p.add_argument("--freeze_backbone", action="store_true")

    args = p.parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # dataset
    transform = default_transform(args.image_size)
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
    ds = MRIBagDataset(cfg, transform=transform)

    train_idx, val_idx, test_idx = make_patient_splits(ds, seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac)
    train_ds = Subset(ds, train_idx.tolist())
    val_ds = Subset(ds, val_idx.tolist())
    test_ds = Subset(ds, test_idx.tolist())

    num_types = len(ds.type_encoder)
    num_grades = len(ds.grade_encoder)
    type_names = [ds.type_encoder.decode(i) for i in range(num_types)]
    grade_names = [ds.grade_encoder.decode(i) for i in range(num_grades)]

    # model
    model = TwoStepClassifier(
        num_types=num_types,
        num_grades=num_grades,
        backbone=args.backbone,
        pretrained=args.pretrained,
        in_channels=1 if args.grayscale else 3,
    ).to(device)

    # load SSL weights if provided
    if args.ssl_ckpt:
        load_simclr_encoder_into_twostep(model, args.ssl_ckpt, strict=False)
        if args.freeze_backbone:
            for p_ in model.backbone.parameters():
                p_.requires_grad = False
            print("[SSL] Backbone frozen.")

    # loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    # optimizer
    opt = torch.optim.AdamW(filter(lambda p_: p_.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)

    # save config
    with (outdir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "data_cfg": asdict(cfg),
                "splits": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
                "num_types": num_types,
                "num_grades": num_grades,
                "type_names": type_names,
                "grade_names": grade_names,
            },
            f,
            indent=2,
        )

    # training with early stop on end2end_all_correct
    best_val = -1.0
    best_path = outdir / "best.pt"
    patience = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, device, num_types, num_grades, args.type_weight, args.grade_weight)
        val_metrics = eval_split(model, val_loader, device, num_types, num_grades)
        val_score = val_metrics["end2end_all_correct"]

        print(f"[sup] epoch {epoch:03d}/{args.epochs} train_loss={tr_loss:.4f} val={val_metrics}")

        if val_score > best_val:
            best_val = val_score
            patience = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val_metrics}, best_path)
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                print(f"Early stopping. Best val end2end_all_correct={best_val:.4f}")
                break

    # test
    ck = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    test_metrics = eval_split(model, test_loader, device, num_types, num_grades)
    with (outdir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"[TEST] {test_metrics}")

    inf = infer_test(model, test_loader, device, num_types, num_grades)
    y_true_type = inf["y_true_type"]
    y_true_grade = inf["y_true_grade"]
    p_type = inf["p_type"]
    p_grade = inf["p_grade"]
    type_pred = inf["type_pred"]
    grade_pred = inf["grade_pred"]

    report = {
        "type_accuracy": float(accuracy_score(y_true_type, type_pred)),
        "type_macro_f1": float(f1_score(y_true_type, type_pred, average="macro")),
        "grade_accuracy_e2e": float(accuracy_score(y_true_grade, grade_pred)),
        "grade_macro_f1_e2e": float(f1_score(y_true_grade, grade_pred, average="macro")),
        "confusion_type": confusion_matrix(y_true_type, type_pred).tolist(),
        "confusion_grade_e2e": confusion_matrix(y_true_grade, grade_pred).tolist(),
        "classification_report_type": classification_report(y_true_type, type_pred, target_names=type_names, output_dict=True, zero_division=0),
        "classification_report_grade_e2e": classification_report(y_true_grade, grade_pred, target_names=grade_names, output_dict=True, zero_division=0),
    }
    with (outdir / "test_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # curves
    aucs_type = plot_multiclass_ovr_roc(y_true_type, p_type, type_names, outdir / "roc_tumor_type_ovr.png", "ROC (OvR): Tumor Type")
    aps_type = plot_multiclass_ovr_pr(y_true_type, p_type, type_names, outdir / "pr_tumor_type_ovr.png", "PR (OvR): Tumor Type")

    aucs_grade = plot_multiclass_ovr_roc(y_true_grade, p_grade, grade_names, outdir / "roc_grade_e2e_ovr.png", "ROC (OvR): Grade (E2E mixture)")
    aps_grade = plot_multiclass_ovr_pr(y_true_grade, p_grade, grade_names, outdir / "pr_grade_e2e_ovr.png", "PR (OvR): Grade (E2E mixture)")

    with (outdir / "curves_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "type_auc_ovr": aucs_type,
                "type_ap_ovr": aps_type,
                "grade_auc_e2e_ovr": aucs_grade,
                "grade_ap_e2e_ovr": aps_grade,
            },
            f,
            indent=2,
        )

    print(f"Saved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
