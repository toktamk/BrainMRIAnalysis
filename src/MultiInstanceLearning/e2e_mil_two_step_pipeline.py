# src/MultiInstanceLearning/e2e_mil_two_step_pipeline.py
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# plotting + metrics
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

# Local imports (your existing modules)
from MRIDataRead import MRIDataConfig, MRIBagTwoStepDataset
from models import TwoStepAttentionMIL


# -----------------------------
# Reproducibility
# -----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism note: can reduce speed / sometimes still non-deterministic on GPU ops.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Splitting
# -----------------------------
def make_splits(n: int, seed: int, train_frac: float, val_frac: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert 0 < train_frac < 1
    assert 0 < val_frac < 1
    assert train_frac + val_frac < 1

    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    tr = idx[:n_train]
    va = idx[n_train : n_train + n_val]
    te = idx[n_train + n_val :]
    return tr, va, te


# -----------------------------
# Training / Eval helpers
# -----------------------------
def train_one_epoch(
    model: TwoStepAttentionMIL,
    loader: DataLoader,
    device: torch.device,
    opt: torch.optim.Optimizer,
    *,
    w_type: float = 1.0,
    w_grade: float = 1.0,
    grad_clip: float | None = None,
) -> Dict[str, float]:
    model.train()

    losses: List[float] = []
    type_losses: List[float] = []
    grade_losses: List[float] = []

    for bag, y_type, y_grade, _meta in loader:
        bag = bag.to(device)
        y_type = y_type.to(device)
        y_grade = y_grade.to(device)

        type_logits, grade_logits = model(bag)

        loss_type = F.cross_entropy(type_logits, y_type)

        idx = torch.arange(bag.size(0), device=device)
        grade_logits_true = grade_logits[idx, y_type]  # teacher forcing
        loss_grade = F.cross_entropy(grade_logits_true, y_grade)

        loss = w_type * loss_type + w_grade * loss_grade

        opt.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))

        opt.step()

        losses.append(float(loss.item()))
        type_losses.append(float(loss_type.item()))
        grade_losses.append(float(loss_grade.item()))

    return {
        "loss": float(np.mean(losses)) if losses else math.nan,
        "loss_type": float(np.mean(type_losses)) if type_losses else math.nan,
        "loss_grade": float(np.mean(grade_losses)) if grade_losses else math.nan,
    }


@torch.no_grad()
def predict_all(
    model: TwoStepAttentionMIL,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Collect probabilities + predictions for:
      - tumor type (multi-class)
      - grade using teacher-forcing head (true type)
      - grade using end-to-end head (predicted type)

    Returns numpy arrays + metadata list.
    """
    model.eval()

    all_meta: List[Dict[str, Any]] = []

    y_type_true: List[int] = []
    y_type_pred: List[int] = []
    p_type: List[np.ndarray] = []

    y_grade_true: List[int] = []
    y_grade_pred_tf: List[int] = []
    y_grade_pred_e2e: List[int] = []
    p_grade_tf: List[np.ndarray] = []
    p_grade_e2e: List[np.ndarray] = []

    for bag, y_type, y_grade, meta in loader:
        bag = bag.to(device)
        y_type = y_type.to(device)
        y_grade = y_grade.to(device)

        type_logits, grade_logits = model(bag)

        # type
        prob_type = torch.softmax(type_logits, dim=1)
        pred_type = prob_type.argmax(dim=1)

        # grade: teacher forcing (true type)
        idx = torch.arange(bag.size(0), device=device)
        grade_logits_tf = grade_logits[idx, y_type]
        prob_grade_tf = torch.softmax(grade_logits_tf, dim=1)
        pred_grade_tf = prob_grade_tf.argmax(dim=1)

        # grade: end-to-end (pred type)
        grade_logits_e2e = grade_logits[idx, pred_type]
        prob_grade_e2e = torch.softmax(grade_logits_e2e, dim=1)
        pred_grade_e2e = prob_grade_e2e.argmax(dim=1)

        # stash
        y_type_true.extend(y_type.cpu().numpy().tolist())
        y_type_pred.extend(pred_type.cpu().numpy().tolist())
        p_type.extend(prob_type.cpu().numpy())

        y_grade_true.extend(y_grade.cpu().numpy().tolist())
        y_grade_pred_tf.extend(pred_grade_tf.cpu().numpy().tolist())
        y_grade_pred_e2e.extend(pred_grade_e2e.cpu().numpy().tolist())
        p_grade_tf.extend(prob_grade_tf.cpu().numpy())
        p_grade_e2e.extend(prob_grade_e2e.cpu().numpy())

        # meta is a dict-of-lists from default collate; convert to per-item dict
        # keys: patient_id, subtype, tumor_type, grade
        bsz = bag.size(0)
        for i in range(bsz):
            m = {k: meta[k][i] for k in meta.keys()}
            all_meta.append(m)

    return {
        "meta": all_meta,
        "y_type_true": np.asarray(y_type_true, dtype=int),
        "y_type_pred": np.asarray(y_type_pred, dtype=int),
        "p_type": np.asarray(p_type, dtype=float),
        "y_grade_true": np.asarray(y_grade_true, dtype=int),
        "y_grade_pred_tf": np.asarray(y_grade_pred_tf, dtype=int),
        "y_grade_pred_e2e": np.asarray(y_grade_pred_e2e, dtype=int),
        "p_grade_tf": np.asarray(p_grade_tf, dtype=float),
        "p_grade_e2e": np.asarray(p_grade_e2e, dtype=float),
    }


def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, digits=4, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# -----------------------------
# Plotting
# -----------------------------
def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    # annotate counts
    thresh = cm.max() * 0.5 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_multiclass_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    title: str,
    out_path: Path,
) -> Dict[str, float]:
    """
    One-vs-rest ROC. Returns per-class AUC + micro/macro AUC.
    """
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr: Dict[int, np.ndarray] = {}
    tpr: Dict[int, np.ndarray] = {}
    roc_auc: Dict[int, float] = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[i] = float(auc(fpr[i], tpr[i]))

    # micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    auc_micro = float(auc(fpr_micro, tpr_micro))

    # macro-average
    # aggregate all fpr
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= max(1, n_classes)
    auc_macro = float(auc(all_fpr, mean_tpr))

    fig = plt.figure(figsize=(9, 7))
    plt.title(title)
    plt.plot(fpr_micro, tpr_micro, label=f"micro-average (AUC={auc_micro:.3f})")
    plt.plot(all_fpr, mean_tpr, label=f"macro-average (AUC={auc_macro:.3f})")

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    out = {f"auc_{class_names[i]}": roc_auc[i] for i in range(n_classes)}
    out["auc_micro"] = auc_micro
    out["auc_macro"] = auc_macro
    return out


def plot_multiclass_pr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    title: str,
    out_path: Path,
) -> Dict[str, float]:
    """
    One-vs-rest Precision-Recall. Returns per-class AP + micro/macro AP.
    """
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    ap: Dict[int, float] = {}
    prec: Dict[int, np.ndarray] = {}
    rec: Dict[int, np.ndarray] = {}

    for i in range(n_classes):
        prec[i], rec[i], _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap[i] = float(average_precision_score(y_bin[:, i], y_prob[:, i]))

    # micro-average
    prec_micro, rec_micro, _ = precision_recall_curve(y_bin.ravel(), y_prob.ravel())
    ap_micro = float(average_precision_score(y_bin, y_prob, average="micro"))
    ap_macro = float(average_precision_score(y_bin, y_prob, average="macro"))

    fig = plt.figure(figsize=(9, 7))
    plt.title(title)
    plt.plot(rec_micro, prec_micro, label=f"micro-average (AP={ap_micro:.3f})")

    for i in range(n_classes):
        plt.plot(rec[i], prec[i], label=f"{class_names[i]} (AP={ap[i]:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    out = {f"ap_{class_names[i]}": ap[i] for i in range(n_classes)}
    out["ap_micro"] = ap_micro
    out["ap_macro"] = ap_macro
    return out


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser("End-to-end Two-Step MIL (type + grade) pipeline with reporting + plots")
    ap.add_argument("--data_root", type=str, required=True)

    ap.add_argument("--out_dir", type=str, default="runs/e2e_mil_two_step")
    ap.add_argument("--seed", type=int, default=7)

    # dataset / preprocessing
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--bag_size", type=int, default=8)
    ap.add_argument("--bag_policy", type=str, default="uniform", choices=["uniform", "first", "center"])
    ap.add_argument("--grayscale", action="store_true")

    # splits
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)

    # optimization
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=0.0)

    # model
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--attn_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)

    # loss weights
    ap.add_argument("--w_type", type=float, default=1.0)
    ap.add_argument("--w_grade", type=float, default=1.0)

    # checkpoint selection: metric to maximize on val
    ap.add_argument(
        "--select_by",
        type=str,
        default="val_end2end_all_correct",
        choices=[
            "val_type_acc",
            "val_grade_acc_given_true_type",
            "val_end2end_grade_acc",
            "val_end2end_all_correct",
        ],
    )

    # dataloader
    ap.add_argument("--num_workers", type=int, default=0)

    args = ap.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Data
    # -----------------------------
    cfg = MRIDataConfig(
        root=args.data_root,
        image_size=args.image_size,
        grayscale=bool(args.grayscale),
        mode="bag",
        bag_size=args.bag_size,
        bag_policy=args.bag_policy,
        seed=args.seed,
    )
    ds = MRIBagTwoStepDataset(cfg)

    type_classes: List[str] = list(ds.type_encoder.classes) if hasattr(ds.type_encoder, "classes") else []
    grade_classes: List[str] = list(ds.grade_encoder.classes) if hasattr(ds.grade_encoder, "classes") else ["g1", "g2", "g3", "g4"]

    tr_idx, va_idx, te_idx = make_splits(len(ds), args.seed, args.train_frac, args.val_frac)

    train_loader = DataLoader(
        Subset(ds, tr_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(ds, va_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        Subset(ds, te_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # -----------------------------
    # Model
    # -----------------------------
    num_types = len(ds.type_encoder)
    num_grades = len(ds.grade_encoder) if hasattr(ds.grade_encoder, "__len__") else 4

    model = TwoStepAttentionMIL(
        in_channels=1 if args.grayscale else 3,
        emb_dim=args.emb_dim,
        attn_dim=args.attn_dim,
        num_types=num_types,
        num_grades=num_grades,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # -----------------------------
    # Training loop + checkpoint
    # -----------------------------
    best_score = -1e9
    best_path = out_dir / "best.pt"
    history: List[Dict[str, Any]] = []

    def score_from_val(pred_pack: Dict[str, Any]) -> Dict[str, float]:
        ytt = pred_pack["y_type_true"]
        ytp = pred_pack["y_type_pred"]
        ygt = pred_pack["y_grade_true"]
        ygp_tf = pred_pack["y_grade_pred_tf"]
        ygp_e2e = pred_pack["y_grade_pred_e2e"]

        type_acc = float(accuracy_score(ytt, ytp))
        grade_tf_acc = float(accuracy_score(ygt, ygp_tf))
        grade_e2e_acc = float(accuracy_score(ygt, ygp_e2e))
        all_correct = float(np.mean((ytt == ytp) & (ygt == ygp_e2e)))

        return {
            "val_type_acc": type_acc,
            "val_grade_acc_given_true_type": grade_tf_acc,
            "val_end2end_grade_acc": grade_e2e_acc,
            "val_end2end_all_correct": all_correct,
        }

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model,
            train_loader,
            device,
            opt,
            w_type=args.w_type,
            w_grade=args.w_grade,
            grad_clip=(args.grad_clip if args.grad_clip > 0 else None),
        )

        # evaluate on train/val (cheap but informative)
        with torch.no_grad():
            pred_tr = predict_all(model, train_loader, device)
            pred_va = predict_all(model, val_loader, device)

        tr_scores = score_from_val(pred_tr)
        va_scores = score_from_val(pred_va)

        row = {
            "epoch": epoch,
            **tr_loss,
            **{f"train_{k.replace('val_', '')}": v for k, v in tr_scores.items()},
            **va_scores,
        }
        history.append(row)

        print(
            f"[epoch {epoch:03d}] "
            f"loss={row['loss']:.4f} "
            f"train(type={row['train_type_acc']:.3f}, grade|trueT={row['train_grade_acc_given_true_type']:.3f}, e2e_grade={row['train_end2end_grade_acc']:.3f}, all={row['train_end2end_all_correct']:.3f}) "
            f"val(type={row['val_type_acc']:.3f}, grade|trueT={row['val_grade_acc_given_true_type']:.3f}, e2e_grade={row['val_end2end_grade_acc']:.3f}, all={row['val_end2end_all_correct']:.3f})"
        )

        cur = float(row[args.select_by])
        if cur > best_score:
            best_score = cur
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_types": num_types,
                    "num_grades": num_grades,
                    "type_classes": type_classes,
                    "grade_classes": grade_classes,
                    "cfg": asdict(cfg),
                    "args": vars(args),
                    "best_score": best_score,
                    "best_by": args.select_by,
                },
                best_path,
            )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved training history -> {out_dir / 'history.json'}")
    print(f"Best checkpoint -> {best_path} (score={best_score:.6f}, by={args.select_by})")

    # -----------------------------
    # Load best + test inference
    # -----------------------------
    ck = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ck["model_state"], strict=True)
    model.to(device)

    pred_te = predict_all(model, test_loader, device)

    # metrics: type
    m_type = compute_basic_metrics(pred_te["y_type_true"], pred_te["y_type_pred"], type_classes)

    # metrics: grade
    # teacher-forced grade (true type head)
    m_grade_tf = compute_basic_metrics(pred_te["y_grade_true"], pred_te["y_grade_pred_tf"], grade_classes)
    # end-to-end grade (pred type head)
    m_grade_e2e = compute_basic_metrics(pred_te["y_grade_true"], pred_te["y_grade_pred_e2e"], grade_classes)

    # combined correctness
    end2end_all_correct = float(
        np.mean((pred_te["y_type_true"] == pred_te["y_type_pred"]) & (pred_te["y_grade_true"] == pred_te["y_grade_pred_e2e"]))
    )

    # ROC/PR plots (OvR)
    roc_type = plot_multiclass_roc(
        pred_te["y_type_true"],
        pred_te["p_type"],
        type_classes,
        title="ROC (Tumor Type) - One-vs-Rest",
        out_path=out_dir / "roc_type.png",
    )
    pr_type = plot_multiclass_pr(
        pred_te["y_type_true"],
        pred_te["p_type"],
        type_classes,
        title="Precision-Recall (Tumor Type) - One-vs-Rest",
        out_path=out_dir / "pr_type.png",
    )

    roc_grade_tf = plot_multiclass_roc(
        pred_te["y_grade_true"],
        pred_te["p_grade_tf"],
        grade_classes,
        title="ROC (Grade | True Type Head) - One-vs-Rest",
        out_path=out_dir / "roc_grade_teacher_forced.png",
    )
    pr_grade_tf = plot_multiclass_pr(
        pred_te["y_grade_true"],
        pred_te["p_grade_tf"],
        grade_classes,
        title="Precision-Recall (Grade | True Type Head) - One-vs-Rest",
        out_path=out_dir / "pr_grade_teacher_forced.png",
    )

    roc_grade_e2e = plot_multiclass_roc(
        pred_te["y_grade_true"],
        pred_te["p_grade_e2e"],
        grade_classes,
        title="ROC (Grade | Pred Type Head) - One-vs-Rest",
        out_path=out_dir / "roc_grade_end2end.png",
    )
    pr_grade_e2e = plot_multiclass_pr(
        pred_te["y_grade_true"],
        pred_te["p_grade_e2e"],
        grade_classes,
        title="Precision-Recall (Grade | Pred Type Head) - One-vs-Rest",
        out_path=out_dir / "pr_grade_end2end.png",
    )

    # Confusion matrices (PNGs)
    plot_confusion_matrix(
        np.asarray(m_type["confusion_matrix"]),
        type_classes,
        title="Confusion Matrix - Tumor Type (Test)",
        out_path=out_dir / "cm_type.png",
    )
    plot_confusion_matrix(
        np.asarray(m_grade_tf["confusion_matrix"]),
        grade_classes,
        title="Confusion Matrix - Grade (Teacher Forced) (Test)",
        out_path=out_dir / "cm_grade_teacher_forced.png",
    )
    plot_confusion_matrix(
        np.asarray(m_grade_e2e["confusion_matrix"]),
        grade_classes,
        title="Confusion Matrix - Grade (End-to-End) (Test)",
        out_path=out_dir / "cm_grade_end2end.png",
    )

    # Save predictions CSV
    # (avoid pandas dependency)
    pred_csv = out_dir / "test_predictions.csv"
    lines = []
    header = [
        "patient_id",
        "subtype",
        "tumor_type_str",
        "grade_int",
        "y_type_true",
        "y_type_pred",
        "y_grade_true",
        "y_grade_pred_tf",
        "y_grade_pred_e2e",
    ]
    # add probabilities columns
    header += [f"p_type_{c}" for c in type_classes]
    header += [f"p_grade_tf_{c}" for c in grade_classes]
    header += [f"p_grade_e2e_{c}" for c in grade_classes]
    lines.append(",".join(header))

    for i, meta in enumerate(pred_te["meta"]):
        row = [
            str(meta.get("patient_id", "")),
            str(meta.get("subtype", "")),
            str(meta.get("tumor_type", "")),
            str(meta.get("grade", "")),
            str(int(pred_te["y_type_true"][i])),
            str(int(pred_te["y_type_pred"][i])),
            str(int(pred_te["y_grade_true"][i])),
            str(int(pred_te["y_grade_pred_tf"][i])),
            str(int(pred_te["y_grade_pred_e2e"][i])),
        ]
        row += [f"{x:.6f}" for x in pred_te["p_type"][i].tolist()]
        row += [f"{x:.6f}" for x in pred_te["p_grade_tf"][i].tolist()]
        row += [f"{x:.6f}" for x in pred_te["p_grade_e2e"][i].tolist()]
        lines.append(",".join(row))

    pred_csv.write_text("\n".join(lines), encoding="utf-8")

    # Save final metrics JSON
    metrics = {
        "test": {
            "type": m_type,
            "grade_teacher_forced": m_grade_tf,
            "grade_end2end": m_grade_e2e,
            "end2end_all_correct": end2end_all_correct,
            "roc_auc_type": roc_type,
            "pr_ap_type": pr_type,
            "roc_auc_grade_teacher_forced": roc_grade_tf,
            "pr_ap_grade_teacher_forced": pr_grade_tf,
            "roc_auc_grade_end2end": roc_grade_e2e,
            "pr_ap_grade_end2end": pr_grade_e2e,
        },
        "artifacts": {
            "checkpoint": str(best_path),
            "predictions_csv": str(pred_csv),
            "plots": {
                "roc_type": str(out_dir / "roc_type.png"),
                "pr_type": str(out_dir / "pr_type.png"),
                "cm_type": str(out_dir / "cm_type.png"),
                "roc_grade_teacher_forced": str(out_dir / "roc_grade_teacher_forced.png"),
                "pr_grade_teacher_forced": str(out_dir / "pr_grade_teacher_forced.png"),
                "cm_grade_teacher_forced": str(out_dir / "cm_grade_teacher_forced.png"),
                "roc_grade_end2end": str(out_dir / "roc_grade_end2end.png"),
                "pr_grade_end2end": str(out_dir / "pr_grade_end2end.png"),
                "cm_grade_end2end": str(out_dir / "cm_grade_end2end.png"),
            },
        },
    }
    (out_dir / "metrics_test.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n==== TEST SUMMARY ====")
    print(f"type_acc            : {m_type['accuracy']:.4f}")
    print(f"grade_acc (TF)       : {m_grade_tf['accuracy']:.4f}")
    print(f"grade_acc (end2end)  : {m_grade_e2e['accuracy']:.4f}")
    print(f"all_correct (end2end): {end2end_all_correct:.4f}")
    print(f"\nSaved outputs under: {out_dir}")


if __name__ == "__main__":
    main()