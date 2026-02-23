from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

# Plots + metrics
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

# Your modules (same folder)
from MRIDataRead import MRIDataConfig, MRIBagTwoStepDataset
from graphs import SliceEncoder, slices_to_pyg_data
from models import TwoStepGNNClassifier, two_step_metrics


# -------------------------
# Repro / utils
# -------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_indices(
    n: int,
    seed: int,
    frac_train: float = 0.70,
    frac_val: float = 0.15,
) -> Tuple[List[int], List[int], List[int]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(n * frac_train))
    n_val = int(round(n * frac_val))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    train = idx[:n_train].tolist()
    val = idx[n_train : n_train + n_val].tolist()
    test = idx[n_train + n_val :].tolist()
    return train, val, test


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def make_inv_freq_weights(labels: List[int], n_classes: int) -> torch.Tensor:
    """
    Inverse-frequency weights normalized to mean=1.
    Useful for imbalanced classification with CrossEntropyLoss(weight=...).
    """
    c = Counter(labels)
    w = np.zeros(n_classes, dtype=np.float32)
    for k in range(n_classes):
        w[k] = 1.0 / max(1, c.get(k, 0))
    w = w / (w.mean() + 1e-8)
    return torch.tensor(w, dtype=torch.float32)


# -------------------------
# Plot helpers
# -------------------------
def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    thresh = cm.max() * 0.5 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_multiclass_roc(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str], title: str, out_path: Path) -> Dict:
    """
    One-vs-rest ROC. Returns dict with per-class AUC + micro/macro.
    """
    n_classes = y_prob.shape[1]
    Y = label_binarize(y_true, classes=list(range(n_classes)))

    roc_info = {"per_class_auc": {}, "micro_auc": None, "macro_auc": None}

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)

    aucs = []
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(Y[:, c], y_prob[:, c])
        c_auc = auc(fpr, tpr)
        aucs.append(c_auc)
        roc_info["per_class_auc"][class_names[c]] = float(c_auc)
        ax.plot(fpr, tpr, label=f"{class_names[c]} (AUC={c_auc:.3f})")

    fpr_m, tpr_m, _ = roc_curve(Y.ravel(), y_prob.ravel())
    micro_auc = auc(fpr_m, tpr_m)
    roc_info["micro_auc"] = float(micro_auc)
    ax.plot(fpr_m, tpr_m, linestyle="--", label=f"micro (AUC={micro_auc:.3f})")

    roc_info["macro_auc"] = float(np.mean(aucs)) if aucs else None

    ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return roc_info


def plot_multiclass_pr(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str], title: str, out_path: Path) -> Dict:
    """
    One-vs-rest Precision-Recall. Returns dict with per-class AP + micro/macro.
    """
    n_classes = y_prob.shape[1]
    Y = label_binarize(y_true, classes=list(range(n_classes)))

    pr_info = {"per_class_ap": {}, "micro_ap": None, "macro_ap": None}

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)

    aps = []
    for c in range(n_classes):
        prec, rec, _ = precision_recall_curve(Y[:, c], y_prob[:, c])
        ap = average_precision_score(Y[:, c], y_prob[:, c])
        aps.append(ap)
        pr_info["per_class_ap"][class_names[c]] = float(ap)
        ax.plot(rec, prec, label=f"{class_names[c]} (AP={ap:.3f})")

    prec_m, rec_m, _ = precision_recall_curve(Y.ravel(), y_prob.ravel())
    ap_m = average_precision_score(Y, y_prob, average="micro")
    pr_info["micro_ap"] = float(ap_m)
    ax.plot(rec_m, prec_m, linestyle="--", label=f"micro (AP={ap_m:.3f})")

    pr_info["macro_ap"] = float(np.mean(aps)) if aps else None

    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return pr_info


# -------------------------
# Graph batching (END-TO-END)
# -------------------------
def collate_to_pyg_batch(
    samples,
    *,
    node_encoder: torch.nn.Module,
    device: torch.device,
    no_grad: bool,
):
    """
    samples: list of (bag, y_type, y_grade, meta) from MRIBagTwoStepDataset.__getitem__

    IMPORTANT:
      - For training: no_grad=False so gradients flow into SliceEncoder.
      - For eval/test: no_grad=True.
    """
    data_list = []
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        for bag, y_type, y_grade, meta in samples:
            bag = bag.to(device)  # (S,C,H,W)
            g = slices_to_pyg_data(
                bag,
                encoder=node_encoder,   # encoder called HERE => end-to-end
                y_type=int(y_type.item()),
                y_grade=int(y_grade.item()),
                meta=meta,
                chain_edges=True,
            )
            data_list.append(g)
    return Batch.from_data_list(data_list).to(device)


@torch.no_grad()
def eval_split(
    ds: MRIBagTwoStepDataset,
    indices: List[int],
    node_encoder: torch.nn.Module,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    node_encoder.eval()
    model.eval()

    idx_loader = torch.utils.data.DataLoader(indices, batch_size=batch_size, shuffle=False, drop_last=False)
    mets_sum = {
        "type_acc": 0.0,
        "grade_acc_given_true_type": 0.0,
        "end2end_grade_acc": 0.0,
        "end2end_all_correct": 0.0,
    }
    n_batches = 0

    for batch_ids in idx_loader:
        samples = [ds[int(i)] for i in batch_ids]
        batch = collate_to_pyg_batch(samples, node_encoder=node_encoder, device=device, no_grad=True)
        type_logits, grade_logits = model(batch)
        m = two_step_metrics(type_logits, grade_logits, batch.y_type.view(-1), batch.y_grade.view(-1))
        for k in mets_sum:
            mets_sum[k] += float(m[k])
        n_batches += 1

    if n_batches > 0:
        for k in mets_sum:
            mets_sum[k] /= n_batches
    return mets_sum


@torch.no_grad()
def predict_split(
    ds: MRIBagTwoStepDataset,
    indices: List[int],
    node_encoder: torch.nn.Module,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    """
    Returns numpy arrays:
      y_type_true, y_grade_true
      y_type_pred, y_grade_pred_e2e
      type_prob (B,T)
      grade_prob_e2e (B,G)  # grade probabilities after choosing predicted type head
      meta_patient_id, meta_subtype, meta_tumor_type_str, meta_grade_int
    """
    node_encoder.eval()
    model.eval()

    idx_loader = torch.utils.data.DataLoader(indices, batch_size=batch_size, shuffle=False, drop_last=False)

    y_type_true, y_grade_true = [], []
    y_type_pred, y_grade_pred = [], []
    type_prob_all, grade_prob_all = [], []

    meta_patient_id, meta_subtype, meta_tumor_type_str, meta_grade_int = [], [], [], []

    for batch_ids in idx_loader:
        samples = [ds[int(i)] for i in batch_ids]
        batch = collate_to_pyg_batch(samples, node_encoder=node_encoder, device=device, no_grad=True)

        type_logits, grade_logits = model(batch)
        type_prob = torch.softmax(type_logits, dim=1)  # (B,T)
        type_pred = torch.argmax(type_prob, dim=1)     # (B,)

        B = type_pred.shape[0]
        chosen_grade_logits = grade_logits[torch.arange(B, device=device), type_pred]  # (B,G)
        grade_prob = torch.softmax(chosen_grade_logits, dim=1)
        grade_pred = torch.argmax(grade_prob, dim=1)

        y_type_true.append(batch.y_type.view(-1).cpu().numpy())
        y_grade_true.append(batch.y_grade.view(-1).cpu().numpy())
        y_type_pred.append(type_pred.cpu().numpy())
        y_grade_pred.append(grade_pred.cpu().numpy())
        type_prob_all.append(type_prob.cpu().numpy())
        grade_prob_all.append(grade_prob.cpu().numpy())

        data_list = batch.to_data_list()
        for d in data_list:
            m = getattr(d, "meta", {}) or {}
            meta_patient_id.append(str(m.get("patient_id", "")))
            meta_subtype.append(str(m.get("subtype", "")))
            meta_tumor_type_str.append(str(m.get("tumor_type", "")))
            meta_grade_int.append(int(m.get("grade", -1)))

    out = {
        "y_type_true": np.concatenate(y_type_true) if y_type_true else np.array([]),
        "y_grade_true": np.concatenate(y_grade_true) if y_grade_true else np.array([]),
        "y_type_pred": np.concatenate(y_type_pred) if y_type_pred else np.array([]),
        "y_grade_pred_e2e": np.concatenate(y_grade_pred) if y_grade_pred else np.array([]),
        "type_prob": np.concatenate(type_prob_all) if type_prob_all else np.zeros((0, len(ds.type_encoder))),
        "grade_prob_e2e": np.concatenate(grade_prob_all) if grade_prob_all else np.zeros((0, len(ds.grade_encoder))),
        "meta_patient_id": np.array(meta_patient_id),
        "meta_subtype": np.array(meta_subtype),
        "meta_tumor_type_str": np.array(meta_tumor_type_str),
        "meta_grade_int": np.array(meta_grade_int),
    }
    return out


def compute_and_save_reports(
    ds: MRIBagTwoStepDataset,
    pred: Dict[str, np.ndarray],
    out_dir: Path,
) -> Dict:
    ensure_dir(out_dir)

    type_classes = ds.type_encoder.classes
    grade_classes = ds.grade_encoder.classes  # ["g1","g2","g3","g4"]

    y_type_true = pred["y_type_true"].astype(int)
    y_type_pred = pred["y_type_pred"].astype(int)
    y_grade_true = pred["y_grade_true"].astype(int)
    y_grade_pred = pred["y_grade_pred_e2e"].astype(int)

    type_prob = pred["type_prob"]
    grade_prob = pred["grade_prob_e2e"]

    T = len(type_classes)
    G = len(grade_classes)

    type_metrics = {
        "acc": float(accuracy_score(y_type_true, y_type_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_type_true, y_type_pred)),
        "macro_f1": float(f1_score(y_type_true, y_type_pred, average="macro")),
        "micro_f1": float(f1_score(y_type_true, y_type_pred, average="micro")),
        "report": classification_report(
            y_type_true, y_type_pred, target_names=type_classes, output_dict=True, zero_division=0
        ),
    }

    grade_metrics = {
        "acc": float(accuracy_score(y_grade_true, y_grade_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_grade_true, y_grade_pred)),
        "macro_f1": float(f1_score(y_grade_true, y_grade_pred, average="macro")),
        "micro_f1": float(f1_score(y_grade_true, y_grade_pred, average="micro")),
        "report": classification_report(
            y_grade_true, y_grade_pred, target_names=grade_classes, output_dict=True, zero_division=0
        ),
    }

    joint_true = y_type_true * G + y_grade_true
    joint_pred = y_type_pred * G + y_grade_pred
    joint_names = [f"{type_classes[t]}-{grade_classes[g]}" for t in range(T) for g in range(G)]

    joint_metrics = {
        "acc": float(accuracy_score(joint_true, joint_pred)),
        "macro_f1": float(f1_score(joint_true, joint_pred, average="macro")),
        "micro_f1": float(f1_score(joint_true, joint_pred, average="micro")),
        "report": classification_report(
            joint_true, joint_pred, target_names=joint_names, output_dict=True, zero_division=0
        ),
    }

    cm_type = confusion_matrix(y_type_true, y_type_pred, labels=list(range(T)))
    plot_confusion_matrix(cm_type, type_classes, "Confusion Matrix: Tumor Type", out_dir / "cm_type.png")

    cm_grade = confusion_matrix(y_grade_true, y_grade_pred, labels=list(range(G)))
    plot_confusion_matrix(cm_grade, grade_classes, "Confusion Matrix: Grade (end-to-end)", out_dir / "cm_grade_e2e.png")

    cm_joint = confusion_matrix(joint_true, joint_pred, labels=list(range(T * G)))
    plot_confusion_matrix(cm_joint, joint_names, "Confusion Matrix: Joint (Type, Grade)", out_dir / "cm_joint_type_grade.png")

    roc_type = plot_multiclass_roc(
        y_true=y_type_true,
        y_prob=type_prob,
        class_names=type_classes,
        title="ROC: Tumor Type (OvR)",
        out_path=out_dir / "roc_type.png",
    )
    pr_type = plot_multiclass_pr(
        y_true=y_type_true,
        y_prob=type_prob,
        class_names=type_classes,
        title="Precision-Recall: Tumor Type (OvR)",
        out_path=out_dir / "pr_type.png",
    )

    roc_grade = plot_multiclass_roc(
        y_true=y_grade_true,
        y_prob=grade_prob,
        class_names=grade_classes,
        title="ROC: Grade (end-to-end, OvR)",
        out_path=out_dir / "roc_grade_e2e.png",
    )
    pr_grade = plot_multiclass_pr(
        y_true=y_grade_true,
        y_prob=grade_prob,
        class_names=grade_classes,
        title="Precision-Recall: Grade (end-to-end, OvR)",
        out_path=out_dir / "pr_grade_e2e.png",
    )

    csv_path = out_dir / "predictions_test.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(
            "patient_id,subtype,true_type,true_grade,true_type_idx,true_grade_idx,"
            "pred_type,pred_grade,pred_type_idx,pred_grade_idx\n"
        )
        for i in range(len(y_type_true)):
            true_type_str = type_classes[y_type_true[i]] if 0 <= y_type_true[i] < T else ""
            true_grade_str = grade_classes[y_grade_true[i]] if 0 <= y_grade_true[i] < G else ""
            pred_type_str = type_classes[y_type_pred[i]] if 0 <= y_type_pred[i] < T else ""
            pred_grade_str = grade_classes[y_grade_pred[i]] if 0 <= y_grade_pred[i] < G else ""
            f.write(
                f"{pred['meta_patient_id'][i]},{pred['meta_subtype'][i]},"
                f"{true_type_str},{true_grade_str},{y_type_true[i]},{y_grade_true[i]},"
                f"{pred_type_str},{pred_grade_str},{y_type_pred[i]},{y_grade_pred[i]}\n"
            )

    summary = {
        "type_metrics": type_metrics,
        "grade_metrics_e2e": grade_metrics,
        "joint_metrics": joint_metrics,
        "roc_type": roc_type,
        "pr_type": pr_type,
        "roc_grade_e2e": roc_grade,
        "pr_grade_e2e": pr_grade,
        "artifacts": {
            "cm_type": "cm_type.png",
            "cm_grade_e2e": "cm_grade_e2e.png",
            "cm_joint": "cm_joint_type_grade.png",
            "roc_type": "roc_type.png",
            "pr_type": "pr_type.png",
            "roc_grade_e2e": "roc_grade_e2e.png",
            "pr_grade_e2e": "pr_grade_e2e.png",
            "predictions_csv": "predictions_test.csv",
        },
    }

    (out_dir / "test_reports.json").write_text(json.dumps(summary, indent=2))
    return summary


# -------------------------
# Main
# -------------------------
def main() -> None:
    p = argparse.ArgumentParser("End-to-end two-step GNN pipeline: type -> grade")
    p.add_argument("--data_root", required=True, type=str, help="Dataset root containing <tumor>-g<1..4>/P*/ slices")
    p.add_argument("--out_dir", type=str, default="runs/gnn_two_step_e2e")

    # data
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--grayscale", action="store_true")
    p.add_argument("--bag_size", type=int, default=8)
    p.add_argument("--bag_policy", type=str, default="uniform", choices=["uniform", "first", "center"])

    # split
    p.add_argument("--frac_train", type=float, default=0.70)
    p.add_argument("--frac_val", type=float, default=0.15)

    # train
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=7)

    # model
    p.add_argument("--node_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lambda_type", type=float, default=1.0)
    p.add_argument("--lambda_grade", type=float, default=1.0)

    # debug
    p.add_argument("--overfit_n", type=int, default=0, help="If >0, train/val/test all use same first N train samples.")
    p.add_argument("--debug_grad", action="store_true", help="Print encoder grad norm on epoch1,batch1.")
    p.add_argument("--save_val_preds_each_epoch", action="store_true", help="Save val predictions CSV each epoch.")

    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

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

    n = len(ds)
    tr_idx, va_idx, te_idx = split_indices(n, args.seed, args.frac_train, args.frac_val)

    if args.overfit_n and args.overfit_n > 0:
        small = tr_idx[: args.overfit_n]
        tr_idx = small
        va_idx = small
        te_idx = small
        print(f"[overfit] using {len(small)} samples for train/val/test")

    (out_dir / "split.json").write_text(json.dumps({"train": tr_idx, "val": va_idx, "test": te_idx}, indent=2))

    # Compute class weights from TRAIN split (imbalanced-data friendly)
    y_type_tr = [int(ds[i][1].item()) for i in tr_idx]
    y_grade_tr = [int(ds[i][2].item()) for i in tr_idx]
    w_type = make_inv_freq_weights(y_type_tr, len(ds.type_encoder)).to(device)
    w_grade = make_inv_freq_weights(y_grade_tr, len(ds.grade_encoder)).to(device)

    print("Train type counts:", Counter(y_type_tr))
    print("Train grade counts:", Counter(y_grade_tr))
    print("Type classes:", ds.type_encoder.classes)
    print("Grade classes:", ds.grade_encoder.classes)
    print("Type weights:", w_type.detach().cpu().numpy().round(3).tolist())
    print("Grade weights:", w_grade.detach().cpu().numpy().round(3).tolist())

    in_ch = 1 if args.grayscale else 3
    node_encoder = SliceEncoder(in_channels=in_ch, out_dim=args.node_dim).to(device)
    model = TwoStepGNNClassifier(
        in_dim=args.node_dim,
        hidden_dim=args.hidden_dim,
        num_types=len(ds.type_encoder),
        num_grades=len(ds.grade_encoder),
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(
        list(node_encoder.parameters()) + list(model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = -1.0
    best_path = out_dir / "best.pt"
    history = {"train_loss": [], "val": []}

    tr_index_loader = torch.utils.data.DataLoader(tr_idx, batch_size=args.batch_size, shuffle=True, drop_last=False)

    for epoch in range(1, args.epochs + 1):
        node_encoder.train()
        model.train()

        total_loss = 0.0
        n_batches = 0

        for batch_ids in tr_index_loader:
            samples = [ds[int(i)] for i in batch_ids]
            batch = collate_to_pyg_batch(samples, node_encoder=node_encoder, device=device, no_grad=False)

            type_logits, grade_logits = model(batch)

            # Weighted type CE
            y_type = batch.y_type.view(-1)
            y_grade = batch.y_grade.view(-1)

            loss_type = F.cross_entropy(type_logits, y_type, weight=w_type)

            # Teacher-forced grade CE (use TRUE type head for stability)
            B = y_type.shape[0]
            chosen_grade_logits = grade_logits[torch.arange(B, device=device), y_type]  # (B,G)
            loss_grade = F.cross_entropy(chosen_grade_logits, y_grade, weight=w_grade)

            loss = args.lambda_type * loss_type + args.lambda_grade * loss_grade

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if args.debug_grad and epoch == 1 and n_batches == 0:
                gnorm = 0.0
                for p in node_encoder.parameters():
                    if p.grad is not None:
                        gnorm += float(p.grad.detach().data.norm(2).item())
                print(f"[debug] encoder grad norm (epoch1,batch1): {gnorm:.6f}")

            opt.step()

            total_loss += float(loss.item())
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)

        val_m = eval_split(ds, va_idx, node_encoder, model, device, args.batch_size)
        val_score = float(val_m["end2end_all_correct"])  # strict: both correct

        history["train_loss"].append(avg_loss)
        history["val"].append(val_m)

        print(
            f"[epoch {epoch:03d}] loss={avg_loss:.4f} "
            f"val_type={val_m['type_acc']:.3f} "
            f"val_gradeTF={val_m['grade_acc_given_true_type']:.3f} "
            f"val_gradeE2E={val_m['end2end_grade_acc']:.3f} "
            f"val_all={val_m['end2end_all_correct']:.3f}"
        )

        if args.save_val_preds_each_epoch:
            # Save val predictions each epoch for inspection
            pred_val = predict_split(ds, va_idx, node_encoder, model, device, args.batch_size)
            val_csv = out_dir / f"predictions_val_epoch_{epoch:03d}.csv"
            type_classes = ds.type_encoder.classes
            grade_classes = ds.grade_encoder.classes
            T = len(type_classes)
            G = len(grade_classes)
            with val_csv.open("w", encoding="utf-8") as f:
                f.write(
                    "patient_id,subtype,true_type,true_grade,true_type_idx,true_grade_idx,"
                    "pred_type,pred_grade,pred_type_idx,pred_grade_idx\n"
                )
                yt = pred_val["y_type_true"].astype(int)
                yg = pred_val["y_grade_true"].astype(int)
                ytp = pred_val["y_type_pred"].astype(int)
                ygp = pred_val["y_grade_pred_e2e"].astype(int)
                for i in range(len(yt)):
                    true_type_str = type_classes[yt[i]] if 0 <= yt[i] < T else ""
                    true_grade_str = grade_classes[yg[i]] if 0 <= yg[i] < G else ""
                    pred_type_str = type_classes[ytp[i]] if 0 <= ytp[i] < T else ""
                    pred_grade_str = grade_classes[ygp[i]] if 0 <= ygp[i] < G else ""
                    f.write(
                        f"{pred_val['meta_patient_id'][i]},{pred_val['meta_subtype'][i]},"
                        f"{true_type_str},{true_grade_str},{yt[i]},{yg[i]},"
                        f"{pred_type_str},{pred_grade_str},{ytp[i]},{ygp[i]}\n"
                    )

        if val_score > best_val:
            best_val = val_score
            torch.save(
                {
                    "node_encoder": node_encoder.state_dict(),
                    "model": model.state_dict(),
                    "type_classes": ds.type_encoder.classes,
                    "grade_classes": ds.grade_encoder.classes,
                    "cfg": asdict(cfg),
                    "seed": args.seed,
                    "epoch": epoch,
                    "best_val_end2end_all_correct": best_val,
                    "args": vars(args),
                    "train_type_counts": dict(Counter(y_type_tr)),
                    "train_grade_counts": dict(Counter(y_grade_tr)),
                    "type_weights": w_type.detach().cpu().tolist(),
                    "grade_weights": w_grade.detach().cpu().tolist(),
                },
                best_path,
            )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best checkpoint -> {best_path} (best_val_all={best_val:.4f})")

    ckpt = torch.load(best_path, map_location="cpu")
    node_encoder.load_state_dict(ckpt["node_encoder"], strict=False)
    model.load_state_dict(ckpt["model"], strict=False)
    node_encoder.to(device)
    model.to(device)

    test_metrics_fast = eval_split(ds, te_idx, node_encoder, model, device, args.batch_size)
    print("[test quick metrics]", test_metrics_fast)
    (out_dir / "test_quick_metrics.json").write_text(json.dumps(test_metrics_fast, indent=2))

    pred = predict_split(ds, te_idx, node_encoder, model, device, args.batch_size)
    reports = compute_and_save_reports(ds, pred, out_dir)

    print("Saved artifacts:")
    for k, v in reports["artifacts"].items():
        print(f" - {k}: {out_dir / v}")


if __name__ == "__main__":
    main()
