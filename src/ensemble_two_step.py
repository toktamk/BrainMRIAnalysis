# src/MultiInstanceLearning/ensemble_two_step.py
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

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

# ---- Your project imports ----
# Attention MIL dataset/model (used in e2e_mil_two_step_pipeline.py)
from MultiInstanceLearning.MRIDataRead import MRIDataConfig as MRIDataConfigRead, MRIBagTwoStepDataset
from MultiInstanceLearning.models import TwoStepAttentionMIL

# GNN pipeline components (used in e2e_two_step_pipeline.py)
from GraphNeuralNetworks.graphs import SliceEncoder, slices_to_pyg_data
from GraphNeuralNetworks.models import TwoStepGNNClassifier
from torch_geometric.data import Batch

# SSL-style TwoStepClassifier (used in e2e_two_step_pipeline_ssl.py)
# NOTE: This assumes your repo has MRIData.py + models.TwoStepClassifier
from ContrastiveLearning.MRIData import MRIDataConfig as MRIDataConfigSSL, MRIBagDataset, default_transform
from ContrastiveLearning.models import TwoStepClassifier


# -----------------------------
# Repro
# -----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Plotting + metrics
# -----------------------------
def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, digits=4, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


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

    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")

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
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[i] = float(auc(fpr[i], tpr[i]))

    # micro
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    auc_micro = float(auc(fpr_micro, tpr_micro))

    # macro
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= max(1, n_classes)
    auc_macro = float(auc(all_fpr, mean_tpr))

    fig = plt.figure(figsize=(9, 7))
    plt.title(title)
    plt.plot(fpr_micro, tpr_micro, label=f"micro (AUC={auc_micro:.3f})")
    plt.plot(all_fpr, mean_tpr, label=f"macro (AUC={auc_macro:.3f})")
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
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    prec, rec, ap = {}, {}, {}
    for i in range(n_classes):
        prec[i], rec[i], _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap[i] = float(average_precision_score(y_bin[:, i], y_prob[:, i]))

    prec_micro, rec_micro, _ = precision_recall_curve(y_bin.ravel(), y_prob.ravel())
    ap_micro = float(average_precision_score(y_bin, y_prob, average="micro"))
    ap_macro = float(average_precision_score(y_bin, y_prob, average="macro"))

    fig = plt.figure(figsize=(9, 7))
    plt.title(title)
    plt.plot(rec_micro, prec_micro, label=f"micro (AP={ap_micro:.3f})")
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
# Utilities: class alignment
# -----------------------------
def build_reindex(src_classes: List[str], tgt_classes: List[str], name: str) -> np.ndarray:
    """
    Returns index array idx such that:
      prob_tgt = prob_src[:, idx]
    i.e. src dimension is re-ordered to match tgt ordering.
    """
    src_set = set(src_classes)
    tgt_set = set(tgt_classes)
    if src_set != tgt_set:
        missing = sorted(list(tgt_set - src_set))
        extra = sorted(list(src_set - tgt_set))
        raise ValueError(
            f"[{name}] Class mismatch.\n"
            f"  missing in src: {missing}\n"
            f"  extra in src:   {extra}\n"
            f"  src={src_classes}\n"
            f"  tgt={tgt_classes}\n"
        )
    src_pos = {c: i for i, c in enumerate(src_classes)}
    return np.asarray([src_pos[c] for c in tgt_classes], dtype=int)


def reindex_probs(p: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return p[:, idx]


# -----------------------------
# Grade probability mixing
# -----------------------------
@torch.no_grad()
def grade_probs_mixture(type_logits: torch.Tensor, grade_logits: torch.Tensor) -> torch.Tensor:
    """
    type_logits:  (B, T)
    grade_logits: (B, T, G)
    Returns:
      p_grade: (B, G) where p(g)=sum_t p(t)*p(g|t)
    """
    p_type = F.softmax(type_logits, dim=1)         # (B, T)
    p_g_t = F.softmax(grade_logits, dim=2)         # (B, T, G)
    return torch.einsum("bt,btg->bg", p_type, p_g_t)


# -----------------------------
# Model 1: Attention MIL
# -----------------------------
@torch.no_grad()
def predict_attention_mil(
    ckpt_path: Path,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    ck = torch.load(ckpt_path, map_location="cpu")

    type_classes = list(ck.get("type_classes", []))
    grade_classes = list(ck.get("grade_classes", []))

    args = ck.get("args", {})
    in_ch = 1 if bool(args.get("grayscale", False)) else 3
    emb_dim = int(args.get("emb_dim", 256))
    attn_dim = int(args.get("attn_dim", 128))
    dropout = float(args.get("dropout", 0.2))

    model = TwoStepAttentionMIL(
        in_channels=in_ch,
        emb_dim=emb_dim,
        attn_dim=attn_dim,
        num_types=len(type_classes),
        num_grades=len(grade_classes),
        dropout=dropout,
    )
    model.load_state_dict(ck["model_state"], strict=True)
    model.to(device).eval()

    metas = []
    y_type_true, y_grade_true = [], []
    p_type_all, p_grade_all = [], []

    for bag, y_type, y_grade, meta in loader:
        bag = bag.to(device)
        y_type = y_type.to(device)
        y_grade = y_grade.to(device)

        type_logits, grade_logits = model(bag)  # grade_logits: (B, T, G)

        p_type = torch.softmax(type_logits, dim=1)
        p_grade = grade_probs_mixture(type_logits, grade_logits)

        y_type_true.extend(y_type.cpu().numpy().tolist())
        y_grade_true.extend(y_grade.cpu().numpy().tolist())
        p_type_all.append(p_type.cpu().numpy())
        p_grade_all.append(p_grade.cpu().numpy())

        bsz = bag.size(0)
        for i in range(bsz):
            metas.append({k: meta[k][i] for k in meta.keys()})

    return {
        "name": "attention_mil",
        "type_classes": type_classes,
        "grade_classes": grade_classes,
        "meta": metas,
        "y_type_true": np.asarray(y_type_true, dtype=int),
        "y_grade_true": np.asarray(y_grade_true, dtype=int),
        "p_type": np.concatenate(p_type_all, axis=0) if p_type_all else np.zeros((0, len(type_classes))),
        "p_grade": np.concatenate(p_grade_all, axis=0) if p_grade_all else np.zeros((0, len(grade_classes))),
    }


# -----------------------------
# Model 2: GNN (SliceEncoder + GNN classifier)
# -----------------------------
def collate_to_pyg_batch(samples, *, node_encoder: torch.nn.Module, device: torch.device) -> Batch:
    data_list = []
    with torch.no_grad():
        for bag, y_type, y_grade, meta in samples:
            bag = bag.to(device)  # (S,C,H,W)
            g = slices_to_pyg_data(
                bag,
                encoder=node_encoder,
                y_type=int(y_type.item()),
                y_grade=int(y_grade.item()),
                meta=meta,
                chain_edges=True,
            )
            data_list.append(g)
    return Batch.from_data_list(data_list).to(device)


@torch.no_grad()
def predict_gnn(
    ckpt_path: Path,
    ds: MRIBagTwoStepDataset,
    indices: List[int],
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    ck = torch.load(ckpt_path, map_location="cpu")
    type_classes = list(ck.get("type_classes", []))
    grade_classes = list(ck.get("grade_classes", []))

    args = ck.get("args", {})
    in_ch = 1 if bool(args.get("grayscale", False)) else 3
    node_dim = int(args.get("node_dim", 64))
    hidden_dim = int(args.get("hidden_dim", 128))
    dropout = float(args.get("dropout", 0.2))

    node_encoder = SliceEncoder(in_channels=in_ch, out_dim=node_dim).to(device)
    model = TwoStepGNNClassifier(
        in_dim=node_dim,
        hidden_dim=hidden_dim,
        num_types=len(type_classes),
        num_grades=len(grade_classes),
        dropout=dropout,
    ).to(device)

    node_encoder.load_state_dict(ck["node_encoder"], strict=False)
    model.load_state_dict(ck["model"], strict=False)
    node_encoder.eval()
    model.eval()

    metas = []
    y_type_true, y_grade_true = [], []
    p_type_all, p_grade_all = [], []

    idx_loader = torch.utils.data.DataLoader(indices, batch_size=batch_size, shuffle=False, drop_last=False)

    for batch_ids in idx_loader:
        samples = [ds[int(i)] for i in batch_ids]
        batch = collate_to_pyg_batch(samples, node_encoder=node_encoder, device=device)

        type_logits, grade_logits = model(batch)  # grade_logits: (B, T, G)
        p_type = torch.softmax(type_logits, dim=1)
        p_grade = grade_probs_mixture(type_logits, grade_logits)

        y_type_true.extend(batch.y_type.view(-1).cpu().numpy().tolist())
        y_grade_true.extend(batch.y_grade.view(-1).cpu().numpy().tolist())
        p_type_all.append(p_type.cpu().numpy())
        p_grade_all.append(p_grade.cpu().numpy())

        for d in batch.to_data_list():
            m = getattr(d, "meta", {}) or {}
            metas.append(m)

    return {
        "name": "gnn",
        "type_classes": type_classes,
        "grade_classes": grade_classes,
        "meta": metas,
        "y_type_true": np.asarray(y_type_true, dtype=int),
        "y_grade_true": np.asarray(y_grade_true, dtype=int),
        "p_type": np.concatenate(p_type_all, axis=0) if p_type_all else np.zeros((0, len(type_classes))),
        "p_grade": np.concatenate(p_grade_all, axis=0) if p_grade_all else np.zeros((0, len(grade_classes))),
    }


# -----------------------------
# Model 3: SSL MIL (TwoStepClassifier-style)
# -----------------------------
@torch.no_grad()
def predict_ssl_mil(
    ckpt_path: Path,
    data_root: str,
    *,
    image_size: int,
    bag_size: int,
    bag_policy: str,
    grayscale: bool,
    seed: int,
    indices: List[int],
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    """
    Uses MRIData.MRIBagDataset + models.TwoStepClassifier to be consistent with the SSL pipeline.
    The checkpoint path is expected to contain either:
      - {"model": state_dict, ...}  (as in e2e_two_step_pipeline_ssl.py), OR
      - raw state_dict
    """
    transform = default_transform(image_size)
    cfg = MRIDataConfigSSL(
        root=data_root,
        image_size=image_size,
        grayscale=grayscale,
        mode="bag",
        target="two_step",
        bag_size=bag_size,
        bag_policy=bag_policy,
        seed=seed,
    )
    ds = MRIBagDataset(cfg, transform=transform)

    num_types = len(ds.type_encoder)
    num_grades = len(ds.grade_encoder)
    type_classes = [ds.type_encoder.decode(i) for i in range(num_types)]
    grade_classes = [ds.grade_encoder.decode(i) for i in range(num_grades)]

    model = TwoStepClassifier(
        num_types=num_types,
        num_grades=num_grades,
        backbone="mobilenet_v2",
        pretrained=False,
        in_channels=1 if grayscale else 3,
    ).to(device).eval()

    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(sd, strict=False)

    subset = Subset(ds, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    metas = []
    y_type_true, y_grade_true = [], []
    p_type_all, p_grade_all = [], []

    for bag, y_type, y_grade, meta in loader:
        # bag: (B, N, C, H, W) -> flatten to slices and pool mean (like SSL pipeline)
        B, N = bag.shape[0], bag.shape[1]
        bag = bag.to(device, non_blocking=True)
        y_type = y_type.to(device, non_blocking=True)
        y_grade = y_grade.to(device, non_blocking=True)

        x = bag.view(B * N, *bag.shape[2:])
        type_logits_s, grade_logits_s = model(x)  # (B*N,T), (B*N,T,G)

        type_logits_s = type_logits_s.view(B, N, num_types)
        grade_logits_s = grade_logits_s.view(B, N, num_types, num_grades)

        # patient pool = mean over slices
        type_logits = type_logits_s.mean(dim=1)         # (B,T)
        grade_logits = grade_logits_s.mean(dim=1)       # (B,T,G)

        p_type = torch.softmax(type_logits, dim=1)
        p_grade = grade_probs_mixture(type_logits, grade_logits)

        y_type_true.extend(y_type.cpu().numpy().tolist())
        y_grade_true.extend(y_grade.cpu().numpy().tolist())
        p_type_all.append(p_type.cpu().numpy())
        p_grade_all.append(p_grade.cpu().numpy())

        bsz = bag.size(0)
        for i in range(bsz):
            metas.append({k: meta[k][i] for k in meta.keys()})

    return {
        "name": "ssl_mil",
        "type_classes": type_classes,
        "grade_classes": grade_classes,
        "meta": metas,
        "y_type_true": np.asarray(y_type_true, dtype=int),
        "y_grade_true": np.asarray(y_grade_true, dtype=int),
        "p_type": np.concatenate(p_type_all, axis=0) if p_type_all else np.zeros((0, num_types)),
        "p_grade": np.concatenate(p_grade_all, axis=0) if p_grade_all else np.zeros((0, num_grades)),
    }


# -----------------------------
# Split loading
# -----------------------------
def load_or_make_split(
    ds_len: int,
    seed: int,
    train_frac: float,
    val_frac: float,
    split_json: Path | None,
) -> Dict[str, List[int]]:
    if split_json is not None and split_json.exists():
        sp = json.loads(split_json.read_text(encoding="utf-8"))
        # Accept either {"train":[...],"val":[...],"test":[...]} or variants
        if all(k in sp for k in ["train", "val", "test"]):
            return {"train": list(map(int, sp["train"])), "val": list(map(int, sp["val"])), "test": list(map(int, sp["test"]))}

    # fallback: deterministic shuffle split
    idx = np.arange(ds_len)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(round(train_frac * ds_len))
    n_val = int(round(val_frac * ds_len))
    tr = idx[:n_train].tolist()
    va = idx[n_train : n_train + n_val].tolist()
    te = idx[n_train + n_val :].tolist()
    return {"train": tr, "val": va, "test": te}


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser("Ensemble: Attention MIL + GNN + SSL-MIL (two-step: type + grade)")
    ap.add_argument("--data_root", type=str, required=True)

    ap.add_argument("--out_dir", type=str, default="runs/ensemble_two_step")
    ap.add_argument("--seed", type=int, default=7)

    # data params (must match what you used to train; used here to instantiate datasets)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--bag_size", type=int, default=8)
    ap.add_argument("--bag_policy", type=str, default="uniform", choices=["uniform", "first", "center"])
    ap.add_argument("--grayscale", action="store_true")

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)

    # split control
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument(
        "--split_json",
        type=str,
        default="runs/gnn_two_step_e2e/split.json",
        help="If exists, uses its train/val/test indices for consistency.",
    )

    # checkpoints
    ap.add_argument("--ckpt_mil", type=str, default="runs/e2e_mil_two_step/best.pt")
    ap.add_argument("--ckpt_gnn", type=str, default="runs/gnn_two_step_e2e/best.pt")
    ap.add_argument("--ckpt_ssl", type=str, default="runs/e2e_simclr_two_step/mil_two_step/mil_two_step_best.pt")

    # ensemble weights
    ap.add_argument("--w_mil", type=float, default=1.0)
    ap.add_argument("--w_gnn", type=float, default=1.0)
    ap.add_argument("--w_ssl", type=float, default=1.0)

    args = ap.parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Dataset for MIL + GNN (MRIDataRead) ----
    cfg_read = MRIDataConfigRead(
        root=args.data_root,
        image_size=args.image_size,
        grayscale=bool(args.grayscale),
        mode="bag",
        bag_size=args.bag_size,
        bag_policy=args.bag_policy,
        seed=args.seed,
    )
    ds_read = MRIBagTwoStepDataset(cfg_read)

    split = load_or_make_split(
        ds_len=len(ds_read),
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        split_json=Path(args.split_json) if args.split_json else None,
    )
    test_idx = split["test"]

    # loaders for Attention MIL (direct DataLoader on Subset)
    test_loader_mil = DataLoader(
        Subset(ds_read, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # ---- Predict each model ----
    p1 = predict_attention_mil(Path(args.ckpt_mil), test_loader_mil, device)
    p2 = predict_gnn(Path(args.ckpt_gnn), ds_read, test_idx, device, batch_size=args.batch_size)
    p3 = predict_ssl_mil(
        Path(args.ckpt_ssl),
        data_root=args.data_root,
        image_size=args.image_size,
        bag_size=args.bag_size,
        bag_policy=args.bag_policy,
        grayscale=bool(args.grayscale),
        seed=args.seed,
        indices=test_idx,
        device=device,
        batch_size=args.batch_size,
    )

    # ---- Choose reference class order (MIL checkpoint) ----
    type_ref = p1["type_classes"]
    grade_ref = p1["grade_classes"]

    # align probs
    idx_t2 = build_reindex(p2["type_classes"], type_ref, "GNN:type")
    idx_g2 = build_reindex(p2["grade_classes"], grade_ref, "GNN:grade")

    idx_t3 = build_reindex(p3["type_classes"], type_ref, "SSL:type")
    idx_g3 = build_reindex(p3["grade_classes"], grade_ref, "SSL:grade")

    p1_type = p1["p_type"]
    p1_grade = p1["p_grade"]

    p2_type = reindex_probs(p2["p_type"], idx_t2)
    p2_grade = reindex_probs(p2["p_grade"], idx_g2)

    p3_type = reindex_probs(p3["p_type"], idx_t3)
    p3_grade = reindex_probs(p3["p_grade"], idx_g3)

    # ---- Weight + normalize ----
    w = np.asarray([args.w_mil, args.w_gnn, args.w_ssl], dtype=float)
    w = np.maximum(w, 0.0)
    w = w / (w.sum() + 1e-12)

    p_type_ens = w[0] * p1_type + w[1] * p2_type + w[2] * p3_type
    p_grade_ens = w[0] * p1_grade + w[1] * p2_grade + w[2] * p3_grade

    y_type_true = p1["y_type_true"]
    y_grade_true = p1["y_grade_true"]

    y_type_pred = p_type_ens.argmax(axis=1)
    y_grade_pred = p_grade_ens.argmax(axis=1)

    # strict end-to-end correctness
    end2end_all_correct = float(np.mean((y_type_true == y_type_pred) & (y_grade_true == y_grade_pred)))

    # ---- Reports ----
    m_type = compute_basic_metrics(y_type_true, y_type_pred, type_ref)
    m_grade = compute_basic_metrics(y_grade_true, y_grade_pred, grade_ref)

    plot_confusion_matrix(np.asarray(m_type["confusion_matrix"]), type_ref, "CM: Tumor Type (Ensemble)", out_dir / "cm_type_ens.png")
    plot_confusion_matrix(np.asarray(m_grade["confusion_matrix"]), grade_ref, "CM: Grade (Ensemble)", out_dir / "cm_grade_ens.png")

    roc_type = plot_multiclass_roc(y_type_true, p_type_ens, type_ref, "ROC: Tumor Type (Ensemble)", out_dir / "roc_type_ens.png")
    pr_type = plot_multiclass_pr(y_type_true, p_type_ens, type_ref, "PR: Tumor Type (Ensemble)", out_dir / "pr_type_ens.png")

    roc_grade = plot_multiclass_roc(y_grade_true, p_grade_ens, grade_ref, "ROC: Grade (Ensemble)", out_dir / "roc_grade_ens.png")
    pr_grade = plot_multiclass_pr(y_grade_true, p_grade_ens, grade_ref, "PR: Grade (Ensemble)", out_dir / "pr_grade_ens.png")

    metrics = {
        "weights": {"mil": float(w[0]), "gnn": float(w[1]), "ssl": float(w[2])},
        "end2end_all_correct": end2end_all_correct,
        "type": m_type,
        "grade": m_grade,
        "roc_auc_type": roc_type,
        "pr_ap_type": pr_type,
        "roc_auc_grade": roc_grade,
        "pr_ap_grade": pr_grade,
        "artifacts": {
            "cm_type_ens": "cm_type_ens.png",
            "cm_grade_ens": "cm_grade_ens.png",
            "roc_type_ens": "roc_type_ens.png",
            "pr_type_ens": "pr_type_ens.png",
            "roc_grade_ens": "roc_grade_ens.png",
            "pr_grade_ens": "pr_grade_ens.png",
        },
        "checkpoints": {
            "mil": str(Path(args.ckpt_mil)),
            "gnn": str(Path(args.ckpt_gnn)),
            "ssl": str(Path(args.ckpt_ssl)),
        },
        "split_source": args.split_json if args.split_json else "generated",
        "data_cfg_read": asdict(cfg_read),
    }
    (out_dir / "metrics_ensemble.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # ---- Save CSV ----
    # We use meta from MIL loader for convenience (patient_id/subtype/tumor_type/grade).
    csv_path = out_dir / "predictions_ensemble.csv"
    header = [
        "patient_id",
        "subtype",
        "tumor_type_str",
        "grade_int",
        "y_type_true",
        "y_grade_true",
        "y_type_pred_ens",
        "y_grade_pred_ens",
    ]
    header += [f"p_type_ens_{c}" for c in type_ref]
    header += [f"p_grade_ens_{c}" for c in grade_ref]

    lines = [",".join(header)]
    for i, meta in enumerate(p1["meta"]):
        row = [
            str(meta.get("patient_id", "")),
            str(meta.get("subtype", "")),
            str(meta.get("tumor_type", "")),
            str(meta.get("grade", "")),
            str(int(y_type_true[i])),
            str(int(y_grade_true[i])),
            str(int(y_type_pred[i])),
            str(int(y_grade_pred[i])),
        ]
        row += [f"{x:.6f}" for x in p_type_ens[i].tolist()]
        row += [f"{x:.6f}" for x in p_grade_ens[i].tolist()]
        lines.append(",".join(row))
    csv_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n==== ENSEMBLE TEST SUMMARY ====")
    print(f"type_acc            : {m_type['accuracy']:.4f}")
    print(f"grade_acc           : {m_grade['accuracy']:.4f}")
    print(f"all_correct (strict): {end2end_all_correct:.4f}")
    print(f"Saved -> {out_dir}")


if __name__ == "__main__":
    main()