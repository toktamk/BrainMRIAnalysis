# scripts/e2e_min_pipeline.py
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Utilities
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic-ish
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(p: Path, obj) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2))


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def accuracy_from_probs(probs: torch.Tensor, y_true: torch.Tensor) -> float:
    pred = probs.argmax(dim=-1)
    return (pred == y_true).float().mean().item()


def macro_f1_from_probs(probs: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> float:
    pred = probs.argmax(dim=-1)
    f1s = []
    for c in range(num_classes):
        tp = ((pred == c) & (y_true == c)).sum().item()
        fp = ((pred == c) & (y_true != c)).sum().item()
        fn = ((pred != c) & (y_true == c)).sum().item()
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


# =============================================================================
# Patient discovery + split
# =============================================================================
_TUMOR_GRADE_RE = None
try:
    import re

    _TUMOR_GRADE_RE = re.compile(r"^(?P<tumor>.+)-g(?P<grade>[1-4])$", re.IGNORECASE)
except Exception:
    _TUMOR_GRADE_RE = None


def parse_folder_label(folder_name: str) -> Tuple[str, int]:
    if _TUMOR_GRADE_RE is None:
        raise RuntimeError("Regex not available")
    m = _TUMOR_GRADE_RE.match(folder_name.strip())
    if not m:
        raise ValueError(f"Folder name '{folder_name}' does not match '<tumor>-g<1..4>'")
    return m.group("tumor").lower(), int(m.group("grade"))


@dataclass(frozen=True)
class PatientInfo:
    patient_id: str
    subtype: str            # e.g. actrocytoma-g1
    tumor_type: str         # e.g. actrocytoma
    grade: int              # 1..4
    patient_dir: Path       # .../subtype/Pxx


def list_patients(data_root: Path) -> List[PatientInfo]:
    out: List[PatientInfo] = []
    for class_dir in sorted([p for p in data_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        tumor, grade = parse_folder_label(class_dir.name)
        for patient_dir in sorted([p for p in class_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            out.append(
                PatientInfo(
                    patient_id=patient_dir.name,
                    subtype=class_dir.name,
                    tumor_type=tumor,
                    grade=grade,
                    patient_dir=patient_dir,
                )
            )
    if not out:
        raise RuntimeError(f"No patient folders found under: {data_root}")
    return out


def stratified_split(
    patients: Sequence[PatientInfo],
    *,
    seed: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[List[PatientInfo], List[PatientInfo], List[PatientInfo]]:
    """
    Stratify by subtype (folder name). Keeps distributions stable.
    """
    rng = random.Random(seed)
    by_cls: Dict[str, List[PatientInfo]] = {}
    for p in patients:
        by_cls.setdefault(p.subtype, []).append(p)

    train, val, test = [], [], []
    for cls, items in by_cls.items():
        items = items[:]
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        cls_train = items[:n_train]
        cls_val = items[n_train : n_train + n_val]
        cls_test = items[n_train + n_val :]
        train.extend(cls_train)
        val.extend(cls_val)
        test.extend(cls_test)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# =============================================================================
# Dataset wrappers (filter by patient split)
# =============================================================================
class FilteredMRIPairDataset(Dataset):
    """
    Wrap ContrastiveLearning.MRIPairDataset and filter pairs to a patient_id set.
    Assumes pair paths look like: root/subtype/patient_id/slice.png
    """
    def __init__(self, base, allowed_patient_ids: set[str]):
        self.base = base
        self.allowed = allowed_patient_ids

        # Precompute allowed indices once for speed/determinism.
        keep = []
        for i in range(len(base)):
            a_path, b_path = base._pairs[i]  # relies on your MRIPairDataset internals (OK for pipeline script)
            pid = a_path.parent.name
            if pid in self.allowed:
                keep.append(i)
        self._idx = keep
        if len(self._idx) == 0:
            raise RuntimeError("FilteredMRIPairDataset has 0 pairs for the given split.")

    def __len__(self) -> int:
        return len(self._idx)

    def __getitem__(self, i: int):
        return self.base[self._idx[i]]


class FilteredBagDataset(Dataset):
    """
    Wrap MultiInstanceLearning/GraphNeuralNetworks MRIBagDataset and filter to patient ids.
    Expects underlying ds has attribute: ds.patients = List[(subtype, patient_dir, tumor, grade)]
    """
    def __init__(self, base, allowed_patient_ids: set[str]):
        self.base = base
        self.allowed = allowed_patient_ids
        keep = []
        for i, tpl in enumerate(base.patients):
            # tpl: (subtype, patient_dir, tumor, grade)
            patient_dir = tpl[1]
            pid = patient_dir.name
            if pid in allowed_patient_ids:
                keep.append(i)
        self._idx = keep
        if len(self._idx) == 0:
            raise RuntimeError("FilteredBagDataset has 0 patients for the given split.")

        # Keep the SAME encoder as base (must match class ordering across splits)
        self.encoder = base.encoder

    def __len__(self) -> int:
        return len(self._idx)

    def __getitem__(self, i: int):
        return self.base[self._idx[i]]


# =============================================================================
# SimCLR helpers (robust output parsing + embedding)
# =============================================================================
def unpack_simclr_output(out):
    """
    Returns (z1, z2) from common SimCLR forward outputs.
    """
    if isinstance(out, dict):
        if "z_i" in out and "z_j" in out:
            return out["z_i"], out["z_j"]
        if "z1" in out and "z2" in out:
            return out["z1"], out["z2"]
        if "proj_i" in out and "proj_j" in out:
            return out["proj_i"], out["proj_j"]
        raise ValueError(f"Unknown SimCLR dict outputs: {list(out.keys())}")

    if isinstance(out, (tuple, list)):
        if len(out) == 2:
            return out[0], out[1]
        if len(out) >= 4:
            return out[-2], out[-1]
        raise ValueError(f"Unexpected SimCLR output tuple length: {len(out)}")

    raise ValueError(f"Unexpected SimCLR output type: {type(out)}")


@torch.no_grad()
def simclr_embed_single(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Produce an embedding for a single view tensor x: (B,C,H,W).
    Best-effort across implementations.
    """
    model.eval()

    # If model exposes an explicit encoder method
    for attr in ["encode", "encode_image", "forward_single", "get_embedding"]:
        if hasattr(model, attr) and callable(getattr(model, attr)):
            z = getattr(model, attr)(x)
            if isinstance(z, (tuple, list)):
                z = z[-1]
            if isinstance(z, dict):
                z = z.get("z") or z.get("z_i") or z.get("proj") or next(iter(z.values()))
            return z

    # Common pattern: model.encoder + optional projector / projection_head
    if hasattr(model, "encoder"):
        h = model.encoder(x)
        # Some encoders return (B,F,1,1) etc.
        if h.ndim > 2:
            h = torch.flatten(h, 1)

        for proj_name in ["projector", "projection_head", "proj_head", "projection"]:
            if hasattr(model, proj_name):
                proj = getattr(model, proj_name)
                if callable(proj):
                    z = proj(h)
                    return z
        return h

    # Fallback: run forward with identical inputs and grab z1
    out = model(x, x)
    z1, _ = unpack_simclr_output(out)
    return z1


# =============================================================================
# Model heads
# =============================================================================
class PatientHead(nn.Module):
    """
    Patient-level classifier on top of slice embeddings:
      - embed each slice -> mean pool -> linear classifier
    """
    def __init__(self, embed_dim: int, num_classes: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, patient_emb: torch.Tensor) -> torch.Tensor:
        return self.net(patient_emb)


# =============================================================================
# Training / evaluation loops
# =============================================================================
def pretrain_simclr(
    *,
    data_root: Path,
    train_patient_ids: set[str],
    out_dir: Path,
    device: torch.device,
    seed: int,
    image_size: int,
    batch_size: int,
    steps: int,
    temperature: float,
    encoder_name: str,
    projection_dim: int,
) -> Tuple[Path, int]:
    """
    Contrastive pretrain on train patients only.
    """
    from ContrastiveLearning.data import ContrastiveDataConfig, MRIPairDataset
    from ContrastiveLearning.models import EncoderConfig, SimCLR
    from ContrastiveLearning.losses import NTXentLoss

    ensure_dir(out_dir)

    data_cfg = ContrastiveDataConfig(
        root=data_root,
        image_size=image_size,
        grayscale=False,
        pair_mode="adjacent",
        strict_images=True,
    )
    base = MRIPairDataset(data_cfg)
    ds = FilteredMRIPairDataset(base, train_patient_ids)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    enc_cfg = EncoderConfig(name=encoder_name, pretrained=False, projection_dim=projection_dim)
    model = SimCLR(enc_cfg, in_channels=3).to(device)

    loss_fn = NTXentLoss(temperature=temperature).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    it = iter(loader)
    for step in range(steps):
        try:
            x1, x2 = next(it)
        except StopIteration:
            it = iter(loader)
            x1, x2 = next(it)

        x1 = x1.to(device)
        x2 = x2.to(device)

        out = model(x1, x2)
        z1, z2 = unpack_simclr_output(out)

        loss = loss_fn(z1, z2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % max(1, steps // 5) == 0:
            print(f"[simclr] step {step+1}/{steps} loss={loss.item():.4f}")

    ckpt = out_dir / "simclr_pretrain.pt"
    torch.save(
        {
            "seed": seed,
            "encoder_cfg": asdict(enc_cfg),
            "data_cfg": asdict(data_cfg),
            "model_state": model.state_dict(),
        },
        ckpt,
    )
    print(f"[simclr] saved -> {ckpt}")
    return ckpt, len(ds)


def make_patient_bags(
    patients: Sequence[PatientInfo],
    *,
    image_size: int,
    bag_size: int,
    target: str,
    seed: int,
    mode_module: str,
) -> Dataset:
    """
    Build MRIBagDataset from either:
      - MultiInstanceLearning.MRIDataRead
      - GraphNeuralNetworks.MRIDataRead

    and then filter to the provided patient_ids.
    """
    if mode_module == "mil":
        from MultiInstanceLearning.MRIDataRead import MRIDataConfig, MRIBagDataset
    elif mode_module == "gnn":
        from GraphNeuralNetworks.MRIDataRead import MRIDataConfig, MRIBagDataset
    else:
        raise ValueError("mode_module must be 'mil' or 'gnn'")

    # IMPORTANT: use the SAME label definition across modules
    cfg = MRIDataConfig(
        root=str(patients[0].patient_dir.parent.parent),  # data_root
        mode="bag",
        target=target,         # e.g. "grade" or "subtype"
        image_size=image_size,
        grayscale=False,
        bag_size=bag_size,
        bag_policy="uniform",
        seed=seed,
    )
    base = MRIBagDataset(cfg)
    allowed = {p.patient_id for p in patients}
    return FilteredBagDataset(base, allowed)


def train_mil(
    train_ds: Dataset,
    val_ds: Dataset,
    *,
    out_dir: Path,
    device: torch.device,
    steps: int,
    batch_size: int,
    lr: float = 1e-3,
) -> Path:
    from MultiInstanceLearning.models import AttentionMIL

    ensure_dir(out_dir)
    model = AttentionMIL().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # derive num_classes from encoder
    num_classes = len(train_ds.encoder)

    best_val = -1.0
    best_path = out_dir / "mil_best.pt"

    model.train()
    it = iter(train_loader)
    for step in range(steps):
        try:
            bag, y, _ = next(it)
        except StopIteration:
            it = iter(train_loader)
            bag, y, _ = next(it)

        bag = bag.to(device)  # (B,N,C,H,W)
        y = y.to(device)

        logits = model(bag)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % max(1, steps // 5) == 0:
            # val
            model.eval()
            all_probs, all_y = [], []
            with torch.no_grad():
                for vb, vy, _ in val_loader:
                    vb = vb.to(device)
                    vy = vy.to(device)
                    v_logits = model(vb)
                    all_probs.append(softmax_probs(v_logits).detach().cpu())
                    all_y.append(vy.detach().cpu())
            probs = torch.cat(all_probs, dim=0)
            y_true = torch.cat(all_y, dim=0)
            val_acc = accuracy_from_probs(probs, y_true)
            print(f"[mil] step {step+1}/{steps} loss={loss.item():.4f} val_acc={val_acc:.4f}")
            if val_acc > best_val:
                best_val = val_acc
                torch.save({"model_state": model.state_dict(), "num_classes": num_classes}, best_path)
            model.train()

    print(f"[mil] saved best -> {best_path} (val_acc={best_val:.4f})")
    return best_path


def train_gnn(
    train_ds: Dataset,
    val_ds: Dataset,
    *,
    out_dir: Path,
    device: torch.device,
    steps: int,
    lr: float = 1e-3,
    node_dim: int = 64,
    hidden_dim: int = 128,
) -> Path:
    from GraphNeuralNetworks.graphs import SliceEncoder, slices_to_pyg_data
    from GraphNeuralNetworks.models import GNNClassifier

    ensure_dir(out_dir)

    encoder = SliceEncoder(out_dim=node_dim).to(device)
    model = GNNClassifier(in_dim=node_dim, hidden_dim=hidden_dim, num_classes=len(train_ds.encoder)).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(model.parameters()), lr=lr)

    best_val = -1.0
    best_path = out_dir / "gnn_best.pt"

    encoder.train()
    model.train()

    # simple index-based iteration (patient-level)
    for step in range(steps):
        idx = step % len(train_ds)
        bag, y, _ = train_ds[idx]  # bag: (N,C,H,W)  y: scalar tensor
        bag = bag.to(device)
        y = int(y.item())

        data = slices_to_pyg_data(bag, y=y, encoder=encoder).to(device)
        logits = model(data)
        target = data.y.view(-1).to(device).long()

        loss = F.cross_entropy(logits, target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % max(1, steps // 5) == 0:
            # val
            encoder.eval()
            model.eval()
            probs_list, y_list = [], []
            with torch.no_grad():
                for j in range(len(val_ds)):
                    vbag, vy, _ = val_ds[j]
                    vbag = vbag.to(device)
                    vy_int = int(vy.item())
                    vdata = slices_to_pyg_data(vbag, y=vy_int, encoder=encoder).to(device)
                    vlogits = model(vdata)
                    probs_list.append(softmax_probs(vlogits).cpu())
                    y_list.append(torch.tensor([vy_int], dtype=torch.long))
            probs = torch.cat(probs_list, dim=0)
            y_true = torch.cat(y_list, dim=0)
            val_acc = accuracy_from_probs(probs, y_true)
            print(f"[gnn] step {step+1}/{steps} loss={loss.item():.4f} val_acc={val_acc:.4f}")
            if val_acc > best_val:
                best_val = val_acc
                torch.save(
                    {"encoder_state": encoder.state_dict(), "model_state": model.state_dict(), "hidden_dim": hidden_dim},
                    best_path,
                )
            encoder.train()
            model.train()

    print(f"[gnn] saved best -> {best_path} (val_acc={best_val:.4f})")
    return best_path


def eval_mil_probs_by_pid(
    ds: Dataset,
    ckpt_path: Path,
    *,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int], int]:
    from MultiInstanceLearning.models import AttentionMIL

    ck = torch.load(ckpt_path, map_location="cpu")
    model = AttentionMIL().to(device)
    model.load_state_dict(ck["model_state"], strict=False)
    model.eval()

    probs_by_pid: Dict[str, torch.Tensor] = {}
    y_by_pid: Dict[str, int] = {}
    for i in range(len(ds)):
        bag, y, meta = ds[i]
        pid = meta["patient_id"]
        bag = bag.unsqueeze(0).to(device)  # (1,N,C,H,W)
        with torch.no_grad():
            logits = model(bag)
            probs = softmax_probs(logits).squeeze(0).detach().cpu()
        probs_by_pid[pid] = probs
        y_by_pid[pid] = int(y.item())
    num_classes = probs.shape[0]
    return probs_by_pid, y_by_pid, num_classes


def eval_gnn_probs_by_pid(
    ds: Dataset,
    ckpt_path: Path,
    *,
    device: torch.device,
    node_dim: int = 64,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int], int]:
    from GraphNeuralNetworks.graphs import SliceEncoder, slices_to_pyg_data
    from GraphNeuralNetworks.models import GNNClassifier

    ck = torch.load(ckpt_path, map_location="cpu")

    encoder = SliceEncoder(out_dim=node_dim).to(device)
    model = GNNClassifier(in_dim=node_dim, hidden_dim=int(ck.get("hidden_dim", 128)), num_classes=len(ds.encoder)).to(device)

    encoder.load_state_dict(ck["encoder_state"], strict=False)
    model.load_state_dict(ck["model_state"], strict=False)
    encoder.eval()
    model.eval()

    probs_by_pid: Dict[str, torch.Tensor] = {}
    y_by_pid: Dict[str, int] = {}
    for i in range(len(ds)):
        bag, y, meta = ds[i]
        pid = meta["patient_id"]
        bag = bag.to(device)  # (N,C,H,W)
        with torch.no_grad():
            data = slices_to_pyg_data(bag, y=int(y.item()), encoder=encoder).to(device)
            logits = model(data)
            probs = softmax_probs(logits).squeeze(0).detach().cpu()
        probs_by_pid[pid] = probs
        y_by_pid[pid] = int(y.item())
    num_classes = probs.shape[0]
    return probs_by_pid, y_by_pid, num_classes


def train_simclr_patient_head(
    *,
    simclr_ckpt: Path,
    train_patients: Sequence[PatientInfo],
    val_patients: Sequence[PatientInfo],
    test_patients: Sequence[PatientInfo],
    target: str,
    image_size: int,
    bag_size_for_eval: int,
    out_dir: Path,
    device: torch.device,
    steps: int,
    lr: float = 1e-3,
) -> Tuple[Path, Dict[str, torch.Tensor], Dict[str, int], int]:
    """
    Minimal “proper” SimCLR usage for classification:
      - freeze SimCLR encoder
      - build patient bags (same as MIL bag selection)
      - embed each slice with SimCLR encoder/proj
      - mean pool -> head -> train on train, select by val
      - return patient probs on test as dict by patient_id
    """
    from ContrastiveLearning.models import EncoderConfig, SimCLR

    ensure_dir(out_dir)

    ck = torch.load(simclr_ckpt, map_location="cpu")
    enc_cfg = EncoderConfig(**ck["encoder_cfg"])
    simclr = SimCLR(enc_cfg, in_channels=3).to(device)
    simclr.load_state_dict(ck["model_state"], strict=False)
    simclr.eval()
    for p in simclr.parameters():
        p.requires_grad = False

    # Use MIL bag dataset code to select the same slices per patient (deterministic by seed)
    train_ds = make_patient_bags(train_patients, image_size=image_size, bag_size=bag_size_for_eval, target=target, seed=7, mode_module="mil")
    val_ds = make_patient_bags(val_patients, image_size=image_size, bag_size=bag_size_for_eval, target=target, seed=7, mode_module="mil")
    test_ds = make_patient_bags(test_patients, image_size=image_size, bag_size=bag_size_for_eval, target=target, seed=7, mode_module="mil")
    num_classes = len(train_ds.encoder)

    # infer embed dim by running one batch
    bag0, _, _ = train_ds[0]
    with torch.no_grad():
        z = simclr_embed_single(simclr, bag0.to(device))  # bag0 is (N,C,H,W); we want embeddings per slice
    # simclr_embed_single expects (B,C,H,W); so we embed per slice
    # We'll do per-slice embed below; for dim:
    embed_dim = int(simclr_embed_single(simclr, bag0[0:1].to(device)).shape[-1])

    head = PatientHead(embed_dim=embed_dim, num_classes=num_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    def patient_batch_to_emb_and_y(ds: Dataset, idx: int) -> Tuple[torch.Tensor, int, str]:
        bag, y, meta = ds[idx]  # bag: (N,C,H,W)
        pid = meta["patient_id"]
        bag = bag.to(device)
        ys = int(y.item())
        # embed each slice (N,embed_dim), then mean pool -> (embed_dim,)
        zs = []
        with torch.no_grad():
            for i in range(bag.size(0)):
                zi = simclr_embed_single(simclr, bag[i : i + 1])  # (1,D)
                zs.append(zi.squeeze(0))
        z_mean = torch.stack(zs, dim=0).mean(dim=0)
        return z_mean, ys, pid

    best_val = -1.0
    best_path = out_dir / "simclr_head_best.pt"

    head.train()
    for step in range(steps):
        idx = step % len(train_ds)
        z_mean, y, _ = patient_batch_to_emb_and_y(train_ds, idx)
        logits = head(z_mean.unsqueeze(0))
        loss = F.cross_entropy(logits, torch.tensor([y], device=device, dtype=torch.long))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % max(1, steps // 5) == 0:
            # val eval
            head.eval()
            probs_list, y_list = [], []
            with torch.no_grad():
                for j in range(len(val_ds)):
                    zv, yv, _ = patient_batch_to_emb_and_y(val_ds, j)
                    lv = head(zv.unsqueeze(0))
                    probs_list.append(softmax_probs(lv).cpu())
                    y_list.append(torch.tensor([yv], dtype=torch.long))
            probs = torch.cat(probs_list, dim=0)
            y_true = torch.cat(y_list, dim=0)
            val_acc = accuracy_from_probs(probs, y_true)
            print(f"[simclr-head] step {step+1}/{steps} loss={loss.item():.4f} val_acc={val_acc:.4f}")
            if val_acc > best_val:
                best_val = val_acc
                torch.save({"head_state": head.state_dict(), "embed_dim": embed_dim, "num_classes": num_classes}, best_path)
            head.train()

    print(f"[simclr-head] saved best -> {best_path} (val_acc={best_val:.4f})")

    # test probs by pid
    ck2 = torch.load(best_path, map_location="cpu")
    head.load_state_dict(ck2["head_state"], strict=True)
    head.eval()

    probs_by_pid: Dict[str, torch.Tensor] = {}
    y_by_pid: Dict[str, int] = {}
    with torch.no_grad():
        for j in range(len(test_ds)):
            zt, yt, pid = patient_batch_to_emb_and_y(test_ds, j)
            lt = head(zt.unsqueeze(0))
            probs_by_pid[pid] = softmax_probs(lt).squeeze(0).cpu()
            y_by_pid[pid] = yt

    return best_path, probs_by_pid, y_by_pid, num_classes


def eval_ensemble(
    simclr_probs_by_pid: Dict[str, torch.Tensor],
    mil_probs_by_pid: Dict[str, torch.Tensor],
    gnn_probs_by_pid: Dict[str, torch.Tensor],
    y_by_pid: Dict[str, int],
    *,
    num_classes: int,
) -> Dict[str, float]:
    common = sorted(set(simclr_probs_by_pid) & set(mil_probs_by_pid) & set(gnn_probs_by_pid))
    if not common:
        raise RuntimeError(
            "No overlapping patient_ids across SimCLR/MIL/GNN predictions. "
            "Ensure all three evaluators are restricted to the SAME test split."
        )

    sim_mat = torch.stack([simclr_probs_by_pid[pid] for pid in common], dim=0)
    mil_mat = torch.stack([mil_probs_by_pid[pid] for pid in common], dim=0)
    gnn_mat = torch.stack([gnn_probs_by_pid[pid] for pid in common], dim=0)
    probs_ens = (sim_mat + mil_mat + gnn_mat) / 3.0

    y_true = torch.tensor([y_by_pid[pid] for pid in common], dtype=torch.long)

    acc = accuracy_from_probs(probs_ens, y_true)
    f1 = macro_f1_from_probs(probs_ens, y_true, num_classes=num_classes)

    return {
        "n_common": len(common),
        "acc": acc,
        "macro_f1": f1,
    }


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=os.environ.get("DATA_ROOT", ""), help="Dataset root (folder with actrocytoma-g1 etc.)")
    p.add_argument("--out_dir", type=str, default="runs/e2e_min_v1")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--image_size", type=int, default=224)

    # label target for final classification
    p.add_argument("--target", type=str, default="grade", choices=["subtype", "type", "grade", "type_grade"])

    # split fractions
    p.add_argument("--train_frac", type=float, default=0.70)
    p.add_argument("--val_frac", type=float, default=0.15)

    # SimCLR pretrain knobs
    p.add_argument("--simclr_steps", type=int, default=50)
    p.add_argument("--simclr_batch", type=int, default=32)
    p.add_argument("--simclr_temp", type=float, default=0.5)
    p.add_argument("--simclr_encoder", type=str, default="small_cnn")
    p.add_argument("--simclr_proj_dim", type=int, default=128)

    # SimCLR head training
    p.add_argument("--simclr_head_steps", type=int, default=80)
    p.add_argument("--simclr_head_bag_size", type=int, default=8)

    # MIL training
    p.add_argument("--bag_size", type=int, default=8)
    p.add_argument("--mil_steps", type=int, default=60)
    p.add_argument("--mil_batch", type=int, default=2)

    # GNN training
    p.add_argument("--gnn_steps", type=int, default=80)
    p.add_argument("--gnn_node_dim", type=int, default=64)
    p.add_argument("--gnn_hidden_dim", type=int, default=128)

    p.add_argument("--force_cpu", action="store_true")

    args = p.parse_args()

    if not args.data_root:
        raise SystemExit("ERROR: provide --data_root or set DATA_ROOT env var.")

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    device = get_device(force_cpu=args.force_cpu)
    set_seed(args.seed)

    print(f"[pipe] data_root={data_root}")
    print(f"[pipe] out_dir={out_dir}")
    print(f"[pipe] device={device.type}")
    print(f"[pipe] seed={args.seed}")

    # -------------------------------------------------------------------------
    # Split (by patient)
    # -------------------------------------------------------------------------
    patients = list_patients(data_root)
    train_p, val_p, test_p = stratified_split(
        patients,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )
    print(f"[split] train={len(train_p)} val={len(val_p)} test={len(test_p)}")

    train_ids = {p.patient_id for p in train_p}
    val_ids = {p.patient_id for p in val_p}
    test_ids = {p.patient_id for p in test_p}

    # -------------------------------------------------------------------------
    # 1) SimCLR pretrain
    # -------------------------------------------------------------------------
    simclr_dir = out_dir / "simclr"
    simclr_ckpt, n_train_pairs = pretrain_simclr(
        data_root=data_root,
        train_patient_ids=train_ids,
        out_dir=simclr_dir,
        device=device,
        seed=args.seed,
        image_size=args.image_size,
        batch_size=args.simclr_batch,
        steps=args.simclr_steps,
        temperature=args.simclr_temp,
        encoder_name=args.simclr_encoder,
        projection_dim=args.simclr_proj_dim,
    )

    # -------------------------------------------------------------------------
    # 2) MIL train/val, evaluate on test (prob dict by patient_id)
    # -------------------------------------------------------------------------
    mil_dir = out_dir / "mil"
    train_mil_ds = make_patient_bags(train_p, image_size=args.image_size, bag_size=args.bag_size, target=args.target, seed=args.seed, mode_module="mil")
    val_mil_ds = make_patient_bags(val_p, image_size=args.image_size, bag_size=args.bag_size, target=args.target, seed=args.seed, mode_module="mil")
    test_mil_ds = make_patient_bags(test_p, image_size=args.image_size, bag_size=args.bag_size, target=args.target, seed=args.seed, mode_module="mil")

    mil_ckpt = train_mil(
        train_mil_ds,
        val_mil_ds,
        out_dir=mil_dir,
        device=device,
        steps=args.mil_steps,
        batch_size=args.mil_batch,
    )
    mil_probs_by_pid, y_by_pid_mil, mil_C = eval_mil_probs_by_pid(test_mil_ds, mil_ckpt, device=device)

    # -------------------------------------------------------------------------
    # 3) GNN train/val, evaluate on test (prob dict by patient_id)
    # -------------------------------------------------------------------------
    gnn_dir = out_dir / "gnn"
    train_gnn_ds = make_patient_bags(train_p, image_size=args.image_size, bag_size=args.bag_size, target=args.target, seed=args.seed, mode_module="gnn")
    val_gnn_ds = make_patient_bags(val_p, image_size=args.image_size, bag_size=args.bag_size, target=args.target, seed=args.seed, mode_module="gnn")
    test_gnn_ds = make_patient_bags(test_p, image_size=args.image_size, bag_size=args.bag_size, target=args.target, seed=args.seed, mode_module="gnn")

    gnn_ckpt = train_gnn(
        train_gnn_ds,
        val_gnn_ds,
        out_dir=gnn_dir,
        device=device,
        steps=args.gnn_steps,
        node_dim=args.gnn_node_dim,
        hidden_dim=args.gnn_hidden_dim,
    )
    gnn_probs_by_pid, y_by_pid_gnn, gnn_C = eval_gnn_probs_by_pid(
        test_gnn_ds,
        gnn_ckpt,
        device=device,
        node_dim=args.gnn_node_dim,
    )

    # -------------------------------------------------------------------------
    # 4) SimCLR patient head (train on train, tune on val, eval on test)
    # -------------------------------------------------------------------------
    simclr_head_dir = out_dir / "simclr_head"
    simclr_head_ckpt, simclr_probs_by_pid, y_by_pid_simclr, simclr_C = train_simclr_patient_head(
        simclr_ckpt=simclr_ckpt,
        train_patients=train_p,
        val_patients=val_p,
        test_patients=test_p,
        target=args.target,
        image_size=args.image_size,
        bag_size_for_eval=args.simclr_head_bag_size,
        out_dir=simclr_head_dir,
        device=device,
        steps=args.simclr_head_steps,
    )

    # -------------------------------------------------------------------------
    # 5) Check class-space consistency
    # -------------------------------------------------------------------------
    C = simclr_C
    if mil_C != C or gnn_C != C:
        raise RuntimeError(
            f"Class-count mismatch: simclr={simclr_C}, mil={mil_C}, gnn={gnn_C}. "
            f"Ensure all modules use the same --target={args.target} and same label encoder mapping."
        )

    # -------------------------------------------------------------------------
    # 6) Ensemble on test (aligned by patient_id)
    # -------------------------------------------------------------------------
    # choose a canonical y_by_pid from MIL (but require consistency)
    # If some patient_id exists but labels differ, that's a bug worth failing fast.
    y_by_pid = {}
    for pid, y in y_by_pid_mil.items():
        y2 = y_by_pid_gnn.get(pid, y)
        y3 = y_by_pid_simclr.get(pid, y)
        if (pid in y_by_pid_gnn and y2 != y) or (pid in y_by_pid_simclr and y3 != y):
            raise RuntimeError(f"Label mismatch for patient_id={pid}: mil={y}, gnn={y2}, simclr={y3}")
        y_by_pid[pid] = y

    ens_metrics = eval_ensemble(simclr_probs_by_pid, mil_probs_by_pid, gnn_probs_by_pid, y_by_pid, num_classes=C)

    # Also per-model test metrics on their own (aligned to their own sets)
    def metrics_from_dict(probs_by_pid: Dict[str, torch.Tensor], y_by_pid: Dict[str, int], C: int) -> Dict[str, float]:
        common = sorted(set(probs_by_pid) & set(y_by_pid))
        probs = torch.stack([probs_by_pid[pid] for pid in common], dim=0)
        y = torch.tensor([y_by_pid[pid] for pid in common], dtype=torch.long)
        return {
            "n": len(common),
            "acc": accuracy_from_probs(probs, y),
            "macro_f1": macro_f1_from_probs(probs, y, num_classes=C),
        }

    simclr_metrics = metrics_from_dict(simclr_probs_by_pid, y_by_pid, C)
    mil_metrics = metrics_from_dict(mil_probs_by_pid, y_by_pid, C)
    gnn_metrics = metrics_from_dict(gnn_probs_by_pid, y_by_pid, C)

    # -------------------------------------------------------------------------
    # Final summary block (requested)
    # -------------------------------------------------------------------------
    summary = {
        "dataset": {
            "patients_total": len(patients),
            "train_patients": len(train_p),
            "val_patients": len(val_p),
            "test_patients": len(test_p),
            "train_pairs_simclr": int(n_train_pairs),
        },
        "device": str(device),
        "config": {
            "target": args.target,
            "image_size": args.image_size,
            "bag_size": args.bag_size,
            "simclr_head_bag_size": args.simclr_head_bag_size,
            "seed": args.seed,
            "splits": {"train_frac": args.train_frac, "val_frac": args.val_frac},
            "simclr": {
                "steps": args.simclr_steps,
                "batch": args.simclr_batch,
                "temp": args.simclr_temp,
                "encoder": args.simclr_encoder,
                "proj_dim": args.simclr_proj_dim,
            },
            "mil": {"steps": args.mil_steps, "batch": args.mil_batch},
            "gnn": {"steps": args.gnn_steps, "node_dim": args.gnn_node_dim, "hidden_dim": args.gnn_hidden_dim},
        },
        "checkpoints": {
            "simclr_pretrain": str(simclr_ckpt),
            "simclr_head": str(simclr_head_ckpt),
            "mil_best": str(mil_ckpt),
            "gnn_best": str(gnn_ckpt),
        },
        "test_metrics": {
            "simclr_head": simclr_metrics,
            "mil": mil_metrics,
            "gnn": gnn_metrics,
            "ensemble": ens_metrics,
        },
    }

    print("\n" + "=" * 72)
    print("[FINAL SUMMARY]")
    print(json.dumps(summary, indent=2))
    print("=" * 72 + "\n")

    save_json(out_dir / "summary.json", summary)
    print(f"[pipe] saved summary -> {out_dir / 'summary.json'}")
    print("[pipe] DONE")


if __name__ == "__main__":
    main()