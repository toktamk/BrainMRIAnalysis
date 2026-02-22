from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# -----------------------------
# Config + label parsing
# -----------------------------
@dataclass(frozen=True)
class MRIDataConfig:
    root: Union[str, Path] = r"D:\datasets\MRI_Mahdieh_Datasets"
    image_size: int = 224
    grayscale: bool = False  # set True if your slices are single-channel
    target: str = "subtype"  # "subtype" | "type" | "grade" | "type_grade"
    mode: str = "slice"      # "slice" | "bag" | "contrastive"
    bag_size: int = 8        # used for mode="bag" (MIL)
    bag_policy: str = "uniform"  # "uniform" | "first" | "center"
    pair_stride: int = 1     # used for mode="contrastive": adjacent slices with step
    seed: int = 7


_TUMOR_GRADE_RE = re.compile(r"^(?P<tumor>.+)-g(?P<grade>[1-4])$", re.IGNORECASE)


def parse_folder_label(folder_name: str) -> Tuple[str, int]:
    """
    Parse labels from folder name like:
      "actrocytoma-g1" -> ("actrocytoma", 1)
      "glioblastoma-g4" -> ("glioblastoma", 4)
    """
    m = _TUMOR_GRADE_RE.match(folder_name.strip())
    if not m:
        raise ValueError(
            f"Folder name '{folder_name}' does not match '<tumor>-g<1..4>' pattern."
        )
    tumor = m.group("tumor").lower()
    grade = int(m.group("grade"))
    return tumor, grade


def default_transform(image_size: int) -> Callable:
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ]
    )


def default_simclr_transform(image_size: int) -> Callable:
    # Conservative augmentation suitable for MRI slices; tune as needed.
    return T.Compose(
        [
            T.ToPILImage(),
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.2),
            T.ToTensor(),
        ]
    )


def _read_image(path: Path, image_size: int, grayscale: bool) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    # Standardize HxWxC
    if grayscale:
        img = img[:, :, None]

    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return img


def _list_patient_dirs(root: Path) -> List[Tuple[str, Path, str, int]]:
    """
    Returns list of (subtype_name, patient_dir, tumor_type, grade).
    subtype_name is class folder name, e.g., actrocytoma-g1
    """
    out: List[Tuple[str, Path, str, int]] = []
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        tumor, grade = parse_folder_label(class_dir.name)
        for patient_dir in sorted([p for p in class_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            out.append((class_dir.name, patient_dir, tumor, grade))
    return out


def _list_slices(patient_dir: Path) -> List[Path]:
    files = [p for p in patient_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda p: p.name)


# -----------------------------
# Label encoders
# -----------------------------
class LabelEncoder:
    def __init__(self, classes: Sequence[str]) -> None:
        self.classes = list(classes)
        self.class_to_index: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}

    def encode(self, c: str) -> int:
        return self.class_to_index[c]

    def __len__(self) -> int:
        return len(self.classes)


# -----------------------------
# Datasets
# -----------------------------
class MRISliceDataset(Dataset):
    """
    Supervised slice-level dataset.
    Each item: (image_tensor, label_index, metadata_dict)
    """

    def __init__(self, cfg: MRIDataConfig, transform: Optional[Callable] = None) -> None:
        self.cfg = cfg
        self.root = Path(cfg.root)
        self.transform = transform or default_transform(cfg.image_size)

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        patients = _list_patient_dirs(self.root)
        if not patients:
            raise RuntimeError(f"No class/patient folders found under: {self.root}")

        # Build records: (slice_path, label_str, patient_id, subtype, tumor_type, grade)
        self.records: List[Tuple[Path, str, str, str, str, int]] = []
        for subtype, patient_dir, tumor, grade in patients:
            slices = _list_slices(patient_dir)
            if not slices:
                continue
            patient_id = patient_dir.name
            for sp in slices:
                label_str = self._label_string(subtype=subtype, tumor=tumor, grade=grade)
                self.records.append((sp, label_str, patient_id, subtype, tumor, grade))

        if not self.records:
            raise RuntimeError(f"Found patient folders but no slice images under: {self.root}")

        # Fit encoder
        classes = sorted({r[1] for r in self.records})
        self.encoder = LabelEncoder(classes)

    def _label_string(self, subtype: str, tumor: str, grade: int) -> str:
        if self.cfg.target == "subtype":
            return subtype
        if self.cfg.target == "type":
            return tumor
        if self.cfg.target == "grade":
            return f"g{grade}"
        if self.cfg.target == "type_grade":
            return f"{tumor}-g{grade}"
        raise ValueError(f"Unknown target='{self.cfg.target}'")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        sp, label_str, patient_id, subtype, tumor, grade = self.records[idx]
        img = _read_image(sp, self.cfg.image_size, self.cfg.grayscale)
        x = self.transform(img)
        y = torch.tensor(self.encoder.encode(label_str), dtype=torch.long)

        meta = {
            "path": str(sp),
            "patient_id": patient_id,
            "subtype": subtype,
            "tumor_type": tumor,
            "grade": grade,
            "label_str": label_str,
        }
        return x, y, meta


class MRIBagDataset(Dataset):
    """
    Patient-level bag dataset for MIL / GNN.
    Each item: (bag_tensor, label_index, metadata)
      - bag_tensor: (N, C, H, W), where N = cfg.bag_size
    """

    def __init__(self, cfg: MRIDataConfig, transform: Optional[Callable] = None) -> None:
        if cfg.mode != "bag":
            raise ValueError("MRIBagDataset requires cfg.mode='bag'")

        self.cfg = cfg
        self.root = Path(cfg.root)
        self.transform = transform or default_transform(cfg.image_size)

        patients = _list_patient_dirs(self.root)
        if not patients:
            raise RuntimeError(f"No class/patient folders found under: {self.root}")

        self.patients: List[Tuple[str, Path, str, int]] = patients

        # Build label universe at patient-level
        label_strs = []
        for subtype, _, tumor, grade in self.patients:
            label_strs.append(self._label_string(subtype, tumor, grade))
        self.encoder = LabelEncoder(sorted(set(label_strs)))

        self.rng = np.random.default_rng(cfg.seed)

    def _label_string(self, subtype: str, tumor: str, grade: int) -> str:
        if self.cfg.target == "subtype":
            return subtype
        if self.cfg.target == "type":
            return tumor
        if self.cfg.target == "grade":
            return f"g{grade}"
        if self.cfg.target == "type_grade":
            return f"{tumor}-g{grade}"
        raise ValueError(f"Unknown target='{self.cfg.target}'")

    def __len__(self) -> int:
        return len(self.patients)

    def _select_indices(self, n_slices: int) -> np.ndarray:
        k = self.cfg.bag_size
        if n_slices <= 0:
            raise ValueError("Patient has no slices.")
        if k <= 0:
            raise ValueError("bag_size must be > 0")

        if n_slices >= k:
            if self.cfg.bag_policy == "uniform":
                # Sample without replacement uniformly
                return self.rng.choice(n_slices, size=k, replace=False)
            if self.cfg.bag_policy == "first":
                return np.arange(k)
            if self.cfg.bag_policy == "center":
                start = max(0, (n_slices - k) // 2)
                return np.arange(start, start + k)
            raise ValueError(f"Unknown bag_policy='{self.cfg.bag_policy}'")

        # If fewer slices than bag_size, pad by sampling with replacement
        return self.rng.choice(n_slices, size=k, replace=True)

    def __getitem__(self, idx: int):
        subtype, patient_dir, tumor, grade = self.patients[idx]
        slices = _list_slices(patient_dir)
        if not slices:
            raise RuntimeError(f"No slices found for patient: {patient_dir}")

        sel = self._select_indices(len(slices))
        chosen = [slices[i] for i in sel]

        bag_tensors: List[torch.Tensor] = []
        for sp in chosen:
            img = _read_image(sp, self.cfg.image_size, self.cfg.grayscale)
            bag_tensors.append(self.transform(img))

        bag = torch.stack(bag_tensors, dim=0)  # (N,C,H,W)
        label_str = self._label_string(subtype, tumor, grade)
        y = torch.tensor(self.encoder.encode(label_str), dtype=torch.long)

        meta = {
            "patient_id": patient_dir.name,
            "subtype": subtype,
            "tumor_type": tumor,
            "grade": grade,
            "label_str": label_str,
        }
        return bag, y, meta


class MRIContrastivePairDataset(Dataset):
    """
    Contrastive dataset that builds positive pairs from adjacent slices within the same patient.
    Each item: (x_i, x_j) tensors (C,H,W)
    """

    def __init__(self, cfg: MRIDataConfig, transform: Optional[Callable] = None) -> None:
        if cfg.mode != "contrastive":
            raise ValueError("MRIContrastivePairDataset requires cfg.mode='contrastive'")

        self.cfg = cfg
        self.root = Path(cfg.root)
        self.transform = transform or default_simclr_transform(cfg.image_size)

        patients = _list_patient_dirs(self.root)
        if not patients:
            raise RuntimeError(f"No class/patient folders found under: {self.root}")

        self.pairs: List[Tuple[Path, Path, str]] = []  # (a,b,patient_id)
        stride = max(1, int(cfg.pair_stride))

        for subtype, patient_dir, tumor, grade in patients:
            slices = _list_slices(patient_dir)
            if len(slices) < 2:
                continue
            for i in range(0, len(slices) - stride, stride):
                a = slices[i]
                b = slices[i + stride]
                self.pairs.append((a, b, patient_dir.name))

        if not self.pairs:
            raise RuntimeError(
                f"No contrastive pairs found under {self.root}. "
                "Check that patient folders contain >=2 slice images."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        a, b, patient_id = self.pairs[idx]
        img_a = _read_image(a, self.cfg.image_size, self.cfg.grayscale)
        img_b = _read_image(b, self.cfg.image_size, self.cfg.grayscale)
        x_i = self.transform(img_a)
        x_j = self.transform(img_b)
        return x_i, x_j


# -----------------------------
# Convenience factory
# -----------------------------
def make_dataset(
    *,
    root: Union[str, Path] = r"D:\datasets\MRI_Mahdieh_Datasets",
    mode: str = "slice",
    target: str = "subtype",
    image_size: int = 224,
    grayscale: bool = False,
    bag_size: int = 8,
    bag_policy: str = "uniform",
    pair_stride: int = 1,
    seed: int = 7,
    transform: Optional[Callable] = None,
):
    cfg = MRIDataConfig(
        root=root,
        mode=mode,
        target=target,
        image_size=image_size,
        grayscale=grayscale,
        bag_size=bag_size,
        bag_policy=bag_policy,
        pair_stride=pair_stride,
        seed=seed,
    )
    if mode == "slice":
        return MRISliceDataset(cfg, transform=transform)
    if mode == "bag":
        return MRIBagDataset(cfg, transform=transform)
    if mode == "contrastive":
        return MRIContrastivePairDataset(cfg, transform=transform)
    raise ValueError(f"Unknown mode='{mode}'. Expected one of: slice|bag|contrastive")