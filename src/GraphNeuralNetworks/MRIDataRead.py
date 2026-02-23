
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import random, numpy as np, torch
random.seed(7); np.random.seed(7); torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

_TUMOR_GRADE_RE = re.compile(r"^(?P<tumor>.+)-g(?P<grade>[1-4])$", re.IGNORECASE)


def parse_folder_label(folder_name: str) -> Tuple[str, int]:
    m = _TUMOR_GRADE_RE.match(folder_name.strip())
    if not m:
        raise ValueError(f"Folder name '{folder_name}' does not match '<tumor>-g<1..4>'")
    return m.group("tumor").lower(), int(m.group("grade"))


@dataclass(frozen=True)
class MRIDataConfig:
    root: Union[str, Path]
    image_size: int = 224
    grayscale: bool = False
    mode: str = "bag"            # "bag" only here (GNN)
    bag_size: int = 8
    bag_policy: str = "first"  # "uniform" | "first" | "center"
    seed: int = 7


def default_transform(image_size: int) -> Callable:
    return T.Compose([T.ToPILImage(), T.Resize((image_size, image_size)), T.ToTensor()])


def _read_image(path: Path, image_size: int, grayscale: bool) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if grayscale:
        img = img[:, :, None]
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return img


def _list_patient_dirs(root: Path) -> List[Tuple[str, Path, str, int]]:
    out: List[Tuple[str, Path, str, int]] = []
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        tumor, grade = parse_folder_label(class_dir.name)
        for patient_dir in sorted([p for p in class_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            out.append((class_dir.name, patient_dir, tumor, grade))
    if not out:
        raise RuntimeError(f"No patient folders found under: {root}")
    return out


def _list_slices(patient_dir: Path) -> List[Path]:
    files = [p for p in patient_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda p: p.name)


class LabelEncoder:
    """Simple stable encoder with explicit class order. """
    def __init__(self, classes: Sequence[str]) -> None:
        self.classes = list(classes)
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}

    def encode(self, c: str) -> int:
        return self.class_to_idx[c]

    def __len__(self) -> int:
        return len(self.classes)


class MRIBagTwoStepDataset(Dataset):
    """
    Two-step bag dataset for GNN:

    Each item:
      bag: (N,C,H,W)
      y_type: scalar long tensor (0..T-1)
      y_grade: scalar long tensor (0..3)
      meta: dict
    """
    def __init__(self, cfg: MRIDataConfig, transform: Optional[Callable] = None) -> None:
        if cfg.mode != "bag":
            raise ValueError("MRIBagTwoStepDataset requires cfg.mode='bag'")

        self.cfg = cfg
        self.root = Path(cfg.root)
        self.transform = transform or default_transform(cfg.image_size)
        self.patients = _list_patient_dirs(self.root)

        # canonical type classes from dataset folders
        tumor_types = sorted({tumor for _, _, tumor, _ in self.patients})
        self.type_encoder = LabelEncoder(tumor_types)

        # grades are always g1..g4 but we keep 0..3
        self.grade_encoder = LabelEncoder(["g1", "g2", "g3", "g4"])

        self.rng = np.random.default_rng(cfg.seed)

    def __len__(self) -> int:
        return len(self.patients)

    def _select_indices(self, n_slices: int) -> np.ndarray:
        k = int(self.cfg.bag_size)
        if n_slices <= 0:
            raise ValueError("Patient has no slices.")
        if n_slices >= k:
            if self.cfg.bag_policy == "uniform":
                return self.rng.choice(n_slices, size=k, replace=False)
            if self.cfg.bag_policy == "first":
                return np.arange(k)
            if self.cfg.bag_policy == "center":
                start = max(0, (n_slices - k) // 2)
                return np.arange(start, start + k)
            raise ValueError(f"Unknown bag_policy='{self.cfg.bag_policy}'")
        return self.rng.choice(n_slices, size=k, replace=True)

    def __getitem__(self, idx: int):
        subtype, patient_dir, tumor, grade = self.patients[idx]
        slices = _list_slices(patient_dir)
        if not slices:
            raise RuntimeError(f"No slices found for patient: {patient_dir}")

        sel = self._select_indices(len(slices))
        chosen = [slices[i] for i in sel]

        bag = torch.stack([self.transform(_read_image(sp, self.cfg.image_size, self.cfg.grayscale)) for sp in chosen], dim=0)

        y_type = torch.tensor(self.type_encoder.encode(tumor), dtype=torch.long)
        y_grade = torch.tensor(self.grade_encoder.encode(f"g{int(grade)}"), dtype=torch.long)

        meta = {
            "patient_id": patient_dir.name,
            "subtype": subtype,
            "tumor_type": tumor,
            "grade": int(grade),
        }
        return bag, y_type, y_grade, meta
