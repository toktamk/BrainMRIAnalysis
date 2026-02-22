# src/contrastivelearning/data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


# ---------------------------
# Config
# ---------------------------

@dataclass(frozen=True)
class ContrastiveDataConfig:
    root: Path
    image_size: int = 224
    grayscale: bool = False          # True => 1 channel (C=1), False => RGB (C=3)
    pair_mode: str = "adjacent"      # "adjacent" (a_i, a_{i+1}) or "same" (a_i, a_i)
    strict_images: bool = True       # if True, only known image extensions are considered


# ---------------------------
# Helpers
# ---------------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _numeric_sort_key(p: Path) -> Tuple[int, str]:
    """
    Robust ordering for filenames like:
      slice_1.png, slice_2.png, ..., slice_10.png
    If digits exist in stem, sort numerically; otherwise fall back to name.
    """
    stem = p.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    num = int(digits) if digits else 10**18
    return (num, p.name)


def _read_image(path: Path, image_size: int, grayscale: bool) -> np.ndarray:
    """
    Reads image with OpenCV and resizes to (image_size, image_size).

    Returns:
      - HxWx1 uint8 if grayscale=True
      - HxWx3 uint8 if grayscale=False (RGB order)
    """
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)

    if grayscale:
        # HxW -> HxWx1
        img = img[:, :, None]
    else:
        # OpenCV loads BGR; convert to RGB for torchvision consistency
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def default_simclr_transform(image_size: int) -> Callable:
    """
    Conservative augmentations (reasonable for MRI sanity checks).
    You can strengthen later once the pipeline is verified.
    """
    return T.Compose(
        [
            T.ToPILImage(),
            T.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.2),
            T.ToTensor(),  # float32 in [0,1], shape (C,H,W)
        ]
    )


# ---------------------------
# Dataset
# ---------------------------

class MRIPairDataset(Dataset):
    """
    Builds positive pairs from slices within the same patient folder.

    Expected directory layout:
        root/
          gradeX/
            patient_id/
              slice_001.png
              slice_002.png
              ...

    pair_mode:
      - "adjacent": (slice_i, slice_{i+1})
      - "same":     (slice_i, slice_i)  (useful for debugging transforms/loss)

    Returns:
      (x_i, x_j) where each is a torch.Tensor with shape (C, H, W)
    """

    def __init__(
        self,
        cfg: ContrastiveDataConfig,
        transform: Optional[Callable] = None,
    ) -> None:
        self.cfg = cfg
        self.transform = transform or default_simclr_transform(cfg.image_size)

        root = cfg.root
        if not root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")

        self._pairs: List[Tuple[Path, Path]] = []
        self._build_pairs()

        if len(self._pairs) == 0:
            raise RuntimeError(f"No slice pairs found under: {root}")

    def _collect_slice_files(self, patient_dir: Path) -> List[Path]:
        files = [p for p in patient_dir.iterdir() if p.is_file()]

        if self.cfg.strict_images:
            files = [p for p in files if p.suffix.lower() in IMG_EXTS]

        # Robust ordering: numeric if possible, else by name
        files = sorted(files, key=_numeric_sort_key)
        return files

    def _build_pairs(self) -> None:
        root = self.cfg.root

        for grade_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            for patient_dir in sorted([p for p in grade_dir.iterdir() if p.is_dir()]):
                slices = self._collect_slice_files(patient_dir)

                if self.cfg.pair_mode == "adjacent":
                    for a, b in zip(slices[:-1], slices[1:]):
                        self._pairs.append((a, b))
                elif self.cfg.pair_mode == "same":
                    for a in slices:
                        self._pairs.append((a, a))
                else:
                    raise ValueError(
                        f"Invalid pair_mode={self.cfg.pair_mode!r}. Use 'adjacent' or 'same'."
                    )

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        a_path, b_path = self._pairs[idx]

        a = _read_image(a_path, self.cfg.image_size, self.cfg.grayscale)
        b = _read_image(b_path, self.cfg.image_size, self.cfg.grayscale)

        x_i = self.transform(a)
        x_j = self.transform(b)

        return x_i, x_j