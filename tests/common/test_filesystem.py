# tests/common/test_filesystem.py
from pathlib import Path
import pytest


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _iter_patient_dirs(root: Path):
    for grade_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for patient_dir in sorted([p for p in grade_dir.iterdir() if p.is_dir()]):
            yield grade_dir, patient_dir


def test_has_grade_folders(data_root: Path):
    grades = [p for p in data_root.iterdir() if p.is_dir()]
    assert len(grades) > 0, f"No grade folders found under: {data_root}"


def test_no_empty_patient_folders(data_root: Path):
    empties = []
    for grade_dir, patient_dir in _iter_patient_dirs(data_root):
        n_imgs = sum(1 for f in patient_dir.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS)
        if n_imgs == 0:
            empties.append((grade_dir.name, patient_dir.name))
    assert len(empties) == 0, f"Found empty patient folders: {empties[:20]}"


def test_reasonable_slice_counts(data_root: Path):
    # Loose bounds: you can tighten later.
    counts = []
    for _, patient_dir in _iter_patient_dirs(data_root):
        n_imgs = sum(1 for f in patient_dir.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS)
        counts.append(n_imgs)

    assert len(counts) > 0
    assert min(counts) >= 1
    assert max(counts) <= 10_000  # catches path mistakes / runaway recursion