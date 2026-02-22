from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PatientStats:
    grade: str
    patient_id: str
    n_slices: int
    unreadable: int


def iter_patients(root: Path) -> List[Tuple[str, Path]]:
    grades = [p for p in root.iterdir() if p.is_dir()]
    out = []
    for g in sorted(grades, key=lambda x: x.name):
        for patient in sorted([p for p in g.iterdir() if p.is_dir()], key=lambda x: x.name):
            out.append((g.name, patient))
    return out


def count_images(patient_dir: Path) -> List[Path]:
    files = [p for p in patient_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda x: x.name)


def try_read_image(p: Path) -> bool:
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    return img is not None


def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset audit: counts, integrity, slice stats.")
    ap.add_argument("--data_root", required=True, type=str)
    ap.add_argument("--max_read", type=int, default=2000, help="Max images to test decode for speed.")
    args = ap.parse_args()

    root = Path(args.data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    patients = iter_patients(root)
    if not patients:
        raise RuntimeError(f"No grade/patient folders found under: {root}")

    grade_counts = Counter()
    patient_stats: List[PatientStats] = []
    unreadable_total = 0

    # For quick decode test
    decode_budget = args.max_read
    decoded = 0

    for grade, patient_dir in patients:
        imgs = count_images(patient_dir)
        grade_counts[grade] += 1

        unreadable = 0
        for img_path in imgs:
            if decoded >= decode_budget:
                break
            decoded += 1
            if not try_read_image(img_path):
                unreadable += 1

        unreadable_total += unreadable
        patient_stats.append(
            PatientStats(
                grade=grade,
                patient_id=patient_dir.name,
                n_slices=len(imgs),
                unreadable=unreadable,
            )
        )

    # Aggregate slice statistics
    slices = np.array([ps.n_slices for ps in patient_stats], dtype=np.int64)
    by_grade: Dict[str, List[int]] = {}
    for ps in patient_stats:
        by_grade.setdefault(ps.grade, []).append(ps.n_slices)

    print("\n=== Dataset Audit ===")
    print(f"Root: {root}")
    print(f"Grades found: {sorted(grade_counts.keys())}")
    print(f"Patients total: {len(patient_stats)}")
    print("\nPatients per grade:")
    for g in sorted(grade_counts.keys()):
        print(f"  {g}: {grade_counts[g]}")

    print("\nSlice count per patient (all grades):")
    print(f"  min={int(slices.min())}  median={int(np.median(slices))}  max={int(slices.max())}  mean={float(slices.mean()):.2f}")

    print("\nSlice count per patient (by grade):")
    for g in sorted(by_grade.keys()):
        arr = np.array(by_grade[g], dtype=np.int64)
        print(
            f"  {g}: min={int(arr.min())} median={int(np.median(arr))} "
            f"max={int(arr.max())} mean={float(arr.mean()):.2f}"
        )

    print("\nIntegrity checks:")
    print(f"  Decode-tested images: {decoded} (budget={decode_budget})")
    print(f"  Unreadable among tested: {unreadable_total}")

    worst = sorted(patient_stats, key=lambda x: (-x.unreadable, -x.n_slices))[:10]
    if any(ps.unreadable > 0 for ps in worst):
        print("\nTop problematic patients (unreadable slices):")
        for ps in worst:
            if ps.unreadable == 0:
                break
            print(f"  grade={ps.grade} patient={ps.patient_id} unreadable={ps.unreadable} total_slices={ps.n_slices}")

    print("\nAudit complete.\n")


if __name__ == "__main__":
    main()