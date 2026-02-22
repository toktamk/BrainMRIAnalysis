# tests/contrastive/test_contrastive_pairs.py
import re
import pytest

from ContrastiveLearning.data import ContrastiveDataConfig, MRIPairDataset, IMG_EXTS


def _digits_int(stem: str):
    nums = re.findall(r"\d+", stem)
    return int(nums[-1]) if nums else None


@pytest.mark.contrastive
def test_pairs_within_patient_and_images_only(data_root):
    ds = MRIPairDataset(
        ContrastiveDataConfig(
            root=data_root,
            image_size=224,
            grayscale=False,
            pair_mode="adjacent",
            strict_images=True,
        )
    )

    N = min(300, len(ds))
    for i in range(N):
        a, b = ds._pairs[i]  # internal, but useful for auditing logic
        assert a.parent == b.parent, f"Pair crosses patient folder: {a} vs {b}"
        assert a.suffix.lower() in IMG_EXTS, f"Non-image in pairs: {a}"
        assert b.suffix.lower() in IMG_EXTS, f"Non-image in pairs: {b}"


@pytest.mark.contrastive
def test_numeric_adjacency_ratio(data_root):
    ds = MRIPairDataset(
        ContrastiveDataConfig(
            root=data_root,
            image_size=224,
            grayscale=False,
            pair_mode="adjacent",
            strict_images=True,
        )
    )

    N = min(500, len(ds))
    checked = 0
    ok = 0

    for i in range(N):
        a, b = ds._pairs[i]
        na = _digits_int(a.stem)
        nb = _digits_int(b.stem)
        if na is None or nb is None:
            continue
        checked += 1
        if nb == na + 1:
            ok += 1

    if checked == 0:
        pytest.skip("No numeric filenames detected; cannot verify numeric adjacency.")

    ratio = ok / checked
    assert ratio >= 0.90, f"Numeric adjacency too low: {ok}/{checked} = {ratio:.2%}"