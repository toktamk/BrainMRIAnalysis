# tests/contrastive/test_contrastive_data_contract.py
import pytest
import torch
from torch.utils.data import DataLoader

# Your package name (you successfully imported this earlier)
from ContrastiveLearning.data import ContrastiveDataConfig, MRIPairDataset


@pytest.mark.contrastive
def test_contrastive_dataset_contract(data_root):
    cfg = ContrastiveDataConfig(
        root=data_root,
        image_size=224,
        grayscale=False,
        pair_mode="adjacent",
        strict_images=True,
    )
    ds = MRIPairDataset(cfg)

    n = len(ds)
    assert n > 0
    assert 500 < n < 200_000, f"Suspicious contrastive pair count: {n}"

    x1, x2 = ds[0]
    assert isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor)
    assert x1.dtype == torch.float32 and x2.dtype == torch.float32
    assert x1.shape == (3, 224, 224)
    assert x2.shape == (3, 224, 224)

    # ToTensor() range check
    assert float(x1.min()) >= 0.0 and float(x1.max()) <= 1.0
    assert float(x2.min()) >= 0.0 and float(x2.max()) <= 1.0

    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    xb1, xb2 = next(iter(loader))
    assert xb1.shape == (8, 3, 224, 224)
    assert xb2.shape == (8, 3, 224, 224)