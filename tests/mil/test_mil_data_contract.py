# tests/mil/test_mil_data_contract.py
import pytest
import torch
from torch.utils.data import DataLoader

pytestmark = pytest.mark.mil


def _make_bag_dataset(data_root):
    from MultiInstanceLearning.MRIDataRead import MRIDataConfig, MRIBagDataset

    cfg = MRIDataConfig(
        root=data_root,
        mode="bag",
        target="subtype",
        image_size=224,
        grayscale=False,
        bag_size=8,
        bag_policy="uniform",
        seed=7,
    )
    ds = MRIBagDataset(cfg)
    return ds, cfg


def test_mil_bag_dataset_contract(data_root):
    ds, cfg = _make_bag_dataset(data_root)

    assert len(ds) > 0, "MRIBagDataset is empty"

    bag, y, meta = ds[0]

    assert isinstance(bag, torch.Tensor), f"bag must be Tensor, got {type(bag)}"
    assert bag.ndim == 4, f"bag must be (N,C,H,W), got {tuple(bag.shape)}"

    N, C, H, W = bag.shape
    assert N == cfg.bag_size, f"Expected N==bag_size=={cfg.bag_size}, got {N}"
    assert C == 3, f"Expected 3 channels (grayscale=False), got {C}"
    assert H == cfg.image_size and W == cfg.image_size, f"Expected {cfg.image_size}x{cfg.image_size}, got {H}x{W}"

    assert bag.dtype == torch.float32, f"Expected float32, got {bag.dtype}"
    assert torch.isfinite(bag).all().item(), "bag contains NaN/Inf"
    # ToTensor() range check
    assert float(bag.min()) >= 0.0 and float(bag.max()) <= 1.0, "bag values not in [0,1]"

    assert isinstance(y, torch.Tensor) and y.dtype == torch.long and y.ndim == 0, f"y must be scalar long tensor, got {y}"
    assert isinstance(meta, dict), f"meta must be dict, got {type(meta)}"
    for k in ["patient_id", "subtype", "tumor_type", "grade", "label_str"]:
        assert k in meta, f"meta missing key: {k}"


def test_mil_bag_size_distribution_smoke(data_root):
    ds, cfg = _make_bag_dataset(data_root)

    n = min(25, len(ds))
    sizes = []
    for i in range(n):
        bag, _, _ = ds[i]
        sizes.append(int(bag.shape[0]))

    assert min(sizes) == cfg.bag_size, f"All bags should have size {cfg.bag_size}, got min={min(sizes)}"
    assert max(sizes) == cfg.bag_size, f"All bags should have size {cfg.bag_size}, got max={max(sizes)}"


def test_mil_dataloader_batch_shape(data_root):
    ds, cfg = _make_bag_dataset(data_root)

    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
    bag_b, y_b, meta_b = next(iter(loader))

    # Default collate stacks bags: (B, N, C, H, W)
    assert bag_b.ndim == 5, f"Expected (B,N,C,H,W), got {tuple(bag_b.shape)}"
    B, N, C, H, W = bag_b.shape

    assert B == 2
    assert N == cfg.bag_size
    assert C == 3
    assert H == cfg.image_size and W == cfg.image_size

    assert y_b.shape == (B,), f"Expected labels (B,), got {tuple(y_b.shape)}"
    assert isinstance(meta_b, dict) or isinstance(meta_b, list), "meta batch should be collated (dict or list)"