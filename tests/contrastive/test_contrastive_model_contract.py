# tests/contrastive/test_contrastive_model_contract.py
import os
import inspect
import pytest
import torch
from torch.utils.data import DataLoader

from ContrastiveLearning.data import ContrastiveDataConfig, MRIPairDataset
from ContrastiveLearning.models import SimCLR, EncoderConfig
from ContrastiveLearning.losses import NTXentLoss


def _build_encoder_cfg() -> EncoderConfig:
    """
    Build EncoderConfig for SimCLR(cfg=EncoderConfig(...)).
    Also tries hard to DISABLE pretrained downloads for unit tests.

    Env vars you can set (optional):
      SIMCLR_ENCODER=small_cnn
      SIMCLR_PROJ_DIM=128
      SIMCLR_FEAT_DIM=128
      SIMCLR_PRETRAINED=0/1
    """
    # Fast path: defaults exist
    try:
        cfg = EncoderConfig()
    except TypeError:
        sig = inspect.signature(EncoderConfig)
        params = sig.parameters

        encoder = os.environ.get("SIMCLR_ENCODER", "small_cnn")
        proj_dim = int(os.environ.get("SIMCLR_PROJ_DIM", "128"))
        feat_dim = int(os.environ.get("SIMCLR_FEAT_DIM", str(proj_dim)))
        pretrained = os.environ.get("SIMCLR_PRETRAINED", "0").strip().lower() in {"1", "true", "yes"}

        kwargs = {}
        for name, p in params.items():
            if name == "self":
                continue
            if p.default is not inspect._empty:
                continue  # optional

            if name in {"encoder", "backbone", "base_encoder", "arch", "name", "model_name"}:
                kwargs[name] = encoder
            elif name in {"proj_dim", "projection_dim", "out_dim", "embedding_dim", "dim"}:
                kwargs[name] = proj_dim
            elif name in {"feat_dim", "feature_dim", "hidden_dim"}:
                kwargs[name] = feat_dim
            elif name in {"pretrained"}:
                kwargs[name] = pretrained
            else:
                raise AssertionError(
                    f"Cannot auto-construct EncoderConfig: required field {name!r} has no default. "
                    "Extend mapping in _build_encoder_cfg()."
                )

        cfg = EncoderConfig(**kwargs)

    # --- IMPORTANT: disable pretrained downloads if EncoderConfig supports it ---
    # We do this by setting known attribute names if present.
    for attr in ["pretrained", "use_pretrained", "imagenet_pretrained", "weights"]:
        if hasattr(cfg, attr):
            try:
                if attr == "weights":
                    setattr(cfg, attr, None)
                else:
                    setattr(cfg, attr, False)
            except Exception:
                pass

    # Some configs store backbone name; if default is mobilenet and triggers download,
    # you can override via env var SIMCLR_ENCODER=small_cnn
    return cfg


def _instantiate_simclr() -> torch.nn.Module:
    cfg = _build_encoder_cfg()
    try:
        return SimCLR(cfg=cfg, in_channels=3)
    except TypeError:
        return SimCLR(cfg, in_channels=3)


def _normalize_to_z_pair(out):
    """
    Expected: SimCLR(x_i, x_j) returns either:
      - (z_i, z_j)
      - dict with keys
      - a single tensor (unsupported for this model signature)
    """
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        z1, z2 = out[0], out[1]
    elif isinstance(out, dict):
        # common key conventions
        for k1, k2 in [("z_i", "z_j"), ("z1", "z2"), ("zi", "zj"), ("z", "z2"), ("p1", "p2")]:
            if k1 in out and k2 in out:
                z1, z2 = out[k1], out[k2]
                break
        else:
            raise AssertionError(f"Dict output keys {list(out.keys())} do not match expected z pair keys.")
    else:
        raise AssertionError(f"Unsupported SimCLR output type: {type(out)}")

    assert isinstance(z1, torch.Tensor) and isinstance(z2, torch.Tensor)
    assert z1.ndim == 2 and z2.ndim == 2, f"Expected (B,D) embeddings, got {z1.shape}, {z2.shape}"
    return z1, z2


@pytest.mark.contrastive
def test_model_forward_loss_backward(data_root):
    # real batch
    data_cfg = ContrastiveDataConfig(
        root=data_root,
        image_size=224,
        grayscale=False,
        pair_mode="adjacent",
        strict_images=True,
    )
    ds = MRIPairDataset(data_cfg)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    x_i, x_j = next(iter(loader))

    assert x_i.shape == x_j.shape
    assert x_i.ndim == 4  # (B,C,H,W)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _instantiate_simclr().to(device)
    model.train()

    x_i = x_i.to(device)
    x_j = x_j.to(device)

    # Forward MUST accept two views
    out = model(x_i, x_j)
    z_i, z_j = _normalize_to_z_pair(out)

    assert z_i.shape[0] == x_i.shape[0]
    assert z_j.shape[0] == x_j.shape[0]
    assert torch.isfinite(z_i).all(), "z_i contains NaN/Inf"
    assert torch.isfinite(z_j).all(), "z_j contains NaN/Inf"

    loss_fn = NTXentLoss(temperature=0.5)
    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(device)

    loss = loss_fn(z_i, z_j)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0, f"Loss must be scalar, got {tuple(loss.shape)}"
    assert torch.isfinite(loss).item(), "Loss is NaN/Inf"
    assert float(loss.item()) > 0.0, "Loss is not positive (suspicious)"

    model.zero_grad(set_to_none=True)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients (graph disconnected?)"
    assert all(torch.isfinite(g).all().item() for g in grads), "NaN/Inf gradients found"
    assert any(g.abs().sum().item() > 0 for g in grads), "All gradients are zero (suspicious)"