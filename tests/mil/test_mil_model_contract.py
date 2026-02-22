# tests/mil/test_mil_model_contract.py
import pytest
import torch

pytestmark = pytest.mark.mil


def test_attentionmil_forward_smoke():
    """
    Contract: AttentionMIL forward expects a bag of instances shaped (B, N, C, H, W).
    It should return a finite tensor output (typically logits per bag).
    """
    try:
        from MultiInstanceLearning.models import AttentionMIL
    except ModuleNotFoundError as e:
        pytest.skip(f"Cannot import MultiInstanceLearning.models.AttentionMIL: {e}")

    # Instantiate model (your AttentionMIL appears to have a no-arg constructor)
    model = AttentionMIL()
    model.eval()

    # Dummy bag: (B, N, C, H, W)
    # Keep it small enough for tests, but compatible with CNNs
    B, N, C, H, W = 2, 4, 3, 224, 224
    bag = torch.randn(B, N, C, H, W)

    with torch.no_grad():
        y = model(bag)

    assert isinstance(y, torch.Tensor), f"Output must be a tensor, got {type(y)}"
    assert torch.isfinite(y).all().item(), "Output contains NaN/Inf"

    # Typical MIL classifier returns (B, num_classes) or (B,) etc.
    assert y.shape[0] == B, f"Batch dim mismatch: expected {B}, got {y.shape[0]}"