# tests/gnn/test_gnn_model_contract.py
import inspect
import pytest
import torch

pytestmark = pytest.mark.gnn


def _instantiate_gnn_classifier():
    from GraphNeuralNetworks import models as m

    # Prefer GNNClassifier if present
    if hasattr(m, "GNNClassifier"):
        Model = m.GNNClassifier
    else:
        # fallback: pick first class containing 'gnn' or 'graph'
        Model = None
        for name, obj in vars(m).items():
            if inspect.isclass(obj) and obj.__module__ == m.__name__:
                if "gnn" in name.lower() or "graph" in name.lower():
                    Model = obj
                    break
        if Model is None:
            raise AssertionError("No GNN model class found in GraphNeuralNetworks.models")

    # Try no-arg construction first; otherwise provide common kwargs
    try:
        return Model()
    except TypeError:
        sig = inspect.signature(Model)
        kwargs = {}
        for k in sig.parameters:
            if k in ("in_dim", "in_channels", "input_dim", "num_features"):
                kwargs[k] = 64
            if k in ("hidden_dim", "hidden", "hid_dim"):
                kwargs[k] = 64
            if k in ("num_classes", "out_dim", "n_classes", "classes"):
                kwargs[k] = 4  # your MIL model shows 4 classes; adjust if needed
        return Model(**kwargs)


@pytest.mark.gnn
def test_gnn_forward_loss_backward(data_root):
    from GraphNeuralNetworks.MRIDataRead import MRIDataConfig, MRIBagDataset
    from GraphNeuralNetworks.graphs import SliceEncoder, slices_to_pyg_data

    # Build a deterministic bag
    cfg = MRIDataConfig(
        root=data_root,
        mode="bag",
        target="subtype",
        image_size=224,
        grayscale=False,
        bag_size=8,
        bag_policy="first",
        seed=7,
    )
    ds = MRIBagDataset(cfg)
    bag, y, meta = ds[0]

    # Build graph with known node feature dim
    enc = SliceEncoder(out_dim=64)
    data = slices_to_pyg_data(bag, y=int(y.item()), encoder=enc)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _instantiate_gnn_classifier().to(device)
    model.train()

    # Move graph tensors to device
    if hasattr(data, "to"):
        data = data.to(device)
    else:
        # minimal fallback
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        if hasattr(data, "y") and isinstance(data.y, torch.Tensor):
            data.y = data.y.to(device)

    # Forward
    out = model(data)

    assert isinstance(out, torch.Tensor), f"Model output must be tensor, got {type(out)}"
    assert torch.isfinite(out).all().item(), "Output contains NaN/Inf"

    # Determine target + loss
    # Common: out is (num_classes,) or (1,num_classes) or (B,num_classes)
    target = data.y
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, dtype=torch.long, device=device)

    if out.ndim == 1:
        out2 = out.unsqueeze(0)          # (1,C)
        tgt2 = target.view(1)            # (1,)
    elif out.ndim == 2:
        out2 = out
        tgt2 = target.view(-1)
        if out2.shape[0] != tgt2.shape[0]:
            # graph-level single label but batched output
            tgt2 = tgt2[: out2.shape[0]]
    else:
        raise AssertionError(f"Unexpected output shape: {tuple(out.shape)}")

    # If output has 1 logit, treat as binary; else multiclass
    if out2.shape[1] == 1:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out2.view(-1), tgt2.float()
        )
    else:
        loss = torch.nn.functional.cross_entropy(out2, tgt2.long())

    assert torch.isfinite(loss).item()
    model.zero_grad(set_to_none=True)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients found"
    assert all(torch.isfinite(g).all().item() for g in grads), "NaN/Inf in gradients"
    assert any(g.abs().sum().item() > 0 for g in grads), "All gradients are zero (suspicious)"