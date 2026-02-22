# tests/gnn/test_gnn_graph_builder_contract.py
import pytest
import torch

pytestmark = pytest.mark.gnn


@pytest.mark.gnn
def test_graph_builder_with_slice_encoder_contract(data_root):
    """
    Contract for GraphNeuralNetworks.graphs.slices_to_pyg_data:

    - Input: bag of slices (N,C,H,W)
    - Uses SliceEncoder(out_dim=64) -> node features x shape (N,64)
    - Chain graph: undirected edges => E = 2*(N-1) when N>1
    - Output is PyG Data-like with x, edge_index, y optional
    """
    from GraphNeuralNetworks.MRIDataRead import MRIDataConfig, MRIBagDataset
    from GraphNeuralNetworks.graphs import SliceEncoder, slices_to_pyg_data

    # Build a small bag dataset
    cfg = MRIDataConfig(
        root=data_root,
        mode="bag",
        target="subtype",
        image_size=224,
        grayscale=False,
        bag_size=8,
        bag_policy="first",  # deterministic for test
        seed=7,
    )
    ds = MRIBagDataset(cfg)

    bag, y, meta = ds[0]
    assert isinstance(bag, torch.Tensor) and bag.ndim == 4
    N, C, H, W = bag.shape
    assert N == cfg.bag_size

    enc = SliceEncoder(out_dim=64)
    data = slices_to_pyg_data(bag, y=int(y.item()), encoder=enc)

    # PyG Data-like
    assert hasattr(data, "x") and hasattr(data, "edge_index")
    assert isinstance(data.x, torch.Tensor)
    assert isinstance(data.edge_index, torch.Tensor)

    # Node feature contract
    assert data.x.shape == (N, 64), f"Expected x=(N,64), got {tuple(data.x.shape)}"
    assert torch.isfinite(data.x).all().item()

    # Edge contract (chain, undirected)
    assert data.edge_index.shape[0] == 2
    E = data.edge_index.shape[1]
    if N == 1:
        assert E == 0
    else:
        assert E == 2 * (N - 1), f"Expected E=2*(N-1)={2*(N-1)}, got {E}"

    # Index validity
    if E > 0:
        assert int(data.edge_index.min()) >= 0
        assert int(data.edge_index.max()) < N

    # Label contract
    assert hasattr(data, "y"), "Expected data.y to exist"
    assert int(data.y) == int(y.item())