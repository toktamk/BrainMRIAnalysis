# tests/gnn/test_gnn_data_contract.py
import pytest
import torch

pytestmark = pytest.mark.gnn


def _make_bag_dataset(data_root):
    from GraphNeuralNetworks.MRIDataRead import MRIDataConfig, MRIBagDataset

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


def _find_graph_builder():
    """
    Try to find a graph builder in GraphNeuralNetworks package.
    Expected outputs: torch_geometric.data.Data-like with x and edge_index.
    """
    import importlib

    # Common locations/names in GNN repos
    module_candidates = [
        "GraphNeuralNetworks.graphs",
        "GraphNeuralNetworks.graph_builder",
        "GraphNeuralNetworks.data",
        "GraphNeuralNetworks.dataset",
        "GraphNeuralNetworks.utils",
        "GraphNeuralNetworks",
    ]

    fn_candidates = [
        "slices_to_pyg_data",
        "bag_to_graph",
        "build_graph",
        "make_graph",
        "to_pyg_data",
        "create_graph",
    ]

    cls_candidates = [
        "GraphDataset",
        "GNNDataset",
        "MRIGraphDataset",
        "GraphBuilder",
    ]

    for mod_name in module_candidates:
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            continue

        for fn in fn_candidates:
            if hasattr(mod, fn) and callable(getattr(mod, fn)):
                return ("fn", getattr(mod, fn))

        for cls in cls_candidates:
            if hasattr(mod, cls):
                return ("cls", getattr(mod, cls))

    return (None, None)


def _extract_graph(obj):
    """
    Accept:
      - PyG Data object with .x and .edge_index
      - dict with x/edge_index
      - tuple (x, edge_index, y?)
    """
    if hasattr(obj, "x") and hasattr(obj, "edge_index"):
        return obj.x, obj.edge_index, getattr(obj, "y", None), obj

    if isinstance(obj, dict):
        return obj.get("x"), obj.get("edge_index"), obj.get("y", None), obj

    if isinstance(obj, (tuple, list)) and len(obj) >= 2:
        x = obj[0]
        edge_index = obj[1]
        y = obj[2] if len(obj) >= 3 else None
        return x, edge_index, y, obj

    raise AssertionError(f"Unsupported graph object type: {type(obj)}")


def test_gnn_data_contract(data_root):
    # Step 1: Bag dataset must be constructible (this is your current GNN reader)
    ds, cfg = _make_bag_dataset(data_root)
    assert len(ds) > 0, "GraphNeuralNetworks MRIBagDataset is empty"

    bag, y, meta = ds[0]
    assert isinstance(bag, torch.Tensor) and bag.ndim == 4, f"Expected (N,C,H,W) bag, got {type(bag)} {tuple(bag.shape)}"
    assert bag.shape[0] == cfg.bag_size
    assert torch.isfinite(bag).all().item()

    # Step 2: Find graph builder and validate real graph object
    kind, builder = _find_graph_builder()
    if builder is None:
        raise AssertionError(
            "No graph builder found in GraphNeuralNetworks.*.\n"
            "Expected a function like bag_to_graph/build_graph/to_pyg_data or a GraphDataset class.\n"
            "Implement a graph conversion step (bag -> torch_geometric.data.Data) and expose it."
        )

    if kind == "fn":
        graph_obj = builder(bag=bag, y=y, meta=meta) if "bag" in builder.__code__.co_varnames else builder(bag)
    else:
        # Dataset-style graph builder: instantiate and index
        # Try common constructor patterns
        try:
            graph_ds = builder(root=data_root)
        except Exception:
            graph_ds = builder(cfg=cfg) if "cfg" in getattr(builder, "__init__").__code__.co_varnames else builder()
        graph_obj = graph_ds[0]

    x, edge_index, y2, _ = _extract_graph(graph_obj)

    assert isinstance(x, torch.Tensor), f"x must be Tensor, got {type(x)}"
    assert x.ndim == 2, f"x must be (num_nodes, num_features), got {tuple(x.shape)}"
    assert x.shape[0] >= 1
    assert torch.isfinite(x).all().item()

    assert isinstance(edge_index, torch.Tensor), f"edge_index must be Tensor, got {type(edge_index)}"
    assert edge_index.ndim == 2 and edge_index.shape[0] == 2, f"edge_index must be (2,E), got {tuple(edge_index.shape)}"
    if edge_index.numel() > 0:
        assert int(edge_index.min()) >= 0
        assert int(edge_index.max()) < x.shape[0], "edge_index references invalid node id"

    # y optional, but if present must be finite scalar-ish
    if y2 is not None:
        if isinstance(y2, torch.Tensor):
            assert y2.numel() in (1, x.shape[0])
            assert torch.isfinite(y2).all().item()