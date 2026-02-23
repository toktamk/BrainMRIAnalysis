
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from MRIDataRead import MRIDataConfig, MRIBagTwoStepDataset
from graphs import SliceEncoder, slices_to_pyg_data
from models import TwoStepGNNClassifier, two_step_loss, two_step_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(n: int, seed: int, frac_train: float = 0.7, frac_val: float = 0.15) -> Tuple[List[int], List[int], List[int]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(n * frac_train))
    n_val = int(round(n * frac_val))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    train = idx[:n_train].tolist()
    val = idx[n_train:n_train + n_val].tolist()
    test = idx[n_train + n_val:].tolist()
    return train, val, test


def main() -> None:
    p = argparse.ArgumentParser(description="Two-step GNN training: tumor type -> grade conditioned on type")
    p.add_argument("--data_root", required=True, type=str)
    p.add_argument("--out_dir", type=str, default="runs/gnn_two_step")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--bag_size", type=int, default=8)
    p.add_argument("--bag_policy", type=str, default="uniform", choices=["uniform", "first", "center"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--node_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--lambda_type", type=float, default=1.0)
    p.add_argument("--lambda_grade", type=float, default=1.0)
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = MRIDataConfig(
        root=args.data_root,
        image_size=args.image_size,
        grayscale=False,
        mode="bag",
        bag_size=args.bag_size,
        bag_policy=args.bag_policy,
        seed=args.seed,
    )
    ds = MRIBagTwoStepDataset(cfg)
    n = len(ds)
    tr_idx, va_idx, te_idx = split_indices(n, args.seed)

    # Build graph list indices only; graphs are built on the fly to allow training node encoder
    def make_loader(idxs: List[int], shuffle: bool) -> DataLoader:
        # wrap as list of indices; DataLoader will collate Data objects we create in collate_fn
        sub = [idx for idx in idxs]

        def collate_fn(batch_indices):
            graphs = []
            for idx in batch_indices:
                bag, y_type, y_grade, meta = ds[int(idx)]
                bag = bag.to(device)
                g = slices_to_pyg_data(
                    bag,
                    encoder=node_encoder,
                    y_type=int(y_type.item()),
                    y_grade=int(y_grade.item()),
                    meta=meta,
                )
                graphs.append(g)
            # PyG DataLoader expects list[Data] if we bypass its collate; but we can just return list and let DataLoader from pyg handle it
            return graphs

        # We'll use  torch_geometric.loader.DataLoader directly on list[Data] per batch -> easiest is prebuild graphs per epoch.
        # So here we keep it simple: prebuild once per epoch in train/eval loops (below).
        return DataLoader([], batch_size=1)

    # Models
    node_encoder = SliceEncoder(in_channels=3, out_dim=args.node_dim).to(device)
    model = TwoStepGNNClassifier(in_dim=args.node_dim, hidden_dim=args.hidden_dim, num_types=len(ds.type_encoder), num_grades=len(ds.grade_encoder)).to(device)

    opt = torch.optim.Adam(list(node_encoder.parameters()) + list(model.parameters()), lr=args.lr)

    def build_graphs(idxs: List[int]) -> List:
        graphs = []
        for idx in idxs:
            bag, y_type, y_grade, meta = ds[int(idx)]
            bag = bag.to(device)
            graphs.append(
                slices_to_pyg_data(
                    bag,
                    encoder=node_encoder,
                    y_type=int(y_type.item()),
                    y_grade=int(y_grade.item()),
                    meta=meta,
                )
            )
        return graphs

    def eval_split(idxs: List[int]) -> dict:
        node_encoder.eval()
        model.eval()
        graphs = build_graphs(idxs)
        loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)
        mets = {"type_acc": 0.0, "grade_acc_given_true_type": 0.0, "end2end_grade_acc": 0.0, "end2end_all_correct": 0.0}
        n_batches = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                type_logits, grade_logits = model(batch)
                m = two_step_metrics(type_logits, grade_logits, batch.y_type.view(-1), batch.y_grade.view(-1))
                for k in mets:
                    mets[k] += m[k]
                n_batches += 1
        if n_batches > 0:
            for k in mets:
                mets[k] /= n_batches
        return mets

    best_val = -1.0
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        node_encoder.train()
        model.train()

        graphs = build_graphs(tr_idx)
        train_loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=True)

        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            type_logits, grade_logits = model(batch)
            loss = two_step_loss(
                type_logits,
                grade_logits,
                batch.y_type.view(-1),
                batch.y_grade.view(-1),
                lambda_type=args.lambda_type,
                lambda_grade=args.lambda_grade,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        val_m = eval_split(va_idx)
        val_score = val_m["end2end_all_correct"]  # strict metric

        print(
            f"[epoch {epoch:03d}] loss={total_loss/max(1,len(train_loader)):.4f} "
            f"val_type={val_m['type_acc']:.3f} val_gradeTF={val_m['grade_acc_given_true_type']:.3f} "
            f"val_e2e_grade={val_m['end2end_grade_acc']:.3f} val_all={val_m['end2end_all_correct']:.3f}"
        )

        if val_score > best_val:
            best_val = val_score
            torch.save(
                {
                    "node_encoder": node_encoder.state_dict(),
                    "model": model.state_dict(),
                    "type_classes": ds.type_encoder.classes,
                    "grade_classes": ds.grade_encoder.classes,
                    "cfg": cfg.__dict__,
                    "epoch": epoch,
                },
                best_path,
            )

    # Test
    ck = torch.load(best_path, map_location="cpu")
    node_encoder.load_state_dict(ck["node_encoder"], strict=False)
    model.load_state_dict(ck["model"], strict=False)

    test_m = eval_split(te_idx)
    print("[test]", test_m)

    (out_dir / "test_metrics.json").write_text(json.dumps(test_m, indent=2))
    print(f"saved -> {out_dir/'test_metrics.json'}")
    print(f"best ckpt -> {best_path}")


if __name__ == "__main__":
    main()
