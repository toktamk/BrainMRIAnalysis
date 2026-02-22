from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader

# Reuse your existing reader if you want, but keep training clean.
import MRIDataRead  # existing module in folder
from graphs import SliceEncoder, slices_to_pyg_data
from models import GNNClassifier


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--bagsize", type=int, default=5)
    p.add_argument("--bags_num", type=int, default=6)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reader = MRIDataRead.MRIDataRead(args.data_root, args.image_size)
    total_mris, targets, channel_num = reader.ReadData()
    data_bags, target_bags = reader.GenBags(total_mris, targets, args.bagsize, args.bags_num)

    # Map grade labels to {0..K-1} deterministically
    # reader.GenBags currently maps grade1..grade4 -> 1..4; we shift to 0..3.
    y = torch.tensor(np.array(target_bags) - 1, dtype=torch.long)

    # Prepare node encoder and graphs
    node_encoder = SliceEncoder(in_channels=max(1, channel_num), out_dim=64).to(device)
    node_encoder.eval()

    graphs = []
    for bag, yi in zip(np.array(data_bags, dtype=np.float32), y.tolist()):
        # bag shape: (bagsize, H*W*C) in current reader; reshape to (N,C,H,W)
        slices = torch.tensor(bag).reshape(args.bagsize, args.image_size, args.image_size, channel_num)
        slices = slices.permute(0, 3, 1, 2).contiguous()  # (N,C,H,W)
        data = slices_to_pyg_data(slices, encoder=node_encoder.cpu(), y=yi)
        graphs.append(data)

    # Train/test split
    idx = np.arange(len(graphs))
    np.random.shuffle(idx)
    split = int(0.8 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]

    train_set = [graphs[i] for i in train_idx]
    test_set = [graphs[i] for i in test_idx]

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = GNNClassifier(in_dim=train_set[0].x.shape[1], hidden_dim=128, num_classes=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    def eval_acc(loader: DataLoader) -> float:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch)
                pred = logits.argmax(dim=1)
                correct += int((pred == batch.y.view(-1)).sum().item())
                total += int(batch.y.numel())
        return 100.0 * correct / max(1, total)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = loss_fn(logits, batch.y.view(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        tr = eval_acc(train_loader)
        te = eval_acc(test_loader)
        print(f"[GNN] epoch={epoch:03d} train_acc={tr:.2f}% test_acc={te:.2f}%")

    torch.save(model.state_dict(), "gnn_model.pt")
    print("Saved: gnn_model.pt")


if __name__ == "__main__":
    main()