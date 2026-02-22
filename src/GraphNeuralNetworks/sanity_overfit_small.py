from __future__ import annotations

import argparse
import numpy as np
import torch
from torch_geometric.loader import DataLoader

import MRIDataRead
from graphs import SliceEncoder, slices_to_pyg_data
from models import GNNClassifier


def main() -> None:
    ap = argparse.ArgumentParser(description="Overfit sanity check for GNN on tiny subset.")
    ap.add_argument("--data_root", required=True, type=str)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--bagsize", type=int, default=5)
    ap.add_argument("--bags_num", type=int, default=12)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reader = MRIDataRead.MRIDataRead(args.data_root, args.image_size)
    total_mris, targets, channel_num = reader.ReadData()
    bags, labels = reader.GenBags(total_mris, targets, args.bagsize, args.bags_num)

    X = np.array(bags, dtype=np.float32)
    y = np.array(labels, dtype=np.int64) - 1

    node_encoder = SliceEncoder(in_channels=max(1, channel_num), out_dim=64)
    node_encoder.eval()

    graphs = []
    for bag, yi in zip(X, y):
        slices = torch.tensor(bag).reshape(args.bagsize, args.image_size, args.image_size, channel_num)
        slices = slices.permute(0, 3, 1, 2).contiguous()
        graphs.append(slices_to_pyg_data(slices, encoder=node_encoder, y=int(yi)))

    # Tiny subset
    graphs = graphs[: min(8, len(graphs))]
    loader = DataLoader(graphs, batch_size=4, shuffle=True)

    model = GNNClassifier(in_dim=graphs[0].x.shape[1], hidden_dim=64, num_classes=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    it = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        batch = batch.to(device)
        logits = model(batch)
        loss = loss_fn(logits, batch.y.view(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 20 == 0:
            pred = logits.argmax(dim=1)
            acc = (pred == batch.y.view(-1)).float().mean().item() * 100.0
            print(f"[GNN Overfit] step={step:03d} loss={loss.item():.4f} batch_acc={acc:.2f}%")

    print("Overfit sanity run complete. Expect loss -> small and acc -> high on tiny set.")


if __name__ == "__main__":
    main()