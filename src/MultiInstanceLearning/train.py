from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import MRIDataRead  # existing module in folder
from models import AttentionMIL


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class BagDataset(Dataset):
    def __init__(self, bags: np.ndarray, labels: np.ndarray, bagsize: int, image_size: int, channels: int):
        self.bags = bags.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.bagsize = bagsize
        self.image_size = image_size
        self.channels = channels

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int):
        bag = torch.tensor(self.bags[idx])
        bag = bag.reshape(self.bagsize, self.image_size, self.image_size, self.channels)
        bag = bag.permute(0, 3, 1, 2).contiguous()  # (N,C,H,W)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return bag, y


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--bagsize", type=int, default=4)
    p.add_argument("--bags_num", type=int, default=50)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reader = MRIDataRead.MRIDataRead(args.data_root, args.image_size)
    total_mris, targets, channel_num = reader.ReadData()
    data_bags, target_bags = reader.GenBags(total_mris, targets, args.bagsize, args.bags_num)

    X = np.array(data_bags, dtype=np.float32)
    y = np.array(target_bags, dtype=np.int64) - 1  # shift {1..4} -> {0..3}

    # split
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(0.8 * len(idx))
    tr_idx, te_idx = idx[:split], idx[split:]

    train_ds = BagDataset(X[tr_idx], y[tr_idx], args.bagsize, args.image_size, channel_num)
    test_ds = BagDataset(X[te_idx], y[te_idx], args.bagsize, args.image_size, channel_num)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = AttentionMIL(in_channels=max(1, channel_num), num_classes=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    def eval_acc(loader: DataLoader) -> float:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bag, label in loader:
                bag, label = bag.to(device), label.to(device)
                logits = model(bag)
                pred = logits.argmax(dim=1)
                correct += int((pred == label).sum().item())
                total += int(label.numel())
        return 100.0 * correct / max(1, total)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for bag, label in train_loader:
            bag, label = bag.to(device), label.to(device)
            logits = model(bag)
            loss = loss_fn(logits, label)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())

        tr = eval_acc(train_loader)
        te = eval_acc(test_loader)
        print(f"[MIL] epoch={epoch:03d} loss={running/max(1,len(train_loader)):.4f} train_acc={tr:.2f}% test_acc={te:.2f}%")

    torch.save(model.state_dict(), "mil_model.pt")
    print("Saved: mil_model.pt")


if __name__ == "__main__":
    main()