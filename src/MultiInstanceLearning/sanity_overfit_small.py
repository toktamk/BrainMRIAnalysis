from __future__ import annotations

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import MRIDataRead
from models import AttentionMIL


class BagDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, bagsize: int, image_size: int, channels: int):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.bagsize = bagsize
        self.image_size = image_size
        self.channels = channels

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        bag = torch.tensor(self.X[idx])
        bag = bag.reshape(self.bagsize, self.image_size, self.image_size, self.channels)
        bag = bag.permute(0, 3, 1, 2).contiguous()  # (N,C,H,W)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return bag, label


def main() -> None:
    ap = argparse.ArgumentParser(description="Overfit sanity check for MIL on tiny subset.")
    ap.add_argument("--data_root", required=True, type=str)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--bagsize", type=int, default=4)
    ap.add_argument("--bags_num", type=int, default=16)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reader = MRIDataRead.MRIDataRead(args.data_root, args.image_size)
    total_mris, targets, channel_num = reader.ReadData()
    bags, labels = reader.GenBags(total_mris, targets, args.bagsize, args.bags_num)

    X = np.array(bags, dtype=np.float32)
    y = np.array(labels, dtype=np.int64) - 1

    # Tiny subset
    X = X[: min(8, len(X))]
    y = y[: min(8, len(y))]

    ds = BagDataset(X, y, args.bagsize, args.image_size, channel_num)
    dl = DataLoader(ds, batch_size=4, shuffle=True)

    model = AttentionMIL(in_channels=max(1, channel_num), num_classes=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    it = iter(dl)
    for step in range(1, args.steps + 1):
        try:
            bag, label = next(it)
        except StopIteration:
            it = iter(dl)
            bag, label = next(it)

        bag, label = bag.to(device), label.to(device)
        logits = model(bag.unsqueeze(0) if bag.ndim == 4 else bag)  # ensure (B,N,C,H,W)
        loss = loss_fn(logits, label)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 20 == 0:
            pred = logits.argmax(dim=1)
            acc = (pred == label).float().mean().item() * 100.0
            print(f"[MIL Overfit] step={step:03d} loss={loss.item():.4f} batch_acc={acc:.2f}%")

    print("Overfit sanity run complete. Expect loss -> small and acc -> high on tiny set.")


if __name__ == "__main__":
    main()