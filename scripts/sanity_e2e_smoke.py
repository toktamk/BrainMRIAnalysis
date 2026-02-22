from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# 1) Contrastive (SimCLR)
# -------------------------
def run_contrastive(
    data_root: Path,
    out_dir: Path,
    steps: int,
    batch_size: int,
    image_size: int,
    seed: int,
) -> Path:
    """
    Minimal SimCLR sanity training using your ContrastiveLearning package.
    """
    from ContrastiveLearning.data import ContrastiveDataConfig, MRIPairDataset
    from ContrastiveLearning.models import EncoderConfig, SimCLR
    from ContrastiveLearning.losses import NTXentLoss

    cfg = ContrastiveDataConfig(
        root=data_root,
        image_size=image_size,
        grayscale=False,
        pair_mode="adjacent",
        strict_images=True,
    )

    ds = MRIPairDataset(cfg)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    device = get_device()

    # Ensure we NEVER download weights in sanity runs
    enc_cfg = EncoderConfig(name="small_cnn", pretrained=False, projection_dim=128)
    model = SimCLR(enc_cfg, in_channels=3).to(device)
    loss_fn = NTXentLoss(temperature=0.5).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    it = iter(loader)

    for step in range(steps):
        try:
            x1, x2 = next(it)
        except StopIteration:
            it = iter(loader)
            x1, x2 = next(it)

        x1 = x1.to(device)
        x2 = x2.to(device)

        # Robust SimCLR output handling:
        # Common patterns:
        #   (z1, z2)
        #   (h1, h2, z1, z2)
        #   dict with keys z_i/z_j or z1/z2 etc.
        out = model(x1, x2)

        if isinstance(out, dict):
            if "z_i" in out and "z_j" in out:
                z1, z2 = out["z_i"], out["z_j"]
            elif "z1" in out and "z2" in out:
                z1, z2 = out["z1"], out["z2"]
            elif "proj_i" in out and "proj_j" in out:
                z1, z2 = out["proj_i"], out["proj_j"]
            else:
                raise ValueError(f"Unknown SimCLR dict outputs: {list(out.keys())}")

        elif isinstance(out, (tuple, list)):
            if len(out) == 2:
                z1, z2 = out
            elif len(out) >= 4:
                # Convention: projections are the last two tensors
                z1, z2 = out[-2], out[-1]
            else:
                raise ValueError(f"Unexpected SimCLR tuple length: {len(out)}")

        else:
            raise ValueError(f"Unexpected SimCLR output type: {type(out)}")

        loss = loss_fn(z1, z2)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % max(1, steps // 5) == 0:
            print(f"[contrastive] step {step+1}/{steps} loss={loss.item():.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "simclr_sanity.pt"
    torch.save(
        {
            "encoder_config": asdict(enc_cfg),
            "model_state": model.state_dict(),
            "data_config": asdict(cfg),
            "seed": seed,
        },
        ckpt,
    )
    print(f"[contrastive] saved checkpoint -> {ckpt}")
    return ckpt


# -------------------------
# 2) MIL (AttentionMIL)
# -------------------------
def run_mil(
    data_root: Path,
    out_dir: Path,
    steps: int,
    batch_size: int,
    image_size: int,
    bag_size: int,
    seed: int,
) -> Path:
    """
    Minimal MIL sanity training.
    """
    from MultiInstanceLearning.MRIDataRead import MRIDataConfig, MRIBagDataset
    from MultiInstanceLearning.models import AttentionMIL

    cfg = MRIDataConfig(
        root=data_root,
        mode="bag",
        target="grade",
        image_size=image_size,
        grayscale=False,
        bag_size=bag_size,
        bag_policy="uniform",
        seed=seed,
    )

    ds = MRIBagDataset(cfg)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    device = get_device()
    model = AttentionMIL().to(device)

    num_classes = len(ds.encoder)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    it = iter(loader)

    for step in range(steps):
        try:
            bag, y, meta = next(it)
        except StopIteration:
            it = iter(loader)
            bag, y, meta = next(it)

        # bag: (B, N, C, H, W)
        bag = bag.to(device)
        y = y.to(device)

        logits = model(bag)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        # Safety check before cross_entropy (prevents CUDA assertion)
        num_logits = logits.shape[-1]
        y_max = int(y.max().item())
        y_min = int(y.min().item())
        if not (0 <= y_min and y_max < num_logits):
            raise ValueError(
                f"[mil] Label out of range for logits. "
                f"y in [{y_min},{y_max}] but logits has {num_logits} classes. "
                f"Fix by using cfg.target='grade' (4 classes) or making AttentionMIL output match."
            )
        loss = torch.nn.functional.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % max(1, steps // 5) == 0:
            print(f"[mil] step {step+1}/{steps} loss={loss.item():.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "mil_sanity.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "num_classes": num_classes,
            "data_config": asdict(cfg),
            "seed": seed,
        },
        ckpt,
    )
    print(f"[mil] saved checkpoint -> {ckpt}")
    return ckpt


# -------------------------
# 3) GNN (graph builder + classifier)
# -------------------------
def run_gnn(
    data_root: Path,
    out_dir: Path,
    steps: int,
    image_size: int,
    bag_size: int,
    seed: int,
) -> Path:
    """
    Minimal GNN sanity training.
    """
    from GraphNeuralNetworks.MRIDataRead import MRIDataConfig, MRIBagDataset
    from GraphNeuralNetworks.graphs import SliceEncoder, slices_to_pyg_data
    from GraphNeuralNetworks.models import GNNClassifier

    cfg = MRIDataConfig(
        root=data_root,
        mode="bag",
        target="grade",
        image_size=image_size,
        grayscale=False,
        bag_size=bag_size,
        bag_policy="uniform",
        seed=seed,
    )

    ds = MRIBagDataset(cfg)

    device = get_device()
    encoder = SliceEncoder(out_dim=64).to(device)
    #model = GNNClassifier(num_classes=len(ds.encoder), in_dim=64).to(device)

    hidden_dim = int(os.environ.get("GNN_HIDDEN_DIM", "64"))
    model = GNNClassifier(num_classes=len(ds.encoder), in_dim=64, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(model.parameters()), lr=1e-3)

    model.train()
    encoder.train()

    for step in range(steps):
        idx = step % len(ds)
        bag, y, meta = ds[idx]  # bag: (N, C, H, W)
        bag = bag.to(device)
        y = y.to(device)

        data = slices_to_pyg_data(bag, y=int(y.item()), encoder=encoder)
        data = data.to(device)

        logits = model(data)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        target = data.y
        if isinstance(target, torch.Tensor) and target.ndim == 0:
            target = target.view(1)

        loss = torch.nn.functional.cross_entropy(logits, target.long())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % max(1, steps // 5) == 0:
            print(f"[gnn] step {step+1}/{steps} loss={loss.item():.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "gnn_sanity.pt"
    torch.save(
        {
            "encoder_state": encoder.state_dict(),
            "model_state": model.state_dict(),
            "data_config": asdict(cfg),
            "seed": seed,
        },
        ckpt,
    )
    print(f"[gnn] saved checkpoint -> {ckpt}")
    return ckpt

# -------------------------
# Summary helpers
# -------------------------
def summarize_dataset_sizes(data_root: Path, image_size: int, bag_size: int, seed: int):
    """
    Builds lightweight datasets to inspect sizes without running training.
    """

    summary = {}

    # Contrastive
    try:
        from ContrastiveLearning.data import ContrastiveDataConfig, MRIPairDataset

        cfg_c = ContrastiveDataConfig(
            root=data_root,
            image_size=image_size,
            grayscale=False,
            pair_mode="adjacent",
            strict_images=True,
        )
        ds_c = MRIPairDataset(cfg_c)
        summary["contrastive_pairs"] = len(ds_c)
    except Exception as e:
        summary["contrastive_pairs"] = f"ERROR: {e}"

    # MIL
    try:
        from MultiInstanceLearning.MRIDataRead import MRIDataConfig, MRIBagDataset

        cfg_m = MRIDataConfig(
            root=data_root,
            mode="bag",
            target="grade",
            image_size=image_size,
            grayscale=False,
            bag_size=bag_size,
            bag_policy="uniform",
            seed=seed,
        )
        ds_m = MRIBagDataset(cfg_m)
        summary["mil_patients"] = len(ds_m)
        summary["mil_num_classes"] = len(ds_m.encoder)
    except Exception as e:
        summary["mil_patients"] = f"ERROR: {e}"

    # GNN
    try:
        from GraphNeuralNetworks.MRIDataRead import MRIDataConfig, MRIBagDataset

        cfg_g = MRIDataConfig(
            root=data_root,
            mode="bag",
            target="grade",
            image_size=image_size,
            grayscale=False,
            bag_size=bag_size,
            bag_policy="uniform",
            seed=seed,
        )
        ds_g = MRIBagDataset(cfg_g)
        summary["gnn_patients"] = len(ds_g)
        summary["gnn_num_classes"] = len(ds_g.encoder)
    except Exception as e:
        summary["gnn_patients"] = f"ERROR: {e}"

    return summary
# -------------------------
# Main
# -------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=os.environ.get("DATA_ROOT", ""), help="Dataset root path")
    p.add_argument("--out_dir", type=str, default="runs/e2e_sanity", help="Output directory")

    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--seed", type=int, default=7)

    # sanity runtime knobs
    p.add_argument("--contrastive_steps", type=int, default=20)
    p.add_argument("--contrastive_batch", type=int, default=16)

    p.add_argument("--mil_steps", type=int, default=10)
    p.add_argument("--mil_batch", type=int, default=2)
    p.add_argument("--bag_size", type=int, default=8)

    p.add_argument("--gnn_steps", type=int, default=10)

    args = p.parse_args()

    if not args.data_root:
        raise SystemExit(
            "ERROR: --data_root not provided and DATA_ROOT env var is not set.\n"
            "Example: set DATA_ROOT or pass --data_root D:\\datasets\\MRI_Mahdieh_Datasets"
        )

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)

    print(f"[e2e] data_root={data_root}")
    print(f"[e2e] out_dir={out_dir}")

    set_seed(args.seed)

    simclr_ckpt = run_contrastive(
        data_root=data_root,
        out_dir=out_dir / "contrastive",
        steps=args.contrastive_steps,
        batch_size=args.contrastive_batch,
        image_size=args.image_size,
        seed=args.seed,
    )

    mil_ckpt = run_mil(
        data_root=data_root,
        out_dir=out_dir / "mil",
        steps=args.mil_steps,
        batch_size=args.mil_batch,
        image_size=args.image_size,
        bag_size=args.bag_size,
        seed=args.seed,
    )

    gnn_ckpt = run_gnn(
        data_root=data_root,
        out_dir=out_dir / "gnn",
        steps=args.gnn_steps,
        image_size=args.image_size,
        bag_size=args.bag_size,
        seed=args.seed,
    )
        # -------------------------
    # Final Summary
    # -------------------------
    device = get_device()
    summary = summarize_dataset_sizes(
        data_root=data_root,
        image_size=args.image_size,
        bag_size=args.bag_size,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("[e2e] FINAL SUMMARY")
    print("=" * 60)

    print(f"Device used:         {device}")
    print(f"Seed:                {args.seed}")
    print(f"Image size:          {args.image_size}")
    print(f"Target task:         grade")
    print(f"Bag size (MIL/GNN):  {args.bag_size}")

    print("\nDataset sizes:")
    print(f"  Contrastive pairs: {summary.get('contrastive_pairs')}")
    print(f"  MIL patients:      {summary.get('mil_patients')}")
    print(f"  MIL classes:       {summary.get('mil_num_classes')}")
    print(f"  GNN patients:      {summary.get('gnn_patients')}")
    print(f"  GNN classes:       {summary.get('gnn_num_classes')}")

    print("\nSteps:")
    print(f"  Contrastive: {args.contrastive_steps}")
    print(f"  MIL:         {args.mil_steps}")
    print(f"  GNN:         {args.gnn_steps}")

    print("=" * 60)
    print("\n[e2e] DONE")
    print(f"  SimCLR: {simclr_ckpt}")
    print(f"  MIL:    {mil_ckpt}")
    print(f"  GNN:    {gnn_ckpt}")


if __name__ == "__main__":
    main()