"""Unified experiment entrypoint.

Goal: provide a stable CLI and config-driven interface, while leaving existing method scripts intact.
This script is a scaffold: map `experiment.method` to your existing training entrypoints over time.

Usage:
  python scripts/run_experiment.py --config configs/base.yaml

Notes:
- This does NOT download data.
- It creates an output directory and logs the resolved config.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from brainmrianalysis.utils.repro import ReproConfig, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/base.yaml", help="Path to YAML config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # Seed & determinism
    r = cfg.get("reproducibility", {}) or {}
    seed_everything(
        ReproConfig(
            seed=int(r.get("seed", 42)),
            deterministic=bool(r.get("deterministic", True)),
            benchmark=bool(r.get("benchmark", False)),
        )
    )

    # Output directory
    exp = cfg.get("experiment", {}) or {}
    out_root = Path(exp.get("output_dir", "runs"))
    exp_name = exp.get("name", "baseline")
    out_dir = out_root / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config (for traceability)
    (out_dir / "config.resolved.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # TODO: route to method-specific trainer
    method = exp.get("method", "cnn")
    print(f"[BrainMRIAnalysis] method={method} | output={out_dir}")
    print("This is a scaffold entrypoint. Next: connect this to your method folders.")

    # Example placeholder of where routing will happen:
    # if method == "cnn":
    #     from MRI_Classification_CNN.train import train
    #     train(cfg, out_dir)


if __name__ == "__main__":
    main()
