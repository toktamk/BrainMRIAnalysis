"""Unified experiment entrypoint (router) — Windows-proof.

Key features:
- Uses the current working directory as REPO_ROOT (prevents path mismatches like GitHub/ vs github/)
- Routes to method-specific scripts (simclr/gnn/mil)
- Optional auto-discovery: if the configured script is missing, it searches for likely entrypoints.

Usage (from repo root):
  python scripts/run_experiment.py --config configs/base.yaml --dry-run
  python scripts/run_experiment.py --config configs/base.yaml

"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

# IMPORTANT: use current working directory as the repo root
REPO_ROOT = Path.cwd().resolve()

# Make `src/` importable
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from brainmrianalysis.utils.repro import ReproConfig, seed_everything  # type: ignore
except Exception:
    # Fallback: allow router to work even if scaffold package isn't installed yet
    ReproConfig = None  # type: ignore
    seed_everything = None  # type: ignore


DEFAULT_SCRIPT_MAP = {
    "simclr": ("ContrastiveLearning", "main_pytorch.py"),
    "gnn": ("GraphNeuralNetworks", "main_gcn_keras.py"),
    "mil": ("MultiInstanceLearning", "main_MultiInstanceLearning.py"),
}

CANDIDATE_PATTERNS = (
    "main*.py",
    "train*.py",
    "run*.py",
    "*simclr*.py",
    "*gnn*.py",
    "*mil*.py",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/base.yaml", help="Path to YAML config")
    p.add_argument("--dry-run", action="store_true", help="Print routed command without executing")
    return p.parse_args()


def _autodiscover_entrypoint(workdir: Path) -> Path | None:
    """Try to find a reasonable entrypoint if the configured file doesn't exist."""
    for pat in CANDIDATE_PATTERNS:
        hits = sorted(workdir.glob(pat))
        hits = [h for h in hits if h.is_file() and not h.name.startswith("__")]
        if hits:
            return hits[0]
    return None


def _resolve_script(method: str, cfg: dict) -> tuple[Path, Path]:
    overrides = (cfg.get("router", {}) or {}).get("script_map", {}) or {}

    if method in overrides:
        wd, script = overrides[method]
    else:
        wd, script = DEFAULT_SCRIPT_MAP.get(method, (None, None))

    if wd is None or script is None:
        raise ValueError(
            f"Unknown method '{method}'. Expected one of: {', '.join(sorted(DEFAULT_SCRIPT_MAP))}"
        )

    workdir = REPO_ROOT / wd
    script_path = workdir / script

    if script_path.exists():
        return workdir, script_path

    # Fallback: autodiscover inside the method folder
    if workdir.exists():
        found = _autodiscover_entrypoint(workdir)
        if found is not None:
            return workdir, found

    # No script found
    raise FileNotFoundError(
        f"Routed script not found: {script_path}\n"
        f"Repo root (cwd): {REPO_ROOT}\n"
        f"Fix it by either:\n"
        f"  (1) editing DEFAULT_SCRIPT_MAP in scripts/run_experiment.py, or\n"
        f"  (2) setting router.script_map.{method} in your config YAML, or\n"
        f"  (3) ensure the folder '{wd}' exists in the repo root.\n"
        f"If you want auto-discovery to pick a different file, rename your entrypoint to match patterns like main*.py or train*.py."
    )


def main() -> None:
    args = parse_args()
    cfg_path = (REPO_ROOT / args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # Seed & determinism (only if scaffold package is present)
    if seed_everything is not None and ReproConfig is not None:
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
    out_dir = (REPO_ROOT / out_root / exp_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.resolved.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    method = str(exp.get("method", "simclr")).lower().strip()
    workdir, script_path = _resolve_script(method, cfg)

    forwarded_args = ["--config", str(cfg_path), "--output_dir", str(out_dir)]
    cmd = [sys.executable, str(script_path), *forwarded_args]

    print(f"[BrainMRIAnalysis] repo_root={REPO_ROOT}")
    print(f"[BrainMRIAnalysis] method={method}")
    print(f"[BrainMRIAnalysis] workdir={workdir}")
    print(f"[BrainMRIAnalysis] script={script_path.name}")
    print(f"[BrainMRIAnalysis] cmd={' '.join(cmd)}")

    if args.dry_run:
        return

    subprocess.run(cmd, cwd=str(workdir), check=True)


if __name__ == "__main__":
    main()
