# tests/conftest.py
import os
import sys
from pathlib import Path
import pytest

import os
os.environ.setdefault("DATA_ROOT", r"D:\datasets\MRI_Mahdieh_Datasets")

# Add <repo_root>/src to sys.path so `import ContrastiveLearning...` works in pytest
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture(scope="session")
def data_root() -> Path:
    p = os.environ.get("DATA_ROOT", "").strip()
    if not p:
        pytest.skip("DATA_ROOT env var not set. Set it to your dataset root.")
    root = Path(p)
    if not root.exists():
        pytest.skip(f"DATA_ROOT does not exist: {root}")
    return root