"""Environment sanity check: Python, PyTorch, CUDA, key libs, dataset symlink."""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

# Allow `python scripts/00_check_environment.py` from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from endoexplain.config import HYPERKVASIR_DIR, PROJECT_ROOT as CFG_ROOT  # noqa: E402


REQUIRED = ["numpy", "pandas", "PIL", "cv2", "torch", "torchvision"]
OPTIONAL = [
    "timm",
    "albumentations",
    "segmentation_models_pytorch",
    "pytorch_grad_cam",
    "matplotlib",
    "plotly",
    "tqdm",
    "rich",
]


def _check_import(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "n/a")
        return True, version
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    print("=" * 60)
    print("EndoExplain environment check")
    print("=" * 60)
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    print(f"Project root: {CFG_ROOT}")
    print()

    print("Required packages:")
    missing = []
    for pkg in REQUIRED:
        ok, info = _check_import(pkg)
        status = "OK" if ok else "FAIL"
        print(f"- [{status}] {pkg}: {info}")
        if not ok:
            missing.append(pkg)

    print()
    print("Optional packages:")
    for pkg in OPTIONAL:
        ok, info = _check_import(pkg)
        status = "OK" if ok else "--"
        print(f"- [{status}] {pkg}: {info}")

    print()
    print("CUDA / GPU:")
    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"cuda.is_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            idx = 0
            props = torch.cuda.get_device_properties(idx)
            gpu_mem_gb = props.total_memory / (1024 ** 3)
            print(f"device[{idx}]: {torch.cuda.get_device_name(idx)}")
            print(f"gpu memory: {gpu_mem_gb:.2f} GB")
            print(f"compute cap: {props.major}.{props.minor}")
    except ImportError:
        print("torch not installed - skipping GPU checks")

    print()
    print("Dataset symlink:")
    print(f"HyperKvasir expected at: {HYPERKVASIR_DIR}")
    if HYPERKVASIR_DIR.exists():
        try:
            entries = sorted(p.name for p in HYPERKVASIR_DIR.iterdir())[:10]
            print(f"exists=True, top entries (max 10): {entries}")
        except PermissionError as e:
            print(f"exists=True but unreadable: {e}")
    else:
        print("exists=False - create the symlink (see README)")

    print()
    if missing:
        print(f"MISSING required packages: {missing}")
        return 1
    print("All required packages importable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
