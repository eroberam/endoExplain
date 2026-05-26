"""Central registry of project paths.

All filesystem layout assumptions live here so scripts and the app never
hard-code paths. Override the project root by setting the environment
variable ``ENDOEXPLAIN_PROJECT_ROOT``.
"""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_project_root() -> Path:
    env_root = os.environ.get("ENDOEXPLAIN_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    # paths.py lives at: <root>/src/endoexplain/config/paths.py
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT: Path = _resolve_project_root()

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
REPORTS_DIR: Path = DATA_DIR / "reports"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
METRICS_DIR: Path = OUTPUTS_DIR / "metrics"
HEATMAPS_DIR: Path = OUTPUTS_DIR / "heatmaps"

MODELS_DIR: Path = PROJECT_ROOT / "models"
CHECKPOINTS_DIR: Path = MODELS_DIR / "checkpoints"

HYPERKVASIR_DIR: Path = RAW_DIR / "hyper-kvasir"
HYPERKVASIR_INDEX_CSV: Path = PROCESSED_DIR / "hyperkvasir_index.csv"

# HyperKvasir ships its own segmented subset with the same `images/` + `masks/`
# layout used by the standalone Kvasir-SEG dataset, so we treat both uniformly.
HYPERKVASIR_SEGMENTED_DIR: Path = HYPERKVASIR_DIR / "segmented-images"
KVASIRSEG_DIR: Path = RAW_DIR / "kvasir-seg"
KVASIRSEG_INDEX_CSV: Path = PROCESSED_DIR / "kvasirseg_index.csv"


def ensure_dirs() -> None:
    """Create the standard output/data folders if they do not exist."""
    for d in (
        DATA_DIR,
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        REPORTS_DIR,
        OUTPUTS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        HEATMAPS_DIR,
        MODELS_DIR,
        CHECKPOINTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
