"""Project-wide constants and lightweight default training settings."""

from __future__ import annotations

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)
VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
)

DEFAULT_SEED: int = 42
DEFAULT_IMAGE_SIZE: int = 256
DEFAULT_BATCH_SIZE: int = 8
DEFAULT_NUM_WORKERS: int = 4
DEFAULT_EPOCHS: int = 20
DEFAULT_EARLY_STOPPING_PATIENCE: int = 5

# Train / val / test split fractions (must sum to 1.0).
DEFAULT_SPLIT: tuple[float, float, float] = (0.70, 0.15, 0.15)

DISCLAIMER: str = (
    "This project is a research and educational prototype built with open "
    "datasets. It is not intended for clinical use, diagnosis, treatment "
    "decisions, or real-time patient care."
)
