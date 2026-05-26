"""1D temporal smoothing helpers for per-frame confidence curves."""

from __future__ import annotations

import numpy as np


def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered moving average, edges padded by reflection."""
    if window <= 1:
        return values.astype(np.float32)
    values = np.asarray(values, dtype=np.float32)
    pad = window // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def ema(values: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Exponential moving average (causal)."""
    values = np.asarray(values, dtype=np.float32)
    out = np.empty_like(values)
    if len(values) == 0:
        return out
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out
