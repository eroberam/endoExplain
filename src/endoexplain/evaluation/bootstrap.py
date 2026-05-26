"""Bootstrap confidence intervals for tabular evaluation outputs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd


MetricFn = Callable[[Any], float]


def percentile_ci(samples: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
    """Return a percentile confidence interval from finite bootstrap samples."""
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(arr, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def _length(data: Any) -> int:
    return int(len(data))


def _take(data: Any, indices: np.ndarray) -> Any:
    if isinstance(data, pd.DataFrame | pd.Series):
        return data.iloc[indices].reset_index(drop=True)
    arr = np.asarray(data)
    return arr[indices]


def _stratified_indices(strata: Sequence[Any]) -> list[np.ndarray]:
    arr = np.asarray(strata)
    groups = []
    for value in pd.unique(arr):
        groups.append(np.flatnonzero(arr == value))
    return groups


def bootstrap_metric(
    data: Any,
    metric_fn: MetricFn,
    n_boot: int = 5000,
    seed: int = 45,
    confidence: float = 0.95,
    strata: Sequence[Any] | None = None,
) -> dict[str, float | int]:
    """Estimate a metric and percentile CI by sampling rows with replacement."""
    n = _length(data)
    if n == 0:
        return {
            "n": 0,
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_boot": int(n_boot),
        }

    estimate = float(metric_fn(data))
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    groups = _stratified_indices(strata) if strata is not None else None

    for _ in range(int(n_boot)):
        if groups is None:
            idx = rng.integers(0, n, size=n)
        else:
            idx = np.concatenate(
                [rng.choice(group, size=len(group), replace=True) for group in groups]
            )
        try:
            value = float(metric_fn(_take(data, idx)))
        except Exception:
            value = float("nan")
        if np.isfinite(value):
            samples.append(value)

    lo, hi = percentile_ci(samples, confidence=confidence)
    return {
        "n": int(n),
        "estimate": estimate,
        "ci_low": lo,
        "ci_high": hi,
        "n_boot": int(n_boot),
    }
