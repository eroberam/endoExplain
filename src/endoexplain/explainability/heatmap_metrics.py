"""Quantitative metrics that compare a heatmap to a ground-truth mask.

These are the core "explanation audit" metrics of the project:

- ``explanation_iou``: IoU between a thresholded heatmap and the mask.
- ``activation_inside_mask``: fraction of heatmap energy that lies inside.
- ``activation_outside_mask``: fraction outside.
- ``pointing_game_hit``: whether the argmax of the heatmap falls inside.
- ``center_of_mass_distance``: normalized distance between heatmap and mask centers.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def _ensure_2d(x: np.ndarray, name: str) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D (H, W), got shape {x.shape}")
    return x


def _normalize_01(heatmap: np.ndarray) -> np.ndarray:
    h = heatmap.astype(np.float32)
    lo, hi = float(h.min()), float(h.max())
    if hi - lo < 1e-12:
        return np.zeros_like(h)
    return (h - lo) / (hi - lo)


def threshold_heatmap(
    heatmap: np.ndarray,
    mode: Literal["top_percent", "fixed", "otsu"] = "top_percent",
    value: float = 0.20,
) -> np.ndarray:
    """Return a boolean mask after thresholding ``heatmap`` in [0, 1].

    - ``top_percent``: keep the top ``value`` fraction of pixels by activation.
    - ``fixed``: keep pixels with activation > ``value``.
    - ``otsu``: use cv2.threshold(_, _, _, cv2.THRESH_OTSU). Falls back to fixed.
    """
    h = _normalize_01(_ensure_2d(heatmap, "heatmap"))
    if mode == "fixed":
        return h > float(value)
    if mode == "top_percent":
        pct = max(0.0, min(1.0, float(value)))
        if pct <= 0:
            return np.zeros_like(h, dtype=bool)
        cutoff = np.quantile(h, 1.0 - pct)
        return h >= cutoff
    if mode == "otsu":
        try:
            import cv2  # type: ignore

            h8 = (h * 255).astype(np.uint8)
            _, binary = cv2.threshold(h8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary > 0
        except Exception:
            return h > float(value)
    raise ValueError(f"unknown threshold mode '{mode}'")


def explanation_iou(thresholded_heatmap: np.ndarray, mask: np.ndarray) -> float:
    th = _ensure_2d(thresholded_heatmap.astype(bool), "thresholded_heatmap")
    m = _ensure_2d(mask.astype(bool), "mask")
    inter = float((th & m).sum())
    union = float((th | m).sum())
    return inter / union if union > 0 else 0.0


def activation_inside_mask(heatmap: np.ndarray, mask: np.ndarray) -> float:
    h = _normalize_01(_ensure_2d(heatmap, "heatmap"))
    m = _ensure_2d(mask.astype(np.float32), "mask")
    total = float(h.sum())
    if total < 1e-12:
        return 0.0
    return float((h * m).sum() / total)


def activation_outside_mask(heatmap: np.ndarray, mask: np.ndarray) -> float:
    h = _normalize_01(_ensure_2d(heatmap, "heatmap"))
    m = _ensure_2d(mask.astype(np.float32), "mask")
    total = float(h.sum())
    if total < 1e-12:
        return 0.0
    return float((h * (1.0 - m)).sum() / total)


def pointing_game_hit(heatmap: np.ndarray, mask: np.ndarray) -> bool:
    """True iff the argmax of the heatmap is inside the mask."""
    h = _ensure_2d(heatmap, "heatmap")
    m = _ensure_2d(mask.astype(bool), "mask")
    flat_idx = int(np.argmax(h))
    yx = np.unravel_index(flat_idx, h.shape)
    return bool(m[yx])


def _center_of_mass(arr: np.ndarray) -> tuple[float, float] | None:
    if arr.sum() < 1e-12:
        return None
    ys, xs = np.indices(arr.shape)
    total = float(arr.sum())
    cy = float((ys * arr).sum() / total)
    cx = float((xs * arr).sum() / total)
    return cy, cx


def center_of_mass_distance(heatmap: np.ndarray, mask: np.ndarray) -> float:
    """Euclidean distance between heatmap and mask centers of mass,
    normalised by the image diagonal. Returns ``-1.0`` if undefined."""
    h = _normalize_01(_ensure_2d(heatmap, "heatmap"))
    m = _ensure_2d(mask.astype(np.float32), "mask")
    com_h = _center_of_mass(h)
    com_m = _center_of_mass(m)
    if com_h is None or com_m is None:
        return -1.0
    dy = com_h[0] - com_m[0]
    dx = com_h[1] - com_m[1]
    diag = float(np.sqrt(h.shape[0] ** 2 + h.shape[1] ** 2))
    return float(np.sqrt(dy * dy + dx * dx) / diag) if diag > 0 else -1.0


def explanation_metrics(
    heatmap: np.ndarray,
    mask: np.ndarray,
    threshold_mode: Literal["top_percent", "fixed", "otsu"] = "top_percent",
    threshold_value: float = 0.20,
) -> dict:
    """Compute the full set of explanation-mask alignment metrics."""
    th = threshold_heatmap(heatmap, mode=threshold_mode, value=threshold_value)
    return {
        "threshold_mode": threshold_mode,
        "threshold_value": threshold_value,
        "explanation_iou": explanation_iou(th, mask),
        "activation_inside_mask": activation_inside_mask(heatmap, mask),
        "activation_outside_mask": activation_outside_mask(heatmap, mask),
        "pointing_game_hit": pointing_game_hit(heatmap, mask),
        "heatmap_mask_center_distance": center_of_mass_distance(heatmap, mask),
    }
