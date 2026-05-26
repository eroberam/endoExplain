"""Overlays for heatmaps and masks (CPU-side, returns numpy uint8 RGB)."""

from __future__ import annotations

import numpy as np


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    img = image
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.5 else img.astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> np.ndarray:
    """Blend a [0,1] heatmap onto an image. Returns uint8 RGB (H, W, 3)."""
    import cv2  # type: ignore

    img = _to_uint8_rgb(image)
    h = _normalize_01(heatmap)
    if h.shape != img.shape[:2]:
        h = cv2.resize(h, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    cm_map = {
        "jet": cv2.COLORMAP_JET,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
    }
    cm = cm_map.get(colormap, cv2.COLORMAP_JET)
    h8 = (h * 255).astype(np.uint8)
    color = cv2.applyColorMap(h8, cm)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    blended = (alpha * color + (1 - alpha) * img).astype(np.uint8)
    return blended


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    """Tint the masked region of ``image`` with the given RGB ``color``."""
    img = _to_uint8_rgb(image).copy()
    m = mask.astype(bool)
    if m.shape != img.shape[:2]:
        import cv2  # type: ignore

        m = (
            cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            > 0
        )
    tint = np.zeros_like(img)
    tint[m] = color
    out = img.copy()
    out[m] = (alpha * tint[m] + (1 - alpha) * img[m]).astype(np.uint8)
    return out
