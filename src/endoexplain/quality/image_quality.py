"""Lightweight image-quality indicators for endoscopy frames.

All functions take a uint8 RGB ``numpy.ndarray`` (H, W, 3) and return
plain floats / ints / bools so they can be serialised to CSV/JSON.
"""

from __future__ import annotations

import numpy as np


# Empirically reasonable defaults for endoscopy frames. They can be tuned
# from a review cockpit or validation dashboard later.
DEFAULT_BLUR_THRESHOLD = 100.0
DEFAULT_DARK_BRIGHTNESS = 40.0
DEFAULT_BRIGHT_BRIGHTNESS = 220.0
DEFAULT_REFLECTION_INTENSITY = 245
DEFAULT_REFLECTION_FRACTION = 0.05


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    return image


def _rgb_to_gray(image: np.ndarray) -> np.ndarray:
    img = _to_rgb_uint8(image)
    # ITU-R BT.601 luma
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)


def blur_score(image: np.ndarray, threshold: float = DEFAULT_BLUR_THRESHOLD) -> dict:
    """Variance of the Laplacian. Lower => blurrier."""
    try:
        import cv2  # type: ignore

        gray = cv2.cvtColor(_to_rgb_uint8(image), cv2.COLOR_RGB2GRAY)
        var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        gray = _rgb_to_gray(image)
        # crude fallback: variance of finite differences
        dy = np.diff(gray, axis=0)
        dx = np.diff(gray, axis=1)
        var = float(dy.var() + dx.var())
    return {
        "blur_variance": var,
        "blur_threshold": float(threshold),
        "blur_flag": var < threshold,
    }


def brightness_metrics(
    image: np.ndarray,
    dark_threshold: float = DEFAULT_DARK_BRIGHTNESS,
    bright_threshold: float = DEFAULT_BRIGHT_BRIGHTNESS,
) -> dict:
    gray = _rgb_to_gray(image)
    mean = float(gray.mean())
    std = float(gray.std())
    return {
        "brightness_mean": mean,
        "brightness_std": std,
        "dark_frame_flag": mean < dark_threshold,
        "overexposed_frame_flag": mean > bright_threshold,
    }


def reflection_metrics(
    image: np.ndarray,
    intensity_cutoff: int = DEFAULT_REFLECTION_INTENSITY,
    ratio_flag_threshold: float = DEFAULT_REFLECTION_FRACTION,
) -> dict:
    """Specular-highlight proxy: fraction of pixels saturated in all channels."""
    img = _to_rgb_uint8(image)
    mask = (img >= intensity_cutoff).all(axis=-1)
    ratio = float(mask.mean())
    return {
        "reflection_ratio": ratio,
        "reflection_intensity_cutoff": int(intensity_cutoff),
        "reflection_flag": ratio > ratio_flag_threshold,
    }


def mask_quality_metrics(mask: np.ndarray | None) -> dict:
    """Predicted-mask quality: area ratio + a fragmentation proxy."""
    if mask is None:
        return {
            "predicted_mask_area_ratio": -1.0,
            "mask_fragmentation_score": -1.0,
        }
    m = (mask > 0).astype(np.uint8)
    if m.size == 0:
        return {
            "predicted_mask_area_ratio": 0.0,
            "mask_fragmentation_score": 0.0,
        }
    area = float(m.sum() / m.size)
    fragments = -1
    try:
        import cv2  # type: ignore

        n_labels, _ = cv2.connectedComponents(m, connectivity=8)
        fragments = max(0, n_labels - 1)
    except Exception:
        # naive fragmentation: count transitions on rows (rough proxy)
        fragments = int(np.diff(m.astype(int), axis=1).clip(min=0).sum())
    # 0 fragments = no mask, 1 = clean blob, >1 = fragmented
    frag_score = 0.0 if fragments <= 0 else 1.0 - 1.0 / float(fragments)
    return {
        "predicted_mask_area_ratio": area,
        "mask_fragmentation_score": float(frag_score),
        "mask_num_fragments": int(fragments),
    }


def compute_image_quality(image: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Run all per-frame indicators and merge them into a flat dict."""
    out: dict = {}
    out.update(blur_score(image))
    out.update(brightness_metrics(image))
    out.update(reflection_metrics(image))
    out.update(mask_quality_metrics(mask))
    # Aggregate "trustworthy frame" flag
    out["quality_flag"] = not (
        out["blur_flag"] or out["dark_frame_flag"] or out["overexposed_frame_flag"]
    )
    return out
