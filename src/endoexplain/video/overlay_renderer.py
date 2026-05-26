"""Professional demo-video renderer for spatial and temporal AI review."""

from __future__ import annotations

import json
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..clinical_targets import (
    cam_target_class,
    pretty_label,
    resolve_target,
    target_probability,
)
from ..config.settings import DEFAULT_IMAGE_SIZE, DISCLAIMER
from ..data.transforms import build_classification_transform
from ..models import build_classifier
from ..quality import compute_image_quality
from ..temporal import EventConfig, event_metrics, group_events, moving_average


@dataclass
class RenderConfig:
    video_path: Path
    output_path: Path
    classifier_ckpt: Path
    segmenter_ckpt: Path | None = None
    image_size: int = DEFAULT_IMAGE_SIZE
    segmenter_image_size: int = 384
    target_fps: float | None = None
    canvas_width: int = 1280
    canvas_height: int = 720
    render_profile: str = "clinical_review"
    source_crop_left: float | None = None
    source_crop_top: float | None = None
    source_crop_right: float | None = None
    source_crop_bottom: float | None = None
    xai_method: str = "gradcam++"
    show_heatmap: bool = True
    show_mask: bool = True
    suspicious_class: str | None = None
    positive_classes: list[str] | None = None
    target_display_name: str | None = None
    confidence_threshold: float = 0.5
    smoothing_window: int = 5
    max_gap_seconds: float = 1.0
    max_frames: int | None = None
    heatmap_alpha: float = 0.28
    heatmap_top_percent: float = 0.18
    heatmap_colormap: str = "magma"
    mask_threshold: float = 0.5
    mask_min_confidence: float = 0.0
    mask_gate_source: str = "target"  # target | smooth | both
    mask_only_in_event: bool = False
    mask_max_fragments: int = 8
    mask_max_area_ratio: float = 0.25
    mask_min_component_area_ratio: float = 0.002
    mask_keep_largest_components: int = 2
    mask_max_border_touch_ratio: float = 0.75
    mask_fill_alpha: float = 0.50
    mask_smoothing_kernel: int = 5
    mask_min_solidity: float = 0.25
    mask_max_aspect_ratio: float = 6.0
    mask_xai_gate: bool = False
    mask_xai_top_percent: float = 0.22
    mask_min_xai_active_inside: float = 0.08
    mask_min_xai_iou: float = 0.0
    mask_xai_dilation_px: int = 8
    mask_min_temporal_iou: float = 0.0
    mask_max_temporal_centroid_shift: float = 1.0
    mask_max_temporal_area_change: float = 100.0
    mask_stability_lookback: int = 4
    mask_fade_frames: int = 0
    device: str | None = None
    history: list[dict] = field(default_factory=list)


def _resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _load_classifier(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_to_idx = ckpt["class_to_idx"]
    model = build_classifier(
        ckpt.get("backbone", "resnet18"),
        num_classes=len(class_to_idx),
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), class_to_idx


def _load_segmenter(ckpt_path: Path, device: torch.device):
    from ..models import build_segmenter

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_segmenter(
        architecture=ckpt.get("architecture", "Unet"),
        encoder_name=ckpt.get("encoder_name", "resnet18"),
        encoder_weights=None,
        num_classes=1,
    )
    model.load_state_dict(ckpt["model_state"])
    model.endoexplain_normalize_input = bool(ckpt.get("normalize", False))
    return model.to(device).eval()


def _format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:05.2f}"


def _normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def _fit_box(src_w: int, src_h: int, x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    scale = min(w / max(src_w, 1), h / max(src_h, 1))
    out_w = max(1, int(src_w * scale))
    out_h = max(1, int(src_h * scale))
    return x + (w - out_w) // 2, y + (h - out_h) // 2, out_w, out_h


def _effective_source_crop(cfg: RenderConfig) -> dict[str, float]:
    explicit = [
        cfg.source_crop_left,
        cfg.source_crop_top,
        cfg.source_crop_right,
        cfg.source_crop_bottom,
    ]
    if any(v is not None for v in explicit):
        left = float(cfg.source_crop_left or 0.0)
        top = float(cfg.source_crop_top or 0.0)
        right = float(cfg.source_crop_right or 0.0)
        bottom = float(cfg.source_crop_bottom or 0.0)
    else:
        profile = (cfg.render_profile or "clinical_review").lower()
        if profile == "legacy":
            left, top, right, bottom = 0.0, 0.0, 0.0, 0.0
        elif profile == "public_demo":
            left, top, right, bottom = 0.15, 0.04, 0.02, 0.09
        elif profile == "media_preview":
            left, top, right, bottom = 0.13, 0.04, 0.02, 0.08
        else:
            left, top, right, bottom = 0.16, 0.045, 0.02, 0.10

    left = float(np.clip(left, 0.0, 0.45))
    top = float(np.clip(top, 0.0, 0.45))
    right = float(np.clip(right, 0.0, 0.45))
    bottom = float(np.clip(bottom, 0.0, 0.45))
    if left + right > 0.8:
        right = max(0.0, 0.8 - left)
    if top + bottom > 0.8:
        bottom = max(0.0, 0.8 - top)
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def _crop_for_display(arr: np.ndarray | None, crop: dict[str, float]) -> np.ndarray | None:
    if arr is None:
        return None
    h, w = arr.shape[:2]
    x0 = int(round(w * crop["left"]))
    x1 = w - int(round(w * crop["right"]))
    y0 = int(round(h * crop["top"]))
    y1 = h - int(round(h * crop["bottom"]))
    if x1 <= x0 + 4 or y1 <= y0 + 4:
        return arr
    return arr[y0:y1, x0:x1]


def _classifier_input(frame_bgr: np.ndarray, image_size: int, transform, device: torch.device):
    import cv2
    from PIL import Image

    rgb = cv2.cvtColor(cv2.resize(frame_bgr, (image_size, image_size)), cv2.COLOR_BGR2RGB)
    x = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
    return x, rgb


def _segmenter_input(
    frame_bgr: np.ndarray,
    image_size: int,
    device: torch.device,
    normalize: bool = False,
):
    import cv2

    rgb = cv2.cvtColor(cv2.resize(frame_bgr, (image_size, image_size)), cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std
    return x.to(device), rgb


def _component_filter(mask: np.ndarray, cfg: RenderConfig) -> tuple[np.ndarray, dict]:
    import cv2

    mask_u = (mask > 0).astype(np.uint8)
    h, w = mask_u.shape
    total = float(h * w)
    if cfg.mask_smoothing_kernel and cfg.mask_smoothing_kernel > 1:
        k = int(cfg.mask_smoothing_kernel)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_u = cv2.morphologyEx(mask_u, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_u = cv2.morphologyEx(mask_u, cv2.MORPH_OPEN, kernel, iterations=1)
    min_area = max(1, int(cfg.mask_min_component_area_ratio * total))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u, connectivity=8)
    components: list[tuple[int, int]] = []
    border_rejected = 0
    shape_rejected = 0
    for label_id in range(1, n_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        aspect = max(bw / max(float(bh), 1.0), bh / max(float(bw), 1.0))
        if cfg.mask_max_aspect_ratio > 0 and aspect > cfg.mask_max_aspect_ratio:
            shape_rejected += 1
            continue
        comp = labels == label_id
        contours, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours and cfg.mask_min_solidity > 0:
            contour_area = max(float(cv2.contourArea(c)) for c in contours)
            hull_area = 0.0
            for c in contours:
                hull = cv2.convexHull(c)
                hull_area = max(hull_area, float(cv2.contourArea(hull)))
            solidity = contour_area / max(hull_area, 1.0)
            if solidity < cfg.mask_min_solidity:
                shape_rejected += 1
                continue
        border_pixels = int(comp[0, :].sum() + comp[-1, :].sum() + comp[:, 0].sum() + comp[:, -1].sum())
        border_ratio = border_pixels / max(float(area), 1.0)
        if border_ratio > cfg.mask_max_border_touch_ratio:
            border_rejected += 1
            continue
        components.append((area, label_id))
    components.sort(reverse=True)
    if cfg.mask_keep_largest_components > 0:
        components = components[: cfg.mask_keep_largest_components]
    if cfg.mask_max_fragments > 0:
        components = components[: cfg.mask_max_fragments]

    filtered = np.zeros_like(mask_u)
    for _, label_id in components:
        filtered[labels == label_id] = 1

    area_ratio = float(filtered.mean())
    reason = ""
    visible_fragments = int(len(components))
    if not components:
        reason = "empty_mask_after_filter"
    elif area_ratio > cfg.mask_max_area_ratio:
        filtered[...] = 0
        visible_fragments = 0
        reason = "mask_area_too_large"

    return filtered, {
        "mask_area_ratio": float(filtered.mean()),
        "mask_fragments": visible_fragments,
        "border_rejected": int(border_rejected),
        "shape_rejected": int(shape_rejected),
        "reason": reason,
    }


def _mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    union = float((aa | bb).sum())
    if union <= 0:
        return 0.0
    return float((aa & bb).sum()) / union


def _temporal_mask_ok(mask: np.ndarray, accepted: deque[np.ndarray], cfg: RenderConfig) -> tuple[bool, str]:
    if not accepted or cfg.mask_stability_lookback <= 0:
        return True, ""
    prev = accepted[-1]
    if cfg.mask_min_temporal_iou > 0 and _mask_iou(mask, prev) < cfg.mask_min_temporal_iou:
        return False, "unstable_mask_iou"
    area_now = float(mask.mean())
    area_prev = max(float(prev.mean()), 1e-6)
    ratio = max(area_now / area_prev, area_prev / max(area_now, 1e-6))
    if ratio > cfg.mask_max_temporal_area_change:
        return False, "unstable_mask_area"
    c1 = _mask_centroid(mask)
    c0 = _mask_centroid(prev)
    if c1 is not None and c0 is not None:
        diag = float(np.hypot(mask.shape[1], mask.shape[0]))
        shift = float(np.hypot(c1[0] - c0[0], c1[1] - c0[1])) / max(diag, 1.0)
        if shift > cfg.mask_max_temporal_centroid_shift:
            return False, "unstable_mask_position"
    return True, ""


def _heatmap_active_mask(heatmap: np.ndarray, top_percent: float, dilation_px: int = 0) -> np.ndarray:
    import cv2

    h = _normalize01(heatmap)
    pct = float(np.clip(top_percent, 0.01, 1.0))
    cutoff = float(np.quantile(h, 1.0 - pct))
    active = (h >= cutoff).astype(np.uint8)
    if dilation_px > 0:
        k = max(1, int(dilation_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        active = cv2.dilate(active, kernel)
    return active


def _xai_mask_agreement(mask: np.ndarray, heatmap: np.ndarray, cfg: RenderConfig) -> tuple[bool, dict]:
    active = _heatmap_active_mask(heatmap, cfg.mask_xai_top_percent, cfg.mask_xai_dilation_px)
    mask_b = mask.astype(bool)
    active_b = active.astype(bool)
    inter = float((mask_b & active_b).sum())
    union = float((mask_b | active_b).sum())
    active_inside = inter / max(float(active_b.sum()), 1.0)
    xai_iou = inter / union if union > 0 else 0.0
    ok = (
        active_inside >= cfg.mask_min_xai_active_inside
        and xai_iou >= cfg.mask_min_xai_iou
    )
    return ok, {"xai_active_inside": active_inside, "xai_mask_iou": xai_iou}


def _colormap_id(name: str) -> int:
    import cv2

    table = {
        "magma": getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_JET),
        "inferno": getattr(cv2, "COLORMAP_INFERNO", cv2.COLORMAP_JET),
        "viridis": getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET),
        "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
        "jet": cv2.COLORMAP_JET,
    }
    return table.get(name.lower(), table["magma"])


def _apply_heatmap(
    frame_bgr: np.ndarray,
    heatmap: np.ndarray,
    cfg: RenderConfig,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    import cv2

    h = _normalize01(heatmap)
    if mask is None:
        pct = float(np.clip(cfg.heatmap_top_percent, 0.04, 1.0))
        cutoff = float(np.quantile(h, 1.0 - pct))
        if float(h.max()) <= cutoff + 1e-6:
            weight = h
        else:
            weight = np.clip((h - cutoff) / (float(h.max()) - cutoff + 1e-6), 0.0, 1.0)
        weight = cv2.GaussianBlur(weight.astype(np.float32), (0, 0), sigmaX=5.0, sigmaY=5.0)
        weight = np.clip(weight, 0.0, 1.0)
        if float(weight.max()) <= 1e-6:
            return frame_bgr.copy()
        color = cv2.applyColorMap((weight * 255).astype(np.uint8), _colormap_id(cfg.heatmap_colormap))
        alpha = float(np.clip(cfg.heatmap_alpha, 0.0, 0.65))
        alpha_map = (alpha * weight)[..., None]
        blend_region = weight > 0.02
        out = frame_bgr.copy()
        out[blend_region] = (
            color[blend_region].astype(np.float32) * alpha_map[blend_region]
            + frame_bgr[blend_region].astype(np.float32) * (1.0 - alpha_map[blend_region])
        ).astype(np.uint8)
        return out

    mask_u = (mask > 0).astype(np.uint8)
    if mask_u.shape != frame_bgr.shape[:2]:
        mask_u = cv2.resize(
            mask_u,
            (frame_bgr.shape[1], frame_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    if int(mask_u.sum()) == 0:
        return frame_bgr.copy()

    dilation = max(4, int(round(min(frame_bgr.shape[:2]) * 0.012)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * dilation + 1, 2 * dilation + 1),
    )
    neighbourhood = cv2.dilate(mask_u, kernel).astype(bool)
    evidence = h * neighbourhood.astype(np.float32)
    values = evidence[neighbourhood]
    values = values[values > 1e-6]
    if values.size == 0:
        return frame_bgr.copy()

    pct = float(np.clip(cfg.heatmap_top_percent, 0.04, 0.50))
    cutoff = float(np.quantile(values, 1.0 - pct))
    peak = float(values.max())
    if peak <= cutoff + 1e-6:
        return frame_bgr.copy()

    active = (evidence >= cutoff) & neighbourhood
    weight = np.zeros_like(h, dtype=np.float32)
    weight[neighbourhood] = np.clip(
        (evidence[neighbourhood] - cutoff) / (peak - cutoff + 1e-6),
        0.0,
        1.0,
    )
    weight = cv2.GaussianBlur(weight, (0, 0), sigmaX=4.5, sigmaY=4.5)
    weight = np.clip(weight, 0.0, 1.0)

    h8 = (weight * 255).astype(np.uint8)
    color = cv2.applyColorMap(h8, _colormap_id(cfg.heatmap_colormap))
    out = frame_bgr.copy()
    alpha = float(np.clip(cfg.heatmap_alpha, 0.0, 0.65))
    alpha_map = (alpha * weight)[..., None]
    blend_region = weight > 0.02
    out[blend_region] = (
        color[blend_region].astype(np.float32) * alpha_map[blend_region]
        + frame_bgr[blend_region].astype(np.float32) * (1.0 - alpha_map[blend_region])
    ).astype(np.uint8)

    return out


def _draw_mask(frame_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.50) -> np.ndarray:
    import cv2

    if mask is None or int(mask.sum()) == 0:
        return frame_bgr.copy()
    alpha = float(np.clip(alpha, 0.0, 0.85))
    mask_u = (mask > 0).astype(np.uint8)
    feather = cv2.GaussianBlur(mask_u.astype(np.float32), (0, 0), sigmaX=1.6, sigmaY=1.6)
    feather = np.clip(feather, 0.0, 1.0)[..., None]
    surgical_mint = np.zeros_like(frame_bgr, dtype=np.float32)
    surgical_mint[...] = (118, 238, 78)  # BGR for a clean surgical green.
    base = frame_bgr.astype(np.float32)
    out = surgical_mint * (alpha * feather) + base * (1.0 - alpha * feather)
    out = np.clip(out, 0, 255).astype(np.uint8)

    grid = np.zeros_like(frame_bgr, dtype=np.uint8)
    h, w = mask_u.shape
    step = max(16, int(round(min(h, w) * 0.055)))
    for gx in range(0, w, step):
        cv2.line(grid, (gx, 0), (gx, h - 1), (126, 245, 132), 1, cv2.LINE_AA)
    for gy in range(0, h, step):
        cv2.line(grid, (0, gy), (w - 1, gy), (126, 245, 132), 1, cv2.LINE_AA)
    grid_region = (mask_u > 0) & (grid.sum(axis=2) > 0)
    if np.any(grid_region):
        out[grid_region] = (
            out[grid_region].astype(np.float32) * 0.58
            + grid[grid_region].astype(np.float32) * 0.42
        ).astype(np.uint8)

    contours, _ = cv2.findContours(mask_u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= 12]
    if contours:
        cv2.drawContours(out, contours, -1, (10, 34, 22), 7, cv2.LINE_AA)
        cv2.drawContours(out, contours, -1, (245, 255, 248), 4, cv2.LINE_AA)
        cv2.drawContours(out, contours, -1, (120, 255, 118), 2, cv2.LINE_AA)
    return out


def _rect(img: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int], color, thickness=1) -> None:
    import cv2

    cv2.rectangle(img, pt1, pt2, color, thickness, cv2.LINE_AA)


def _text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float,
    color,
    thickness: int = 1,
) -> None:
    import cv2

    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_bar(img: np.ndarray, x: int, y: int, w: int, h: int, value: float, threshold: float) -> None:
    import cv2

    value = float(np.clip(value, 0.0, 1.0))
    cv2.rectangle(img, (x, y), (x + w, y + h), (34, 36, 32), -1)
    _rect(img, (x, y), (x + w, y + h), (94, 96, 88), 1)
    fill_w = int((w - 2) * value)
    if fill_w > 0:
        cv2.rectangle(img, (x + 1, y + 1), (x + fill_w, y + h - 1), (94, 215, 110), -1)
        highlight_y = max(y + 2, y + h // 3)
        cv2.rectangle(img, (x + 1, y + 1), (x + fill_w, highlight_y), (132, 245, 145), -1)
    tx = x + int(w * float(np.clip(threshold, 0.0, 1.0)))
    cv2.line(img, (tx, y - 6), (tx, y + h + 6), (42, 220, 255), 2, cv2.LINE_AA)


def _draw_soft_panel(
    img: np.ndarray,
    box: tuple[int, int, int, int],
    fill=(6, 13, 18),
    border=(38, 104, 128),
) -> None:
    import cv2

    def bgr(rgb):
        return (rgb[2], rgb[1], rgb[0])

    x, y, w, h = box
    cv2.rectangle(img, (x + 3, y + 4), (x + w + 3, y + h + 4), (2, 4, 7), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), bgr(fill), -1)
    _rect(img, (x, y), (x + w, y + h), bgr(border), 1)
    cv2.line(img, (x + 1, y + 1), (x + w - 1, y + 1), bgr((16, 170, 180)), 1, cv2.LINE_AA)


def _draw_mini_bar(img: np.ndarray, x: int, y: int, w: int, value: float) -> None:
    import cv2

    value = float(np.clip(value, 0.0, 1.0))
    cv2.rectangle(img, (x, y), (x + w, y + 5), (54, 45, 21), -1)
    fill = int(w * value)
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + 5), (154, 220, 4), -1)
        cv2.circle(img, (x + fill, y + 2), 3, (197, 255, 35), -1, cv2.LINE_AA)


def _draw_confidence_gauge(
    img: np.ndarray,
    center: tuple[int, int],
    radius: int,
    value: float,
    label: str,
) -> None:
    import cv2

    value = float(np.clip(value, 0.0, 1.0))
    cv2.ellipse(img, center, (radius, radius), 0, 180, 360, (68, 50, 31), 8, cv2.LINE_AA)
    cv2.ellipse(img, center, (radius, radius), 0, 180, 180 + int(180 * value), (235, 72, 133), 8, cv2.LINE_AA)
    cv2.ellipse(img, center, (radius, radius), 0, 180 + int(120 * value), 180 + int(180 * value), (158, 229, 21), 8, cv2.LINE_AA)
    _text(img, f"{int(round(value * 100))}%", (center[0] - 34, center[1] + 12), 0.78, (246, 250, 250), 2)
    _text(img, label, (center[0] - 52, center[1] + 36), 0.32, (174, 239, 11), 1)


def _confidence_label(value: float) -> str:
    if value >= 0.85:
        return "HIGH CONFIDENCE"
    if value >= 0.60:
        return "MODERATE CONFIDENCE"
    return "LOW CONFIDENCE"


def _quality_score(quality: dict) -> float:
    blur_var = float(quality.get("blur_variance", 0.0))
    blur_threshold = max(float(quality.get("blur_threshold", 100.0)), 1.0)
    focus = float(np.clip(blur_var / (blur_threshold * 2.0), 0.0, 1.0))

    brightness = float(quality.get("brightness_mean", 128.0))
    exposure = 1.0 - min(abs(brightness - 128.0) / 128.0, 1.0) * 0.45
    if quality.get("dark_frame_flag", False) or quality.get("overexposed_frame_flag", False):
        exposure *= 0.55

    reflection_ratio = float(quality.get("reflection_ratio", 0.0))
    reflection_penalty = min(reflection_ratio / 0.05, 1.0) * 0.35
    if quality.get("reflection_flag", False):
        reflection_penalty = max(reflection_penalty, 0.45)

    score = 0.45 * focus + 0.40 * exposure + 0.15 * (1.0 - reflection_penalty)
    return float(np.clip(score, 0.0, 1.0))


def _display_mode(mask: np.ndarray | None, heatmap: np.ndarray | None, in_event: bool) -> str:
    if mask is not None or heatmap is not None:
        return "AI OVERLAY" if not in_event else "REVIEW"
    return "ENHANCED"


def _family_badge(target_name: str) -> str:
    text = target_name.upper().replace("_", " ")
    if "POLYP" in text:
        return "POLYP FAMILY"
    if "COLITIS" in text or "INFLAM" in text:
        return "INFLAMMATION"
    if "NORMAL" in text:
        return "NORMAL MUCOSA"
    return "OTHER FINDING"


def _resize_letterbox(frame: np.ndarray, w: int, h: int, fill=(3, 6, 8)) -> np.ndarray:
    import cv2

    canvas = np.full((h, w, 3), fill, dtype=np.uint8)
    src_h, src_w = frame.shape[:2]
    x, y, out_w, out_h = _fit_box(src_w, src_h, 0, 0, w, h)
    resized = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
    if resized.ndim == 2:
        resized = np.dstack([resized] * 3)
    canvas[y : y + out_h, x : x + out_w] = resized
    return canvas


def _focus_preview(
    frame: np.ndarray,
    heatmap: np.ndarray | None,
    mask: np.ndarray | None,
    cfg: RenderConfig,
    w: int,
    h: int,
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    raw = _resize_letterbox(frame, w, h)
    focus = raw.copy()
    mask_small = None
    if mask is not None:
        mask_small = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    if heatmap is not None:
        focus = _apply_heatmap(focus, cv2.resize(heatmap, (w, h)), cfg, mask_small)
    if mask_small is not None:
        focus = _draw_mask(focus, mask_small, min(cfg.mask_fill_alpha, 0.42))
    return raw, focus


def _attention_activation_panel(
    heatmap: np.ndarray | None,
    mask: np.ndarray | None,
    cfg: RenderConfig,
    w: int,
    h: int,
) -> np.ndarray:
    import cv2

    panel = np.full((h, w, 3), (10, 8, 5), dtype=np.uint8)
    for gx in range(0, w, max(18, w // 9)):
        cv2.line(panel, (gx, 0), (gx, h - 1), (27, 31, 22), 1, cv2.LINE_AA)
    for gy in range(0, h, max(18, h // 7)):
        cv2.line(panel, (0, gy), (w - 1, gy), (27, 31, 22), 1, cv2.LINE_AA)

    if heatmap is None:
        _text(panel, "NO HEATMAP", (max(8, w // 2 - 46), h // 2), 0.34, (122, 142, 146), 1)
        return panel

    hm = cv2.resize(_normalize01(heatmap).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    hm = cv2.GaussianBlur(hm, (0, 0), sigmaX=max(2.0, w / 42.0), sigmaY=max(2.0, h / 42.0))
    low = float(np.quantile(hm, 0.30))
    high = max(float(hm.max()), low + 1e-6)
    activation = np.clip((hm - low) / (high - low), 0.0, 1.0)
    activation = np.power(activation, 0.62)

    color = cv2.applyColorMap((activation * 255).astype(np.uint8), _colormap_id(cfg.heatmap_colormap))
    alpha = (0.18 + 0.78 * activation)[..., None]
    panel = np.clip(color.astype(np.float32) * alpha + panel.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)

    glow = cv2.GaussianBlur(activation, (0, 0), sigmaX=max(3.0, w / 28.0), sigmaY=max(3.0, h / 28.0))
    glow_color = np.zeros_like(panel, dtype=np.float32)
    glow_color[...] = (78, 238, 198)
    panel = np.clip(panel.astype(np.float32) + glow_color * (glow[..., None] * 0.18), 0, 255).astype(np.uint8)

    cell = max(18, min(w, h) // 6)
    for gx in range(0, w, cell):
        strength = float(np.clip(activation[:, gx : min(gx + 2, w)].mean() if gx < w else 0.0, 0.0, 1.0))
        cv2.line(panel, (gx, 0), (gx, h - 1), (35 + int(95 * strength), 55 + int(120 * strength), 56), 1, cv2.LINE_AA)
    for gy in range(0, h, cell):
        strength = float(np.clip(activation[gy : min(gy + 2, h), :].mean() if gy < h else 0.0, 0.0, 1.0))
        cv2.line(panel, (0, gy), (w - 1, gy), (35 + int(95 * strength), 55 + int(120 * strength), 56), 1, cv2.LINE_AA)

    for level, contour_color, thick in (
        (0.42, (70, 182, 255), 1),
        (0.60, (42, 246, 185), 1),
        (0.78, (36, 255, 64), 2),
    ):
        binary = (activation >= level).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > max(8, 0.002 * w * h)]
        if contours:
            cv2.drawContours(panel, contours, -1, contour_color, thick, cv2.LINE_AA)

    if mask is not None and int(mask.sum()) > 0:
        mask_small = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > max(8, 0.003 * w * h)]
        if contours:
            cv2.drawContours(panel, contours, -1, (10, 23, 18), 5, cv2.LINE_AA)
            cv2.drawContours(panel, contours, -1, (230, 255, 246), 2, cv2.LINE_AA)
            cv2.drawContours(panel, contours, -1, (116, 255, 122), 1, cv2.LINE_AA)

    flat_order = np.argsort(activation.ravel())[::-1]
    points: list[tuple[int, int, float]] = []
    min_dist = max(18.0, min(w, h) * 0.18)
    for idx in flat_order[: min(len(flat_order), 500)]:
        value = float(activation.ravel()[idx])
        if value < 0.48:
            break
        yy, xx = divmod(int(idx), w)
        if all(np.hypot(xx - px, yy - py) >= min_dist for px, py, _ in points):
            points.append((xx, yy, value))
        if len(points) >= 8:
            break
    for i, (x0, y0, v0) in enumerate(points):
        neighbours = sorted(
            ((np.hypot(x0 - x1, y0 - y1), x1, y1, v1) for j, (x1, y1, v1) in enumerate(points) if j != i),
            key=lambda item: item[0],
        )[:2]
        for _, x1, y1, v1 in neighbours:
            line_alpha = 0.22 + 0.42 * min(v0, v1)
            overlay = panel.copy()
            cv2.line(overlay, (x0, y0), (x1, y1), (64, 238, 216), 1, cv2.LINE_AA)
            panel = cv2.addWeighted(overlay, line_alpha, panel, 1.0 - line_alpha, 0.0)
    for x0, y0, value in points:
        radius = max(4, int(round(4 + 8 * value)))
        cv2.circle(panel, (x0, y0), radius + 3, (11, 26, 20), -1, cv2.LINE_AA)
        cv2.circle(panel, (x0, y0), radius, (48, 242, 187), 2, cv2.LINE_AA)
        cv2.circle(panel, (x0, y0), max(2, radius // 2), (42, 255, 63), -1, cv2.LINE_AA)

    return panel


def _segmentation_insight_panel(mask: np.ndarray | None, w: int, h: int) -> np.ndarray:
    import cv2

    panel = np.full((h, w, 3), (7, 6, 4), dtype=np.uint8)
    for gx in range(0, w, max(18, w // 7)):
        cv2.line(panel, (gx, 0), (gx, h - 1), (21, 31, 29), 1, cv2.LINE_AA)
    for gy in range(0, h, max(18, h // 6)):
        cv2.line(panel, (0, gy), (w - 1, gy), (21, 31, 29), 1, cv2.LINE_AA)

    if mask is None or int(mask.sum()) == 0:
        _text(panel, "LOCALIZATION WITHHELD", (max(8, w // 2 - 78), h // 2), 0.32, (124, 144, 150), 1)
        return panel

    mask_small = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    mask_small = cv2.morphologyEx(
        mask_small,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    dist = cv2.distanceTransform(mask_small, cv2.DIST_L2, 3)
    dist = _normalize01(dist)
    fill = np.zeros_like(panel, dtype=np.float32)
    fill[..., 0] = 46 + 76 * dist
    fill[..., 1] = 148 + 100 * dist
    fill[..., 2] = 54 + 34 * dist
    region = mask_small > 0
    panel[region] = np.clip(
        panel[region].astype(np.float32) * 0.20 + fill[region] * 0.80,
        0,
        255,
    ).astype(np.uint8)

    grid = np.zeros_like(panel)
    step = max(14, min(w, h) // 7)
    for gx in range(0, w, step):
        cv2.line(grid, (gx, 0), (gx, h - 1), (120, 255, 132), 1, cv2.LINE_AA)
    for gy in range(0, h, step):
        cv2.line(grid, (0, gy), (w - 1, gy), (120, 255, 132), 1, cv2.LINE_AA)
    grid_region = region & (grid.sum(axis=2) > 0)
    panel[grid_region] = np.clip(
        panel[grid_region].astype(np.float32) * 0.52 + grid[grid_region].astype(np.float32) * 0.48,
        0,
        255,
    ).astype(np.uint8)

    contours, _ = cv2.findContours(mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > max(8, 0.003 * w * h)]
    if contours:
        cv2.drawContours(panel, contours, -1, (11, 25, 18), 5, cv2.LINE_AA)
        cv2.drawContours(panel, contours, -1, (235, 255, 246), 2, cv2.LINE_AA)
        cv2.drawContours(panel, contours, -1, (98, 255, 104), 1, cv2.LINE_AA)
    return panel


def _draw_badge(img: np.ndarray, text: str, x: int, y: int, ok: bool) -> int:
    import cv2

    color = (70, 115, 80) if ok else (70, 72, 160)
    w = max(76, 18 + len(text) * 12)
    cv2.rectangle(img, (x, y), (x + w, y + 30), color, -1)
    _rect(img, (x, y), (x + w, y + 30), (100, 104, 96), 1)
    _text(img, text, (x + 10, y + 21), 0.55, (232, 232, 224), 1)
    return x + w + 10


def _display_gate_status(mask_info: dict) -> str:
    if mask_info.get("visible"):
        return "localization accepted"
    reason = str(mask_info.get("reason", "not_visible"))
    labels = {
        "below_mask_min_confidence": "withheld: confidence",
        "empty_mask_after_filter": "withheld: no stable mask",
        "mask_area_too_large": "withheld: large mask",
        "outside_event": "withheld: outside event",
        "low_xai_mask_agreement": "withheld: XAI gate",
        "unstable_mask_iou": "withheld: unstable mask",
        "unstable_mask_area": "withheld: unstable area",
        "unstable_mask_position": "withheld: unstable position",
        "temporal_fade": "localization held",
        "not_visible": "withheld",
        "unknown": "withheld",
    }
    return labels.get(reason, reason.replace("_", " "))


def _draw_timeline(
    canvas: np.ndarray,
    raw: np.ndarray,
    smooth: np.ndarray,
    threshold: float,
    events: list[tuple[float, float]],
    timestamps: np.ndarray,
    frame_idx: int,
    box: tuple[int, int, int, int],
) -> None:
    import cv2

    x, y, w, h = box
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (16, 18, 17), -1)
    _rect(canvas, (x, y), (x + w, y + h), (74, 82, 76), 1)
    _text(canvas, "TEMPORAL REVIEW", (x + 16, y + 28), 0.55, (205, 204, 196), 1)
    plot_x0, plot_y0 = x + 155, y + 18
    plot_w, plot_h = w - 190, h - 36
    cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x0 + plot_w, plot_y0 + plot_h), (38, 35, 31), -1)
    duration = max(float(timestamps[-1]) if len(timestamps) else 1.0, 1e-6)
    for start, end in events:
        ex0 = plot_x0 + int((start / duration) * plot_w)
        ex1 = plot_x0 + int((end / duration) * plot_w)
        cv2.rectangle(canvas, (ex0, plot_y0), (ex1, plot_y0 + plot_h), (44, 95, 78), -1)

    def y_of(v: float) -> int:
        return plot_y0 + plot_h - int(float(np.clip(v, 0, 1)) * plot_h)

    ty = y_of(threshold)
    for sx in range(plot_x0, plot_x0 + plot_w, 18):
        cv2.line(canvas, (sx, ty), (min(sx + 9, plot_x0 + plot_w), ty), (65, 190, 230), 1, cv2.LINE_AA)

    if len(raw) > 1:
        xs = np.linspace(plot_x0, plot_x0 + plot_w, len(raw)).astype(int)
        for arr, color, thick in ((raw, (86, 86, 86), 1), (smooth, (112, 238, 112), 2)):
            ys = np.asarray([y_of(v) for v in arr], dtype=int)
            pts = np.column_stack([xs, ys]).astype(np.int32)
            cv2.polylines(canvas, [pts], False, color, thick, cv2.LINE_AA)
    px = plot_x0 + int((frame_idx / max(len(raw) - 1, 1)) * plot_w)
    cv2.line(canvas, (px, plot_y0 - 5), (px, plot_y0 + plot_h + 5), (48, 218, 255), 4, cv2.LINE_AA)
    _text(canvas, "raw", (x + 16, y + 55), 0.45, (145, 145, 140), 1)
    _text(canvas, "smooth", (x + 62, y + 55), 0.45, (112, 238, 112), 1)


def _draw_public_timeline(
    canvas: np.ndarray,
    raw: np.ndarray,
    smooth: np.ndarray,
    threshold: float,
    events: list[tuple[float, float]],
    timestamps: np.ndarray,
    frame_idx: int,
    box: tuple[int, int, int, int],
    timeline_thumbnails: list[np.ndarray] | None,
) -> None:
    import cv2

    x, y, w, h = box
    _draw_soft_panel(canvas, box, fill=(5, 12, 17), border=(25, 72, 91))
    _text(canvas, "TEMPORAL REVIEW", (x + 18, y + 28), 0.43, (238, 244, 242), 1)
    plot_x0, plot_y0 = x + 18, y + 50
    plot_w, plot_h = w - 36, max(54, int(h * 0.34))
    cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x0 + plot_w, plot_y0 + plot_h), (4, 13, 17), -1)
    for gx in range(plot_x0, plot_x0 + plot_w + 1, max(1, plot_w // 6)):
        cv2.line(canvas, (gx, plot_y0), (gx, plot_y0 + plot_h), (14, 35, 44), 1, cv2.LINE_AA)
    duration = max(float(timestamps[-1]) if len(timestamps) else 1.0, 1e-6)
    for start, end in events:
        ex0 = plot_x0 + int((start / duration) * plot_w)
        ex1 = plot_x0 + int((end / duration) * plot_w)
        cv2.rectangle(canvas, (ex0, plot_y0), (ex1, plot_y0 + plot_h), (8, 72, 52), -1)

    def y_of(v: float) -> int:
        return plot_y0 + plot_h - int(float(np.clip(v, 0, 1)) * plot_h)

    ty = y_of(threshold)
    for sx in range(plot_x0, plot_x0 + plot_w, 16):
        cv2.line(canvas, (sx, ty), (min(sx + 8, plot_x0 + plot_w), ty), (30, 178, 169), 1, cv2.LINE_AA)
    if len(raw) > 1:
        xs = np.linspace(plot_x0, plot_x0 + plot_w, len(raw)).astype(int)
        for arr, color, thick in ((raw, (120, 98, 158), 1), (smooth, (11, 218, 154), 2)):
            ys = np.asarray([y_of(v) for v in arr], dtype=int)
            pts = np.column_stack([xs, ys]).astype(np.int32)
            cv2.polylines(canvas, [pts], False, color, thick, cv2.LINE_AA)
        for idx in np.linspace(0, len(raw) - 1, min(12, len(raw))).astype(int):
            px = int(xs[idx])
            py = y_of(float(smooth[idx]))
            cv2.circle(canvas, (px, py), 3, (12, 224, 165), -1, cv2.LINE_AA)
    px = plot_x0 + int((frame_idx / max(len(raw) - 1, 1)) * plot_w)
    cv2.line(canvas, (px, plot_y0 - 5), (px, plot_y0 + plot_h + 7), (31, 200, 255), 5, cv2.LINE_AA)
    _text(canvas, "raw", (x + 18, y + 42), 0.34, (203, 139, 159), 1)
    _text(canvas, "smooth", (x + 58, y + 42), 0.34, (154, 218, 11), 1)
    _text(canvas, "00:00", (plot_x0, plot_y0 + plot_h + 24), 0.34, (164, 155, 138), 1)
    _text(canvas, _format_time(duration), (plot_x0 + plot_w - 64, plot_y0 + plot_h + 24), 0.34, (164, 155, 138), 1)

    thumbs = timeline_thumbnails or []
    if thumbs:
        count = min(9, len(thumbs))
        gap = 10
        tw = max(58, (w - 36 - gap * (count - 1)) // count)
        th = max(42, h - (plot_y0 + plot_h + 44 - y) - 16)
        start_y = y + h - th - 16
        selected = int(round((frame_idx / max(len(raw) - 1, 1)) * (count - 1)))
        for i in range(count):
            tx = x + 18 + i * (tw + gap)
            thumb = _resize_letterbox(thumbs[i], tw, th)
            canvas[start_y : start_y + th, tx : tx + tw] = thumb
            border = (14, 236, 177) if i == selected else (26, 57, 70)
            _rect(canvas, (tx, start_y), (tx + tw, start_y + th), border, 2 if i == selected else 1)


def _draw_public_header_metrics(
    canvas: np.ndarray,
    box: tuple[int, int, int, int],
    frame_idx: int,
    total_frames: int,
    mode: str,
    source_fps: float,
    latency_ms: float,
    inference_ms: float,
) -> None:
    import cv2

    x, y, w, h = box
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (35, 23, 7), -1)
    _rect(canvas, (x, y), (x + w, y + h), (96, 86, 42), 1)
    cv2.line(canvas, (x + 1, y + 1), (x + w - 1, y + 1), (130, 112, 56), 1, cv2.LINE_AA)

    items = [
        ("FRAME", f"{frame_idx + 1:03d} / {total_frames}"),
        ("MODE", mode),
        ("FPS", f"{source_fps:.0f}" if abs(source_fps - round(source_fps)) < 0.05 else f"{source_fps:.1f}"),
        ("LATENCY", f"{max(latency_ms, 0.0):.0f} ms"),
        ("INFERENCE", f"{max(inference_ms, 0.0):.0f} ms"),
    ]
    item_w = w / len(items)
    icon_color = (210, 238, 238)
    label_color = (169, 194, 202)
    value_color = (240, 247, 246)
    accent = (164, 238, 116)

    for i, (label, value) in enumerate(items):
        ix = int(x + i * item_w)
        if i > 0:
            cv2.line(canvas, (ix, y + 9), (ix, y + h - 9), (72, 56, 28), 1, cv2.LINE_AA)
        icon_x = ix + 16
        icon_y = y + h // 2
        if label == "FRAME":
            _rect(canvas, (icon_x - 7, icon_y - 8), (icon_x + 7, icon_y + 8), icon_color, 1)
            cv2.line(canvas, (icon_x + 7, icon_y - 4), (icon_x + 14, icon_y - 8), icon_color, 1, cv2.LINE_AA)
            cv2.line(canvas, (icon_x + 7, icon_y + 4), (icon_x + 14, icon_y + 8), icon_color, 1, cv2.LINE_AA)
        elif label == "MODE":
            cv2.circle(canvas, (icon_x, icon_y), 11, icon_color, 1, cv2.LINE_AA)
            cv2.line(canvas, (icon_x - 11, icon_y), (icon_x + 11, icon_y), icon_color, 1, cv2.LINE_AA)
            cv2.line(canvas, (icon_x, icon_y - 11), (icon_x, icon_y + 11), icon_color, 1, cv2.LINE_AA)
        elif label == "FPS":
            cv2.circle(canvas, (icon_x, icon_y), 11, icon_color, 1, cv2.LINE_AA)
            cv2.line(canvas, (icon_x, icon_y), (icon_x, icon_y - 8), icon_color, 1, cv2.LINE_AA)
            cv2.line(canvas, (icon_x, icon_y), (icon_x + 7, icon_y + 3), icon_color, 1, cv2.LINE_AA)
        elif label == "LATENCY":
            pts = np.array(
                [
                    [icon_x - 14, icon_y],
                    [icon_x - 7, icon_y],
                    [icon_x - 4, icon_y - 9],
                    [icon_x + 1, icon_y + 9],
                    [icon_x + 5, icon_y - 4],
                    [icon_x + 9, icon_y],
                    [icon_x + 14, icon_y],
                ],
                dtype=np.int32,
            )
            cv2.polylines(canvas, [pts], False, icon_color, 1, cv2.LINE_AA)
        else:
            _rect(canvas, (icon_x - 9, icon_y - 9), (icon_x + 9, icon_y + 9), icon_color, 1)
            _rect(canvas, (icon_x - 4, icon_y - 4), (icon_x + 4, icon_y + 4), icon_color, 1)
            for d in (-13, 13):
                cv2.line(canvas, (icon_x + d, icon_y - 5), (icon_x + d, icon_y + 5), icon_color, 1, cv2.LINE_AA)
                cv2.line(canvas, (icon_x - 5, icon_y + d), (icon_x + 5, icon_y + d), icon_color, 1, cv2.LINE_AA)

        tx = icon_x + 30
        _text(canvas, label, (tx, y + 18), 0.32, label_color, 1)
        value_col = accent if label == "MODE" else value_color
        _text(canvas, value, (tx, y + 39), 0.42, value_col, 1)


def _draw_public_radar(
    canvas: np.ndarray,
    center: tuple[int, int],
    radius: int,
    frame_idx: int,
    active: bool,
    mask: np.ndarray | None,
) -> None:
    import cv2

    color = (170, 226, 7) if active else (116, 104, 72)
    for r in (radius // 3, (2 * radius) // 3, radius):
        cv2.circle(canvas, center, r, color, 1, cv2.LINE_AA)
    angle = np.deg2rad((frame_idx * 8) % 360)
    tip = (center[0] + int(np.cos(angle) * radius), center[1] + int(np.sin(angle) * radius))
    cv2.line(canvas, center, tip, color, 2, cv2.LINE_AA)
    cv2.circle(canvas, center, 4, color, -1, cv2.LINE_AA)
    if mask is not None and int(mask.sum()) > 0:
        small = _resize_letterbox((mask > 0).astype(np.uint8) * 255, radius, radius, fill=(0, 0, 0))
        if small.ndim == 2:
            small = np.dstack([small] * 3)
        roi_x, roi_y = center[0] - radius // 2, center[1] - radius // 2
        alpha = (small[:, :, 0] > 0).astype(np.float32)[..., None] * 0.45
        roi = canvas[roi_y : roi_y + radius, roi_x : roi_x + radius].astype(np.float32)
        tint = np.zeros_like(roi)
        tint[...] = (150, 235, 35)
        canvas[roi_y : roi_y + radius, roi_x : roi_x + radius] = np.clip(
            tint * alpha + roi * (1.0 - alpha), 0, 255
        ).astype(np.uint8)


def _draw_public_insights(
    canvas: np.ndarray,
    box: tuple[int, int, int, int],
    view_frame: np.ndarray,
    view_heatmap: np.ndarray | None,
    view_mask: np.ndarray | None,
    cfg: RenderConfig,
) -> None:
    import cv2

    x, y, w, h = box
    _draw_soft_panel(canvas, box, fill=(5, 12, 17), border=(25, 72, 91))
    _text(canvas, "MODEL INSIGHTS", (x + 16, y + 28), 0.43, (238, 244, 242), 1)
    half = (w - 44) // 2
    ph = max(92, h - 72)
    attention = _attention_activation_panel(view_heatmap, view_mask, cfg, half, ph)
    mask_panel = _segmentation_insight_panel(view_mask, half, ph)
    px1, px2 = x + 16, x + 28 + half
    py = y + 58
    canvas[py : py + ph, px1 : px1 + half] = attention
    canvas[py : py + ph, px2 : px2 + half] = mask_panel
    _rect(canvas, (px1, py), (px1 + half, py + ph), (28, 86, 98), 1)
    _rect(canvas, (px2, py), (px2 + half, py + ph), (28, 86, 98), 1)
    _text(canvas, "ATTENTION HEATMAP", (px1 + 6, py - 8), 0.27, (210, 204, 184), 1)
    _text(canvas, "SEGMENTATION MASK", (px2 + 6, py - 8), 0.27, (210, 204, 184), 1)


def _draw_public_summary(
    canvas: np.ndarray,
    box: tuple[int, int, int, int],
    cfg: RenderConfig,
    raw: np.ndarray,
    events: list[tuple[float, float]],
    duration_seconds: float,
) -> None:
    import cv2

    x, y, w, h = box
    _draw_soft_panel(canvas, box, fill=(5, 12, 17), border=(25, 72, 91))
    _text(canvas, "SESSION SUMMARY", (x + 16, y + 28), 0.43, (238, 244, 242), 1)
    rows = [
        ("Procedure", "Colonoscopy"),
        ("Video ID", cfg.video_path.stem[:8]),
        ("Duration", _format_time(duration_seconds)),
        ("Frames analyzed", str(len(raw))),
        ("Events detected", str(len(events))),
        ("Max confidence", f"{100 * float(np.max(raw)):.0f}%"),
    ]
    yy = y + 56
    for label, value in rows:
        _text(canvas, label, (x + 18, yy), 0.34, (141, 160, 170), 1)
        _text(canvas, value, (x + w - 120, yy), 0.34, (230, 238, 238), 1)
        yy += 21


def _compose_public_demo_frame(
    frame_bgr: np.ndarray,
    heatmap: np.ndarray | None,
    mask: np.ndarray | None,
    frame_idx: int,
    t_seconds: float,
    duration_seconds: float,
    pred_label: str,
    target_name: str,
    target_classes: tuple[str, ...],
    confidence: float,
    smoothed_confidence: float,
    threshold: float,
    in_event: bool,
    event_text: str,
    quality: dict,
    mask_info: dict,
    raw_curve: np.ndarray,
    smooth_curve: np.ndarray,
    events: list[tuple[float, float]],
    timestamps: np.ndarray,
    cfg: RenderConfig,
    timeline_thumbnails: list[np.ndarray] | None,
    source_fps: float,
    latency_ms: float,
    inference_ms: float,
) -> np.ndarray:
    import cv2

    cw, ch = int(cfg.canvas_width), int(cfg.canvas_height)
    canvas = np.full((ch, cw, 3), (28, 18, 5), dtype=np.uint8)
    margin = max(14, int(cw * 0.013))
    header_h = max(58, int(ch * 0.075))
    bottom_h = max(176, int(ch * 0.21))
    gap = max(10, int(cw * 0.009))
    right_w = max(390, int(cw * 0.285))
    left_w = cw - margin * 2 - right_w - gap
    top_y = header_h + 8
    top_h = ch - top_y - bottom_h - gap - margin
    left = (margin, top_y, left_w, top_h)
    right = (margin + left_w + gap, top_y, right_w, top_h)
    bottom_y = top_y + top_h + gap
    summary_w = max(240, int(cw * 0.165))
    temporal_w = cw - margin * 2 - summary_w - gap
    temporal = (margin, bottom_y, temporal_w, bottom_h)
    summary = (margin + temporal_w + gap, bottom_y, summary_w, bottom_h)

    _text(canvas, "EndoExplain", (margin + 8, 32), 0.78, (248, 250, 250), 2)
    _text(canvas, "AI-POWERED ENDOSCOPY ANALYSIS", (margin + 8, 54), 0.38, (169, 190, 202), 1)
    live_x = cw - 268
    header_x = margin + 300
    header_w = max(500, live_x - header_x - 22)
    mode_text = _display_mode(mask, heatmap, in_event)
    _draw_public_header_metrics(
        canvas,
        (header_x, 11, header_w, 42),
        frame_idx=frame_idx,
        total_frames=len(raw_curve),
        mode=mode_text,
        source_fps=source_fps,
        latency_ms=latency_ms,
        inference_ms=inference_ms,
    )
    cv2.rectangle(canvas, (live_x, 12), (live_x + 82, 42), (35, 23, 7), -1)
    _rect(canvas, (live_x, 12), (live_x + 82, 42), (96, 86, 42), 1)
    cv2.circle(canvas, (live_x + 19, 27), 5, (114, 238, 172), -1, cv2.LINE_AA)
    _text(canvas, "LIVE", (live_x + 34, 33), 0.50, (244, 249, 248), 1)
    _text(canvas, f"{_format_time(t_seconds)} / {_format_time(duration_seconds)}", (cw - 172, 28), 0.47, (232, 240, 242), 1)
    _text(canvas, "VIDEO TIME", (cw - 172, 51), 0.34, (126, 146, 158), 1)

    _draw_soft_panel(canvas, left, fill=(3, 12, 18), border=(38, 86, 96))
    crop = _effective_source_crop(cfg)
    view_frame = _crop_for_display(frame_bgr, crop)
    if view_frame is None:
        view_frame = frame_bgr
    view_mask = _crop_for_display(mask.astype(np.uint8), crop) if mask is not None else None
    view_heatmap = _crop_for_display(heatmap, crop) if heatmap is not None else None
    lx, ly, lw, lh = left
    ix, iy, iw, ih = _fit_box(view_frame.shape[1], view_frame.shape[0], lx + 12, ly + 10, lw - 24, lh - 20)
    display_raw = cv2.resize(view_frame, (iw, ih), interpolation=cv2.INTER_AREA)
    display = display_raw.copy()
    mask_display = None
    if view_mask is not None:
        mask_display = cv2.resize(view_mask.astype(np.uint8), (iw, ih), interpolation=cv2.INTER_NEAREST)
    if view_heatmap is not None:
        display = _apply_heatmap(display, cv2.resize(view_heatmap, (iw, ih)), cfg, mask_display)
    if mask_display is not None:
        display = _draw_mask(display, mask_display, min(cfg.mask_fill_alpha, 0.46))
    canvas[iy : iy + ih, ix : ix + iw] = display
    thumb_w, thumb_h = max(98, iw // 7), max(68, ih // 7)
    thumb = _resize_letterbox(display_raw, thumb_w, thumb_h)
    tx, ty = ix + 24, iy + ih - thumb_h - 24
    canvas[ty : ty + thumb_h, tx : tx + thumb_w] = thumb
    _rect(canvas, (tx, ty), (tx + thumb_w, ty + thumb_h), (184, 211, 213), 1)

    rx, ry, rw, rh = right
    _draw_soft_panel(canvas, right, fill=(4, 14, 21), border=(38, 88, 100))
    cv2.circle(canvas, (rx + 34, ry + 31), 12, (174, 226, 6), 2, cv2.LINE_AA)
    _text(canvas, "AI ANALYSIS", (rx + 58, ry + 36), 0.42, (236, 243, 243), 1)
    _text(canvas, "Primary finding", (rx + 24, ry + 82), 0.30, (142, 162, 173), 1)
    finding = "POLYPS" if target_classes and confidence >= threshold else pretty_label(pred_label).upper()
    _text(canvas, finding, (rx + 24, ry + 114), 0.66, (248, 250, 250), 2)
    _text(canvas, f"Prediction: {pretty_label(pred_label)}", (rx + 24, ry + 146), 0.39, (202, 216, 220), 1)
    _text(canvas, "Model: endoExplain-v0.1", (rx + 24, ry + 168), 0.34, (169, 188, 198), 1)

    card = (rx + 18, ry + 190, rw - 36, 136)
    _draw_soft_panel(canvas, card, fill=(10, 16, 31), border=(28, 73, 101))
    cx, cy, cw2, _ = card
    _text(canvas, "CONFIDENCE OVERVIEW", (cx + 18, cy + 26), 0.34, (246, 248, 248), 1)
    _text(canvas, f"{int(round(confidence * 100))}%", (cx + 18, cy + 74), 0.96, (248, 250, 250), 2)
    _text(canvas, _confidence_label(confidence), (cx + 18, cy + 100), 0.31, (174, 239, 11), 1)
    score_x = cx + 150
    bar_w = cw2 - 172
    quality_value = _quality_score(quality)
    _text(canvas, f"AI SCORE  {confidence:.2f}", (score_x, cy + 52), 0.32, (202, 216, 220), 1)
    _draw_mini_bar(canvas, score_x, cy + 61, bar_w, confidence)
    _text(canvas, f"SMOOTHED  {smoothed_confidence:.2f}", (score_x, cy + 79), 0.32, (202, 216, 220), 1)
    _draw_mini_bar(canvas, score_x, cy + 88, bar_w, smoothed_confidence)
    _text(canvas, f"QUALITY  {quality_value:.2f}", (score_x, cy + 106), 0.32, (202, 216, 220), 1)
    _draw_mini_bar(canvas, score_x, cy + 115, bar_w, quality_value)

    event_box = (rx + 18, ry + 326, rw - 36, 50)
    _draw_soft_panel(canvas, event_box, fill=(7, 20, 29), border=(31, 82, 105))
    _text(canvas, "EVENT CONTEXT", (event_box[0] + 16, event_box[1] + 22), 0.34, (238, 244, 242), 1)
    _text(canvas, event_text if in_event else "No active event", (event_box[0] + 16, event_box[1] + 40), 0.36, (178, 198, 207), 1)

    insights_y = ry + 388
    insights_box = (rx + 18, insights_y, rw - 36, max(160, ry + rh - insights_y - 12))
    _draw_public_insights(canvas, insights_box, view_frame, view_heatmap, view_mask, cfg)

    _draw_public_timeline(canvas, raw_curve, smooth_curve, threshold, events, timestamps, frame_idx, temporal, timeline_thumbnails)
    _draw_public_summary(canvas, summary, cfg, raw_curve, events, duration_seconds)
    return canvas


def _compose_frame(
    frame_bgr: np.ndarray,
    heatmap: np.ndarray | None,
    mask: np.ndarray | None,
    frame_idx: int,
    t_seconds: float,
    duration_seconds: float,
    pred_label: str,
    target_name: str,
    target_classes: tuple[str, ...],
    confidence: float,
    smoothed_confidence: float,
    threshold: float,
    in_event: bool,
    event_text: str,
    quality: dict,
    mask_info: dict,
    raw_curve: np.ndarray,
    smooth_curve: np.ndarray,
    events: list[tuple[float, float]],
    timestamps: np.ndarray,
    cfg: RenderConfig,
    timeline_thumbnails: list[np.ndarray] | None = None,
    source_fps: float = 0.0,
    latency_ms: float = 0.0,
    inference_ms: float = 0.0,
) -> np.ndarray:
    import cv2

    cw, ch = int(cfg.canvas_width), int(cfg.canvas_height)
    profile = (cfg.render_profile or "clinical_review").lower()
    if profile == "public_demo":
        return _compose_public_demo_frame(
            frame_bgr=frame_bgr,
            heatmap=heatmap,
            mask=mask,
            frame_idx=frame_idx,
            t_seconds=t_seconds,
            duration_seconds=duration_seconds,
            pred_label=pred_label,
            target_name=target_name,
            target_classes=target_classes,
            confidence=confidence,
            smoothed_confidence=smoothed_confidence,
            threshold=threshold,
            in_event=in_event,
            event_text=event_text,
            quality=quality,
            mask_info=mask_info,
            raw_curve=raw_curve,
            smooth_curve=smooth_curve,
            events=events,
            timestamps=timestamps,
            cfg=cfg,
            timeline_thumbnails=timeline_thumbnails,
            source_fps=source_fps,
            latency_ms=latency_ms,
            inference_ms=inference_ms,
        )
    left_fraction = 0.56 if profile == "legacy" else (0.64 if profile == "media_preview" else 0.62)
    canvas = np.full((ch, cw, 3), (18, 22, 23), dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (cw, 46), (16, 19, 20), -1)
    cv2.line(canvas, (14, 46), (cw - 14, 46), (76, 112, 102), 1, cv2.LINE_AA)
    _text(canvas, "EndoExplain", (14, 34), 0.74, (244, 246, 242), 2)
    _text(canvas, "spatial and temporal AI review", (200, 32), 0.46, (184, 194, 188), 1)
    _text(canvas, f"{_format_time(t_seconds)} / {_format_time(duration_seconds)}", (cw - 235, 34), 0.58, (222, 226, 222), 1)

    left = (14, 54, int(cw * left_fraction), ch - 152)
    right = (left[0] + left[2] + 22, 54, cw - (left[0] + left[2] + 36), ch - 152)
    timeline = (14, ch - 92, cw - 28, 78)
    for box in (left, right):
        bx, by, bw, bh = box
        cv2.rectangle(canvas, (bx + 2, by + 2), (bx + bw + 2, by + bh + 2), (10, 12, 12), -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (22, 27, 28), -1)
        _rect(canvas, (bx, by), (bx + bw, by + bh), (78, 92, 90), 1)
        cv2.line(canvas, (bx + 1, by + 1), (bx + bw - 1, by + 1), (82, 142, 122), 1, cv2.LINE_AA)

    crop = _effective_source_crop(cfg)
    view_frame = _crop_for_display(frame_bgr, crop)
    view_mask = _crop_for_display(mask.astype(np.uint8), crop) if mask is not None else None
    view_heatmap = _crop_for_display(heatmap, crop) if heatmap is not None else None
    if view_frame is None:
        view_frame = frame_bgr

    src_h, src_w = view_frame.shape[:2]
    ix, iy, iw, ih = _fit_box(src_w, src_h, left[0] + 12, left[1] + 12, left[2] - 24, left[3] - 24)
    display = cv2.resize(view_frame, (iw, ih), interpolation=cv2.INTER_AREA)
    mask_display = None
    if view_mask is not None:
        mask_display = cv2.resize(view_mask.astype(np.uint8), (iw, ih), interpolation=cv2.INTER_NEAREST)
    if view_heatmap is not None:
        display = _apply_heatmap(display, cv2.resize(view_heatmap, (iw, ih)), cfg, mask_display)
    if mask_display is not None:
        display = _draw_mask(display, mask_display, cfg.mask_fill_alpha)
    canvas[iy : iy + ih, ix : ix + iw] = display

    rx, ry, rw, rh = right
    _text(canvas, "AI REVIEW", (rx + 22, ry + 34), 0.50, (180, 190, 184), 1)
    if in_event:
        cv2.rectangle(canvas, (rx + rw - 128, ry + 18), (rx + rw - 34, ry + 58), (26, 52, 44), -1)
        cv2.rectangle(canvas, (rx + rw - 124, ry + 22), (rx + rw - 38, ry + 54), (62, 130, 92), -1)
        _rect(canvas, (rx + rw - 124, ry + 22), (rx + rw - 38, ry + 54), (132, 220, 166), 1)
        _text(canvas, "EVENT", (rx + rw - 116, ry + 47), 0.58, (245, 245, 240), 1)
    _text(canvas, f"Target: {target_name}", (rx + 22, ry + 78), 0.70, (246, 248, 244), 2)
    _text(canvas, f"Prediction: {pretty_label(pred_label)}", (rx + 22, ry + 110), 0.50, (200, 204, 198), 1)
    if target_classes:
        _text(canvas, "Score: " + " + ".join(target_classes), (rx + 22, ry + 136), 0.40, (170, 178, 172), 1)

    card_y = ry + 150
    card_w = (rw - 58) // 2
    for title, value, x0 in (
        ("SCORE", confidence, rx + 22),
        ("SMOOTHED", smoothed_confidence, rx + 42 + card_w),
    ):
        cv2.rectangle(canvas, (x0, card_y), (x0 + card_w, card_y + 82), (24, 29, 29), -1)
        _rect(canvas, (x0, card_y), (x0 + card_w, card_y + 82), (82, 96, 92), 1)
        _text(canvas, title, (x0 + 14, card_y + 27), 0.44, (174, 184, 178), 1)
        col = (108, 236, 118) if value >= threshold else (218, 216, 208)
        _text(canvas, f"{value:.2f}", (x0 + 14, card_y + 64), 0.88, col, 2)

    _text(canvas, "Target probability", (rx + 22, card_y + 106), 0.45, (182, 190, 184), 1)
    _draw_bar(canvas, rx + 22, card_y + 118, rw - 44, 20, smoothed_confidence, threshold)

    y = card_y + 160
    _text(canvas, "Event context", (rx + 22, y), 0.57, (222, 218, 210), 1)
    _text(canvas, event_text, (rx + 22, y + 38), 0.62, (202, 198, 190), 1)
    y += 70
    _text(canvas, "Frame quality", (rx + 22, y), 0.57, (222, 218, 210), 1)
    bx = rx + 22
    bx = _draw_badge(canvas, "FOCUS", bx, y + 25, not quality.get("blur_flag", False))
    bx = _draw_badge(
        canvas,
        "EXPOSURE",
        bx,
        y + 25,
        not (quality.get("dark_frame_flag", False) or quality.get("overexposed_frame_flag", False)),
    )
    _draw_badge(canvas, "GLARE", bx, y + 25, not quality.get("reflection_flag", False))

    y += 66
    stat_w = (rw - 64) // 2
    for title, value, x0 in (
        ("MASK AREA", f"{100.0 * mask_info.get('mask_area_ratio', 0.0):.1f}%", rx + 22),
        ("FRAGMENTS", f"{int(mask_info.get('mask_fragments', 0))}", rx + 42 + stat_w),
    ):
        cv2.rectangle(canvas, (x0, y), (x0 + stat_w, y + 62), (24, 29, 29), -1)
        _rect(canvas, (x0, y), (x0 + stat_w, y + 62), (82, 96, 92), 1)
        _text(canvas, title, (x0 + 16, y + 24), 0.43, (172, 168, 160), 1)
        _text(canvas, value, (x0 + 16, y + 52), 0.72, (118, 255, 118), 2)
    evidence = _display_gate_status(mask_info)
    _text(canvas, f"Localization: {evidence}", (rx + 22, ry + rh - 42), 0.40, (178, 186, 180), 1)
    _text(canvas, "Research prototype - not for diagnosis or patient care", (rx + 22, ry + rh - 22), 0.40, (160, 156, 150), 1)

    _draw_timeline(canvas, raw_curve, smooth_curve, threshold, events, timestamps, frame_idx, timeline)
    return canvas


def _sample_timeline_thumbnails(video_path: Path, frame_count: int, count: int = 9) -> list[np.ndarray]:
    import cv2

    if frame_count <= 0 or count <= 0:
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    indices = np.linspace(0, max(frame_count - 1, 0), min(count, frame_count)).astype(int)
    thumbs: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            thumbs.append(frame)
    cap.release()
    return thumbs


def render_overlay_video(
    cfg: RenderConfig,
    on_pass1_progress=None,
    on_pass2_progress=None,
) -> dict:
    """Render an annotated MP4 and sidecar CSV/JSON files."""
    import cv2
    import pandas as pd
    from ..explainability.gradcam import generate_heatmap

    device = _resolve_device(cfg.device)
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(cfg.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {cfg.video_path}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    target_fps = cfg.target_fps or src_fps

    classifier, class_to_idx = _load_classifier(cfg.classifier_ckpt, device)
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    target = resolve_target(
        class_to_idx,
        suspicious_class=cfg.suspicious_class,
        positive_classes=cfg.positive_classes,
        target_display_name=cfg.target_display_name,
    )
    segmenter = _load_segmenter(cfg.segmenter_ckpt, device) if cfg.segmenter_ckpt and cfg.show_mask else None
    transform = build_classification_transform(cfg.image_size, train=False)

    t0 = time.time()
    confidences: list[float] = []
    preds: list[int] = []
    cap = cv2.VideoCapture(str(cfg.video_path))
    frame_count = 0
    with torch.no_grad():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if cfg.max_frames is not None and frame_count >= cfg.max_frames:
                break
            x, _ = _classifier_input(frame_bgr, cfg.image_size, transform, device)
            probs = torch.softmax(classifier(x), dim=1)[0].detach().cpu().numpy()
            pred = int(np.argmax(probs))
            preds.append(pred)
            confidences.append(target_probability(probs, target, pred))
            frame_count += 1
            if on_pass1_progress is not None:
                on_pass1_progress(frame_count, src_total if src_total > 0 else frame_count)
    cap.release()
    pass1_dt = time.time() - t0
    if not confidences:
        raise RuntimeError("video produced no frames")

    raw = np.asarray(confidences, dtype=np.float32)
    smooth = moving_average(raw, window=cfg.smoothing_window)
    timestamps = np.arange(len(raw), dtype=np.float32) / src_fps
    frames_df = pd.DataFrame(
        {
            "frame_id": np.arange(len(raw)),
            "timestamp": timestamps,
            "confidence": raw,
            "confidence_smoothed": smooth,
        }
    )
    events_df = group_events(
        frames_df,
        EventConfig(confidence_threshold=cfg.confidence_threshold, max_gap_seconds=cfg.max_gap_seconds),
    )
    event_intervals = [
        (float(r["start_time"]), float(r["end_time"])) for _, r in events_df.iterrows()
    ]
    timeline_thumbnails = (
        _sample_timeline_thumbnails(cfg.video_path, len(raw), count=9)
        if (cfg.render_profile or "").lower() == "public_demo"
        else None
    )

    def _event_row_for(t: float):
        for _, row in events_df.iterrows():
            if float(row["start_time"]) <= t <= float(row["end_time"]):
                return row
        return None

    cap = cv2.VideoCapture(str(cfg.video_path))
    writer = cv2.VideoWriter(
        str(cfg.output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        target_fps,
        (int(cfg.canvas_width), int(cfg.canvas_height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer for {cfg.output_path}")

    accepted_masks: deque[np.ndarray] = deque(maxlen=max(1, cfg.mask_stability_lookback))
    last_mask: np.ndarray | None = None
    last_mask_age = 999
    frame_rows: list[dict] = []
    best_per_event: dict[int, tuple[tuple[int, float], np.ndarray]] = {}
    preview_frames: dict[str, tuple[float, np.ndarray]] = {}
    skip_reasons: Counter[str] = Counter()
    rendered = 0
    timing_alpha = 0.20
    latency_ewma_ms = 0.0
    inference_ewma_ms = 0.0
    t0 = time.time()

    for frame_idx in range(len(raw)):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_t0 = time.perf_counter()
        _synchronize_if_cuda(device)
        inference_t0 = time.perf_counter()
        t_seconds = float(timestamps[frame_idx])
        x_cls, rgb_small = _classifier_input(frame_bgr, cfg.image_size, transform, device)
        with torch.no_grad():
            probs = torch.softmax(classifier(x_cls), dim=1)[0].detach().cpu().numpy()
        pred = int(np.argmax(probs))
        pred_label = idx_to_class.get(pred, str(pred))
        confidence = target_probability(probs, target, pred)
        smoothed_confidence = float(smooth[frame_idx])
        event_row = _event_row_for(t_seconds)
        in_event = event_row is not None

        heatmap_full = None
        if cfg.show_heatmap:
            try:
                cam_idx = cam_target_class(class_to_idx, target, pred_label)
                heatmap = generate_heatmap(
                    model=classifier,
                    input_tensor=x_cls,
                    target_class=cam_idx,
                    method=cfg.xai_method,
                )
                heatmap_full = cv2.resize(heatmap, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
            except Exception:
                heatmap_full = None

        mask_full = None
        mask_info = {
            "visible": False,
            "reason": "no_segmenter",
            "mask_area_ratio": 0.0,
            "mask_fragments": 0,
            "xai_active_inside": 0.0,
            "xai_mask_iou": 0.0,
        }
        if segmenter is not None:
            gate_target = confidence >= cfg.mask_min_confidence
            gate_smooth = smoothed_confidence >= cfg.mask_min_confidence
            if cfg.mask_gate_source == "both":
                gate_ok = gate_target and gate_smooth
            elif cfg.mask_gate_source == "smooth":
                gate_ok = gate_smooth
            else:
                gate_ok = gate_target
            if cfg.mask_only_in_event and not in_event:
                gate_ok = False
                mask_info["reason"] = "outside_event"
            elif not gate_ok:
                mask_info["reason"] = "below_mask_min_confidence"
            else:
                x_seg, _ = _segmenter_input(
                    frame_bgr,
                    cfg.segmenter_image_size,
                    device,
                    normalize=bool(getattr(segmenter, "endoexplain_normalize_input", False)),
                )
                with torch.no_grad():
                    seg_prob = torch.sigmoid(segmenter(x_seg))[0, 0].detach().cpu().numpy()
                raw_mask = (seg_prob >= cfg.mask_threshold).astype(np.uint8)
                raw_mask = cv2.resize(raw_mask, (src_w, src_h), interpolation=cv2.INTER_NEAREST)
                filtered, info = _component_filter(raw_mask, cfg)
                mask_info.update(info)
                if not mask_info["reason"] and cfg.mask_xai_gate and heatmap_full is not None:
                    xai_ok, xai_info = _xai_mask_agreement(filtered, heatmap_full, cfg)
                    mask_info.update(xai_info)
                    if not xai_ok:
                        mask_info["reason"] = "low_xai_mask_agreement"
                if not mask_info["reason"]:
                    temporal_ok, temporal_reason = _temporal_mask_ok(filtered, accepted_masks, cfg)
                    if not temporal_ok:
                        mask_info["reason"] = temporal_reason
                if not mask_info["reason"]:
                    mask_full = filtered
                    mask_info["visible"] = True
                    accepted_masks.append(filtered.copy())
                    last_mask = filtered.copy()
                    last_mask_age = 0

        allow_fade = (
            mask_info.get("reason") == "empty_mask_after_filter"
            and in_event
            and confidence >= cfg.mask_min_confidence
            and smoothed_confidence >= cfg.mask_min_confidence
        )
        if (
            mask_full is None
            and allow_fade
            and last_mask is not None
            and last_mask_age < cfg.mask_fade_frames
        ):
            mask_full = last_mask.copy()
            mask_info["visible"] = True
            mask_info["reason"] = "temporal_fade"
            mask_info["mask_area_ratio"] = float(mask_full.mean())
            mask_info["mask_fragments"] = int(compute_image_quality(rgb_small, mask_full).get("mask_num_fragments", 0))
        else:
            last_mask_age += 1

        _synchronize_if_cuda(device)
        inference_ms_raw = (time.perf_counter() - inference_t0) * 1000.0
        if rendered == 0:
            inference_ewma_ms = inference_ms_raw
        else:
            inference_ewma_ms = timing_alpha * inference_ms_raw + (1.0 - timing_alpha) * inference_ewma_ms

        if not mask_info.get("visible"):
            skip_reasons[str(mask_info.get("reason", "unknown"))] += 1

        quality = compute_image_quality(rgb_small, mask_full)
        if event_row is None:
            event_text = "No active suspicious event"
        else:
            event_text = (
                f"#{int(event_row['event_id']):02d}  "
                f"{_format_time(float(event_row['start_time']))}-"
                f"{_format_time(float(event_row['end_time']))}"
            )

        display_latency_ms = latency_ewma_ms if rendered > 0 else inference_ewma_ms
        canvas = _compose_frame(
            frame_bgr=frame_bgr,
            heatmap=heatmap_full,
            mask=mask_full,
            frame_idx=frame_idx,
            t_seconds=t_seconds,
            duration_seconds=float(len(raw) / src_fps),
            pred_label=pred_label,
            target_name=target.display_name,
            target_classes=target.class_labels,
            confidence=confidence,
            smoothed_confidence=smoothed_confidence,
            threshold=cfg.confidence_threshold,
            in_event=in_event,
            event_text=event_text,
            quality=quality,
            mask_info=mask_info,
            raw_curve=raw,
            smooth_curve=smooth,
            events=event_intervals,
            timestamps=timestamps,
            cfg=cfg,
            timeline_thumbnails=timeline_thumbnails,
            source_fps=src_fps,
            latency_ms=display_latency_ms,
            inference_ms=inference_ewma_ms,
        )
        latency_ms_raw = (time.perf_counter() - frame_t0) * 1000.0
        if rendered == 0:
            latency_ewma_ms = latency_ms_raw
        else:
            latency_ewma_ms = timing_alpha * latency_ms_raw + (1.0 - timing_alpha) * latency_ewma_ms
        writer.write(canvas)
        rendered += 1

        def _remember_preview(name: str, score: float) -> None:
            prev = preview_frames.get(name)
            if prev is None or score > prev[0]:
                preview_frames[name] = (float(score), canvas.copy())

        if frame_idx == 0:
            _remember_preview("start", 1.0)
        area = float(mask_info.get("mask_area_ratio", 0.0))
        reason = str(mask_info.get("reason", ""))
        if mask_info.get("visible"):
            _remember_preview("localization_accepted_large", area)
            _remember_preview("localization_accepted_small", 1.0 / max(area, 1e-6))
        elif reason == "below_mask_min_confidence":
            _remember_preview("withheld_confidence", float(confidence))
        elif reason == "mask_area_too_large":
            _remember_preview("withheld_large_mask", float(confidence))
        elif reason == "empty_mask_after_filter":
            _remember_preview("withheld_no_stable_mask", float(confidence))

        if event_row is not None:
            ev_id = int(event_row["event_id"])
            prev = best_per_event.get(ev_id)
            frame_score = (1 if mask_info.get("visible") else 0, float(confidence))
            if prev is None or frame_score > prev[0]:
                best_per_event[ev_id] = (frame_score, canvas.copy())

        frame_rows.append(
            {
                "frame_id": frame_idx,
                "timestamp": t_seconds,
                "predicted_class": pred,
                "predicted_label": pred_label,
                "target_name": target.display_name,
                "target_probability": confidence,
                "target_probability_smoothed": smoothed_confidence,
                "inference_ms": inference_ms_raw,
                "latency_ms": latency_ms_raw,
                "in_event": bool(in_event),
                "mask_visible": bool(mask_info.get("visible", False)),
                "mask_skip_reason": mask_info.get("reason", ""),
                "mask_area_ratio": mask_info.get("mask_area_ratio", 0.0),
                "mask_fragments": mask_info.get("mask_fragments", 0),
                "xai_active_inside": mask_info.get("xai_active_inside", 0.0),
                "xai_mask_iou": mask_info.get("xai_mask_iou", 0.0),
                **{k: v for k, v in quality.items() if isinstance(v, (int, float, bool))},
            }
        )
        if on_pass2_progress is not None:
            on_pass2_progress(rendered, len(raw))

    cap.release()
    writer.release()
    pass2_dt = time.time() - t0

    thumb_dir = cfg.output_path.parent / (cfg.output_path.stem + ".thumbs")
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_paths: dict[int, str] = {}
    for ev_id, (_, frame) in sorted(best_per_event.items()):
        out_thumb = thumb_dir / f"event_{ev_id:03d}.jpg"
        cv2.imwrite(str(out_thumb), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        thumb_paths[ev_id] = str(out_thumb)
    if thumb_paths and "event_id" in events_df.columns:
        events_df["thumbnail_path"] = events_df["event_id"].map(thumb_paths).fillna("")

    preview_dir = cfg.output_path.parent / (cfg.output_path.stem + ".previews")
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_paths: dict[str, str] = {}
    for name, (_, frame) in sorted(preview_frames.items()):
        out_preview = preview_dir / f"{name}.jpg"
        cv2.imwrite(str(out_preview), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        preview_paths[name] = str(out_preview)

    frames_out = pd.DataFrame(frame_rows)
    frames_csv = cfg.output_path.with_suffix(".frames.csv")
    frames_out.to_csv(frames_csv, index=False)
    events_csv = cfg.output_path.with_suffix(".events.csv")
    events_df.to_csv(events_csv, index=False)

    summary = {
        "video_path": str(cfg.video_path),
        "output_path": str(cfg.output_path),
        "source_fps": src_fps,
        "source_total_frames": src_total,
        "source_width": src_w,
        "source_height": src_h,
        "rendered_frames": int(rendered),
        "pass1_seconds": pass1_dt,
        "pass2_seconds": pass2_dt,
        "mean_inference_ms": float(frames_out["inference_ms"].mean()) if "inference_ms" in frames_out else 0.0,
        "mean_latency_ms": float(frames_out["latency_ms"].mean()) if "latency_ms" in frames_out else 0.0,
        "num_events": int(len(events_df)),
        "events": event_intervals,
        "target": asdict(target),
        "mask_visible_frames": int(frames_out["mask_visible"].sum()) if not frames_out.empty else 0,
        "mask_skip_reasons": dict(skip_reasons),
        "temporal_summary": event_metrics(frames_df),
        "classes": list(class_to_idx.keys()),
        "effective_source_crop": _effective_source_crop(cfg),
        "frames_csv": str(frames_csv),
        "events_csv": str(events_csv),
        "thumb_dir": str(thumb_dir),
        "thumbnails": thumb_paths,
        "preview_dir": str(preview_dir),
        "previews": preview_paths,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "disclaimer": DISCLAIMER,
    }
    side_json = cfg.output_path.with_suffix(".summary.json")
    side_json.write_text(json.dumps(summary, indent=2))
    return summary
