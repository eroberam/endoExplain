"""High-level wrapper: image → prediction → heatmap → metrics → overlays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..data.transforms import build_classification_transform
from .gradcam import generate_heatmap
from .heatmap_metrics import explanation_metrics


@dataclass
class ExplanationResult:
    predicted_class: int
    confidence: float
    heatmap: np.ndarray  # (H, W) in [0, 1] at model input resolution
    image_rgb: np.ndarray  # uint8 (H, W, 3) at model input resolution
    mask_rgb: np.ndarray | None  # uint8 (H, W) at model input resolution, or None
    metrics: dict | None  # explanation-mask metrics, or None if no mask


def _load_image_as_rgb(image: str | Path | np.ndarray | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return Image.open(image).convert("RGB")


def _load_mask_2d(mask: str | Path | np.ndarray | None, size: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    if isinstance(mask, np.ndarray):
        arr = mask
    else:
        arr = np.asarray(Image.open(mask).convert("L"))
    binary = (arr > 127).astype(np.uint8) * 255
    if binary.shape != size:
        binary_img = Image.fromarray(binary).resize(size[::-1], Image.NEAREST)
        binary = np.asarray(binary_img)
    return (binary > 127).astype(np.uint8)


def explain_image(
    model: torch.nn.Module,
    image: str | Path | np.ndarray | Image.Image,
    image_size: int = 256,
    method: str = "gradcam++",
    target_class: int | None = None,
    mask: str | Path | np.ndarray | None = None,
    device: str | torch.device | None = None,
    target_layer: torch.nn.Module | None = None,
    threshold_mode: str = "top_percent",
    threshold_value: float = 0.20,
) -> ExplanationResult:
    """Run a single image through model + CAM and compute metrics if mask is given."""
    device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device).eval()

    pil = _load_image_as_rgb(image)
    pil_resized = pil.resize((image_size, image_size), Image.BILINEAR)
    image_rgb = np.asarray(pil_resized)

    transform = build_classification_transform(image_size=image_size, train=False)
    x = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        predicted_class = int(pred.item())
        confidence = float(conf.item())

    cls_for_cam = target_class if target_class is not None else predicted_class
    heatmap = generate_heatmap(
        model=model,
        input_tensor=x,
        target_class=cls_for_cam,
        method=method,
        target_layer=target_layer,
    )
    if heatmap.shape != (image_size, image_size):
        # pytorch_grad_cam usually upscales already; safety net.
        from PIL import Image as PILImage

        heatmap_img = PILImage.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (image_size, image_size), PILImage.BILINEAR
        )
        heatmap = np.asarray(heatmap_img).astype(np.float32) / 255.0

    mask_arr = _load_mask_2d(mask, (image_size, image_size))
    metrics = None
    if mask_arr is not None:
        metrics = explanation_metrics(
            heatmap=heatmap,
            mask=mask_arr,
            threshold_mode=threshold_mode,
            threshold_value=threshold_value,
        )

    return ExplanationResult(
        predicted_class=predicted_class,
        confidence=confidence,
        heatmap=heatmap,
        image_rgb=image_rgb,
        mask_rgb=mask_arr,
        metrics=metrics,
    )
