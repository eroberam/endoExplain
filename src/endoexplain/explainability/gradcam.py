"""Wrappers around ``pytorch_grad_cam`` for the supported XAI methods."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

_METHOD_REGISTRY = {
    "gradcam": "GradCAM",
    "gradcam++": "GradCAMPlusPlus",
    "eigencam": "EigenCAM",
    "scorecam": "ScoreCAM",
    "ablationcam": "AblationCAM",
    "xgradcam": "XGradCAM",
}


def available_methods() -> list[str]:
    return sorted(_METHOD_REGISTRY)


def default_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Best-effort pick of the last conv block for popular backbones."""
    # torchvision ResNets
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    # torchvision MobileNetV3 / EfficientNet (via timm) / generic
    if hasattr(model, "features"):
        features = model.features
        for module in reversed(list(features.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module
    if hasattr(model, "blocks"):  # timm ConvNeXt / EfficientNet
        return model.blocks[-1]
    # Last resort: last Conv2d in the whole model
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError(
            "could not auto-detect a target layer for Grad-CAM; pass one explicitly"
        )
    return last_conv


def generate_heatmap(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int | None = None,
    method: str = "gradcam++",
    target_layer: torch.nn.Module | None = None,
) -> np.ndarray:
    """Return a (H, W) heatmap in [0, 1] for the given input.

    Parameters
    ----------
    model : nn.Module
        Trained classifier in eval mode.
    input_tensor : torch.Tensor
        Shape (1, C, H, W), already normalised.
    target_class : int | None
        Class index for which to compute the attribution. If None,
        uses the predicted class.
    method : str
        One of :func:`available_methods`.
    target_layer : nn.Module | None
        Conv layer to hook. If None, picked automatically.
    """
    if method not in _METHOD_REGISTRY:
        raise ValueError(f"unknown method '{method}'. Available: {available_methods()}")

    try:
        import pytorch_grad_cam as gc  # type: ignore
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "grad-cam is required: `pip install grad-cam`"
        ) from e

    cam_cls = getattr(gc, _METHOD_REGISTRY[method])
    layer = target_layer if target_layer is not None else default_target_layer(model)

    targets: Iterable | None = None
    if target_class is not None:
        targets = [ClassifierOutputTarget(int(target_class))]

    model.eval()
    with cam_cls(model=model, target_layers=[layer]) as cam:
        out = cam(input_tensor=input_tensor, targets=targets)  # (B, H, W)
    return out[0]
