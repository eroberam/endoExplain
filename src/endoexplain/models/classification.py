"""Image classifier builders."""

from __future__ import annotations

from typing import Iterable

_TORCHVISION_BACKBONES = {"resnet18", "resnet34", "mobilenet_v3_small", "mobilenet_v3_large"}
_TIMM_BACKBONES = {
    "efficientnet_b0",
    "mobilenetv3_small_100",
    "convnext_tiny",
    "deit_tiny_patch16_224",
    "vit_tiny_patch16_224",
    "swin_tiny_patch4_window7_224",
}


def available_classifier_backbones() -> list[str]:
    return sorted(_TORCHVISION_BACKBONES | _TIMM_BACKBONES)


def build_classifier(
    backbone: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = True,
):
    """Return a torch.nn.Module classifier head over the requested backbone.

    Raises
    ------
    ValueError
        If the backbone name is unknown.
    ImportError
        If a timm backbone is requested but timm is not installed.
    """
    import torch.nn as nn  # Local import keeps `import endoexplain` cheap.

    backbone = backbone.lower()

    if backbone in _TORCHVISION_BACKBONES:
        return _build_torchvision(backbone, num_classes, pretrained)

    if backbone in _TIMM_BACKBONES:
        try:
            import timm  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                f"backbone '{backbone}' needs timm: `pip install timm`"
            ) from e
        return timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)

    raise ValueError(
        f"unknown backbone '{backbone}'. Available: {available_classifier_backbones()}"
    )


def _build_torchvision(backbone: str, num_classes: int, pretrained: bool):
    import torch.nn as nn
    from torchvision import models

    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if backbone == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if backbone == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
        return model
    if backbone == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
        return model
    raise ValueError(backbone)  # unreachable
