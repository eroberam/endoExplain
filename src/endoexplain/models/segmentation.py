"""Segmentation model factory built on top of ``segmentation_models_pytorch``."""

from __future__ import annotations

_AVAILABLE_ARCHITECTURES = (
    "Unet",
    "UnetPlusPlus",
    "DeepLabV3Plus",
    "FPN",
    "Linknet",
    "MAnet",
    "MANet",
    "Segformer",
)
_DEFAULT_ENCODERS = (
    "resnet18",
    "resnet34",
    "mobilenet_v2",
    "efficientnet-b0",
    "efficientnet-b1",
    "mit_b0",
    "timm-mobilenetv3_small_100",
)


def available_segmentation_architectures() -> tuple[str, ...]:
    return _AVAILABLE_ARCHITECTURES


def available_segmentation_encoders() -> tuple[str, ...]:
    return _DEFAULT_ENCODERS


def build_segmenter(
    architecture: str = "Unet",
    encoder_name: str = "resnet18",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    num_classes: int = 1,
):
    """Return a ``segmentation_models_pytorch`` model.

    Defaults to U-Net + ResNet18 for fast experimentation.
    """
    try:
        import segmentation_models_pytorch as smp  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "segmentation_models_pytorch is required: `pip install segmentation-models-pytorch`"
        ) from e

    if architecture == "MANet":
        architecture = "MAnet"
    if architecture not in _AVAILABLE_ARCHITECTURES:
        raise ValueError(
            f"unknown architecture '{architecture}'. Available: {_AVAILABLE_ARCHITECTURES}"
        )

    factory = getattr(smp, architecture)
    return factory(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )
