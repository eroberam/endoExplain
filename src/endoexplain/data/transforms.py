"""Image preprocessing for classification / segmentation models."""

from __future__ import annotations

from typing import Callable

from torchvision import transforms

# ImageNet stats for pretrained torchvision/timm backbones.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_classification_transform(
    image_size: int = 256,
    train: bool = True,
    augment_level: str = "standard",
) -> Callable:
    if train:
        level = augment_level.lower()
        if level == "strong":
            return transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(degrees=12),
                    transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.94, 1.06)),
                    transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.08, hue=0.02),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                    transforms.RandomErasing(p=0.12, scale=(0.01, 0.04), ratio=(0.5, 2.0), value=0),
                ]
            )
        if level == "light":
            return transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.08, contrast=0.08),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
