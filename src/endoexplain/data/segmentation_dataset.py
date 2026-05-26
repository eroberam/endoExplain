"""PyTorch Dataset for paired image/mask segmentation."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class SegmentationPairDataset(Dataset):
    """Yields ``(image_tensor, mask_tensor)`` from a CSV produced by
    :func:`endoexplain.data.index_segmentation.index_segmentation`.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``image_path`` and ``mask_path`` columns.
    image_size : int
        Square resize side.
    augment : bool
        Apply light geometric/photometric augmentation when True.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 256,
        augment: bool = False,
        augment_level: str = "light",
        normalize: bool = False,
    ) -> None:
        missing = {"image_path", "mask_path"} - set(df.columns)
        if missing:
            raise ValueError(f"dataframe missing required columns: {missing}")
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.augment = augment
        self.augment_level = augment_level
        self.normalize = normalize
        self._augment_pipeline: Callable | None = None
        if augment:
            self._augment_pipeline = self._build_albumentations()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image = np.asarray(Image.open(row["image_path"]).convert("RGB"))
        mask = np.asarray(Image.open(row["mask_path"]).convert("L"))
        mask = (mask > 127).astype(np.float32)

        if self._augment_pipeline is not None:
            out = self._augment_pipeline(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        else:
            image, mask = self._resize_pair(image, mask)

        # to tensors, CHW, [0,1]
        image_t = torch.from_numpy(image.copy().transpose(2, 0, 1)).float() / 255.0
        if self.normalize:
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_t.dtype).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=image_t.dtype).view(3, 1, 1)
            image_t = (image_t - mean) / std
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)  # (1, H, W)
        return image_t, mask_t

    def _resize_pair(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from PIL import Image as PILImage

        size = (self.image_size, self.image_size)
        image_r = np.asarray(
            PILImage.fromarray(image).resize(size, PILImage.BILINEAR)
        )
        mask_r = np.asarray(
            PILImage.fromarray((mask * 255).astype(np.uint8)).resize(size, PILImage.NEAREST)
        ) > 127
        return image_r, mask_r.astype(np.float32)

    def _build_albumentations(self):
        import albumentations as A

        if self.augment_level.lower() == "strong":
            return A.Compose(
                [
                    A.Resize(self.image_size, self.image_size),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Affine(
                        scale=(0.90, 1.12),
                        translate_percent=(-0.04, 0.04),
                        rotate=(-18, 18),
                        p=0.45,
                    ),
                    A.ElasticTransform(alpha=12, sigma=8, p=0.15),
                    A.GridDistortion(num_steps=5, distort_limit=0.08, p=0.12),
                    A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.35),
                    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=12, val_shift_limit=8, p=0.20),
                    A.GaussNoise(p=0.10),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.08),
                ]
            )

        return A.Compose(
            [
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ]
        )
