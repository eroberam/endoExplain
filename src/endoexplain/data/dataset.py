"""PyTorch Dataset wrappers around the indexed CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class IndexedImageClassificationDataset(Dataset):
    """Reads rows from a CSV produced by ``index_hyperkvasir``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``absolute_path``, ``inferred_label`` (and
        optionally ``split``).
    class_to_idx : dict[str, int]
        Stable mapping from label string to integer class id.
    transform : callable | None
        Image transform (PIL → tensor). Defaults to ToTensor().
    """

    def __init__(
        self,
        df: pd.DataFrame,
        class_to_idx: dict[str, int],
        transform: Callable | None = None,
    ) -> None:
        missing = {"absolute_path", "inferred_label"} - set(df.columns)
        if missing:
            raise ValueError(f"dataframe missing required columns: {missing}")
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["absolute_path"]
        label = row["inferred_label"]
        target = self.class_to_idx[label]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    @staticmethod
    def build_class_map(labels: Sequence[str]) -> dict[str, int]:
        return {label: i for i, label in enumerate(sorted(set(labels)))}
