"""Train/val/test splitting helpers.

For image classification we can split per-record. For video frames the
caller MUST group by video first to avoid temporal leakage.
"""

from __future__ import annotations

import random
from typing import Sequence

import pandas as pd

from ..config.settings import DEFAULT_SEED, DEFAULT_SPLIT


def stratified_split(
    df: pd.DataFrame,
    label_col: str = "inferred_label",
    fractions: tuple[float, float, float] = DEFAULT_SPLIT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Return ``df`` with a new ``split`` column in {train, val, test}.

    Stratifies per ``label_col`` so each class is represented in all splits.
    Does NOT group adjacent video frames. Only use for image-level data.
    """
    train_frac, val_frac, test_frac = fractions
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0 (got {total})")

    rng = random.Random(seed)
    out = df.copy()
    out["split"] = ""

    for label, group in out.groupby(label_col):
        indices = list(group.index)
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]
        out.loc[train_idx, "split"] = "train"
        out.loc[val_idx, "split"] = "val"
        out.loc[test_idx, "split"] = "test"

    return out


def group_split_by_video(
    df: pd.DataFrame,
    video_col: str,
    fractions: tuple[float, float, float] = DEFAULT_SPLIT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Split so that all frames of one video land in the same split.

    Prevents temporal leakage when training a frame-level classifier on
    extracted video frames.
    """
    train_frac, val_frac, _ = fractions
    rng = random.Random(seed)
    videos: Sequence[str] = sorted(df[video_col].unique().tolist())
    videos = list(videos)
    rng.shuffle(videos)
    n = len(videos)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_vids = set(videos[:n_train])
    val_vids = set(videos[n_train : n_train + n_val])

    def assign(v: str) -> str:
        if v in train_vids:
            return "train"
        if v in val_vids:
            return "val"
        return "test"

    out = df.copy()
    out["split"] = out[video_col].map(assign)
    return out
