"""Aggregate temporal metrics over a frame-level prediction CSV."""

from __future__ import annotations

import numpy as np
import pandas as pd


def event_metrics(frames_df: pd.DataFrame, confidence_column: str = "confidence") -> dict:
    """Compute global temporal-stability indicators for a video."""
    if frames_df.empty:
        return {
            "num_frames": 0,
            "confidence_mean": float("nan"),
            "confidence_std": float("nan"),
            "confidence_variance": float("nan"),
            "isolated_spikes": 0,
            "temporal_smoothness": float("nan"),
        }
    conf = frames_df[confidence_column].astype(float).to_numpy()
    diffs = np.diff(conf) if len(conf) > 1 else np.array([0.0])
    isolated = int(((diffs > 0.5) | (diffs < -0.5)).sum())
    smoothness = float(np.exp(-np.mean(np.abs(diffs))))
    return {
        "num_frames": int(len(conf)),
        "confidence_mean": float(conf.mean()),
        "confidence_std": float(conf.std()),
        "confidence_variance": float(conf.var()),
        "isolated_spikes": isolated,
        "temporal_smoothness": smoothness,
    }
