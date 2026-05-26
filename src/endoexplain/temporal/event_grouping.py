"""Group consecutive suspicious frames into reviewable events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EventConfig:
    confidence_threshold: float = 0.5
    max_gap_seconds: float = 1.0
    min_event_duration_seconds: float = 0.2
    smoothed_column: str = "confidence_smoothed"
    timestamp_column: str = "timestamp"
    confidence_column: str = "confidence"
    quality_column: str | None = "quality_flag"


def group_events(
    frames_df: pd.DataFrame,
    config: EventConfig | None = None,
) -> pd.DataFrame:
    """Aggregate frame-level rows into events.

    Parameters
    ----------
    frames_df : pd.DataFrame
        Must contain at least ``timestamp`` and ``confidence`` columns. If a
        ``confidence_smoothed`` column is present (see
        :mod:`endoexplain.temporal.smoothing`) it is used to decide
        suspicious frames; otherwise raw ``confidence`` is used.

    Returns
    -------
    pd.DataFrame
        Columns: event_id, start_time, end_time, duration, num_frames,
        mean_confidence, max_confidence, representative_frame.
    """
    cfg = config or EventConfig()
    df = frames_df.copy().sort_values(cfg.timestamp_column).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "start_time",
                "end_time",
                "duration",
                "num_frames",
                "mean_confidence",
                "max_confidence",
                "representative_frame",
            ]
        )

    conf_col = (
        cfg.smoothed_column if cfg.smoothed_column in df.columns else cfg.confidence_column
    )
    df["_suspicious"] = df[conf_col] >= cfg.confidence_threshold

    events: list[dict] = []
    current: list[int] = []
    last_ts: float | None = None

    def _flush() -> None:
        if not current:
            return
        sub = df.iloc[current]
        start = float(sub[cfg.timestamp_column].iloc[0])
        end = float(sub[cfg.timestamp_column].iloc[-1])
        duration = end - start
        if duration < cfg.min_event_duration_seconds and len(current) < 2:
            return
        rep_idx = int(sub[conf_col].idxmax())
        rep_path = (
            str(df.iloc[rep_idx].get("frame_path", ""))
            if "frame_path" in df.columns
            else ""
        )
        events.append(
            {
                "event_id": len(events) + 1,
                "start_time": start,
                "end_time": end,
                "duration": duration,
                "num_frames": int(len(current)),
                "mean_confidence": float(sub[conf_col].mean()),
                "max_confidence": float(sub[conf_col].max()),
                "representative_frame": rep_path,
            }
        )

    for i, row in df.iterrows():
        ts = float(row[cfg.timestamp_column])
        gap = ts - last_ts if last_ts is not None else 0.0
        if row["_suspicious"]:
            if current and gap > cfg.max_gap_seconds:
                _flush()
                current = []
            current.append(int(i))
        else:
            if current and gap > cfg.max_gap_seconds:
                _flush()
                current = []
        last_ts = ts
    _flush()
    df.drop(columns=["_suspicious"], inplace=True)
    return pd.DataFrame(events)
