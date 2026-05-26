"""Event-level temporal benchmark utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


VIDEO_COLUMNS = {
    "video_id",
    "relative_path",
    "source_label",
    "target_binary",
    "fps",
    "frame_count",
    "duration_seconds",
    "role",
}
EVENT_COLUMNS = {"video_id", "event_id", "start_s", "end_s"}


@dataclass(frozen=True)
class TemporalMatch:
    truth_index: int
    pred_index: int
    overlap_seconds: float
    latency_seconds: float


def validate_temporal_inputs(videos: pd.DataFrame, events: pd.DataFrame) -> list[str]:
    """Return schema/consistency issues for temporal benchmark CSVs."""
    issues: list[str] = []
    missing_video = sorted(VIDEO_COLUMNS - set(videos.columns))
    missing_event = sorted(EVENT_COLUMNS - set(events.columns))
    if missing_video:
        issues.append(f"videos CSV missing columns: {', '.join(missing_video)}")
    if missing_event:
        issues.append(f"events CSV missing columns: {', '.join(missing_event)}")
    if "video_id" in videos.columns and videos["video_id"].duplicated().any():
        issues.append("videos CSV has duplicated video_id values")
    if {"video_id", "start_s", "end_s"}.issubset(events.columns):
        known = set(videos.get("video_id", []))
        unknown = sorted(set(events["video_id"].dropna()) - known)
        if unknown:
            issues.append(f"events CSV references unknown videos: {', '.join(unknown[:5])}")
        complete = events.dropna(subset=["start_s", "end_s"]).copy()
        if not complete.empty:
            bad = complete[pd.to_numeric(complete["end_s"]) <= pd.to_numeric(complete["start_s"])]
            if not bad.empty:
                issues.append("events CSV has rows with end_s <= start_s")
    return issues


def events_from_frame_scores(
    frames: pd.DataFrame,
    score_column: str,
    threshold: float,
    timestamp_column: str = "timestamp",
    max_gap_seconds: float = 1.0,
    min_event_duration_seconds: float = 0.2,
) -> pd.DataFrame:
    """Group suprathreshold frame scores into temporal events."""
    if frames.empty:
        return pd.DataFrame(columns=["event_id", "start_time", "end_time", "duration"])
    df = frames.sort_values(timestamp_column).reset_index(drop=True)
    active = pd.to_numeric(df[score_column], errors="coerce").fillna(0.0) >= threshold
    timestamps = pd.to_numeric(df[timestamp_column], errors="coerce").fillna(0.0).to_numpy()

    events: list[dict[str, float | int]] = []
    current: list[int] = []
    last_ts: float | None = None

    def flush() -> None:
        if not current:
            return
        start = float(timestamps[current[0]])
        end = float(timestamps[current[-1]])
        duration = end - start
        if duration < min_event_duration_seconds and len(current) < 2:
            return
        events.append(
            {
                "event_id": len(events) + 1,
                "start_time": start,
                "end_time": end,
                "duration": duration,
                "num_frames": int(len(current)),
            }
        )

    for idx, is_active in enumerate(active):
        ts = float(timestamps[idx])
        gap = ts - last_ts if last_ts is not None else 0.0
        if is_active:
            if current and gap > max_gap_seconds:
                flush()
                current = []
            current.append(idx)
        elif current and gap > max_gap_seconds:
            flush()
            current = []
        last_ts = ts
    flush()
    return pd.DataFrame(events)


def _intervals(df: pd.DataFrame, start_col: str, end_col: str) -> list[tuple[float, float]]:
    if df.empty:
        return []
    complete = df.dropna(subset=[start_col, end_col])
    return [
        (float(row[start_col]), float(row[end_col]))
        for _, row in complete.iterrows()
        if float(row[end_col]) > float(row[start_col])
    ]


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def match_temporal_events(
    truth_events: pd.DataFrame,
    pred_events: pd.DataFrame,
    min_overlap_seconds: float = 0.0,
) -> tuple[list[TemporalMatch], list[int], list[int]]:
    """Greedy one-to-one matching by temporal overlap."""
    truth = _intervals(truth_events, "start_s", "end_s")
    pred = _intervals(pred_events, "start_time", "end_time")
    candidates: list[tuple[float, int, int]] = []
    for ti, t in enumerate(truth):
        for pi, p in enumerate(pred):
            ov = _overlap(t, p)
            if ov > min_overlap_seconds:
                candidates.append((ov, ti, pi))
    candidates.sort(reverse=True)

    used_truth: set[int] = set()
    used_pred: set[int] = set()
    matches: list[TemporalMatch] = []
    for ov, ti, pi in candidates:
        if ti in used_truth or pi in used_pred:
            continue
        latency = max(0.0, pred[pi][0] - truth[ti][0])
        matches.append(TemporalMatch(ti, pi, float(ov), float(latency)))
        used_truth.add(ti)
        used_pred.add(pi)

    false_negatives = [i for i in range(len(truth)) if i not in used_truth]
    false_positives = [i for i in range(len(pred)) if i not in used_pred]
    return matches, false_positives, false_negatives


def _f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return 0.0 if denom == 0.0 else float(2.0 * precision * recall / denom)


def evaluate_temporal_benchmark(
    videos: pd.DataFrame,
    truth_events: pd.DataFrame,
    frame_predictions: dict[str, pd.DataFrame],
    threshold: float = 0.85,
    raw_score_column: str = "target_probability",
    smoothed_score_column: str = "target_probability_smoothed",
    timestamp_column: str = "timestamp",
    max_gap_seconds: float = 1.0,
    min_event_duration_seconds: float = 0.2,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Evaluate event detection over videos with complete annotations."""
    rows: list[dict[str, float | int | str]] = []
    all_latencies: list[float] = []
    totals = {"tp": 0, "fp": 0, "fn": 0, "raw_events": 0, "pred_events": 0}
    total_minutes = 0.0
    skipped = 0

    for _, video in videos.iterrows():
        video_id = str(video["video_id"])
        frames = frame_predictions.get(video_id)
        if frames is None or frames.empty:
            skipped += 1
            continue
        truth = truth_events[truth_events["video_id"].astype(str) == video_id]
        if int(video.get("target_binary", 0)) == 1 and truth.dropna(subset=["start_s", "end_s"]).empty:
            skipped += 1
            continue

        raw = events_from_frame_scores(
            frames,
            score_column=raw_score_column,
            threshold=threshold,
            timestamp_column=timestamp_column,
            max_gap_seconds=max_gap_seconds,
            min_event_duration_seconds=min_event_duration_seconds,
        )
        score_col = smoothed_score_column if smoothed_score_column in frames.columns else raw_score_column
        pred = events_from_frame_scores(
            frames,
            score_column=score_col,
            threshold=threshold,
            timestamp_column=timestamp_column,
            max_gap_seconds=max_gap_seconds,
            min_event_duration_seconds=min_event_duration_seconds,
        )
        matches, fps, fns = match_temporal_events(truth, pred)
        tp = len(matches)
        fp = len(fps)
        fn = len(fns)
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        duration = float(video.get("duration_seconds", frames[timestamp_column].max()))
        minutes = max(duration / 60.0, 1e-9)
        latencies = [m.latency_seconds for m in matches]
        all_latencies.extend(latencies)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["raw_events"] += int(len(raw))
        totals["pred_events"] += int(len(pred))
        total_minutes += minutes
        rows.append(
            {
                "video_id": video_id,
                "target_binary": int(video.get("target_binary", 0)),
                "duration_seconds": duration,
                "truth_events": int(len(truth.dropna(subset=["start_s", "end_s"]))),
                "raw_events": int(len(raw)),
                "pred_events": int(len(pred)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": float(precision),
                "recall": float(recall),
                "f1": _f1(float(precision), float(recall)),
                "false_events_per_min": float(fp / minutes),
                "median_latency_seconds": float(np.median(latencies)) if latencies else np.nan,
            }
        )

    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 1.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 1.0
    raw_events = totals["raw_events"]
    pred_events = totals["pred_events"]
    summary = {
        "videos_evaluated": len(rows),
        "videos_skipped": int(skipped),
        "tp": int(totals["tp"]),
        "fp": int(totals["fp"]),
        "fn": int(totals["fn"]),
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": _f1(float(precision), float(recall)),
        "false_events_per_min": float(totals["fp"] / max(total_minutes, 1e-9)),
        "raw_events": int(raw_events),
        "smoothed_events": int(pred_events),
        "spike_reduction": float(1.0 - pred_events / raw_events) if raw_events else 0.0,
        "median_latency_seconds": float(np.median(all_latencies)) if all_latencies else np.nan,
        "latency_iqr_seconds": float(np.quantile(all_latencies, 0.75) - np.quantile(all_latencies, 0.25))
        if all_latencies
        else np.nan,
    }
    return pd.DataFrame(rows), summary
