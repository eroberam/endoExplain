"""Quality-stratified summaries for endoscopy evaluation tables."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


QUALITY_STRATA = ("good", "blurred", "overexposed", "reflective", "mixed_low_quality")


def _flag(row: Mapping[str, Any], name: str) -> bool:
    value = row.get(name, False)
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def assign_quality_stratum(row: Mapping[str, Any]) -> str:
    """Map lightweight frame indicators to one quality stratum."""
    issues: list[str] = []
    if _flag(row, "blur_flag"):
        issues.append("blurred")
    if _flag(row, "overexposed_frame_flag") or _flag(row, "dark_frame_flag"):
        issues.append("overexposed")
    if _flag(row, "reflection_flag"):
        issues.append("reflective")
    if not issues:
        return "good"
    if len(issues) == 1:
        return issues[0]
    return "mixed_low_quality"


def add_quality_strata(df: pd.DataFrame, output_column: str = "quality_stratum") -> pd.DataFrame:
    """Return a copy of ``df`` with a quality-stratum column."""
    out = df.copy()
    out[output_column] = [assign_quality_stratum(row) for row in out.to_dict("records")]
    return out


def _safe_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        if y_true.nunique(dropna=True) < 2:
            return float("nan")
        return float(roc_auc_score(y_true.astype(int), y_score.astype(float)))
    except Exception:
        return float("nan")


def summarize_quality_strata(
    df: pd.DataFrame,
    stratum_column: str = "quality_stratum",
    min_n: int = 20,
) -> pd.DataFrame:
    """Summarize available metrics per quality stratum."""
    if stratum_column not in df.columns:
        df = add_quality_strata(df, output_column=stratum_column)

    rows: list[dict[str, float | int | str | bool]] = []
    for stratum in QUALITY_STRATA:
        sub = df[df[stratum_column] == stratum]
        row: dict[str, float | int | str | bool] = {
            "quality_stratum": stratum,
            "n": int(len(sub)),
            "descriptive_only": bool(len(sub) < min_n),
        }
        if len(sub) == 0:
            rows.append(row)
            continue
        if "correct" in sub.columns:
            row["accuracy"] = float(sub["correct"].astype(float).mean())
        if {"y_true_binary", "y_score_binary"}.issubset(sub.columns):
            row["roc_auc"] = _safe_auc(sub["y_true_binary"], sub["y_score_binary"])
        for col in (
            "dice",
            "iou",
            "precision",
            "recall",
            "explanation_iou",
            "activation_inside_mask",
            "pointing_game_hit",
        ):
            if col in sub.columns:
                values = pd.to_numeric(sub[col], errors="coerce")
                row[f"mean_{col}"] = float(values.mean()) if values.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
