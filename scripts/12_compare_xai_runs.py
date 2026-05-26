"""Summarise explainability CSV outputs from scripts/04_run_explainability.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = PROJECT_ROOT / "outputs" / "metrics"


def _write_markdown(df: pd.DataFrame, path: Path) -> None:
    try:
        text = df.to_markdown(index=False)
    except ImportError:
        text = df.to_csv(index=False)
    path.write_text(text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare XAI alignment metric CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--metrics_dir", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--glob", default="explainability_*.csv")
    p.add_argument("--output_csv", type=Path, default=DEFAULT_METRICS / "xai_comparison.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for csv_path in sorted(args.metrics_dir.glob(args.glob)):
        df = pd.read_csv(csv_path)
        metrics = {}
        for col in (
            "explanation_iou",
            "activation_inside_mask",
            "activation_outside_mask",
            "pointing_game_hit",
            "heatmap_mask_center_distance",
        ):
            if col in df.columns:
                metrics[col] = float(df[col].mean())
        rows.append(
            {
                "run": csv_path.stem,
                "rows": len(df),
                "with_mask": int(df.get("has_mask", pd.Series(dtype=bool)).sum())
                if "has_mask" in df.columns
                else "",
                **metrics,
                "csv_path": str(csv_path),
            }
        )
    if not rows:
        raise SystemExit(f"no XAI CSVs found under {args.metrics_dir}")
    out = pd.DataFrame(rows)
    sort_col = next(
        (
            c
            for c in ("activation_inside_mask", "explanation_iou", "pointing_game_hit")
            if c in out.columns
        ),
        "rows",
    )
    out = out.sort_values(sort_col, ascending=False)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    _write_markdown(out, args.output_csv.with_suffix(".md"))
    print(f"Wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
