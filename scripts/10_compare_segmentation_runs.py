"""Compare segmentation run summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "models" / "checkpoints"


def _write_markdown(df: pd.DataFrame, path: Path) -> None:
    try:
        text = df.to_markdown(index=False)
    except ImportError:
        text = df.to_csv(index=False)
    path.write_text(text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a CSV/Markdown comparison from segmentation summary.json files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--runs_root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output_csv", type=Path, default=DEFAULT_ROOT / "segmentation_comparison.csv")
    p.add_argument("--glob", default="seg*/summary.json")
    return p.parse_args()


def _metric(data: dict, key: str) -> float | str:
    if key in data:
        return data[key]
    if key.startswith("test_"):
        short = key[5:]
        if short in data.get("test_metrics_at_0_5", {}):
            return data["test_metrics_at_0_5"][short]
        if short in data.get("test_metrics", {}):
            return data["test_metrics"][short]
    return ""


def main() -> int:
    args = parse_args()
    rows = []
    for summary_path in sorted(args.runs_root.glob(args.glob)):
        data = json.loads(summary_path.read_text())
        cfg = data.get("config", {})
        rows.append(
            {
                "run": summary_path.parent.name,
                "architecture": cfg.get("architecture", data.get("architecture", "")),
                "encoder": cfg.get("encoder_name", data.get("encoder_name", "")),
                "image_size": cfg.get("image_size", ""),
                "seed": cfg.get("seed", ""),
                "best_epoch": data.get("best_epoch", data.get("checkpoint_epoch", "")),
                "threshold": data.get("recommended_threshold", data.get("best_threshold", 0.5)),
                "val_dice": data.get("best_val_dice", data.get("best_val_dice_at_0_5", "")),
                "test_dice": _metric(data, "test_dice"),
                "test_iou": _metric(data, "test_iou"),
                "test_precision": _metric(data, "test_precision"),
                "test_recall": _metric(data, "test_recall"),
                "checkpoint": data.get("checkpoint", ""),
                "summary_path": str(summary_path),
            }
        )
    if not rows:
        raise SystemExit(f"no summaries found under {args.runs_root}")
    df = pd.DataFrame(rows).sort_values("test_dice", ascending=False)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    _write_markdown(df, args.output_csv.with_suffix(".md"))
    print(f"Wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
