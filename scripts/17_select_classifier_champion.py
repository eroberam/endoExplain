"""Select the current classifier champion and write classifier_mvp.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "models" / "checkpoints"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select a classifier champion from classification_comparison.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--comparison_csv", type=Path, default=DEFAULT_ROOT / "classification_comparison.csv")
    p.add_argument("--metric", default="test_acc")
    p.add_argument("--output_json", type=Path, default=DEFAULT_ROOT / "classifier_mvp.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.comparison_csv)
    if args.metric not in df.columns:
        raise SystemExit(f"metric not found: {args.metric}")
    best = df.sort_values(args.metric, ascending=False).iloc[0].to_dict()
    summary_path = Path(str(best.get("summary_path", "")))
    if str(summary_path) in ("", "nan"):
        checkpoint = Path(str(best.get("checkpoint", "")))
        summary_path = checkpoint.parent / "summary.json" if str(checkpoint) else Path()
    if summary_path and not summary_path.is_absolute():
        summary_path = PROJECT_ROOT / summary_path
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    cfg = summary.get("config", {})
    checkpoint = best.get("checkpoint", "")
    run_dir = summary_path.parent if summary_path.exists() else DEFAULT_ROOT / str(best["run"])
    out = {
        "run": best["run"],
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "backbone": best.get("backbone", cfg.get("backbone", "")),
        "image_size": int(best.get("image_size", cfg.get("image_size", 0)) or 0),
        "num_classes": int(best.get("num_classes", summary.get("num_classes", 0)) or 0),
        "selection_metric": args.metric,
        "selection_metric_value": float(best[args.metric]),
        "best_val_acc": float(best.get("best_val_acc", 0.0) or 0.0),
        "summary_path": str(summary_path),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2))
    print(f"Selected {out['run']} -> {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
