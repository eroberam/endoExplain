"""Compare classifier run summaries."""

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
        description="Build a CSV/Markdown comparison from classifier summary.json files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--runs_root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output_csv", type=Path, default=DEFAULT_ROOT / "classification_comparison.csv")
    p.add_argument("--glob", default="classifier*/summary.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for summary_path in sorted(args.runs_root.glob(args.glob)):
        data = json.loads(summary_path.read_text())
        cfg = data.get("config", {})
        rows.append(
            {
                "run": summary_path.parent.name,
                "backbone": cfg.get("backbone", data.get("backbone", "")),
                "image_size": cfg.get("image_size", ""),
                "batch_size": cfg.get("batch_size", ""),
                "epochs": cfg.get("epochs", ""),
                "best_epoch": data.get("best_epoch", data.get("checkpoint_epoch", "")),
                "best_val_acc": data.get("best_val_acc", ""),
                "test_acc": data.get("test_acc", data.get("test_metrics", {}).get("accuracy", "")),
                "test_loss": data.get("test_loss", data.get("test_metrics", {}).get("loss", "")),
                "num_classes": data.get("num_classes", ""),
                "n_train": data.get("n_train", ""),
                "n_val": data.get("n_val", ""),
                "n_test": data.get("n_test", data.get("test_metrics", {}).get("num_samples", "")),
                "checkpoint": data.get("checkpoint", ""),
                "summary_path": str(summary_path),
            }
        )
    if not rows:
        raise SystemExit(f"no summaries found under {args.runs_root}")
    df = pd.DataFrame(rows).sort_values("test_acc", ascending=False)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    _write_markdown(df, args.output_csv.with_suffix(".md"))
    print(f"Wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
