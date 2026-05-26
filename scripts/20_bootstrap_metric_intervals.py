"""Bootstrap confidence intervals from exported evaluation CSVs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from endoexplain.evaluation import bootstrap_metric  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute percentile bootstrap intervals for exported metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_csv", type=Path, required=True)
    p.add_argument("--kind", choices=["classification", "segmentation", "xai", "temporal"], required=True)
    p.add_argument("--output_json", type=Path, default=None)
    p.add_argument("--n_boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def _safe_metric(fn):
    try:
        return float(fn())
    except Exception:
        return float("nan")


def _classification_metrics(df: pd.DataFrame, n_boot: int, seed: int, threshold: float) -> dict:
    strata = df["y_true_binary"].astype(int).to_numpy() if "y_true_binary" in df.columns else None
    return {
        "accuracy": bootstrap_metric(df, lambda x: x["correct"].astype(float).mean(), n_boot, seed),
        "roc_auc": bootstrap_metric(
            df,
            lambda x: _safe_metric(lambda: roc_auc_score(x["y_true_binary"], x["y_score_binary"])),
            n_boot,
            seed,
            strata=strata,
        ),
        "average_precision": bootstrap_metric(
            df,
            lambda x: _safe_metric(lambda: average_precision_score(x["y_true_binary"], x["y_score_binary"])),
            n_boot,
            seed,
            strata=strata,
        ),
        "f1": bootstrap_metric(
            df,
            lambda x: _safe_metric(
                lambda: f1_score(
                    x["y_true_binary"],
                    (x["y_score_binary"].astype(float) >= threshold).astype(int),
                )
            ),
            n_boot,
            seed,
            strata=strata,
        ),
        "brier": bootstrap_metric(
            df,
            lambda x: _safe_metric(lambda: brier_score_loss(x["y_true_binary"], x["y_score_binary"])),
            n_boot,
            seed,
            strata=strata,
        ),
    }


def _mean_metrics(df: pd.DataFrame, columns: list[str], n_boot: int, seed: int) -> dict:
    return {
        col: bootstrap_metric(
            df,
            lambda x, c=col: pd.to_numeric(x[c], errors="coerce").mean(),
            n_boot,
            seed,
        )
        for col in columns
        if col in df.columns
    }


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    if args.kind == "classification":
        metrics = _classification_metrics(df, args.n_boot, args.seed, args.threshold)
    elif args.kind == "segmentation":
        metrics = _mean_metrics(df, ["dice", "iou", "precision", "recall"], args.n_boot, args.seed)
    elif args.kind == "xai":
        metrics = _mean_metrics(
            df,
            ["explanation_iou", "activation_inside_mask", "pointing_game_hit"],
            args.n_boot,
            args.seed,
        )
    else:
        metrics = _mean_metrics(
            df,
            ["precision", "recall", "f1", "false_events_per_min", "median_latency_seconds"],
            args.n_boot,
            args.seed,
        )

    payload = {
        "input_csv": str(args.input_csv),
        "kind": args.kind,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "metrics": metrics,
    }
    out_json = args.output_json or args.input_csv.with_suffix(".bootstrap.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
