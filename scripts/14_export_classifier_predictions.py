"""Export classifier predictions to CSV for external metric dashboards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from endoexplain.clinical_targets import binary_label_for_target, resolve_target  # noqa: E402
from endoexplain.config import HYPERKVASIR_INDEX_CSV, METRICS_DIR  # noqa: E402
from endoexplain.data.transforms import build_classification_transform  # noqa: E402
from endoexplain.models import build_classifier  # noqa: E402


class _ImageRows(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = build_classification_transform(image_size, train=False)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["absolute_path"]).convert("RGB")
        return self.transform(img), idx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export image-level classifier predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--index_csv", type=Path, default=HYPERKVASIR_INDEX_CSV)
    p.add_argument("--output_csv", type=Path, default=None)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--classes", nargs="*", default=None)
    p.add_argument("--split_csv", type=Path, default=None)
    p.add_argument("--split", choices=["train", "val", "test"], default=None)
    p.add_argument("--positive_classes", nargs="*", default=["polyps", "dyed-lifted-polyps"])
    p.add_argument("--positive_name", default="polyp_family")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    return p.parse_args()


def _load_classifier(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    class_to_idx = ckpt["class_to_idx"]
    model = build_classifier(
        ckpt.get("backbone", "resnet18"),
        num_classes=len(class_to_idx),
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), class_to_idx


def _input_rows(index_csv: Path, split_csv: Path | None, split: str | None) -> pd.DataFrame:
    df = pd.read_csv(split_csv if split_csv and split_csv.exists() else index_csv)
    df = df[df["file_type"] == "image"].copy() if "file_type" in df.columns else df.copy()
    if split and "split" in df.columns:
        df = df[df["split"] == split].copy()
    return df.reset_index(drop=True)


def main() -> int:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, class_to_idx = _load_classifier(args.checkpoint, device)
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    df = _input_rows(args.index_csv, args.split_csv, args.split)
    if args.classes:
        df = df[df["inferred_label"].isin(args.classes)].reset_index(drop=True)
    if df.empty:
        raise SystemExit("no rows to score")

    ds = _ImageRows(df, args.image_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    rows: list[dict] = []
    target = resolve_target(
        class_to_idx=class_to_idx,
        positive_classes=args.positive_classes,
        target_display_name=args.positive_name,
    )

    with torch.no_grad():
        for xb, indices in loader:
            xb = xb.to(device, non_blocking=True)
            probs = torch.softmax(model(xb), dim=1).cpu().numpy()
            pred_idx = probs.argmax(axis=1)
            for j, row_idx in enumerate(indices.numpy().tolist()):
                src = df.iloc[int(row_idx)]
                label = str(src.get("inferred_label", ""))
                pred_label = idx_to_class[int(pred_idx[j])]
                target_score = float(probs[j, list(target.class_indices)].sum())
                out = {
                    "sample_id": src.get("filename", Path(src["absolute_path"]).name),
                    "image_path": src["absolute_path"],
                    "split": src.get("split", ""),
                    "y_true": label,
                    "y_pred": pred_label,
                    "correct": int(label == pred_label),
                    "target_name": target.display_name,
                    "target_positive_classes": "+".join(target.class_labels),
                    "y_true_binary": binary_label_for_target(label, target.class_labels),
                    "y_score_binary": target_score,
                    "y_pred_binary": int(target_score >= args.threshold),
                    "threshold": args.threshold,
                    "confidence": float(probs[j, int(pred_idx[j])]),
                }
                for cls, cls_idx in class_to_idx.items():
                    out[f"proba_{cls}"] = float(probs[j, cls_idx])
                rows.append(out)

    out_csv = args.output_csv or (
        METRICS_DIR
        / "classifier_predictions"
        / args.checkpoint.parent.name
        / f"predictions_{args.split or 'all'}.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)

    y_true = out_df["y_true_binary"].astype(int).to_numpy()
    y_score = out_df["y_score_binary"].astype(float).to_numpy()
    y_pred_binary = out_df["y_pred_binary"].astype(int).to_numpy()
    y_true_multi = out_df["y_true"].astype(str).to_numpy()
    y_pred_multi = out_df["y_pred"].astype(str).to_numpy()

    def _safe_metric(fn, default: float = float("nan")) -> float:
        try:
            return float(fn())
        except Exception:
            return default

    metrics = {
        "n": int(len(out_df)),
        "multiclass_accuracy": float(out_df["correct"].astype(int).mean()),
        "multiclass_balanced_accuracy": _safe_metric(
            lambda: balanced_accuracy_score(y_true_multi, y_pred_multi)
        ),
        f"{args.positive_name}_roc_auc": _safe_metric(lambda: roc_auc_score(y_true, y_score)),
        f"{args.positive_name}_average_precision": _safe_metric(
            lambda: average_precision_score(y_true, y_score)
        ),
        f"{args.positive_name}_f1_at_{str(args.threshold).replace('.', '_')}": _safe_metric(
            lambda: f1_score(y_true, y_pred_binary)
        ),
        f"{args.positive_name}_brier": _safe_metric(lambda: brier_score_loss(y_true, y_score)),
        "threshold": float(args.threshold),
        "positives": int(y_true.sum()),
        "negatives": int((1 - y_true).sum()),
    }

    summary = {
        "checkpoint": str(args.checkpoint),
        "rows": len(rows),
        "output_csv": str(out_csv),
        "class_to_idx": class_to_idx,
        "target": target.__dict__,
        "metrics_json": str(out_csv.with_suffix(".metrics.json")),
    }
    out_csv.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    out_csv.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out_csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
