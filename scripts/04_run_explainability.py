"""Run class-activation maps on a trained classifier checkpoint.

If the index CSV contains an ``inferred_label`` column, the script can be
restricted to a subset of classes. If a segmentation index CSV is supplied
via ``--seg_index_csv``, paired masks are loaded and explanation-mask
metrics are written next to the heatmaps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from endoexplain.config import HEATMAPS_DIR, METRICS_DIR, HYPERKVASIR_INDEX_CSV  # noqa: E402
from endoexplain.explainability import (  # noqa: E402
    available_methods,
    explain_image,
    overlay_heatmap,
    overlay_mask,
)
from endoexplain.models import build_classifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate CAM heatmaps and (optionally) explanation-mask metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to classifier best.pt")
    p.add_argument("--index_csv", type=Path, default=HYPERKVASIR_INDEX_CSV)
    p.add_argument(
        "--seg_index_csv",
        type=Path,
        default=None,
        help="Optional segmentation index CSV to attach ground-truth masks via filename match.",
    )
    p.add_argument("--method", choices=available_methods(), default="gradcam++")
    p.add_argument("--num_samples", type=int, default=12)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--classes", nargs="*", default=None)
    p.add_argument(
        "--threshold_mode",
        choices=("top_percent", "fixed", "otsu"),
        default="top_percent",
    )
    p.add_argument("--threshold_value", type=float, default=0.20)
    p.add_argument(
        "--output_dir",
        type=Path,
        default=HEATMAPS_DIR,
        help="Folder to write overlays into.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    return p.parse_args()


def _load_classifier(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, int]]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_to_idx: dict[str, int] = ckpt["class_to_idx"]
    backbone: str = ckpt["backbone"]
    model = build_classifier(backbone, num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), class_to_idx


def _sample_rows(df: pd.DataFrame, classes: list[str] | None, n: int, seed: int) -> pd.DataFrame:
    df = df.copy()
    if "absolute_path" not in df.columns and "image_path" in df.columns:
        df["absolute_path"] = df["image_path"]
    if "file_type" not in df.columns:
        df["file_type"] = "image"
    if "inferred_label" not in df.columns:
        df["inferred_label"] = "polyps"
    pool = df[df["file_type"] == "image"]
    if classes:
        pool = pool[pool["inferred_label"].isin(classes)]
    if pool.empty:
        raise RuntimeError("no rows match the requested classes")
    return pool.sample(n=min(n, len(pool)), random_state=seed).reset_index(drop=True)


def _build_mask_lookup(seg_csv: Path | None) -> dict[str, str]:
    if seg_csv is None or not seg_csv.exists():
        return {}
    seg_df = pd.read_csv(seg_csv)
    out: dict[str, str] = {}
    for _, row in seg_df.iterrows():
        if not row.get("has_mask", True):
            continue
        stem = Path(row["image_path"]).stem
        out[stem] = row["mask_path"]
    return out


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    model, class_to_idx = _load_classifier(args.checkpoint, device)
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    print(f"Loaded checkpoint with {len(class_to_idx)} classes: {sorted(class_to_idx)}")

    df = pd.read_csv(args.index_csv)
    samples = _sample_rows(df, args.classes, args.num_samples, args.seed)
    mask_lookup = _build_mask_lookup(args.seg_index_csv)
    if mask_lookup:
        print(f"Mask lookup loaded: {len(mask_lookup)} pairs")

    rows_out: list[dict] = []
    for i, row in samples.iterrows():
        img_path = row["absolute_path"] if "absolute_path" in row else row["image_path"]
        label = row.get("inferred_label", "")
        stem = Path(img_path).stem
        mask_path = row.get("mask_path", "") or mask_lookup.get(stem)
        result = explain_image(
            model=model,
            image=img_path,
            image_size=args.image_size,
            method=args.method,
            mask=mask_path,
            device=device,
            threshold_mode=args.threshold_mode,
            threshold_value=args.threshold_value,
        )
        pred_label = idx_to_class.get(result.predicted_class, str(result.predicted_class))

        overlay = overlay_heatmap(result.image_rgb, result.heatmap)
        if result.mask_rgb is not None:
            overlay = overlay_mask(overlay, result.mask_rgb)

        out_path = args.output_dir / f"{i:03d}_{stem}_{args.method}.png"
        Image.fromarray(overlay).save(out_path)

        entry = {
            "sample_idx": int(i),
            "image_path": img_path,
            "filename": Path(img_path).name,
            "true_label": label,
            "predicted_label": pred_label,
            "confidence": result.confidence,
            "method": args.method,
            "overlay_path": str(out_path),
            "has_mask": result.mask_rgb is not None,
        }
        if result.metrics is not None:
            entry.update(result.metrics)
        rows_out.append(entry)
        metric_text = ""
        if result.metrics:
            metric_text = f" iou={entry.get('explanation_iou', float('nan')):.3f}"
        print(
            f"  [{i + 1}/{len(samples)}] {Path(img_path).name} -> "
            f"{pred_label} ({result.confidence:.3f}){metric_text}"
        )

    metrics_csv = METRICS_DIR / f"explainability_{args.method}.csv"
    pd.DataFrame(rows_out).to_csv(metrics_csv, index=False)
    summary = {
        "method": args.method,
        "num_samples": len(rows_out),
        "with_mask": int(sum(1 for r in rows_out if r["has_mask"])),
        "metrics_csv": str(metrics_csv),
        "output_dir": str(args.output_dir),
    }
    summary_json = METRICS_DIR / f"explainability_{args.method}_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {metrics_csv}")
    print(f"Wrote {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
