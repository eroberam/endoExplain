"""Export segmentation predictions and per-image metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from endoexplain.config import METRICS_DIR, PROCESSED_DIR  # noqa: E402
from endoexplain.models import build_segmenter  # noqa: E402


DEFAULT_INDEX = PROCESSED_DIR / "hyperkvasir_segmented_index.csv"


class _SegRows(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int, normalize: bool = False) -> None:
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        mask = Image.open(row["mask_path"]).convert("L").resize(
            (self.image_size, self.image_size), Image.NEAREST
        )
        x = torch.from_numpy(np.asarray(image).copy().transpose(2, 0, 1)).float() / 255.0
        if self.normalize:
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype).view(3, 1, 1)
            x = (x - mean) / std
        y = (torch.from_numpy(np.asarray(mask).copy()).float() > 127).float().unsqueeze(0)
        return x, y, idx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export segmentation probabilities/masks and metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--index_csv", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--output_csv", type=Path, default=None)
    p.add_argument("--split_csv", type=Path, default=None)
    p.add_argument("--split", choices=["train", "val", "test"], default=None)
    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_masks", action="store_true")
    p.add_argument("--mask_dir", type=Path, default=None)
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    return p.parse_args()


def _load_segmenter(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_segmenter(
        architecture=ckpt.get("architecture", "Unet"),
        encoder_name=ckpt.get("encoder_name", "resnet18"),
        encoder_weights=None,
        num_classes=1,
    )
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


def _metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred_b = pred.astype(bool)
    target_b = target.astype(bool)
    tp = float((pred_b & target_b).sum())
    fp = float((pred_b & ~target_b).sum())
    fn = float((~pred_b & target_b).sum())
    inter = tp
    pred_sum = float(pred_b.sum())
    target_sum = float(target_b.sum())
    union = float((pred_b | target_b).sum())
    return {
        "dice": (2.0 * inter + 1.0) / (pred_sum + target_sum + 1.0),
        "iou": (inter + 1.0) / (union + 1.0),
        "precision": (tp + 1.0) / (tp + fp + 1.0),
        "recall": (tp + 1.0) / (tp + fn + 1.0),
        "pred_area_ratio": pred_sum / max(float(pred_b.size), 1.0),
        "true_area_ratio": target_sum / max(float(target_b.size), 1.0),
    }


def _input_rows(index_csv: Path, split_csv: Path | None, split: str | None) -> pd.DataFrame:
    df = pd.read_csv(split_csv if split_csv and split_csv.exists() else index_csv)
    if split and "split" in df.columns:
        df = df[df["split"] == split].copy()
    if "has_mask" in df.columns:
        df = df[df["has_mask"].astype(bool)].copy()
    return df.reset_index(drop=True)


def main() -> int:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ckpt = _load_segmenter(args.checkpoint, device)
    image_size = int(args.image_size or ckpt.get("image_size", 384))
    normalize = bool(ckpt.get("normalize", False))
    df = _input_rows(args.index_csv, args.split_csv, args.split)
    if df.empty:
        raise SystemExit("no segmentation rows with masks")

    loader = DataLoader(
        _SegRows(df, image_size, normalize=normalize),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    out_csv = args.output_csv or (
        METRICS_DIR
        / "segmentation_predictions"
        / args.checkpoint.parent.name
        / f"predictions_thr{args.threshold:.2f}.csv"
    )
    mask_dir = args.mask_dir or out_csv.parent / "masks"
    if args.save_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with torch.no_grad():
        for xb, yb, indices in loader:
            logits = model(xb.to(device, non_blocking=True))
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
            targets = yb.numpy()[:, 0]
            for j, row_idx in enumerate(indices.numpy().tolist()):
                src = df.iloc[int(row_idx)]
                pred = probs[j] >= args.threshold
                rec = {
                    "sample_id": src.get("pair_id", Path(src["image_path"]).stem),
                    "image_path": src["image_path"],
                    "mask_path": src["mask_path"],
                    "checkpoint": str(args.checkpoint),
                    "architecture": ckpt.get("architecture", ""),
                    "encoder_name": ckpt.get("encoder_name", ""),
                    "image_size": image_size,
                    "normalize": normalize,
                    "threshold": args.threshold,
                    "mean_probability": float(probs[j].mean()),
                    "max_probability": float(probs[j].max()),
                }
                rec.update(_metrics(pred, targets[j] > 0.5))
                if args.save_masks:
                    pred_img = Image.fromarray((pred.astype(np.uint8) * 255))
                    pred_path = mask_dir / f"{Path(src['image_path']).stem}_pred.png"
                    pred_img.save(pred_path)
                    rec["prediction_mask_path"] = str(pred_path)
                rows.append(rec)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    summary = {
        "checkpoint": str(args.checkpoint),
        "rows": len(rows),
        "threshold": args.threshold,
        "output_csv": str(out_csv),
        "mean_metrics": pd.DataFrame(rows)[["dice", "iou", "precision", "recall"]]
        .mean()
        .to_dict(),
    }
    out_csv.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
