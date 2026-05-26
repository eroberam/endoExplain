"""CLI for the U-Net segmentation trainer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from endoexplain.config import CHECKPOINTS_DIR, PROCESSED_DIR  # noqa: E402
from endoexplain.models import (  # noqa: E402
    available_segmentation_architectures,
    available_segmentation_encoders,
)
from endoexplain.training import SegTrainConfig, train_segmenter  # noqa: E402


DEFAULT_INDEX = PROCESSED_DIR / "hyperkvasir_segmented_index.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a polyp segmenter (U-Net / smp).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--index_csv", type=Path, default=DEFAULT_INDEX)
    p.add_argument(
        "--output_dir", type=Path, default=CHECKPOINTS_DIR / "unet_resnet18"
    )
    p.add_argument(
        "--architecture",
        choices=available_segmentation_architectures(),
        default="Unet",
    )
    p.add_argument(
        "--encoder_name",
        choices=available_segmentation_encoders(),
        default="resnet18",
    )
    p.add_argument(
        "--encoder_weights",
        default="imagenet",
        help="Pretrained weights tag. Pass 'none' for random init.",
    )
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--min_learning_rate", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--augment_level", choices=["light", "strong"], default="light")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--max_grad_norm", type=float, default=0.0)
    p.add_argument("--loss_bce_weight", type=float, default=1.0)
    p.add_argument("--loss_dice_weight", type=float, default=1.0)
    p.add_argument("--loss_focal_weight", type=float, default=0.0)
    p.add_argument("--loss_tversky_weight", type=float, default=0.0)
    p.add_argument("--loss_tversky_alpha", type=float, default=0.3)
    p.add_argument("--loss_tversky_beta", type=float, default=0.7)
    p.add_argument("--threshold_min", type=float, default=0.30)
    p.add_argument("--threshold_max", type=float, default=0.80)
    p.add_argument("--threshold_steps", type=int, default=11)
    p.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    p.add_argument(
        "--debug",
        action="store_true",
        help="1 epoch, batch=2, image=128, 20 samples.",
    )
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.index_csv.exists():
        print(f"ERROR: index CSV not found: {args.index_csv}", file=sys.stderr)
        print("Run: python scripts/01b_index_segmentation.py", file=sys.stderr)
        return 2

    if args.debug:
        args.epochs = 1
        args.batch_size = 2
        args.image_size = 128
        args.max_samples = 20
        print("[debug] forcing epochs=1 batch_size=2 image_size=128 max_samples=20")

    encoder_weights = None if args.encoder_weights == "none" else args.encoder_weights
    cfg = SegTrainConfig(
        index_csv=args.index_csv,
        output_dir=args.output_dir,
        architecture=args.architecture,
        encoder_name=args.encoder_name,
        encoder_weights=encoder_weights,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        max_samples=args.max_samples,
        device=args.device,
        augment_level=args.augment_level,
        normalize=args.normalize,
        max_grad_norm=args.max_grad_norm,
        loss_bce_weight=args.loss_bce_weight,
        loss_dice_weight=args.loss_dice_weight,
        loss_focal_weight=args.loss_focal_weight,
        loss_tversky_weight=args.loss_tversky_weight,
        loss_tversky_alpha=args.loss_tversky_alpha,
        loss_tversky_beta=args.loss_tversky_beta,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_steps=args.threshold_steps,
    )

    print("SegTrainConfig:")
    for k, v in cfg.__dict__.items():
        if k == "history":
            continue
        print(f"{k}: {v}")

    if args.dry_run:
        print("[dry_run] not starting training.")
        return 0

    summary = train_segmenter(cfg)
    print("Best val Dice:", summary["best_val_dice"])
    print("Test Dice:", summary["test_dice"])
    print("Test IoU:", summary["test_iou"])
    print("Checkpoint:", summary["checkpoint"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
