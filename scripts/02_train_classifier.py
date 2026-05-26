"""CLI for classifier training.

Use ``--debug`` to do a 1-epoch smoke test on a tiny subset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from endoexplain.config import HYPERKVASIR_INDEX_CSV, CHECKPOINTS_DIR  # noqa: E402
from endoexplain.models import available_classifier_backbones  # noqa: E402
from endoexplain.training import TrainConfig, train_classifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a baseline image classifier on indexed HyperKvasir data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--index_csv", type=Path, default=HYPERKVASIR_INDEX_CSV)
    p.add_argument(
        "--output_dir",
        type=Path,
        default=CHECKPOINTS_DIR / "classifier_resnet18",
    )
    p.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=available_classifier_backbones(),
    )
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--min_learning_rate", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Enable AMP autocast/GradScaler when on CUDA.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--classes",
        type=str,
        nargs="*",
        default=None,
        help="Whitelist of inferred_label values to keep. Omit for all.",
    )
    p.add_argument(
        "--max_samples_per_class",
        type=int,
        default=None,
        help="Cap per-class samples (use to balance large vs small classes).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        choices=[None, "cpu", "cuda"],
        help="Force a device. Defaults to cuda if available, else cpu.",
    )
    p.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Skip ImageNet pretrained weights (faster cold start, worse acc).",
    )
    p.add_argument("--augment_level", choices=["light", "standard", "strong"], default="standard")
    p.add_argument(
        "--debug",
        action="store_true",
        help="Tiny smoke test: 1 epoch, batch=2, image=128, 20 samples/class.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Build config, validate inputs, but do not start training.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.index_csv.exists():
        print(f"ERROR: index CSV not found: {args.index_csv}", file=sys.stderr)
        print("Run: python scripts/01_index_hyperkvasir.py", file=sys.stderr)
        return 2

    if args.debug:
        args.epochs = 1
        args.batch_size = 2
        args.image_size = 128
        args.max_samples_per_class = 20
        print("[debug] forcing epochs=1 batch_size=2 image_size=128 max_samples_per_class=20")

    cfg = TrainConfig(
        index_csv=args.index_csv,
        output_dir=args.output_dir,
        backbone=args.backbone,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        label_smoothing=args.label_smoothing,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        classes=args.classes,
        max_samples_per_class=args.max_samples_per_class,
        device=args.device,
        pretrained=not args.no_pretrained,
        augment_level=args.augment_level,
    )

    print("TrainConfig:")
    for k, v in cfg.__dict__.items():
        if k == "history":
            continue
        print(f"{k}: {v}")

    if args.dry_run:
        print("[dry_run] not starting training.")
        return 0

    summary = train_classifier(cfg)
    print("Best val acc:", summary["best_val_acc"])
    print("Test acc:", summary["test_acc"])
    print("Checkpoint:", summary["checkpoint"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
