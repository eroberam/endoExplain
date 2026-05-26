"""Run a segmentation experiment from a YAML config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from endoexplain.training import SegTrainConfig, train_segmenter  # noqa: E402


def _path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a segmentation experiment from configs/segmentation/*.yaml.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg_raw = yaml.safe_load(args.config.read_text()) or {}
    if "index_csv" not in cfg_raw or "output_dir" not in cfg_raw:
        raise SystemExit("config must define index_csv and output_dir")

    cfg = SegTrainConfig(
        index_csv=_path(cfg_raw["index_csv"]),
        output_dir=_path(cfg_raw["output_dir"]),
        architecture=cfg_raw.get("architecture", "Unet"),
        encoder_name=cfg_raw.get("encoder_name", "resnet18"),
        encoder_weights=None
        if str(cfg_raw.get("encoder_weights", "imagenet")).lower() == "none"
        else cfg_raw.get("encoder_weights", "imagenet"),
        image_size=int(cfg_raw.get("image_size", 256)),
        batch_size=int(cfg_raw.get("batch_size", 4)),
        num_workers=int(cfg_raw.get("num_workers", 4)),
        epochs=int(cfg_raw.get("epochs", 30)),
        learning_rate=float(cfg_raw.get("learning_rate", 1e-4)),
        min_learning_rate=float(cfg_raw.get("min_learning_rate", 1e-6)),
        weight_decay=float(cfg_raw.get("weight_decay", 1e-4)),
        scheduler=cfg_raw.get("scheduler", "cosine"),
        mixed_precision=bool(cfg_raw.get("mixed_precision", True)),
        seed=int(cfg_raw.get("seed", 42)),
        max_samples=cfg_raw.get("max_samples"),
        device=args.device or cfg_raw.get("device"),
        early_stopping_patience=int(cfg_raw.get("early_stopping_patience", 5)),
        val_fraction=float(cfg_raw.get("val_fraction", 0.15)),
        test_fraction=float(cfg_raw.get("test_fraction", 0.15)),
        augment_level=cfg_raw.get("augment_level", "light"),
        normalize=bool(cfg_raw.get("normalize", False)),
        max_grad_norm=float(cfg_raw.get("max_grad_norm", 0.0)),
        loss_bce_weight=float(cfg_raw.get("loss_bce_weight", 1.0)),
        loss_dice_weight=float(cfg_raw.get("loss_dice_weight", 1.0)),
        loss_focal_weight=float(cfg_raw.get("loss_focal_weight", 0.0)),
        loss_tversky_weight=float(cfg_raw.get("loss_tversky_weight", 0.0)),
        loss_tversky_alpha=float(cfg_raw.get("loss_tversky_alpha", 0.3)),
        loss_tversky_beta=float(cfg_raw.get("loss_tversky_beta", 0.7)),
        threshold_min=float(cfg_raw.get("threshold_min", 0.30)),
        threshold_max=float(cfg_raw.get("threshold_max", 0.80)),
        threshold_steps=int(cfg_raw.get("threshold_steps", 11)),
    )

    print("Segmentation experiment")
    print(f"config: {args.config}")
    print(f"output_dir: {cfg.output_dir}")
    print(f"architecture: {cfg.architecture}")
    print(f"encoder: {cfg.encoder_name}")
    if args.dry_run:
        print("[dry_run] no training started")
        return 0

    summary = train_segmenter(cfg)
    summary["run_name"] = cfg_raw.get("run_name", cfg.output_dir.name)
    summary["config_path"] = str(args.config)
    summary["recommended_threshold"] = cfg_raw.get("recommended_threshold", 0.5)
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Checkpoint: {summary['checkpoint']}")
    print(f"Test Dice: {summary['test_dice']:.4f}")
    print(f"Test IoU: {summary['test_iou']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
