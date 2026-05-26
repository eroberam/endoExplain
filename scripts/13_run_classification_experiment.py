"""Run a classifier experiment from a YAML config."""

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

from endoexplain.training import TrainConfig, train_classifier  # noqa: E402


def _path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a classification experiment from configs/classification/*.yaml.",
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

    cfg = TrainConfig(
        index_csv=_path(cfg_raw["index_csv"]),
        output_dir=_path(cfg_raw["output_dir"]),
        backbone=cfg_raw.get("backbone", "resnet18"),
        image_size=int(cfg_raw.get("image_size", 256)),
        batch_size=int(cfg_raw.get("batch_size", 8)),
        num_workers=int(cfg_raw.get("num_workers", 4)),
        epochs=int(cfg_raw.get("epochs", 20)),
        learning_rate=float(cfg_raw.get("learning_rate", 1e-4)),
        min_learning_rate=float(cfg_raw.get("min_learning_rate", 1e-6)),
        weight_decay=float(cfg_raw.get("weight_decay", 1e-4)),
        scheduler=cfg_raw.get("scheduler", "cosine"),
        label_smoothing=float(cfg_raw.get("label_smoothing", 0.0)),
        mixed_precision=bool(cfg_raw.get("mixed_precision", True)),
        seed=int(cfg_raw.get("seed", 42)),
        classes=cfg_raw.get("classes"),
        max_samples_per_class=cfg_raw.get("max_samples_per_class"),
        device=args.device or cfg_raw.get("device"),
        pretrained=bool(cfg_raw.get("pretrained", True)),
        early_stopping_patience=int(cfg_raw.get("early_stopping_patience", 5)),
        augment_level=cfg_raw.get("augment_level", "standard"),
    )

    print("Classification experiment")
    print(f"config: {args.config}")
    print(f"output_dir: {cfg.output_dir}")
    print(f"backbone: {cfg.backbone}")
    print(f"classes: {len(cfg.classes or []) or 'all'}")
    if args.dry_run:
        print("[dry_run] no training started")
        return 0

    summary = train_classifier(cfg)
    summary["run_name"] = cfg_raw.get("run_name", cfg.output_dir.name)
    summary["config_path"] = str(args.config)
    summary["clinical_target"] = cfg_raw.get("clinical_target", {})
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Checkpoint: {summary['checkpoint']}")
    print(f"Test accuracy: {summary['test_acc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
