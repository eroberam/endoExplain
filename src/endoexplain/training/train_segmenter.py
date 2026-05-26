"""Segmentation training loop based on segmentation_models_pytorch."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config.settings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEED,
)
from ..data.segmentation_dataset import SegmentationPairDataset
from ..models import CompositeSegLoss, binary_segmentation_stats, build_segmenter, dice_score, iou_score


@dataclass
class SegTrainConfig:
    index_csv: Path
    output_dir: Path
    architecture: str = "Unet"
    encoder_name: str = "resnet18"
    encoder_weights: str | None = "imagenet"
    image_size: int = DEFAULT_IMAGE_SIZE
    batch_size: int = 4
    num_workers: int = DEFAULT_NUM_WORKERS
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    mixed_precision: bool = True
    seed: int = DEFAULT_SEED
    max_samples: int | None = None
    device: str | None = None
    early_stopping_patience: int = 5
    history: list[dict] = field(default_factory=list)
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    augment_level: str = "light"
    normalize: bool = False
    max_grad_norm: float = 0.0
    loss_bce_weight: float = 1.0
    loss_dice_weight: float = 1.0
    loss_focal_weight: float = 0.0
    loss_tversky_weight: float = 0.0
    loss_tversky_alpha: float = 0.3
    loss_tversky_beta: float = 0.7
    threshold_min: float = 0.30
    threshold_max: float = 0.80
    threshold_steps: int = 11


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _split_df(df: pd.DataFrame, cfg: SegTrainConfig) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    n = len(indices)
    n_test = int(n * cfg.test_fraction)
    n_val = int(n * cfg.val_fraction)
    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]
    return {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }


def _build_loaders(cfg: SegTrainConfig, splits: dict[str, pd.DataFrame]):
    pin = torch.cuda.is_available()
    train_ds = SegmentationPairDataset(
        splits["train"],
        cfg.image_size,
        augment=True,
        augment_level=cfg.augment_level,
        normalize=cfg.normalize,
    )
    val_ds = SegmentationPairDataset(
        splits["val"],
        cfg.image_size,
        augment=False,
        normalize=cfg.normalize,
    )
    test_ds = SegmentationPairDataset(
        splits["test"],
        cfg.image_size,
        augment=False,
        normalize=cfg.normalize,
    )
    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=pin)
    return (
        DataLoader(train_ds, shuffle=True, drop_last=False, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
    )


def _run_epoch(model, loader, device, optimizer, scaler, criterion, max_grad_norm: float = 0.0):
    training = optimizer is not None
    model.train(training)
    use_amp = scaler is not None
    n = 0
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
            else:
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                if use_amp:
                    scaler.scale(loss).backward()
                    if max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
        bs = x.size(0)
        n += bs
        total_loss += float(loss.detach().item()) * bs
        total_dice += dice_score(logits.detach().float(), y) * bs
        total_iou += iou_score(logits.detach().float(), y) * bs
    return total_loss / max(n, 1), total_dice / max(n, 1), total_iou / max(n, 1)


def _threshold_values(cfg: SegTrainConfig) -> list[float]:
    if cfg.threshold_steps <= 1:
        return [float(cfg.threshold_max)]
    return np.linspace(cfg.threshold_min, cfg.threshold_max, cfg.threshold_steps).round(4).tolist()


@torch.no_grad()
def _evaluate_thresholds(model, loader, device, criterion, thresholds: list[float]) -> dict:
    model.eval()
    totals = {
        float(t): {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "n": 0}
        for t in thresholds
    }
    total_loss = 0.0
    total_n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        bs = int(x.size(0))
        total_loss += float(loss.detach().item()) * bs
        total_n += bs
        for threshold in thresholds:
            stats = binary_segmentation_stats(logits.detach().float(), y, threshold=threshold)
            bucket = totals[float(threshold)]
            for key in ("dice", "iou", "precision", "recall"):
                bucket[key] += stats[key] * bs
            bucket["n"] += bs
    rows = []
    for threshold, bucket in totals.items():
        n = max(int(bucket["n"]), 1)
        rows.append(
            {
                "threshold": float(threshold),
                "loss": total_loss / max(total_n, 1),
                "num_samples": int(total_n),
                "dice": bucket["dice"] / n,
                "iou": bucket["iou"] / n,
                "precision": bucket["precision"] / n,
                "recall": bucket["recall"] / n,
            }
        )
    rows.sort(key=lambda r: (r["dice"], r["iou"], r["recall"]), reverse=True)
    return {"best": rows[0], "rows": rows}


def train_segmenter(cfg: SegTrainConfig) -> dict:
    _seed_everything(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)

    df = pd.read_csv(cfg.index_csv)
    df = df[df["has_mask"].astype(bool)] if "has_mask" in df.columns else df
    if cfg.max_samples is not None and len(df) > cfg.max_samples:
        df = df.sample(n=cfg.max_samples, random_state=cfg.seed).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("no usable rows in segmentation index CSV")

    splits = _split_df(df, cfg)
    train_loader, val_loader, test_loader = _build_loaders(cfg, splits)

    model = build_segmenter(
        architecture=cfg.architecture,
        encoder_name=cfg.encoder_name,
        encoder_weights=cfg.encoder_weights,
        num_classes=1,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    criterion = CompositeSegLoss(
        bce_weight=cfg.loss_bce_weight,
        dice_weight=cfg.loss_dice_weight,
        focal_weight=cfg.loss_focal_weight,
        tversky_weight=cfg.loss_tversky_weight,
        tversky_alpha=cfg.loss_tversky_alpha,
        tversky_beta=cfg.loss_tversky_beta,
    ).to(device)
    scheduler = None
    if cfg.scheduler.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, cfg.epochs),
            eta_min=cfg.min_learning_rate,
        )
    use_amp = cfg.mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_val = -1.0
    patience_left = cfg.early_stopping_patience
    best_ckpt_path = cfg.output_dir / "best.pt"
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_dice, tr_iou = _run_epoch(
            model, train_loader, device, optimizer, scaler, criterion, cfg.max_grad_norm
        )
        va_loss, va_dice, va_iou = _run_epoch(model, val_loader, device, None, None, criterion)
        dt = time.time() - t0
        entry = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_dice": tr_dice,
            "train_iou": tr_iou,
            "val_loss": va_loss,
            "val_dice": va_dice,
            "val_iou": va_iou,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": dt,
        }
        history.append(entry)
        print(
            f"[epoch {epoch}/{cfg.epochs}] "
            f"loss={tr_loss:.4f} dice={tr_dice:.4f} iou={tr_iou:.4f} | "
            f"val_loss={va_loss:.4f} val_dice={va_dice:.4f} val_iou={va_iou:.4f} | {dt:.1f}s"
        )

        if va_dice > best_val:
            best_val = va_dice
            patience_left = cfg.early_stopping_patience
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": cfg.architecture,
                    "encoder_name": cfg.encoder_name,
                    "image_size": cfg.image_size,
                    "epoch": epoch,
                    "val_dice": va_dice,
                    "normalize": cfg.normalize,
                },
                best_ckpt_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}.")
                break
        if scheduler is not None:
            scheduler.step()

    best_epoch = ""
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        best_epoch = int(ckpt.get("epoch", -1))

    thresholds = _threshold_values(cfg)
    if 0.5 not in thresholds:
        thresholds = sorted(set(thresholds + [0.5]))
    val_eval = _evaluate_thresholds(model, val_loader, device, criterion, thresholds)
    best_threshold = float(val_eval["best"]["threshold"])
    test_eval = _evaluate_thresholds(model, test_loader, device, criterion, [best_threshold, 0.5])
    test_metrics = next(r for r in test_eval["rows"] if float(r["threshold"]) == best_threshold)
    test_metrics_at_0_5 = next(r for r in test_eval["rows"] if float(r["threshold"]) == 0.5)
    print(
        "Test: "
        f"thr={best_threshold:.2f} loss={test_metrics['loss']:.4f} "
        f"dice={test_metrics['dice']:.4f} iou={test_metrics['iou']:.4f} "
        f"precision={test_metrics['precision']:.4f} recall={test_metrics['recall']:.4f}"
    )

    summary = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "best_epoch": best_epoch,
        "best_val_dice": best_val,
        "best_threshold": best_threshold,
        "best_val_metrics_at_threshold": val_eval["best"],
        "val_threshold_sweep": val_eval["rows"],
        "test_metrics": test_metrics,
        "test_metrics_at_0_5": test_metrics_at_0_5,
        "test_loss": test_metrics["loss"],
        "test_dice": test_metrics["dice"],
        "test_iou": test_metrics["iou"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "history": history,
        "checkpoint": str(best_ckpt_path),
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
    }
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (cfg.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    pd.concat(
        [split_df.assign(split=name) for name, split_df in splits.items()],
        axis=0,
    ).to_csv(cfg.output_dir / "splits.csv", index=False)
    return summary
