"""Classification training loop."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config.settings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEED,
)
from ..data.dataset import IndexedImageClassificationDataset
from ..data.split_dataset import stratified_split
from ..data.transforms import build_classification_transform
from ..models import build_classifier


@dataclass
class TrainConfig:
    index_csv: Path
    output_dir: Path
    backbone: str = "resnet18"
    image_size: int = DEFAULT_IMAGE_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    label_smoothing: float = 0.0
    mixed_precision: bool = True
    seed: int = DEFAULT_SEED
    classes: Sequence[str] | None = None
    max_samples_per_class: int | None = None
    device: str | None = None
    pretrained: bool = True
    early_stopping_patience: int = 5
    augment_level: str = "standard"
    history: list[dict] = field(default_factory=list)


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


def _filter_classes(df: pd.DataFrame, classes: Sequence[str] | None) -> pd.DataFrame:
    df = df[df["file_type"] == "image"].copy()
    df = df[df["inferred_label"].astype(str).str.len() > 0]
    if classes:
        df = df[df["inferred_label"].isin(list(classes))].copy()
    return df


def _cap_per_class(df: pd.DataFrame, cap: int | None, seed: int) -> pd.DataFrame:
    if cap is None:
        return df
    parts = []
    rng = np.random.default_rng(seed)
    for label, grp in df.groupby("inferred_label"):
        if len(grp) > cap:
            idx = rng.choice(grp.index.to_numpy(), size=cap, replace=False)
            parts.append(grp.loc[idx])
        else:
            parts.append(grp)
    return pd.concat(parts, axis=0).reset_index(drop=True)


def _build_loaders(cfg: TrainConfig, df: pd.DataFrame, class_to_idx: dict[str, int]):
    train_tf = build_classification_transform(
        cfg.image_size,
        train=True,
        augment_level=cfg.augment_level,
    )
    eval_tf = build_classification_transform(cfg.image_size, train=False)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_ds = IndexedImageClassificationDataset(train_df, class_to_idx, train_tf)
    val_ds = IndexedImageClassificationDataset(val_df, class_to_idx, eval_tf)
    test_ds = IndexedImageClassificationDataset(test_df, class_to_idx, eval_tf)

    pin = torch.cuda.is_available()
    common = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=pin,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    criterion: nn.Module,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    use_amp = scaler is not None

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
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += float(loss.detach().item()) * x.size(0)
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_seen += int(x.size(0))

    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


def train_classifier(cfg: TrainConfig) -> dict:
    """Train a classifier and write checkpoint + metrics under ``cfg.output_dir``."""
    _seed_everything(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)

    df = pd.read_csv(cfg.index_csv)
    df = _filter_classes(df, cfg.classes)
    df = _cap_per_class(df, cfg.max_samples_per_class, cfg.seed)
    if df.empty:
        raise RuntimeError(
            "No usable rows after filtering. Check --classes and the index CSV."
        )

    class_to_idx = IndexedImageClassificationDataset.build_class_map(
        df["inferred_label"].tolist()
    )
    num_classes = len(class_to_idx)

    df = stratified_split(df, label_col="inferred_label", seed=cfg.seed)
    train_loader, val_loader, test_loader = _build_loaders(cfg, df, class_to_idx)

    model = build_classifier(cfg.backbone, num_classes=num_classes, pretrained=cfg.pretrained).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.label_smoothing))
    scheduler = None
    if cfg.scheduler.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, cfg.epochs),
            eta_min=cfg.min_learning_rate,
        )
    use_amp = cfg.mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_val_acc = -1.0
    patience_left = cfg.early_stopping_patience
    best_ckpt_path = cfg.output_dir / "best.pt"

    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = _run_epoch(
            model, train_loader, device, optimizer, scaler, criterion
        )
        val_loss, val_acc = _run_epoch(model, val_loader, device, None, None, criterion)
        dt = time.time() - t0
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": dt,
        }
        history.append(entry)
        print(
            f"[epoch {epoch}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {dt:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_left = cfg.early_stopping_patience
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "backbone": cfg.backbone,
                    "epoch": epoch,
                    "val_acc": val_acc,
                },
                best_ckpt_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} (no val improvement).")
                break
        if scheduler is not None:
            scheduler.step()

    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

    test_loss, test_acc = _run_epoch(model, test_loader, device, None, None, criterion)
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")

    summary = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "num_classes": num_classes,
        "class_to_idx": class_to_idx,
        "best_val_acc": best_val_acc,
        "best_epoch": int(ckpt.get("epoch", -1)) if best_ckpt_path.exists() else "",
        "test_loss": test_loss,
        "test_acc": test_acc,
        "history": history,
        "checkpoint": str(best_ckpt_path),
        "n_train": int((df["split"] == "train").sum()),
        "n_val": int((df["split"] == "val").sum()),
        "n_test": int((df["split"] == "test").sum()),
    }
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (cfg.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    df.to_csv(cfg.output_dir / "splits.csv", index=False)
    return summary
