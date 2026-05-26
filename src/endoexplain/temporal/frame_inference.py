"""Frame-by-frame inference of a classifier over a folder of frames.

Outputs a per-frame CSV with prediction, confidence, smoothed confidence and
quality indicators.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ..data.transforms import build_classification_transform
from ..quality import compute_image_quality
from .smoothing import moving_average


@dataclass
class FrameInferenceConfig:
    frames_dir: Path
    output_csv: Path
    image_size: int = 256
    target_fps: float = 5.0
    batch_size: int = 8
    device: str | None = None
    suspicious_class_index: int | None = None
    suspicious_class_indices: tuple[int, ...] | None = None
    smoothing_window: int = 5


def _iter_frame_files(folder: Path) -> list[Path]:
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    )


def _batch(iterable: Iterable, n: int):
    batch: list = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def run_frame_inference(
    model: torch.nn.Module,
    class_to_idx: dict[str, int],
    cfg: FrameInferenceConfig,
) -> pd.DataFrame:
    """Run ``model`` over every frame in ``cfg.frames_dir`` and persist a CSV.

    Returns the resulting DataFrame.
    """
    device = torch.device(cfg.device) if cfg.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device).eval()
    transform = build_classification_transform(cfg.image_size, train=False)
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    frame_paths = _iter_frame_files(cfg.frames_dir)
    if not frame_paths:
        raise RuntimeError(f"no frames found under {cfg.frames_dir}")

    rows: list[dict] = []
    with torch.no_grad():
        for batch in _batch(frame_paths, cfg.batch_size):
            pil_images = [Image.open(p).convert("RGB") for p in batch]
            tensors = torch.stack([transform(p) for p in pil_images]).to(device)
            logits = model(tensors)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            for path, pil, prob, pred in zip(batch, pil_images, probs, preds):
                arr = np.asarray(pil)
                quality = compute_image_quality(arr, mask=None)
                pred_label = idx_to_class.get(int(pred), str(int(pred)))
                if cfg.suspicious_class_indices:
                    suspicious_conf = float(prob[list(cfg.suspicious_class_indices)].sum())
                elif cfg.suspicious_class_index is not None:
                    suspicious_conf = float(prob[cfg.suspicious_class_index])
                else:
                    suspicious_conf = float(prob[int(pred)])
                rows.append(
                    {
                        "frame_path": str(path),
                        "frame_id": int(path.stem.split("_")[-1]) if "_" in path.stem else len(rows),
                        "timestamp": len(rows) / max(cfg.target_fps, 0.1),
                        "predicted_class": int(pred),
                        "predicted_label": pred_label,
                        "confidence": suspicious_conf,
                        **{f"prob_{idx_to_class[i]}": float(prob[i]) for i in range(len(prob))},
                        **quality,
                    }
                )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["confidence_smoothed"] = moving_average(
            df["confidence"].to_numpy(), window=cfg.smoothing_window
        )
    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.output_csv, index=False)
    return df
