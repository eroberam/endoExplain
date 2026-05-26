"""Run an event-level benchmark over annotated endoscopy videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from endoexplain.clinical_targets import resolve_target, target_probability  # noqa: E402
from endoexplain.data.transforms import build_classification_transform  # noqa: E402
from endoexplain.evaluation import evaluate_temporal_benchmark, validate_temporal_inputs  # noqa: E402
from endoexplain.models import build_classifier  # noqa: E402
from endoexplain.quality import compute_image_quality  # noqa: E402
from endoexplain.temporal.smoothing import moving_average  # noqa: E402


DEFAULT_VIDEOS_CSV = PROJECT_ROOT / "configs" / "evaluation" / "temporal_benchmark_v01_videos.csv"
DEFAULT_EVENTS_CSV = PROJECT_ROOT / "configs" / "evaluation" / "temporal_benchmark_v01_events.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate temporal event grouping against interval annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--videos_csv", type=Path, default=DEFAULT_VIDEOS_CSV)
    p.add_argument("--events_csv", type=Path, default=DEFAULT_EVENTS_CSV)
    p.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "outputs" / "temporal_benchmark")
    p.add_argument("--predictions_dir", type=Path, default=None)
    p.add_argument("--classifier_ckpt", type=Path, default=None)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--sample_fps", type=float, default=5.0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--positive_classes", nargs="*", default=["polyps", "dyed-lifted-polyps"])
    p.add_argument("--positive_name", default="polyp_family")
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--smoothing_window", type=int, default=5)
    p.add_argument("--max_gap_seconds", type=float, default=1.0)
    p.add_argument("--min_event_duration_seconds", type=float, default=0.2)
    p.add_argument("--limit_videos", type=int, default=None)
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


def _batch(items: list[tuple[int, float, np.ndarray]], size: int) -> Iterable[list[tuple[int, float, np.ndarray]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _resolve_video_path(row: pd.Series) -> Path:
    for col in ("absolute_path", "video_path"):
        if col in row and pd.notna(row[col]) and Path(str(row[col])).exists():
            return Path(str(row[col]))
    rel = Path(str(row["relative_path"]))
    for candidate in (
        PROJECT_ROOT / rel,
        PROJECT_ROOT / "data" / "raw" / "hyper-kvasir" / rel,
    ):
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / "data" / "raw" / "hyper-kvasir" / rel


def _score_video(
    row: pd.Series,
    model: torch.nn.Module,
    class_to_idx: dict[str, int],
    image_size: int,
    sample_fps: float,
    batch_size: int,
    smoothing_window: int,
    target,
    device: torch.device,
) -> pd.DataFrame:
    video_path = _resolve_video_path(row)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(row.get("fps", 25.0) or 25.0)
    stride = max(1, int(round(fps / max(sample_fps, 0.1))))
    transform = build_classification_transform(image_size, train=False)
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    sampled: list[tuple[int, float, np.ndarray]] = []
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            sampled.append((frame_idx, frame_idx / fps, rgb))
        frame_idx += 1
    cap.release()

    rows: list[dict] = []
    with torch.no_grad():
        for batch in _batch(sampled, batch_size):
            tensors = torch.stack(
                [transform(Image.fromarray(rgb)) for _, _, rgb in batch]
            ).to(device)
            probs = torch.softmax(model(tensors), dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            for (frame_id, ts, rgb), prob, pred in zip(batch, probs, preds):
                quality = compute_image_quality(rgb, mask=None)
                score = float(target_probability(prob, target, int(pred)))
                rows.append(
                    {
                        "video_id": row["video_id"],
                        "frame_id": int(frame_id),
                        "timestamp": float(ts),
                        "predicted_label": idx_to_class.get(int(pred), str(int(pred))),
                        "target_probability": score,
                        **quality,
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["target_probability_smoothed"] = moving_average(
            out["target_probability"].to_numpy(), window=smoothing_window
        )
    return out


def _load_or_score_predictions(args: argparse.Namespace, videos: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = None
    class_to_idx = None
    target = None
    if args.classifier_ckpt is not None:
        model, class_to_idx = _load_classifier(args.classifier_ckpt, device)
        target = resolve_target(
            class_to_idx=class_to_idx,
            positive_classes=args.positive_classes,
            target_display_name=args.positive_name,
        )

    for _, row in videos.iterrows():
        video_id = str(row["video_id"])
        pred_path = args.predictions_dir / f"{video_id}.csv" if args.predictions_dir else None
        if pred_path is not None and pred_path.exists():
            out[video_id] = pd.read_csv(pred_path)
            continue
        if model is None or class_to_idx is None or target is None:
            continue
        frames = _score_video(
            row=row,
            model=model,
            class_to_idx=class_to_idx,
            image_size=args.image_size,
            sample_fps=args.sample_fps,
            batch_size=args.batch_size,
            smoothing_window=args.smoothing_window,
            target=target,
            device=device,
        )
        out[video_id] = frames
        args.output_dir.mkdir(parents=True, exist_ok=True)
        frames.to_csv(args.output_dir / f"{video_id}.frames.csv", index=False)
    return out


def main() -> int:
    args = parse_args()
    videos = pd.read_csv(args.videos_csv)
    events = pd.read_csv(args.events_csv)
    if args.limit_videos is not None:
        videos = videos.head(args.limit_videos).copy()
        events = events[events["video_id"].astype(str).isin(videos["video_id"].astype(str))].copy()
    issues = validate_temporal_inputs(videos, events)
    if issues:
        raise SystemExit("; ".join(issues))

    predictions = _load_or_score_predictions(args, videos)
    per_video, summary = evaluate_temporal_benchmark(
        videos=videos,
        truth_events=events,
        frame_predictions=predictions,
        threshold=args.threshold,
        max_gap_seconds=args.max_gap_seconds,
        min_event_duration_seconds=args.min_event_duration_seconds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_video_csv = args.output_dir / "temporal_benchmark_per_video.csv"
    summary_json = args.output_dir / "temporal_benchmark_summary.json"
    per_video.to_csv(per_video_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {per_video_csv} ({len(per_video)} rows)")
    print(f"Wrote {summary_json}")
    if summary["videos_evaluated"] == 0:
        print("No videos evaluated. Provide predictions or a checkpoint and complete annotations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
