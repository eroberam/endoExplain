"""Temporal review of a video: frame inference + smoothing + event grouping."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from endoexplain.config import METRICS_DIR  # noqa: E402
from endoexplain.clinical_targets import resolve_target  # noqa: E402
from endoexplain.models import build_classifier  # noqa: E402
from endoexplain.temporal import EventConfig, event_metrics, group_events  # noqa: E402
from endoexplain.temporal.frame_inference import (  # noqa: E402
    FrameInferenceConfig,
    run_frame_inference,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run frame-level inference + temporal aggregation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--frames_dir", type=Path, required=True)
    p.add_argument(
        "--output_csv",
        type=Path,
        default=METRICS_DIR / "video_frame_predictions.csv",
    )
    p.add_argument("--events_csv", type=Path, default=METRICS_DIR / "video_events.csv")
    p.add_argument("--summary_json", type=Path, default=METRICS_DIR / "video_temporal_summary.json")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--target_fps", type=float, default=5.0)
    p.add_argument("--smoothing_window", type=int, default=5)
    p.add_argument(
        "--suspicious_class",
        type=str,
        default=None,
        help="Name of the class whose probability is treated as 'suspicious confidence'.",
    )
    p.add_argument("--confidence_threshold", type=float, default=0.5)
    p.add_argument("--max_gap_seconds", type=float, default=1.0)
    p.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    return p.parse_args()


def _load_classifier(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_to_idx: dict[str, int] = ckpt["class_to_idx"]
    backbone: str = ckpt["backbone"]
    model = build_classifier(backbone, num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), class_to_idx


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    if not args.frames_dir.is_dir():
        print(f"ERROR: frames directory not found: {args.frames_dir}", file=sys.stderr)
        return 2

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model, class_to_idx = _load_classifier(args.checkpoint, device)
    target = resolve_target(class_to_idx, suspicious_class=args.suspicious_class)
    suspicious_idx = target.class_indices[0] if len(target.class_indices) == 1 else None

    cfg = FrameInferenceConfig(
        frames_dir=args.frames_dir,
        output_csv=args.output_csv,
        image_size=args.image_size,
        target_fps=args.target_fps,
        batch_size=args.batch_size,
        device=str(device),
        suspicious_class_index=suspicious_idx,
        suspicious_class_indices=target.class_indices if len(target.class_indices) > 1 else None,
        smoothing_window=args.smoothing_window,
    )
    df = run_frame_inference(model=model, class_to_idx=class_to_idx, cfg=cfg)
    print(f"Wrote frame predictions: {args.output_csv} ({len(df)} rows)")

    events = group_events(
        df,
        EventConfig(
            confidence_threshold=args.confidence_threshold,
            max_gap_seconds=args.max_gap_seconds,
        ),
    )
    events.to_csv(args.events_csv, index=False)
    print(f"Wrote events: {args.events_csv} ({len(events)} events)")

    summary = event_metrics(df)
    summary.update(
        {
            "num_events": int(len(events)),
            "frame_csv": str(args.output_csv),
            "events_csv": str(args.events_csv),
        }
    )
    args.summary_json.write_text(json.dumps(summary, indent=2))
    print(f"Wrote temporal summary: {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
