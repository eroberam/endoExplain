"""Render an endoscopy video with AI annotations burned onto every frame.

Pipeline:
  1) Decode the source video.
  2) Pass 1: run the classifier to build per-frame confidence + events.
  3) Pass 2: re-decode, run classifier + (optional) segmenter + Grad-CAM++,
     compose heatmap blend + mask contour + HUD + mini-timeline, write MP4.

Output:
  outputs/videos/<name>.mp4
  outputs/videos/<name>.summary.json
  outputs/videos/<name>.frames.csv
  outputs/videos/<name>.events.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from endoexplain.config import OUTPUTS_DIR  # noqa: E402
from endoexplain.explainability import available_methods  # noqa: E402
from endoexplain.video import RenderConfig, render_overlay_video  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render an annotated demo video (mask + Grad-CAM++ + HUD + timeline).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video_path", type=Path, required=True)
    p.add_argument("--classifier_ckpt", type=Path, required=True)
    p.add_argument("--segmenter_ckpt", type=Path, default=None,
                   help="Optional U-Net checkpoint to draw a polyp mask contour.")
    p.add_argument("--output_path", type=Path, default=None,
                   help="Defaults to outputs/videos/<input_stem>_overlay.mp4")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--segmenter_image_size", type=int, default=384)
    p.add_argument("--target_fps", type=float, default=None,
                   help="Output FPS. If omitted, matches source.")
    p.add_argument("--canvas_width", type=int, default=1280)
    p.add_argument("--canvas_height", type=int, default=720)
    p.add_argument(
        "--render_profile",
        choices=("clinical_review", "public_demo", "media_preview", "legacy"),
        default="clinical_review",
        help="Visual layout profile for the burned-in review interface.",
    )
    p.add_argument(
        "--source-crop-left",
        dest="source_crop_left",
        type=float,
        default=None,
        help="Fraction of source frame cropped from the left before display. "
             "If omitted, the selected render profile chooses a default.",
    )
    p.add_argument(
        "--source-crop-top",
        dest="source_crop_top",
        type=float,
        default=None,
        help="Fraction of source frame cropped from the top before display.",
    )
    p.add_argument(
        "--source-crop-right",
        dest="source_crop_right",
        type=float,
        default=None,
        help="Fraction of source frame cropped from the right before display.",
    )
    p.add_argument(
        "--source-crop-bottom",
        dest="source_crop_bottom",
        type=float,
        default=None,
        help="Fraction of source frame cropped from the bottom before display.",
    )
    p.add_argument("--method", choices=available_methods(), default="gradcam++")
    p.add_argument("--no_heatmap", action="store_true")
    p.add_argument("--no_mask", action="store_true")
    p.add_argument("--suspicious_class", type=str, default=None,
                   help="Name of the class whose probability is shown as 'confidence'. "
                        "If omitted, uses the predicted-class probability.")
    p.add_argument("--confidence_threshold", type=float, default=0.5)
    p.add_argument("--max_gap_seconds", type=float, default=1.0)
    p.add_argument("--smoothing_window", type=int, default=5)
    p.add_argument("--heatmap_alpha", type=float, default=0.28)
    p.add_argument("--heatmap_top_percent", type=float, default=0.18)
    p.add_argument("--heatmap_colormap", type=str, default="magma")
    p.add_argument("--mask_threshold", type=float, default=0.5)
    p.add_argument("--mask_min_confidence", type=float, default=0.0)
    p.add_argument(
        "--mask_gate_source",
        choices=("target", "smooth", "both"),
        default="target",
        help="Confidence signal used to permit mask drawing.",
    )
    p.add_argument("--mask-only-in-event", dest="mask_only_in_event", action="store_true")
    p.add_argument("--mask-max-fragments", dest="mask_max_fragments", type=int, default=8)
    p.add_argument("--mask-max-area-ratio", dest="mask_max_area_ratio", type=float, default=0.25)
    p.add_argument(
        "--mask-min-component-area-ratio",
        dest="mask_min_component_area_ratio",
        type=float,
        default=0.002,
    )
    p.add_argument(
        "--mask-keep-largest-components",
        dest="mask_keep_largest_components",
        type=int,
        default=2,
    )
    p.add_argument(
        "--mask-max-border-touch-ratio",
        dest="mask_max_border_touch_ratio",
        type=float,
        default=0.75,
    )
    p.add_argument("--mask-fill-alpha", dest="mask_fill_alpha", type=float, default=0.50)
    p.add_argument("--mask-smoothing-kernel", dest="mask_smoothing_kernel", type=int, default=5)
    p.add_argument("--mask-min-solidity", dest="mask_min_solidity", type=float, default=0.25)
    p.add_argument("--mask-max-aspect-ratio", dest="mask_max_aspect_ratio", type=float, default=6.0)
    p.add_argument("--mask-xai-gate", dest="mask_xai_gate", action="store_true")
    p.add_argument("--mask-xai-top-percent", dest="mask_xai_top_percent", type=float, default=0.22)
    p.add_argument(
        "--mask-min-xai-active-inside",
        dest="mask_min_xai_active_inside",
        type=float,
        default=0.08,
    )
    p.add_argument("--mask-min-xai-iou", dest="mask_min_xai_iou", type=float, default=0.0)
    p.add_argument("--mask-xai-dilation-px", dest="mask_xai_dilation_px", type=int, default=8)
    p.add_argument("--mask-min-temporal-iou", dest="mask_min_temporal_iou", type=float, default=0.0)
    p.add_argument(
        "--mask-max-temporal-centroid-shift",
        dest="mask_max_temporal_centroid_shift",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--mask-max-temporal-area-change",
        dest="mask_max_temporal_area_change",
        type=float,
        default=100.0,
    )
    p.add_argument("--mask-stability-lookback", dest="mask_stability_lookback", type=int, default=4)
    p.add_argument("--mask-fade-frames", dest="mask_fade_frames", type=int, default=0)
    p.add_argument("--max_frames", type=int, default=None,
                   help="Cap on number of frames rendered (useful for fast demos).")
    p.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.video_path.exists():
        print(f"ERROR: video not found: {args.video_path}", file=sys.stderr)
        return 2
    if not args.classifier_ckpt.exists():
        print(f"ERROR: classifier checkpoint not found: {args.classifier_ckpt}", file=sys.stderr)
        return 2
    if args.segmenter_ckpt is not None and not args.segmenter_ckpt.exists():
        print(f"ERROR: segmenter checkpoint not found: {args.segmenter_ckpt}", file=sys.stderr)
        return 2

    output_path = args.output_path or (
        OUTPUTS_DIR / "videos" / f"{args.video_path.stem}_overlay.mp4"
    )
    cfg = RenderConfig(
        video_path=args.video_path,
        output_path=output_path,
        classifier_ckpt=args.classifier_ckpt,
        segmenter_ckpt=args.segmenter_ckpt,
        image_size=args.image_size,
        segmenter_image_size=args.segmenter_image_size,
        target_fps=args.target_fps,
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
        render_profile=args.render_profile,
        source_crop_left=args.source_crop_left,
        source_crop_top=args.source_crop_top,
        source_crop_right=args.source_crop_right,
        source_crop_bottom=args.source_crop_bottom,
        xai_method=args.method,
        show_heatmap=not args.no_heatmap,
        show_mask=not args.no_mask,
        suspicious_class=args.suspicious_class,
        confidence_threshold=args.confidence_threshold,
        smoothing_window=args.smoothing_window,
        max_gap_seconds=args.max_gap_seconds,
        heatmap_alpha=args.heatmap_alpha,
        heatmap_top_percent=args.heatmap_top_percent,
        heatmap_colormap=args.heatmap_colormap,
        mask_threshold=args.mask_threshold,
        mask_min_confidence=args.mask_min_confidence,
        mask_gate_source=args.mask_gate_source,
        mask_only_in_event=args.mask_only_in_event,
        mask_max_fragments=args.mask_max_fragments,
        mask_max_area_ratio=args.mask_max_area_ratio,
        mask_min_component_area_ratio=args.mask_min_component_area_ratio,
        mask_keep_largest_components=args.mask_keep_largest_components,
        mask_max_border_touch_ratio=args.mask_max_border_touch_ratio,
        mask_fill_alpha=args.mask_fill_alpha,
        mask_smoothing_kernel=args.mask_smoothing_kernel,
        mask_min_solidity=args.mask_min_solidity,
        mask_max_aspect_ratio=args.mask_max_aspect_ratio,
        mask_xai_gate=args.mask_xai_gate,
        mask_xai_top_percent=args.mask_xai_top_percent,
        mask_min_xai_active_inside=args.mask_min_xai_active_inside,
        mask_min_xai_iou=args.mask_min_xai_iou,
        mask_xai_dilation_px=args.mask_xai_dilation_px,
        mask_min_temporal_iou=args.mask_min_temporal_iou,
        mask_max_temporal_centroid_shift=args.mask_max_temporal_centroid_shift,
        mask_max_temporal_area_change=args.mask_max_temporal_area_change,
        mask_stability_lookback=args.mask_stability_lookback,
        mask_fade_frames=args.mask_fade_frames,
        max_frames=args.max_frames,
        device=args.device,
    )

    print("RenderConfig:")
    for k, v in cfg.__dict__.items():
        if k == "history":
            continue
        print(f"{k}: {v}")

    summary = render_overlay_video(cfg)
    print()
    print("Done.")
    print(f"rendered frames: {summary['rendered_frames']}")
    print(f"events: {summary['num_events']}")
    print(f"pass1 seconds: {summary['pass1_seconds']:.1f}")
    print(f"pass2 seconds: {summary['pass2_seconds']:.1f}")
    print(f"output video: {summary['output_path']}")
    print(f"side JSON: {Path(summary['output_path']).with_suffix('.summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
