"""Extract frames from a single endoscopy video at a target FPS."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from endoexplain.data import extract_frames  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample frames from a video file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video_path", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--max_frames", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.video_path.exists():
        print(f"ERROR: video not found: {args.video_path}", file=sys.stderr)
        return 2

    print(f"Video: {args.video_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Target FPS: {args.fps}")

    frames = extract_frames(
        video_path=args.video_path,
        output_dir=args.output_dir,
        target_fps=args.fps,
        max_frames=args.max_frames,
    )
    manifest = args.output_dir / "frames.json"
    manifest.write_text(json.dumps([asdict(f) for f in frames], indent=2))
    print(f"Extracted {len(frames)} frames")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
