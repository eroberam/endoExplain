"""Sample frames from an endoscopy video at a fixed target FPS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class FrameRecord:
    frame_id: int
    timestamp: float
    output_path: str


def extract_frames(
    video_path: Path,
    output_dir: Path,
    target_fps: float = 5.0,
    max_frames: int | None = None,
) -> list[FrameRecord]:
    """Decode ``video_path`` and write every N-th frame to ``output_dir``.

    Returns the list of saved frames (id, timestamp, path).
    """
    try:
        import cv2  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("opencv is required: `pip install opencv-python-headless`") from e

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    step = max(1, int(round(fps / max(target_fps, 0.1))))

    saved: list[FrameRecord] = []
    out_idx = 0
    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            ts = frame_idx / fps
            out_path = output_dir / f"frame_{out_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame_bgr)
            saved.append(FrameRecord(frame_id=out_idx, timestamp=ts, output_path=str(out_path)))
            out_idx += 1
            if max_frames is not None and out_idx >= max_frames:
                break
        frame_idx += 1
    cap.release()
    return saved
