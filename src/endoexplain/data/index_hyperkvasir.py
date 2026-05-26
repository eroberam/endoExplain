"""Recursively index a local HyperKvasir dataset folder into a CSV.

The dataset layout on disk is not assumed to be fixed: labels are inferred
from the parent folder name, and the inner directory chain (e.g.
``labeled-images/lower-gi-tract/polyps``) is preserved so downstream code
can group records by anatomical region or by polyp/normal class.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Iterable, Iterator

from ..config.settings import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS


@dataclass
class FileRecord:
    absolute_path: str
    relative_path: str
    filename: str
    extension: str
    file_type: str  # "image" | "video" | "other"
    inferred_label: str
    parent_folder: str
    top_category: str  # e.g. "labeled-images", "labeled-videos"
    subcategory: str  # second-level folder, if present
    width: int = -1
    height: int = -1
    fps: float = -1.0
    frame_count: int = -1
    duration_seconds: float = -1.0
    size_bytes: int = 0


@dataclass
class _Probes:
    """Lazy-loaded readers so we don't pay the import cost when not needed."""

    _pil_image: object = field(default=None, repr=False)
    _cv2: object = field(default=None, repr=False)

    def pil(self):
        if self._pil_image is None:
            from PIL import Image  # local import

            self._pil_image = Image
        return self._pil_image

    def cv(self):
        if self._cv2 is None:
            try:
                import cv2  # local import
            except ImportError:
                cv2 = None
            self._cv2 = cv2
        return self._cv2


def _classify_extension(ext: str) -> str:
    ext = ext.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return "other"


def _probe_image(path: Path, probes: _Probes) -> tuple[int, int]:
    try:
        Image = probes.pil()
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return -1, -1


def _probe_video(path: Path, probes: _Probes) -> tuple[int, int, float, int, float]:
    """Return (width, height, fps, frame_count, duration_seconds)."""
    cv2 = probes.cv()
    if cv2 is None:
        return -1, -1, -1.0, -1, -1.0
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return -1, -1, -1.0, -1, -1.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration = frames / fps if fps and fps > 0 else -1.0
        return width, height, fps, frames, duration
    except Exception:
        return -1, -1, -1.0, -1, -1.0


def _iter_files(root: Path) -> Iterator[Path]:
    yield from (p for p in root.rglob("*") if p.is_file())


def _build_record(path: Path, root: Path, probes: _Probes, probe_media: bool) -> FileRecord:
    rel = path.relative_to(root)
    parts = rel.parts
    parent_folder = parts[-2] if len(parts) >= 2 else ""
    top_category = parts[0] if len(parts) >= 1 else ""
    subcategory = parts[1] if len(parts) >= 3 else ""
    ext = path.suffix.lower()
    ftype = _classify_extension(ext)

    rec = FileRecord(
        absolute_path=str(path.resolve()),
        relative_path=str(rel).replace("\\", "/"),
        filename=path.name,
        extension=ext,
        file_type=ftype,
        inferred_label=parent_folder,
        parent_folder=parent_folder,
        top_category=top_category,
        subcategory=subcategory,
        size_bytes=path.stat().st_size if path.exists() else 0,
    )

    if probe_media:
        if ftype == "image":
            rec.width, rec.height = _probe_image(path, probes)
        elif ftype == "video":
            (
                rec.width,
                rec.height,
                rec.fps,
                rec.frame_count,
                rec.duration_seconds,
            ) = _probe_video(path, probes)

    return rec


def index_hyperkvasir(
    root: Path,
    output_csv: Path,
    probe_media: bool = True,
    limit: int | None = None,
    progress: bool = True,
) -> int:
    """Walk ``root`` and write a CSV index. Returns the number of rows written.

    Parameters
    ----------
    root : Path
        Top-level HyperKvasir folder (or its symlink).
    output_csv : Path
        Destination CSV. Parent folders are created if missing.
    probe_media : bool
        If True, open each image/video to record dimensions/duration.
    limit : int | None
        Optional cap on number of files processed (useful for --debug).
    progress : bool
        Show a progress bar if ``rich`` or ``tqdm`` is installed.
    """
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    probes = _Probes()

    files: Iterable[Path] = _iter_files(root)
    if limit is not None:
        files = (p for i, p in enumerate(files) if i < limit)

    iterator = _maybe_progress(files, progress=progress, description="Indexing HyperKvasir")

    field_names = [f.name for f in fields(FileRecord)]
    rows = 0
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_names)
        writer.writeheader()
        for path in iterator:
            rec = _build_record(path, root, probes, probe_media=probe_media)
            writer.writerow(asdict(rec))
            rows += 1

    return rows


def _maybe_progress(iterable: Iterable, progress: bool, description: str) -> Iterable:
    if not progress:
        return iterable
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, desc=description, unit="file")
    except ImportError:
        return iterable
