"""Index a polyp-segmentation dataset (images/ + masks/ side-by-side).

Compatible with both the standalone Kvasir-SEG layout (``data/raw/kvasir-seg``)
and the ``segmented-images`` subset of HyperKvasir, which share the same
structure:

    <root>/images/<name>.jpg
    <root>/masks/<name>.jpg
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from ..config.settings import IMAGE_EXTENSIONS


@dataclass
class SegRecord:
    pair_id: str
    image_path: str
    mask_path: str
    filename: str
    extension: str
    width: int = -1
    height: int = -1
    mask_pixel_count: int = -1
    mask_area_ratio: float = -1.0
    has_mask: bool = True


def _list_images(folder: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        out[p.stem] = p
    return out


def _probe_image(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return -1, -1


def _probe_mask(path: Path) -> tuple[int, float]:
    """Return (positive_pixel_count, area_ratio) for a binary-ish mask."""
    try:
        from PIL import Image
        import numpy as np

        with Image.open(path) as im:
            arr = np.asarray(im.convert("L"))
        positive = int((arr > 127).sum())
        total = int(arr.size)
        return positive, (positive / total) if total else -1.0
    except Exception:
        return -1, -1.0


def index_segmentation(
    root: Path,
    output_csv: Path,
    probe_masks: bool = True,
    progress: bool = True,
) -> int:
    """Walk ``<root>/images`` and ``<root>/masks``, write a paired CSV."""
    root = root.resolve()
    images_dir = root / "images"
    masks_dir = root / "masks"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"missing folder: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"missing folder: {masks_dir}")

    images = _list_images(images_dir)
    masks = _list_images(masks_dir)
    paired = sorted(set(images) & set(masks))
    only_image = sorted(set(images) - set(masks))

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    iterator: list[str] = paired + only_image
    iterator_pretty = _maybe_progress(iterator, progress=progress, description="Indexing segmentation")

    field_names = [f.name for f in fields(SegRecord)]
    rows = 0
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_names)
        writer.writeheader()
        for stem in iterator_pretty:
            img = images[stem]
            mask = masks.get(stem)
            w, h = _probe_image(img) if probe_masks else (-1, -1)
            pixels, ratio = _probe_mask(mask) if (probe_masks and mask is not None) else (-1, -1.0)
            rec = SegRecord(
                pair_id=stem,
                image_path=str(img),
                mask_path=str(mask) if mask is not None else "",
                filename=img.name,
                extension=img.suffix.lower(),
                width=w,
                height=h,
                mask_pixel_count=pixels,
                mask_area_ratio=ratio,
                has_mask=mask is not None,
            )
            writer.writerow(asdict(rec))
            rows += 1
    return rows


def _maybe_progress(iterable, progress: bool, description: str):
    if not progress:
        return iterable
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, desc=description, unit="pair")
    except ImportError:
        return iterable
