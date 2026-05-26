"""Index a polyp-segmentation dataset (images/ + masks/ layout) into a CSV.

Works with HyperKvasir's ``segmented-images`` folder out of the box, and
also with a standalone Kvasir-SEG download placed at ``data/raw/kvasir-seg``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from endoexplain.config import (  # noqa: E402
    HYPERKVASIR_SEGMENTED_DIR,
    KVASIRSEG_DIR,
    KVASIRSEG_INDEX_CSV,
    PROCESSED_DIR,
)
from endoexplain.data import index_segmentation  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Index a polyp-segmentation dataset (images/ + masks/).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source",
        choices=("hyperkvasir-segmented", "kvasir-seg", "custom"),
        default="hyperkvasir-segmented",
    )
    p.add_argument(
        "--data_root",
        type=Path,
        default=None,
        help="Required when --source=custom. Folder containing images/ and masks/.",
    )
    p.add_argument("--output_csv", type=Path, default=None)
    p.add_argument(
        "--no_probe_masks",
        action="store_true",
        help="Skip reading mask pixels (faster, but mask_pixel_count stays -1).",
    )
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.source == "hyperkvasir-segmented":
        root = HYPERKVASIR_SEGMENTED_DIR
        out = args.output_csv or (PROCESSED_DIR / "hyperkvasir_segmented_index.csv")
    elif args.source == "kvasir-seg":
        root = KVASIRSEG_DIR
        out = args.output_csv or KVASIRSEG_INDEX_CSV
    else:
        if args.data_root is None:
            raise SystemExit("--data_root is required when --source=custom")
        root = args.data_root
        out = args.output_csv or (PROCESSED_DIR / f"{root.name}_segmentation_index.csv")
    return root, out


def main() -> int:
    args = parse_args()
    root, output_csv = _resolve_paths(args)

    if not root.exists():
        print(f"ERROR: source folder not found: {root}", file=sys.stderr)
        return 2

    print(f"Source: {root}")
    print(f"Output: {output_csv}")
    print(f"Probe masks: {not args.no_probe_masks}")
    if args.dry_run:
        print("[dry_run] no CSV will be written")
        return 0

    t0 = time.time()
    n = index_segmentation(
        root=root,
        output_csv=output_csv,
        probe_masks=not args.no_probe_masks,
        progress=True,
    )
    dt = time.time() - t0
    print(f"Wrote {n} rows to {output_csv} in {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
