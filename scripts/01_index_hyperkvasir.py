"""CLI wrapper around ``endoexplain.data.index_hyperkvasir``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from endoexplain.config import HYPERKVASIR_DIR, HYPERKVASIR_INDEX_CSV  # noqa: E402
from endoexplain.data import index_hyperkvasir  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Index a local HyperKvasir folder into a CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_root", type=Path, default=HYPERKVASIR_DIR)
    p.add_argument("--output_csv", type=Path, default=HYPERKVASIR_INDEX_CSV)
    p.add_argument(
        "--no_probe_media",
        action="store_true",
        help="Skip opening images/videos (faster, but width/height/fps stay -1).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Index only the first 200 files (quick smoke test).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override max number of files to index. Wins over --debug if both set.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="List what would be done without writing the CSV.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root: Path = args.data_root
    output: Path = args.output_csv
    limit = args.limit if args.limit is not None else (200 if args.debug else None)

    if not root.exists():
        print(f"ERROR: dataset root not found: {root}", file=sys.stderr)
        print(
            "Place or symlink the HyperKvasir dataset at data/raw/hyper-kvasir, "
            "or pass --data_root /path/to/hyper-kvasir.",
            file=sys.stderr,
        )
        return 2

    print(f"Indexing: {root}")
    print(f"Output: {output}")
    print(f"Probe media: {not args.no_probe_media}")
    if limit is not None:
        print(f"Limit: {limit} files")

    if args.dry_run:
        print("[dry_run] no CSV will be written")
        return 0

    t0 = time.time()
    n = index_hyperkvasir(
        root=root,
        output_csv=output,
        probe_media=not args.no_probe_media,
        limit=limit,
        progress=True,
    )
    dt = time.time() - t0
    print(f"Wrote {n} rows to {output} in {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
