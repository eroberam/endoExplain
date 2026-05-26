"""Add lightweight endoscopy quality strata to an evaluation CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from endoexplain.evaluation import add_quality_strata, summarize_quality_strata  # noqa: E402
from endoexplain.quality import compute_image_quality  # noqa: E402


QUALITY_COLUMNS = {
    "blur_flag",
    "overexposed_frame_flag",
    "dark_frame_flag",
    "reflection_flag",
}
CORE_QUALITY_COLUMNS = {"blur_flag", "overexposed_frame_flag", "reflection_flag"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute quality strata and per-stratum metric summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_csv", type=Path, required=True)
    p.add_argument("--output_csv", type=Path, default=None)
    p.add_argument("--image_path_col", default="image_path")
    p.add_argument("--min_n", type=int, default=20)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def _ensure_quality_columns(df: pd.DataFrame, image_path_col: str) -> pd.DataFrame:
    if QUALITY_COLUMNS.issubset(df.columns):
        return df.copy()
    if CORE_QUALITY_COLUMNS.issubset(df.columns):
        out = df.copy()
        for col in QUALITY_COLUMNS - set(out.columns):
            out[col] = False
        return out
    if image_path_col not in df.columns:
        missing = ", ".join(sorted(QUALITY_COLUMNS - set(df.columns)))
        raise SystemExit(f"quality columns missing and no {image_path_col} column found: {missing}")

    rows = []
    for row in df.to_dict("records"):
        image_path = Path(str(row[image_path_col]))
        if not image_path.exists():
            rows.append(row)
            continue
        with Image.open(image_path) as img:
            quality = compute_image_quality(image=np.asarray(img.convert("RGB")), mask=None)
        row.update(quality)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    if args.limit is not None:
        df = df.head(args.limit).copy()

    df = _ensure_quality_columns(df, args.image_path_col)
    out_df = add_quality_strata(df)
    out_csv = args.output_csv or args.input_csv.with_name(f"{args.input_csv.stem}.quality_strata.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    summary = summarize_quality_strata(out_df, min_n=args.min_n)
    summary_csv = out_csv.with_suffix(".summary.csv")
    summary_json = out_csv.with_suffix(".summary.json")
    summary.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary.to_dict("records"), indent=2), encoding="utf-8")
    print(f"Wrote {out_csv} ({len(out_df)} rows)")
    print(f"Wrote {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
