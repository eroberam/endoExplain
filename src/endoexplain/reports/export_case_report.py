"""Export a per-case report (markdown + JSON + images) under outputs/reports/."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ..config.settings import DISCLAIMER


@dataclass
class CaseReport:
    case_id: str
    input_type: str  # "image" | "video"
    model_version: str
    prediction: str
    confidence: float
    mask_area_ratio: float | None = None
    explanation_metrics: dict | None = None
    quality_metrics: dict | None = None
    temporal_summary: dict | None = None
    images: dict[str, np.ndarray] = field(default_factory=dict)  # name -> RGB uint8


def _save_image(arr: np.ndarray, path: Path) -> None:
    img = arr
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8) if img.max() > 1.5 else (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def export_case_report(report: CaseReport, root_dir: Path) -> Path:
    """Write the case-report folder. Returns the folder path."""
    case_dir = root_dir / f"case_{report.case_id}"
    case_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "case_id": report.case_id,
        "input_type": report.input_type,
        "model_version": report.model_version,
        "prediction": report.prediction,
        "confidence": report.confidence,
        "mask_area_ratio": report.mask_area_ratio,
        "explanation_metrics": report.explanation_metrics,
        "quality_metrics": report.quality_metrics,
        "temporal_summary": report.temporal_summary,
        "disclaimer": DISCLAIMER,
    }
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    saved_images: dict[str, str] = {}
    for name, arr in report.images.items():
        out = case_dir / f"{name}.png"
        _save_image(arr, out)
        saved_images[name] = out.name

    lines = [f"# Case report: {report.case_id}", ""]
    lines.append(f"- Input type: `{report.input_type}`")
    lines.append(f"- Model version: `{report.model_version}`")
    lines.append(f"- Prediction: **{report.prediction}**")
    lines.append(f"- Confidence: `{report.confidence:.4f}`")
    if report.mask_area_ratio is not None:
        lines.append(f"- Mask area ratio: `{report.mask_area_ratio:.4f}`")

    if report.explanation_metrics:
        lines += ["", "## Explanation-mask alignment", ""]
        for k, v in report.explanation_metrics.items():
            lines.append(f"- `{k}` = `{v}`")
    if report.quality_metrics:
        lines += ["", "## Image quality", ""]
        for k, v in report.quality_metrics.items():
            lines.append(f"- `{k}` = `{v}`")
    if report.temporal_summary:
        lines += ["", "## Temporal summary", ""]
        for k, v in report.temporal_summary.items():
            lines.append(f"- `{k}` = `{v}`")

    if saved_images:
        lines += ["", "## Figures", ""]
        for name, path in saved_images.items():
            lines.append(f"### {name}")
            lines.append(f"![{name}]({path})")
            lines.append("")

    lines += ["", "---", "", "## Disclaimer", "", DISCLAIMER, ""]
    (case_dir / "report.md").write_text("\n".join(lines))
    return case_dir
