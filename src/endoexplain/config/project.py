"""Project-level YAML configuration helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .paths import PROJECT_ROOT


DEFAULT_PROJECT_CONFIG = PROJECT_ROOT / "configs" / "project.yaml"


@lru_cache(maxsize=8)
def load_project_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the central project configuration.

    The configuration intentionally stores stable repository conventions:
    public labels, clinical target families, default local paths and demo
    visualization policy. Training hyperparameters live in experiment YAMLs.
    """
    path = Path(config_path) if config_path else DEFAULT_PROJECT_CONFIG
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Project config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Project config must be a mapping: {path}")
    return data


def project_path(key: str, config_path: str | Path | None = None) -> Path:
    """Resolve a named path from ``configs/project.yaml``."""
    cfg = load_project_config(config_path)
    try:
        raw = cfg["paths"][key]
    except KeyError as exc:
        raise KeyError(f"Unknown project path key: {key}") from exc
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def classification_labels(config_path: str | Path | None = None) -> list[str]:
    """Return the public multiclass label order."""
    cfg = load_project_config(config_path)
    labels = cfg.get("labels", {}).get("classification", [])
    if not labels:
        raise ValueError("Missing labels.classification in project config")
    return list(labels)


def reference_targets(config_path: str | Path | None = None) -> dict[str, Any]:
    """Return optional reference metrics used for experiment manifests."""
    cfg = load_project_config(config_path)
    return dict(cfg.get("reference_targets", {}))
