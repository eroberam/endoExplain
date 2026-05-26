"""Run append-only training experiments with a persistent registry.

This script is intentionally conservative: it never writes into
``models/checkpoints``. Each attempt receives a unique output directory under
``models/experiments/<run_id>/`` and is recorded in a registry CSV/JSONL.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from endoexplain.config import classification_labels, project_path, reference_targets  # noqa: E402
from endoexplain.training import SegTrainConfig, TrainConfig, train_classifier, train_segmenter  # noqa: E402


CLASS_LABELS = classification_labels()
REFERENCE_TARGETS = reference_targets()

REGISTRY_FIELDS = [
    "run_id",
    "task",
    "name",
    "status",
    "started_at",
    "ended_at",
    "seconds",
    "output_dir",
    "checkpoint",
    "primary_metric",
    "best_val_acc",
    "test_acc",
    "test_loss",
    "best_val_dice",
    "test_dice",
    "test_iou",
    "n_train",
    "n_val",
    "n_test",
    "error",
]


@dataclass(frozen=True)
class ExperimentSpec:
    task: str
    name: str
    params: dict[str, Any]


class Tee:
    """Write training output both to the terminal and to a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except ValueError:
                # Some third-party loggers keep a reference to a previous
                # redirected stream after a per-experiment log file closes.
                # Ignore that stale stream and keep the runner alive.
                continue
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            try:
                stream.flush()
            except ValueError:
                continue


for _logger_name in ("huggingface_hub", "timm"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)


def _safe_id(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    return value.strip("_")


def _timestamp_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _base_classifier_params() -> dict[str, Any]:
    return {
        "index_csv": project_path("image_index"),
        "image_size": 256,
        "batch_size": 4,
        "num_workers": 4,
        "epochs": 40,
        "learning_rate": 5e-5,
        "mixed_precision": True,
        "classes": CLASS_LABELS,
        "max_samples_per_class": None,
        "pretrained": True,
        "early_stopping_patience": 10,
    }


def _base_segmenter_params() -> dict[str, Any]:
    return {
        "index_csv": project_path("segmentation_index"),
        "batch_size": 4,
        "num_workers": 4,
        "epochs": 70,
        "learning_rate": 2e-4,
        "mixed_precision": True,
        "max_samples": None,
        "early_stopping_patience": 12,
        "val_fraction": 0.15,
        "test_fraction": 0.15,
    }


def _plan_config_path(plan_name: str) -> Path:
    return PROJECT_ROOT / "configs" / "experiments" / f"{plan_name}.yaml"


def _merge_params(*parts: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for part in parts:
        if part:
            out.update(part)
    return out


def _resolve_param_paths(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, str) and (
            key.endswith("_csv") or key.endswith("_path") or key.endswith("_dir")
        ):
            path = Path(value)
            out[key] = path if path.is_absolute() else PROJECT_ROOT / path
        else:
            out[key] = value
    return out


def _lr_label(value: Any) -> str:
    return f"{float(value):g}".replace("e-0", "e-").replace("e+0", "e+")


def _specs_from_yaml(plan_config: Path) -> list[ExperimentSpec]:
    if not plan_config.is_absolute():
        plan_config = PROJECT_ROOT / plan_config
    with plan_config.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    specs: list[ExperimentSpec] = []

    clf_section = data.get("classification") or {}
    clf_defaults = dict(clf_section.get("defaults") or {})
    clf_defaults.setdefault("classes", CLASS_LABELS)
    for grid in clf_section.get("grids") or []:
        for seed in grid.get("seeds", []):
            for lr in grid.get("learning_rates", []):
                for augment in grid.get("augment_levels", ["standard"]):
                    params = _merge_params(
                        clf_defaults,
                        grid.get("params"),
                        {
                            "backbone": grid["backbone"],
                            "image_size": grid.get("image_size", clf_defaults.get("image_size")),
                            "batch_size": grid.get("batch_size", clf_defaults.get("batch_size")),
                            "seed": seed,
                            "learning_rate": lr,
                            "augment_level": augment,
                        },
                    )
                    name = grid.get("name_template")
                    if name:
                        exp_name = name.format(
                            backbone=grid["backbone"],
                            image_size=params["image_size"],
                            seed=seed,
                            lr=_lr_label(lr),
                            augment=augment,
                        )
                    else:
                        exp_name = (
                            f"{grid['backbone']}_img{params['image_size']}_"
                            f"s{seed}_lr{_lr_label(lr)}_{augment}"
                        )
                    specs.append(ExperimentSpec("classification", exp_name, _resolve_param_paths(params)))
    for entry in clf_section.get("experiments") or []:
        params = _merge_params(clf_defaults, entry.get("params"))
        specs.append(
            ExperimentSpec("classification", entry["name"], _resolve_param_paths(params))
        )

    seg_section = data.get("segmentation") or {}
    seg_defaults = dict(seg_section.get("defaults") or {})
    loss_profiles = dict(seg_section.get("loss_profiles") or {})
    for grid in seg_section.get("grids") or []:
        for seed in grid.get("seeds", []):
            for profile in grid.get("profiles", ["balanced"]):
                for augment in grid.get("augment_levels", ["light"]):
                    params = _merge_params(
                        seg_defaults,
                        loss_profiles.get(profile),
                        grid.get("params"),
                        {
                            "architecture": grid["architecture"],
                            "encoder_name": grid["encoder_name"],
                            "image_size": grid.get("image_size", seg_defaults.get("image_size")),
                            "seed": seed,
                            "augment_level": augment,
                        },
                    )
                    exp_name = (
                        f"{grid['architecture']}_{grid['encoder_name']}_"
                        f"img{params['image_size']}_s{seed}_{profile}_{augment}"
                    )
                    specs.append(ExperimentSpec("segmentation", exp_name, _resolve_param_paths(params)))
    for entry in seg_section.get("experiments") or []:
        profile = entry.get("profile")
        params = _merge_params(seg_defaults, loss_profiles.get(profile), entry.get("params"))
        specs.append(ExperimentSpec("segmentation", entry["name"], _resolve_param_paths(params)))

    if not specs:
        raise ValueError(f"No experiments found in plan config: {plan_config}")
    return specs


def benchmark_reference_plan() -> list[ExperimentSpec]:
    yaml_plan = _plan_config_path("benchmark_reference")
    if yaml_plan.exists():
        return _specs_from_yaml(yaml_plan)
    clf = _base_classifier_params()
    seg = _base_segmenter_params()
    return [
        ExperimentSpec(
            "classification",
            "convnext_tiny_s45_lr5e-5",
            {**clf, "backbone": "convnext_tiny", "seed": 45, "learning_rate": 5e-5},
        ),
        ExperimentSpec(
            "classification",
            "convnext_tiny_s145_lr3e-5",
            {**clf, "backbone": "convnext_tiny", "seed": 145, "learning_rate": 3e-5},
        ),
        ExperimentSpec(
            "classification",
            "efficientnet_b0_s146_lr5e-5",
            {**clf, "backbone": "efficientnet_b0", "seed": 146, "batch_size": 8},
        ),
        ExperimentSpec(
            "classification",
            "resnet34_s147_lr1e-4",
            {**clf, "backbone": "resnet34", "seed": 147, "batch_size": 8, "learning_rate": 1e-4},
        ),
        ExperimentSpec(
            "segmentation",
            "unetpp_effb1_384_s53",
            {
                **seg,
                "architecture": "UnetPlusPlus",
                "encoder_name": "efficientnet-b1",
                "encoder_weights": "imagenet",
                "image_size": 384,
                "seed": 53,
            },
        ),
        ExperimentSpec(
            "segmentation",
            "unetpp_effb1_384_s153",
            {
                **seg,
                "architecture": "UnetPlusPlus",
                "encoder_name": "efficientnet-b1",
                "encoder_weights": "imagenet",
                "image_size": 384,
                "seed": 153,
            },
        ),
        ExperimentSpec(
            "segmentation",
            "deeplabv3plus_resnet34_352_s43",
            {
                **seg,
                "architecture": "DeepLabV3Plus",
                "encoder_name": "resnet34",
                "encoder_weights": "imagenet",
                "image_size": 352,
                "seed": 43,
                "epochs": 60,
            },
        ),
        ExperimentSpec(
            "segmentation",
            "fpn_effb1_384_s152",
            {
                **seg,
                "architecture": "FPN",
                "encoder_name": "efficientnet-b1",
                "encoder_weights": "imagenet",
                "image_size": 384,
                "seed": 152,
                "epochs": 60,
            },
        ),
    ]


def excellence_sweep_plan() -> list[ExperimentSpec]:
    yaml_plan = _plan_config_path("excellence_sweep")
    if yaml_plan.exists():
        return _specs_from_yaml(yaml_plan)
    clf = _base_classifier_params()
    seg = _base_segmenter_params()
    specs: list[ExperimentSpec] = []

    classifier_grid = [
        ("convnext_tiny", 256, 4, [45, 145, 245], [5e-5, 3e-5], ["standard"]),
        ("efficientnet_b0", 256, 8, [46, 146, 246], [5e-5, 1e-4], ["standard"]),
        ("resnet34", 256, 8, [47, 147, 247], [1e-4, 5e-5], ["standard"]),
        ("resnet18", 256, 8, [48, 148], [1e-4], ["standard"]),
        ("deit_tiny_patch16_224", 224, 4, [49, 149], [5e-5], ["standard"]),
    ]
    for backbone, image_size, batch_size, seeds, lrs, augments in classifier_grid:
        for seed in seeds:
            for lr in lrs:
                for augment in augments:
                    specs.append(
                        ExperimentSpec(
                            "classification",
                            f"{backbone}_img{image_size}_s{seed}_lr{lr:g}_{augment}",
                            {
                                **clf,
                                "backbone": backbone,
                                "image_size": image_size,
                                "batch_size": batch_size,
                                "seed": seed,
                                "learning_rate": lr,
                                "augment_level": augment,
                                "scheduler": "cosine",
                                "label_smoothing": 0.03,
                                "weight_decay": 1e-4,
                                "epochs": 45,
                                "early_stopping_patience": 10,
                            },
                        )
                    )

    for backbone, seed, lr in (
        ("convnext_tiny", 45, 5e-5),
        ("convnext_tiny", 145, 3e-5),
        ("efficientnet_b0", 46, 5e-5),
    ):
        specs.append(
            ExperimentSpec(
                "classification",
                f"{backbone}_img256_s{seed}_lr{lr:g}_strong",
                {
                    **clf,
                    "backbone": backbone,
                    "image_size": 256,
                    "batch_size": 4 if backbone == "convnext_tiny" else 8,
                    "seed": seed,
                    "learning_rate": lr,
                    "augment_level": "strong",
                    "scheduler": "cosine",
                    "label_smoothing": 0.03,
                    "weight_decay": 1e-4,
                    "epochs": 45,
                    "early_stopping_patience": 10,
                },
            )
        )

    loss_profiles = {
        "balanced": {
            "loss_bce_weight": 0.50,
            "loss_dice_weight": 1.00,
            "loss_focal_weight": 0.15,
            "loss_tversky_weight": 0.00,
        },
        "recall": {
            "loss_bce_weight": 0.35,
            "loss_dice_weight": 1.00,
            "loss_focal_weight": 0.35,
            "loss_tversky_weight": 0.50,
            "loss_tversky_alpha": 0.30,
            "loss_tversky_beta": 0.70,
        },
        "tight": {
            "loss_bce_weight": 0.70,
            "loss_dice_weight": 1.00,
            "loss_focal_weight": 0.20,
            "loss_tversky_weight": 0.25,
            "loss_tversky_alpha": 0.50,
            "loss_tversky_beta": 0.50,
        },
    }
    segmenter_grid = [
        ("UnetPlusPlus", "efficientnet-b1", 384, [53, 153, 253], ["balanced", "recall"], ["light"]),
        ("DeepLabV3Plus", "resnet34", 352, [43, 143, 243], ["balanced", "recall"], ["light"]),
        ("FPN", "efficientnet-b1", 384, [52, 152], ["balanced", "recall"], ["light"]),
        ("UnetPlusPlus", "efficientnet-b0", 384, [42, 142], ["balanced"], ["light"]),
        ("Unet", "resnet34", 352, [44, 144], ["balanced", "tight"], ["light"]),
        ("Segformer", "mit_b0", 352, [51, 151], ["balanced"], ["light"]),
    ]
    for architecture, encoder, image_size, seeds, profiles, augments in segmenter_grid:
        for seed in seeds:
            for profile in profiles:
                for augment in augments:
                    specs.append(
                        ExperimentSpec(
                            "segmentation",
                            f"{architecture}_{encoder}_img{image_size}_s{seed}_{profile}_{augment}",
                            {
                                **seg,
                                **loss_profiles[profile],
                                "architecture": architecture,
                                "encoder_name": encoder,
                                "encoder_weights": "imagenet",
                                "image_size": image_size,
                                "seed": seed,
                                "augment_level": augment,
                                "normalize": True,
                                "scheduler": "cosine",
                                "weight_decay": 1e-4,
                                "max_grad_norm": 1.0,
                                "epochs": 80 if architecture == "UnetPlusPlus" else 65,
                                "early_stopping_patience": 14,
                                "threshold_min": 0.25,
                                "threshold_max": 0.85,
                                "threshold_steps": 13,
                            },
                        )
                    )

    for architecture, encoder, image_size, seed, profile in (
        ("UnetPlusPlus", "efficientnet-b1", 384, 53, "recall"),
        ("UnetPlusPlus", "efficientnet-b1", 384, 153, "balanced"),
        ("FPN", "efficientnet-b1", 384, 52, "recall"),
    ):
        specs.append(
            ExperimentSpec(
                "segmentation",
                f"{architecture}_{encoder}_img{image_size}_s{seed}_{profile}_strong",
                {
                    **seg,
                    **loss_profiles[profile],
                    "architecture": architecture,
                    "encoder_name": encoder,
                    "encoder_weights": "imagenet",
                    "image_size": image_size,
                    "seed": seed,
                    "augment_level": "strong",
                    "normalize": True,
                    "scheduler": "cosine",
                    "weight_decay": 1e-4,
                    "max_grad_norm": 1.0,
                    "epochs": 80 if architecture == "UnetPlusPlus" else 65,
                    "early_stopping_patience": 14,
                    "threshold_min": 0.25,
                    "threshold_max": 0.85,
                    "threshold_steps": 13,
                },
            )
        )

    return specs


def smoke_plan() -> list[ExperimentSpec]:
    specs = benchmark_reference_plan()
    out: list[ExperimentSpec] = []
    for spec in specs[:1] + specs[4:5]:
        params = dict(spec.params)
        params["epochs"] = 1
        params["num_workers"] = 0
        if spec.task == "classification":
            params["max_samples_per_class"] = 10
        else:
            params["max_samples"] = 24
            params["image_size"] = min(int(params["image_size"]), 160)
        out.append(ExperimentSpec(spec.task, "smoke_" + spec.name, params))
    return out


def _summary_to_row(
    run_id: str,
    spec: ExperimentSpec,
    status: str,
    started_at: str,
    ended_at: str,
    seconds: float,
    output_dir: Path,
    summary: dict[str, Any] | None,
    error: str = "",
) -> dict[str, Any]:
    summary = summary or {}
    if spec.task == "classification":
        primary = summary.get("test_acc", "")
    else:
        primary = summary.get("test_dice", "")
    return {
        "run_id": run_id,
        "task": spec.task,
        "name": spec.name,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "seconds": f"{seconds:.1f}",
        "output_dir": _rel(output_dir),
        "checkpoint": summary.get("checkpoint", ""),
        "primary_metric": primary,
        "best_val_acc": summary.get("best_val_acc", ""),
        "test_acc": summary.get("test_acc", ""),
        "test_loss": summary.get("test_loss", ""),
        "best_val_dice": summary.get("best_val_dice", ""),
        "test_dice": summary.get("test_dice", ""),
        "test_iou": summary.get("test_iou", ""),
        "n_train": summary.get("n_train", ""),
        "n_val": summary.get("n_val", ""),
        "n_test": summary.get("n_test", ""),
        "error": error[:500],
    }


def _write_registry(rows: list[dict[str, Any]], registry_csv: Path, registry_jsonl: Path) -> None:
    registry_csv.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if registry_csv.exists():
        with registry_csv.open(newline="") as f:
            existing = list(csv.DictReader(f))
    combined = existing + rows
    with registry_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in REGISTRY_FIELDS} for row in combined])
    with registry_jsonl.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_batch_tables(rows: list[dict[str, Any]], run_dir: Path) -> None:
    import pandas as pd

    if not rows:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for task, metric in (("classification", "test_acc"), ("segmentation", "test_dice")):
        part = df[(df["task"] == task) & (df["status"] == "completed")].copy()
        if part.empty:
            continue
        part[metric] = part[metric].astype(float)
        part = part.sort_values(metric, ascending=False)
        out_csv = run_dir / f"{task}_comparison.csv"
        part.to_csv(out_csv, index=False)
        try:
            out_csv.with_suffix(".md").write_text(part.to_markdown(index=False))
        except ImportError:
            out_csv.with_suffix(".md").write_text(part.to_csv(index=False))


def _run_one(
    spec: ExperimentSpec,
    run_id: str,
    run_root: Path,
    log_root: Path,
    device: str | None,
) -> dict[str, Any]:
    name = _safe_id(spec.name)
    output_dir = run_root / spec.task / name
    log_path = log_root / f"{spec.task}_{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if (output_dir / "summary.json").exists():
        now = datetime.now().isoformat(timespec="seconds")
        return _summary_to_row(
            run_id,
            spec,
            "skipped_existing",
            now,
            now,
            0.0,
            output_dir,
            json.loads((output_dir / "summary.json").read_text()),
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        now = datetime.now().isoformat(timespec="seconds")
        return _summary_to_row(
            run_id,
            spec,
            "blocked_existing_incomplete",
            now,
            now,
            0.0,
            output_dir,
            None,
            "Output directory exists without summary.json. Use a new run_id.",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    params = dict(spec.params)
    params["output_dir"] = output_dir
    if device is not None:
        params["device"] = device

    started = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    summary: dict[str, Any] | None = None
    status = "completed"
    error = ""
    with log_path.open("w", encoding="utf-8") as log:
        tee_out = Tee(sys.stdout, log)
        tee_err = Tee(sys.stderr, log)
        try:
            print(f"\n=== {spec.task}: {spec.name} ===", file=tee_out)
            print(f"output_dir: {_rel(output_dir)}", file=tee_out)
            with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                if spec.task == "classification":
                    cfg = TrainConfig(**params)
                    summary = train_classifier(cfg)
                elif spec.task == "segmentation":
                    cfg = SegTrainConfig(**params)
                    summary = train_segmenter(cfg)
                else:
                    raise ValueError(f"unknown task: {spec.task}")
        except Exception as exc:  # noqa: BLE001 - registry must capture failures
            status = "failed"
            error = repr(exc)
            print(f"\nFAILED {spec.task}: {spec.name}: {error}", file=tee_err)
        finally:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    ended = datetime.now().isoformat(timespec="seconds")
    return _summary_to_row(
        run_id,
        spec,
        status,
        started,
        ended,
        time.time() - t0,
        output_dir,
        summary,
        error,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run append-only classifier/segmenter experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--plan",
        choices=["benchmark_reference", "excellence_sweep", "smoke"],
        default="benchmark_reference",
    )
    p.add_argument(
        "--plan_config",
        type=Path,
        default=None,
        help="Optional experiment YAML. Overrides --plan except for registry naming.",
    )
    p.add_argument("--run_id", default=None)
    p.add_argument("--only", choices=["all", "classification", "segmentation"], default="all")
    p.add_argument("--include", nargs="*", default=None, help="Optional exact experiment names to run.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--stop_on_failure", action="store_true")
    p.add_argument("--experiments_root", type=Path, default=PROJECT_ROOT / "models" / "experiments")
    p.add_argument("--logs_root", type=Path, default=PROJECT_ROOT / "logs" / "experiments")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_id = _safe_id(args.run_id or _timestamp_id())
    if args.plan_config is not None:
        specs = _specs_from_yaml(args.plan_config)
    elif args.plan == "smoke":
        specs = smoke_plan()
    elif args.plan == "excellence_sweep":
        specs = excellence_sweep_plan()
    else:
        specs = benchmark_reference_plan()
    if args.only != "all":
        specs = [s for s in specs if s.task == args.only]
    if args.include:
        wanted = set(args.include)
        specs = [s for s in specs if s.name in wanted]
    if args.limit is not None:
        specs = specs[: args.limit]
    if not specs:
        raise SystemExit("no experiments selected")

    run_root = args.experiments_root / run_id
    log_root = args.logs_root / run_id
    registry_csv = args.experiments_root / "registry.csv"
    registry_jsonl = args.experiments_root / "registry.jsonl"
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "plan": args.plan,
        "plan_config": _rel(args.plan_config) if args.plan_config else "",
        "only": args.only,
        "reference_targets": REFERENCE_TARGETS,
        "experiments": [
            {"task": s.task, "name": s.name, "params": {k: str(v) for k, v in s.params.items()}}
            for s in specs
        ],
    }

    print(f"run_id: {run_id}")
    print(f"experiments_root: {_rel(args.experiments_root)}")
    print(f"logs_root: {_rel(args.logs_root)}")
    print(f"registry: {_rel(registry_csv)}")
    print(f"selected: {len(specs)} experiments")
    for spec in specs:
        print(f"- {spec.task}: {spec.name}")

    if args.dry_run:
        print("\n[dry_run] no training started")
        return 0

    run_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True))

    rows: list[dict[str, Any]] = []
    for spec in specs:
        row = _run_one(spec, run_id, run_root, log_root, args.device)
        rows.append(row)
        _write_registry([row], registry_csv, registry_jsonl)
        if row["status"] == "failed" and args.stop_on_failure:
            break

    _write_batch_tables(rows, run_root)
    completed = sum(row["status"] == "completed" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    print(f"\nDone: completed={completed}, failed={failed}, total={len(rows)}")
    print(f"Batch folder: {_rel(run_root)}")
    print(f"Registry: {_rel(registry_csv)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
