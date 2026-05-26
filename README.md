# EndoExplain

EndoExplain is a research prototype for explainable gastrointestinal
endoscopy review. It combines multiclass frame classification, polyp
segmentation, Grad-CAM++ visual evidence, temporal event grouping, frame
quality indicators and a cockpit-style video overlay.

This project is for research and education only. It is not a medical device
and must not be used for diagnosis, treatment decisions or patient care.

## Install

The validated environment is WSL2 Ubuntu with conda Python 3.10.

```bash
conda env create -f environment.yml
conda activate endoExplain
pip install -e ".[dev]"
```

Place or symlink the public HyperKvasir release at:

```text
data/raw/hyper-kvasir
```

## Quick Start

```bash
python scripts/01_index_hyperkvasir.py \
  --data_root data/raw/hyper-kvasir \
  --no_probe_media

python scripts/01b_index_segmentation.py \
  --source hyperkvasir-segmented

python scripts/16_run_registered_training.py \
  --plan excellence_sweep \
  --dry_run
```

The registered experiment plan writes local outputs under
`models/experiments/<run_id>/`. Local datasets, checkpoints, metrics,
videos and logs are intentionally excluded from Git.

## Documentation

- `docs/REPRODUCIBILITY.md`: environment, data layout and technical workflow.
- `docs/EXPERIMENTS.md`: training, comparison and model-selection commands.
- `docs/RESULTS_REFERENCE.md`: reference run metrics and publication boundary.

## Repository Layout

```text
configs/                 Reproducible project and experiment configs
data/                    Local datasets and generated indexes, ignored
docs/                    Public technical runbooks
models/                  Local checkpoints and experiment registry, ignored
outputs/                 Local metrics and rendered videos, ignored
scripts/                 Command-line workflows
src/endoexplain/         Python package
tests/                   Unit tests
```

## Citation

If you use this work, cite the repository metadata in `CITATION.cff`.
