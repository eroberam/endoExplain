# Reproducibility

EndoExplain is designed to be reproducible from public code and the public
HyperKvasir dataset. The repository does not include datasets, model weights,
rendered videos, logs or local experiment outputs.

## Environment

The validated environment is WSL2 Ubuntu with conda Python 3.10.

```bash
conda env create -f environment.yml
conda activate endoExplain
pip install -e ".[dev]"
```

## Data Layout

Place or symlink HyperKvasir at:

```text
data/raw/hyper-kvasir
```

Expected top-level entries include `labeled-images`, `labeled-videos`,
`segmented-images` and `unlabeled-images`.

## Indexes

```bash
python scripts/01_index_hyperkvasir.py \
  --data_root data/raw/hyper-kvasir \
  --no_probe_media

python scripts/01b_index_segmentation.py \
  --source hyperkvasir-segmented
```

## Registered Experiment Plan

Inspect the plan:

```bash
python scripts/16_run_registered_training.py \
  --plan excellence_sweep \
  --dry_run
```

Run the full local sweep:

```bash
RUN_ID="excellence_$(date +%Y%m%d_%H%M)"

python scripts/16_run_registered_training.py \
  --plan excellence_sweep \
  --run_id "$RUN_ID" \
  --device cuda
```

## Model Selection

```bash
python scripts/15_compare_classifier_runs.py \
  --runs_root "models/experiments/${RUN_ID}/classification" \
  --output_csv "models/experiments/${RUN_ID}/classification_comparison.csv" \
  --glob "*/summary.json"

python scripts/10_compare_segmentation_runs.py \
  --runs_root "models/experiments/${RUN_ID}/segmentation" \
  --output_csv "models/experiments/${RUN_ID}/segmentation_comparison.csv" \
  --glob "*/summary.json"

python scripts/17_select_classifier_champion.py \
  --comparison_csv "models/experiments/${RUN_ID}/classification_comparison.csv" \
  --metric test_acc \
  --output_json "models/experiments/${RUN_ID}/classifier_mvp.json"

python scripts/11_select_segmentation_champion.py \
  --comparison_csv "models/experiments/${RUN_ID}/segmentation_comparison.csv" \
  --metric test_dice \
  --output_json "models/experiments/${RUN_ID}/segmenter_mvp.json"
```

## Explainability And Video Review

```bash
python scripts/04_run_explainability.py \
  --checkpoint models/experiments/${RUN_ID}/classification/<classifier_run>/best.pt \
  --index_csv data/processed/hyperkvasir_segmented_index.csv \
  --seg_index_csv data/processed/hyperkvasir_segmented_index.csv \
  --method "gradcam++" \
  --num_samples 1000 \
  --image_size 256 \
  --threshold_mode top_percent \
  --threshold_value 0.20 \
  --output_dir outputs/heatmaps/${RUN_ID}_gradcampp_top20_segmented \
  --device cuda
```

```bash
python scripts/07_render_demo_video.py \
  --video_path data/raw/hyper-kvasir/labeled-videos/lower-gi-tract/pathological-findings/polyps/11305ea5-389a-46c8-95f3-94a1af84247d.avi \
  --classifier_ckpt models/experiments/${RUN_ID}/classification/<classifier_run>/best.pt \
  --segmenter_ckpt models/experiments/${RUN_ID}/segmentation/<segmenter_run>/best.pt \
  --output_path outputs/videos/11305_polyp_family_overlay.mp4 \
  --render_profile public_demo \
  --suspicious_class polyps \
  --confidence_threshold 0.85 \
  --method "gradcam++"
```

## Quality, Intervals And Temporal Benchmark

Add lightweight quality strata to any exported metrics CSV:

```bash
python scripts/18_export_quality_strata.py \
  --input_csv outputs/metrics/classifier_predictions/<run>/predictions_test.csv
```

Compute percentile bootstrap intervals:

```bash
python scripts/20_bootstrap_metric_intervals.py \
  --kind classification \
  --input_csv outputs/metrics/classifier_predictions/<run>/predictions_test.csv \
  --n_boot 5000
```

The temporal benchmark manifest is in `configs/evaluation/`. Positive videos
need manual `start_s` and `end_s` interval annotations before event precision,
recall or latency should be reported.

```bash
python scripts/19_run_temporal_benchmark.py \
  --classifier_ckpt models/experiments/${RUN_ID}/classification/<classifier_run>/best.pt \
  --sample_fps 5 \
  --threshold 0.85
```
