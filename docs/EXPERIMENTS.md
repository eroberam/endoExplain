# Experiments

The reference workflow uses a registered sweep defined in
`configs/experiments/excellence_sweep.yaml`.

## Classifier

The selected classifier family is EfficientNet-B0 over the ten retained
HyperKvasir classes. The clinical review target `polyp_family` is computed by
summing the probabilities for `polyps` and `dyed-lifted-polyps`.

Reference local metrics from the completed run:

```text
multiclass test accuracy: 0.9280
polyp_family ROC-AUC: 0.9969
polyp_family AP: 0.9946
polyp_family F1: 0.9738
```

## Segmenter

The selected segmenter family is U-Net++ with an EfficientNet-B1 encoder.

Reference local metrics from the completed run:

```text
test Dice: 0.9318
test IoU: 0.8826
test precision: 0.9404
test recall: 0.9374
```

## Explainability

The reference audit uses Grad-CAM++ over the 1000 HyperKvasir segmented polyp
images and compares top-20% classifier attribution with the lesion mask.
This audit evaluates classifier evidence alignment, not segmentation quality.

```text
mean IoU: 0.0559
median IoU: 0.0080
Q90 IoU: 0.1813
inside energy: 0.0969
pointing-game hit: 0.0950
```

## Video Review

The demo renderer groups frame-level `polyp_family` confidence into temporal
events and draws a conservative segmentation overlay only when confidence,
event membership and mask geometry gates pass.

The reference local case contains one event across 661 source frames, with
297 frames showing an accepted mask overlay.
