# Reference Results

The reference run identifier is:

```text
excellence_20260525_1051
```

The run completed all 59 registered experiments in
`configs/experiments/excellence_sweep.yaml`.

## Selected Models

Classifier:

```text
backbone: EfficientNet-B0
test acc: 0.9280
95% bootstrap CI: 0.9116-0.9435
test n: 1097
```

Segmenter:

```text
model: U-Net++ EfficientNet-B1
threshold: 0.60
test Dice: 0.9318
Dice 95% bootstrap CI: 0.9164-0.9448
test IoU: 0.8826
IoU 95% bootstrap CI: 0.8611-0.9017
precision: 0.9404
recall: 0.9374
test n: 150
```

Binary `polyp_family` export:

```text
ROC-AUC: 0.9969
ROC-AUC 95% bootstrap CI: 0.9932-0.9993
average precision: 0.9946
F1 at 0.5: 0.9738
Brier score: 0.0122
positives/test n: 306 / 1097
```

## Explainability Audit

Grad-CAM++ top-20% attribution on 1000 segmented polyp images:

```text
mean IoU: 0.0559
mean IoU 95% bootstrap CI: 0.0501-0.0617
median IoU: 0.0080
Q90 IoU: 0.1813
inside energy: 0.0969
inside energy 95% bootstrap CI: 0.0896-0.1045
pointing-game hit: 0.0950
pointing-game 95% bootstrap CI: 0.0770-0.1130
```

## Quality Stratification

The lightweight quality proxies are blur, overexposure/darkness and specular
reflection. On the reference exports, the main descriptive strata were:

```text
classifier good n/acc: 817 / 0.9400
classifier blurred n/acc: 267 / 0.8951
segmentation good n/Dice: 148 / 0.9312
Grad-CAM++ good n/IoU: 959 / 0.0566
Grad-CAM++ blurred n/IoU: 23 / 0.0302
```

## Demo Case

The reference local video render is a single retrospective HyperKvasir
`polyps` case:

```text
source frames: 661
source FPS: 25
events: 1
event span: 0.00-26.32 s
mean/max confidence: 0.9092 / 0.9864
mask visible frames: 297 / 661
```

The demo case illustrates the pipeline behavior; it is not a temporal
benchmark.

The repository includes a 60-video temporal benchmark manifest and annotation
template under `configs/evaluation/`. The positive intervals must be manually
reviewed before reporting event precision, recall or latency.
