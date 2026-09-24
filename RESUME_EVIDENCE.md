# Resume Evidence

## Project title

PyTorch Object Detection & Failure Analysis Pipeline

## Verified stack

Python 3.11, PyTorch 2.5.1, Torchvision 0.20.1, Faster R-CNN, MobileNetV3, FPN, COCO transfer learning, pycocotools, MLflow, OpenCV, pytest, Ruff, Docker, GitHub Actions, and GitHub Pages workflow configuration.

## Dataset size

The Penn-Fudan Pedestrian Detection dataset was downloaded from the source used by the official Torchvision tutorial. The validated local copy contains 170 images and 423 mask-derived person instances. Seed 42 produced non-overlapping splits of 119 training images / 290 instances, 25 validation images / 67 instances, and 26 test images / 66 instances. The split manifest SHA-256 is `631066345ec7961628fba7d13948b860c3f37ee54e93cd106bae5331c5c7677e`.

## Model architecture

`fasterrcnn_mobilenet_v3_large_fpn` initialized from Torchvision `COCO_V1` weights, with its classification predictor replaced for background and person. The fine-tuned model has 18,871,333 trainable parameters.

## Training methodology

Two six-epoch CUDA experiments used the same seed, training split, validation split, optimizer, learning-rate schedule, augmentation, and model initialization. The baseline used a 512-768 pixel resize range. Experiment 2 changed only the resize range to 640-960 after the baseline validation audit found low-confidence and localization failures involving small, partial, and crowded pedestrians. Checkpoint selection and the operating threshold used validation data only. The held-out test was evaluated once after model selection.

## Actual final metrics

At the validation-selected confidence threshold of 0.45, the selected model produced the following results on 26 held-out test images containing 66 annotated people:

| Metric | Result |
|---|---:|
| mAP@0.50:0.95 | 0.6850 |
| mAP@0.50 | 0.8937 |
| Precision | 0.9219 |
| Recall | 0.8939 |
| F1 | 0.9077 |
| TP / FP / FN | 59 / 5 / 7 |

The selected checkpoint SHA-256 is `9978477659599ffab35279f3ad81b0800bf8e0e7cf2b1c89ec7029d797291d8e`.

## Failure-analysis evidence

The evaluator records matched detections and classifies false positives, false negatives, localization failures, duplicate detections, and low-confidence correct detections. The baseline validation audit produced two false negatives, three localization failures, and two low-confidence correct detections, with annotated galleries and machine-readable CSV/JSON evidence. The final held-out audit recorded seven false negatives, three unmatched false-positive cases, two localization cases, and three low-confidence correct detections.

## Testing evidence

The complete local test suite passed: **57 passed**. Coverage includes dataset parsing, deterministic splitting, zero split leakage, bounding-box validation, transforms, detector construction, one-batch forward execution, metrics, failure categorization, checkpoint round trips, malformed input, image and video inference, MLflow integration, model loading, experiment comparison, and configuration parsing. Ruff also passed with no violations.

## Docker evidence

Image `podfa:local` built successfully from `python:3.11-slim`. Its default non-root CPU smoke test constructed the detector and ran inference successfully. The complete test suite also passed inside the container: **57 passed**.

## Experiment-tracking evidence

MLflow was exercised against a local SQLite backend. The baseline run ID is `fbdbeccc1f55403aada2cf94d696685c`; the improved run ID is `e83256649d384c30b7c9f661a140756b`. Both runs contain configuration parameters, epoch losses, validation metrics, device, seed, duration, and generated artifacts.

## Video-inference evidence

The OpenCV CLI validates video input, performs frame-by-frame detector inference, overlays labeled boxes and confidence scores, writes an annotated video, and reports source and measured inference FPS. A real checkpoint smoke run processed 10 frames on CUDA at 14.86 average inference FPS; this short repeated-frame smoke result is not presented as a general video benchmark.

## Deployment/publication evidence

The repository contains push/pull-request CI and GitHub Pages deployment workflows. Public repository, CI, release, and Pages URLs are recorded here only after remote verification so this file does not pre-claim deployment success.

## Three candidate resume bullets

- Built a reproducible PyTorch/Torchvision pedestrian-detection pipeline that fine-tuned Faster R-CNN MobileNetV3-FPN on 170 Penn-Fudan street images using deterministic, leakage-tested 119/25/26 train/validation/test splits.
- Designed a validation-driven failure audit and controlled resolution experiment that raised validation mAP@0.50:0.95 from 0.7705 to 0.7913; the once-evaluated held-out test achieved 0.6850 mAP@0.50:0.95, 0.9219 precision, and 0.8939 recall.
- Engineered MLflow-tracked training, checkpoint resume, COCO evaluation, OpenCV video inference, Docker packaging, and 57 automated tests; verified the full test suite both locally and inside the built container.
