# Model card: Penn-Fudan pedestrian detector

## Model details

- Architecture: Torchvision Faster R-CNN MobileNetV3 Large FPN
- Weights initialization: COCO `DEFAULT` / `COCO_V1`
- Head: two classes (background, person)
- Selected experiment: 640–960 px resize, seed 42, best epoch 4
- Checkpoint SHA-256: `9978477659599ffab35279f3ad81b0800bf8e0e7cf2b1c89ec7029d797291d8e`
- Operating confidence threshold: 0.45, selected on validation only
- Framework: PyTorch 2.5.1+cu121 / Torchvision 0.20.1+cu121

## Dataset

Penn-Fudan Pedestrian Detection, downloaded from the University of Pennsylvania URL
used by the official Torchvision object-detection tutorial. The exact archive used
contained 170 paired images/masks and 423 mask-derived instances. Deterministic seed-42
split: 119 train (290 instances), 25 validation (67), 26 test (66). Raw data is not
redistributed by this repository.

## Training methodology

Transfer learning replaced the COCO classification head and fine-tuned the detector
for six epochs with SGD (learning rate 0.005, momentum 0.9, weight decay 0.0005),
StepLR gamma 0.1 after epoch 3, batch size 2, random horizontal flips, and CUDA AMP.
The best checkpoint was selected by validation mAP@0.50:0.95. Local MLflow run ID:
`e83256649d384c30b7c9f661a140756b`. Training took 147.351 seconds on the listed host.

## Evaluation

| Split | mAP@0.50:0.95 | mAP@0.50 | Precision | Recall | TP / FP / FN |
|---|---:|---:|---:|---:|---:|
| Validation | 0.7913 | 0.9768 | 0.9559 | 0.9701 | 65 / 3 / 2 |
| Held-out test | 0.6850 | 0.8937 | 0.9219 | 0.8939 | 59 / 5 / 7 |

COCO AP uses all detector scores. Precision/recall/counts use one-to-one matching at
IoU 0.50 and the frozen confidence threshold 0.45. The test evaluation was executed
once after model selection.

## Intended use

Learning, portfolio review, reproducible computer-vision experimentation, and
non-safety-critical pedestrian-detection prototyping on images similar to Penn-Fudan.
The checkpoint can support local image/video demonstrations and failure-analysis
experiments.

## Out-of-scope use

This model is not production ready and must not be used alone for surveillance,
identity inference, access control, law enforcement, safety intervention, or decisions
about people. It does not recognize identities and must not be extended to do so
without a separate ethical and legal review.

## Known failure cases

- Occluded pedestrians in dense groups can receive geometrically correct boxes but
  very low confidence.
- Overlapping pedestrians can produce poorly separated boxes.
- Unlabeled distant/partial pedestrians may be scored as false positives by the
  available annotations.
- Generalization drops from validation to the small held-out test partition.

## Ethical and privacy limitations

Pedestrian imagery can reveal appearance, location, behavior, and associations.
Penn-Fudan does not represent modern global camera populations. Dataset annotation
scope, demographic coverage, capture consent, and performance parity were not audited.
Any real deployment requires lawful purpose, data minimization, access controls,
retention limits, subgroup evaluation, human review, and a way for affected people to
challenge outcomes.

## Hardware and reproducibility

Training host: Intel Core i9-13900H, 31.71 GB RAM, NVIDIA RTX 4050 Laptop GPU with
6141 MiB VRAM, Windows, Python 3.11.9. Seeds are fixed, but CUDA ROI Align backward in
PyTorch 2.5.1 has no deterministic implementation; exact floating-point reproduction
across hardware is not guaranteed.

## Inference

Download the `v1.0.0` release checkpoint, then run:

```bash
python scripts/predict_image.py --config configs/improved.yaml \
  --checkpoint best.pth --input image.jpg --output prediction.jpg --threshold 0.45

python scripts/predict_video.py --config configs/improved.yaml \
  --checkpoint best.pth --input video.mp4 --output annotated.mp4 --threshold 0.45
```
