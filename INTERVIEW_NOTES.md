# Interview notes

## Why Faster R-CNN MobileNetV3 Large FPN?

Penn-Fudan is small and bounding-box accuracy matters more than maximum throughput.
Faster R-CNN is a mature two-stage detector with interpretable classification,
regression, proposal, and NMS behavior. MobileNetV3 keeps two complete experiments
practical on a 6 GB laptop GPU; FPN preserves multi-scale features needed for people
at different apparent sizes. I did not claim an SSDLite comparison because I did not
run one.

## Transfer learning

The detector started from Torchvision COCO weights, which already encode useful edge,
texture, shape, and person representations. I replaced the COCO box classifier with a
two-output head (background/person) and fine-tuned the detector. This reduces the data
and compute needed relative to random initialization; the download-free smoke tests
use random weights only to verify code paths.

## Backbone and FPN

MobileNetV3 is the convolutional feature extractor. A Feature Pyramid Network combines
deep semantic features with higher-resolution earlier features, producing multiple
scales for the Region Proposal Network and ROI heads. That is relevant to Penn-Fudan
because pedestrians vary in distance and size.

## Bounding-box regression and classification loss

The ROI head classifies each proposal as background/person and regresses offsets from
the proposal to a tighter box. Torchvision exposes `loss_classifier` and
`loss_box_reg`; the Region Proposal Network adds objectness and proposal box-regression
losses. The trainer logs each component and their sum per epoch.

## IoU and matching

Intersection over Union is intersection area divided by union area. The operating
metrics sort predictions by score and greedily match each to one unused ground-truth
box at IoU >= 0.50. This prevents duplicate boxes from becoming multiple true
positives. COCO mAP separately evaluates thresholds 0.50:0.95.

## Non-maximum suppression

NMS removes lower-scoring boxes that overlap a higher-scoring box above a configured
IoU. It operates before the repository's evaluator. NMS can suppress a real person in
a crowded group if two people overlap strongly, while a loose setting can leave
duplicates.

## mAP

Average precision summarizes the precision-recall curve for one IoU threshold; mean
AP averages classes (one foreground class here) and, for COCO mAP@0.50:0.95, ten IoU
thresholds. The final test mAP@0.50 is 0.8937, while stricter mAP@0.50:0.95 is 0.6850,
showing that localization quality is materially harder than finding a roughly correct
person box.

## Precision vs recall and confidence thresholds

Precision asks what fraction of emitted detections are correct; recall asks what
fraction of labeled people were detected. Lowering the confidence threshold usually
raises recall and lowers precision. I selected the threshold only on validation by
maximum F1, then recall, precision, and threshold. Experiment 2 selected 0.45; the
test split was not searched.

## Data leakage

The seed-42 manifest stores image IDs and SHA-256 hashes. Tests reject overlap by ID or
content hash. Training reads only the train partition; checkpoints and thresholds use
validation; the final test is evaluated once after selection. Images from validation
were never copied into training.

## Augmentation and overfitting

Both runs use paired random horizontal flips, updating images, boxes, and masks
together without mutating caller-owned targets. With 119 training images, overfitting
is a serious risk: loss kept falling while validation mAP did not improve monotonically.
Best-checkpoint selection limits, but does not eliminate, that risk.

## False positives and false negatives

The baseline at threshold 0.30 had 65 TP, 3 FP, and 2 FN on validation. The two FNs
also had geometrically correct low-score boxes on occluded pedestrians. The three FP
records overlapped labeled boxes at IoU 0.10–0.25, but visual inspection showed some
boxes corresponded to partial or unlabeled people. That is why I describe both model
and annotation limitations instead of treating every error label as ground truth.

## Why Experiment 2?

The two actionable misses were occluded people in dense groups, with correct boxes at
scores 0.069 and 0.066. I changed only the resize range from 512–768 to 640–960 to
preserve more spatial detail. Validation mAP@0.50:0.95 improved 0.7705 → 0.7913, while
mAP@0.50 fell 0.9940 → 0.9768 and operating counts were unchanged. The result supports
selecting the improved checkpoint under the prespecified metric, but not claiming that
all failure modes improved.

## Limitations

The dataset is small, single-class, geographically narrow, old, and inconsistently
complete for distant/occluded people. A 25-image validation and 26-image test split
produce noisy estimates. Only one architecture and one seed were fully trained. CUDA
ROI Align backward is not fully deterministic. No calibration, subgroup fairness, or
real CCTV domain-shift study was performed.

## With 10× more data

I would stratify by site, sequence, person count, scale, and occlusion; preserve groups
to avoid adjacent-frame leakage; create explicit ignore regions; run multiple seeds;
calibrate confidence; report size/occlusion slices; and test a stronger backbone plus
a one-stage latency baseline. I would version labels and dataset checksums in a data
registry rather than keep one local archive.

## For real-time CCTV/video

I would benchmark end-to-end decode, preprocess, inference, NMS, rendering, and encode
latency on target hardware. Likely changes include SSDLite/YOLO/RT-DETR-style one-stage
models, lower resolution with accuracy slices, batching across streams, mixed precision,
TensorRT/ONNX export, asynchronous decode, frame sampling, tracking-by-detection, and
backpressure. Privacy and retention controls would be designed before deployment.

## Scaling the pipeline

Move immutable data/manifests to versioned object storage, run containerized jobs on a
scheduler, log to a shared MLflow-compatible service, use a model registry with approval
gates, parallelize hyperparameter trials without sharing test data, add automated slice
reports, and promote artifacts by digest. CI remains a smoke gate; expensive training
runs belong in a separate GPU workflow with budgets and cancellation.

## Zero-shot/open-vocabulary detection

This detector has a fixed learned label space: background/person. An open-vocabulary
detector connects image features to language embeddings and can score text prompts it
was not specifically fine-tuned on. That flexibility can help exploration, but it does
not mean prompt scores are calibrated or that a model understands a deployment's label
policy. No zero-shot comparison was run in this repository.

## Ten questions to rehearse

1. How did you prevent train/validation/test leakage?
2. Why did you select Faster R-CNN MobileNetV3 FPN for this dataset and hardware?
3. How do mAP@0.50 and mAP@0.50:0.95 differ, and what do your numbers imply?
4. How does your greedy matching distinguish TP, duplicate, localization FP, and FN?
5. Why was the confidence threshold 0.45, and why must it come from validation?
6. What evidence motivated the resolution experiment, and what regressed?
7. Why did validation and test performance differ?
8. How do the checkpoint and MLflow designs support reproducibility and resume?
9. What changes would you make for real-time multi-camera inference?
10. Which dataset and ethics limitations would block production deployment?
