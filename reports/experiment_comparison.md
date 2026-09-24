# Experiment comparison

Both experiments use the same seed-42 train/validation split. The preferred model was selected using validation data only; the held-out test split was not consulted.

| Metric | Baseline | Improved | Delta |
|---|---:|---:|---:|
| mAP@0.50:0.95 | 0.7705 | 0.7913 | +0.0208 |
| mAP@0.50 | 0.9940 | 0.9768 | -0.0172 |
| Precision | 0.9559 | 0.9559 | +0.0000 |
| Recall | 0.9701 | 0.9701 | +0.0000 |
| F1 | 0.9630 | 0.9630 | +0.0000 |
| True positives | 65 | 65 | +0 |
| False positives | 3 | 3 | +0 |
| False negatives | 2 | 2 | +0 |

**Preferred experiment:** `improved`.

Selection rule: mAP@0.50:0.95, then mAP@0.50, then recall.
