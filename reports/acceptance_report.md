# Acceptance report

Generated after the final local publication gate on 2026-09-24. Remote-only
checks are recorded after the first push; this report does not pre-claim them.

| Mandatory gate | Status | Evidence |
|---|---|---|
| Dataset downloader works | PASS | Official archive downloaded over HTTPS, extracted atomically, and validated as 170 paired images/masks. |
| Deterministic train/validation/test split | PASS | Seed-42 JSON and CSV manifests; SHA-256 `631066345ec7961628fba7d13948b860c3f37ee54e93cd106bae5331c5c7677e`. |
| No split overlap | PASS | Automated leakage test passed; 119/25/26 unique images. |
| PyTorch detector runs | PASS | Real Faster R-CNN train and inference forwards completed on CUDA; CPU install smoke also passed. |
| Model actually trained | PASS | Baseline and improved runs each completed six epochs from COCO transfer weights. |
| Validation evaluation | PASS | COCO mAP plus thresholded precision/recall/TP/FP/FN generated for both experiments. |
| Real failure analysis | PASS | CSV/JSON failure records and annotated galleries generated from validation predictions. |
| Second experiment | PASS | Controlled 640-960 resize experiment completed after the rationale was frozen. |
| Experiment comparison | PASS | `reports/experiment_comparison.md`. |
| Final held-out test | PASS | Test split evaluated once after selection at validation-selected threshold 0.45. |
| Actual metrics recorded | PASS | Test mAP@0.50:0.95 0.6850; mAP@0.50 0.8937; precision 0.9219; recall 0.8939. |
| OpenCV video inference | PASS | Corrupt-input tests plus real 10-frame selected-checkpoint output smoke. |
| MLflow exercised | PASS | Baseline run `fbdbeccc1f55403aada2cf94d696685c`; improved run `e83256649d384c30b7c9f661a140756b`. |
| Tests pass | PASS | 57 passed locally. |
| Ruff passes | PASS | `ruff check .` returned no violations. |
| Docker image builds | PASS | Local image `podfa:local` built successfully. |
| Docker smoke passes | PASS | Default detector inference smoke passed; 57 tests passed inside the image. |
| README and architecture diagram | PASS | Complete README with Mermaid lifecycle diagram and reproduction commands. |
| Model card | PASS | `MODEL_CARD.md`. |
| Resume evidence | PASS | `RESUME_EVIDENCE.md`, limited to measured claims. |
| Interview notes | PASS | `INTERVIEW_NOTES.md`, grounded in this repository. |
| No invented metrics | PASS | Documentation values trace to machine-readable run artifacts. |
| No secrets | PASS | Publishable-file credential/email/path pattern scan returned no findings. |
| No raw dataset | PASS | `data/raw/` is ignored; Git candidate inventory contains no raw images/masks. |
| No accidental personal files | PASS | Candidate inventory contains only project source, tests, configuration, evidence, and documentation. |

## Remote verification

Pending the deliberate first push: GitHub Actions, release assets, repository
accessibility, and GitHub Pages. These do not weaken the completed local gate;
their verified URLs and statuses will replace this paragraph after publication.
