# Reviewer Experiment Runbook

This runbook prepares and documents the experiments requested by the reviewers. The full reviewer batch was run on 2026-06-01. Large YOLO training run folders were written to `C:\BSF_reviewer_experiments\yolo_runs` to avoid filling the repository drive; compact CSV/JSON summaries were copied back into `reports/reviewer_experiments/`.

## 1. Build the GPU Docker Image

Run once if the image is not available or dependencies changed:

```powershell
.\scripts\run_reviewer_experiments_gpu.ps1 -BuildImage
```

By default this runs only non-training experiments after building the image.

## 2. Non-Training Experiments

These can be run with the existing multi-seed YOLO weights.

```powershell
.\scripts\run_reviewer_experiments_gpu.ps1
```

This executes:

1. `scripts/benchmark_yolo_postprocessing_sweep.py`
   - Sweeps confidence thresholds, NMS IoU thresholds, and agnostic NMS.
   - Selects the best setting on validation MAE.
   - Reports the selected test performance.
   - Output: `reports/reviewer_experiments/postprocessing/reviewer_experiments_postprocessing/`

2. `scripts/benchmark_yolo_robustness.py`
   - Evaluates existing weights under deterministic test-time perturbations.
   - Conditions: original, dark, bright, low contrast, high contrast, blur, noise, and synthetic occlusion.
   - Output: `reports/reviewer_experiments/robustness/reviewer_experiments_robustness/`

These two experiments answer reviewer concerns on post-processing and robustness without training new models.

## 3. Training Experiments

Run this only when you are ready to train or reproduce the completed reviewer batch:

```powershell
.\scripts\run_reviewer_experiments_gpu.ps1 -RunTraining
```

Default training cases:

| Case | Purpose |
| --- | --- |
| `default20` | Reproduces the original controlled 20-epoch detector benchmark. |
| `default50` | Tests whether the original 20-epoch budget underfit. |
| `minimal_aug20` | Tests sensitivity to reduced augmentation. |
| `robust_aug20` | Tests stronger color/geometric augmentation. |

Default models and seeds:

| Setting | Value |
| --- | --- |
| Models | `yolov8n.pt`, `yolo11n.pt` |
| Seeds | `0`, `1`, `2` |
| Batch | `16` |
| Image size | `640` |

To run all prepared training cases:

```powershell
.\scripts\run_reviewer_experiments_gpu.ps1 -RunTraining -TrainingCases default20,default50,minimal_aug20,robust_aug20
```

To train only YOLO11n for a faster first check:

```powershell
.\scripts\run_reviewer_experiments_gpu.ps1 -RunTraining -Models yolo11n.pt -TrainingCases default50,minimal_aug20
```

Training output:

```text
C:\BSF_reviewer_experiments\training\<benchmark_name>\
reports/reviewer_experiments/training/<benchmark_name>/   # compact copied summary
```

Important files:

| File | Use |
| --- | --- |
| `run_level_results.csv` | Per-case, per-model, per-seed metrics. |
| `aggregate_results.csv` | Mean/std table for manuscript. |
| `predictions.csv` | Image-level count predictions. |
| `summary.json` | Machine-readable summary for later table generation. |

## 4. Direct Python Commands Inside Docker

Post-processing only:

```powershell
docker run --rm --gpus all --shm-size=8g -v "${PWD}:/workspace" -w /workspace larvae-cv-gpu python scripts/benchmark_yolo_postprocessing_sweep.py --device cuda --include-agnostic-nms --benchmark-name postprocessing_review
```

Robustness only:

```powershell
docker run --rm --gpus all --shm-size=8g -v "${PWD}:/workspace" -w /workspace larvae-cv-gpu python scripts/benchmark_yolo_robustness.py --device cuda --benchmark-name robustness_review
```

Training ablation only:

```powershell
docker run --rm --gpus all --shm-size=8g -v "${PWD}:/workspace" -v "C:\BSF_reviewer_experiments:/reviewer_runs" -w /workspace larvae-cv-gpu python scripts/benchmark_yolo_reviewer_training.py --models yolov8n.pt yolo11n.pt --seeds 0 1 2 --cases default20 default50 minimal_aug20 robust_aug20 --train-device 0 --eval-device cuda --batch 16 --workers 2 --benchmark-name reviewer_training --run-project /reviewer_runs/yolo_runs --output-root /reviewer_runs/training
```

Statistical summary:

```powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace larvae-cv-gpu python scripts/analyze_reviewer_experiments.py
```

## 5. How to Use Results in the Revision

Use the outputs as follows:

| Reviewer point | File to cite |
| --- | --- |
| NMS/confidence/post-processing | `postprocessing/.../selected_results.csv` and `aggregate_selected_test.csv` |
| Robustness to illumination/occlusion | `robustness/.../aggregate_condition_results.csv` |
| Epoch justification | `training/.../aggregate_results.csv`, comparing `default50` with the existing 20-epoch result |
| Augmentation sensitivity | `training/.../aggregate_results.csv`, comparing `minimal_aug20` or `robust_aug20` with default augmentation |

If training results improve the paper, update:

1. `reports/paper_q4_prep/manuscript_draft.md`
2. `reports/paper_q4_prep/tables_for_manuscript_multiseed.md`
3. `reports/paper_q4_prep/response_to_reviewers.md`

Completed run outputs:

| Experiment | Output |
| --- | --- |
| Post-processing sweep | `reports/reviewer_experiments/postprocessing/reviewer_full_20260601_postprocessing/` |
| Robustness perturbations | `reports/reviewer_experiments/robustness/reviewer_full_20260601_robustness/` |
| Training ablations | `reports/reviewer_experiments/training/reviewer_full_20260601_training_c/` |
| Statistical summary | `reports/paper_q4_prep/reviewer_experiment_results_summary.md` |
| Statistical tests | `reports/paper_q4_prep/statistical_tests.csv` |

## 6. Suggested Minimal Run

For the strongest reviewer response with moderate compute:

```powershell
.\scripts\run_reviewer_experiments_gpu.ps1
.\scripts\run_reviewer_experiments_gpu.ps1 -RunTraining -Models yolo11n.pt -TrainingCases default50,minimal_aug20
```

This gives post-processing, robustness, longer-epoch, and augmentation-sensitivity evidence while avoiding a large model sweep.
