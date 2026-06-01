# bsfl-detection-counting-benchmark

This repository accompanies the manuscript **"Automatic Detection and Counting of Black Soldier Fly Larvae Using Classical Computer Vision and Multi-Seed Lightweight YOLO Evaluation"**. The paper-facing components are the BSFL detection/counting dataset pipeline, classical counting baselines, lightweight YOLO benchmarks, reviewer-requested post-processing and robustness experiments, and statistical analyses.

The repository also contains exploratory scripts for weight/length regression, size classification, and sex classification. Those exploratory tasks are retained for transparency but are **not** part of the manuscript's main detection/counting benchmark.

## Dataset

The image data are based on **BSF_Larvae_v1**, available from Zenodo:

<https://zenodo.org/records/13359376>

Download and extract the raw dataset into `data/raw/` before running the pipeline. Raw data, processed datasets, trained weights, and YOLO run folders are intentionally excluded from Git.

## Paper Workflow

Install the package and dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Prepare the detection/counting dataset and manifest:

```bash
python scripts/prepare_datasets.py
```

Run classical detection/counting baselines:

```bash
python scripts/benchmark_detection_counting_methods.py
```

Run the initial YOLO variant benchmark:

```bash
python scripts/benchmark_yolo_variants.py --models yolov8n.pt yolo11n.pt yolo11s.pt --epochs 20 --train-device 0 --eval-device cuda --workers 2 --batch 16
```

Run the multi-seed YOLO benchmark reported in the manuscript:

```bash
python scripts/benchmark_yolo_multiseed.py --models yolov8n.pt yolo11n.pt --seeds 0 1 2 --epochs 20 --train-device 0 --eval-device cuda --workers 2 --batch 16
```

Build the manuscript figures:

```bash
python scripts/build_ieee_split_figures.py
```

## Reviewer Experiments

The revision adds reviewer-requested experiments for post-processing, robustness, training length, augmentation sensitivity, and statistics.

Post-processing sweep using existing multi-seed YOLO weights:

```bash
python scripts/benchmark_yolo_postprocessing_sweep.py --device cuda --include-agnostic-nms --benchmark-name reviewer_postprocessing
```

Robustness sensitivity under brightness, contrast, blur, noise, and synthetic occlusion:

```bash
python scripts/benchmark_yolo_robustness.py --device cuda --benchmark-name reviewer_robustness
```

Training ablations for 20 epochs, 50 epochs, minimal augmentation, and stronger augmentation:

```bash
python scripts/benchmark_yolo_reviewer_training.py --models yolov8n.pt yolo11n.pt --seeds 0 1 2 --cases default20 default50 minimal_aug20 robust_aug20 --train-device 0 --eval-device cuda --batch 16 --workers 2
```

Aggregate reviewer experiments and statistical tests:

```bash
python scripts/analyze_reviewer_experiments.py
```

One-command GPU Docker launcher:

```powershell
.\scripts\run_reviewer_experiments_gpu.ps1 -RunTraining -BenchmarkName reviewer_full
```

By default, the launcher writes large reviewer training artifacts to `C:\BSF_reviewer_experiments` and compact summaries to `reports/reviewer_experiments/`.

## Main Paper Outputs

Revision-ready manuscript materials are in:

- `reports/paper_q4_prep/manuscript_draft.md`
- `reports/paper_q4_prep/response_to_reviewers.md`
- `reports/paper_q4_prep/tables_for_manuscript_multiseed.md`
- `reports/paper_q4_prep/reviewer_experiment_results_summary.md`
- `reports/paper_q4_prep/statistical_tests.csv`

## Repository Structure

- `scripts/`: dataset preparation, baseline benchmarks, YOLO training/evaluation, reviewer experiments, and figure generation.
- `src/larvae_cv/`: reusable project code.
- `reports/paper_q4_prep/`: manuscript tables, figure captions, response letter, and reviewer-experiment summaries.
- `data/raw/`: raw dataset location, ignored by Git.
- `data/processed/`: processed dataset location, ignored by Git.
- `reports/yolo_runs/`: YOLO training artifacts, ignored by Git.
- `reports/reviewer_experiments/`: reviewer experiment artifacts, ignored by Git.

## Docker

CPU container:

```bash
docker build -t larvae-cv .
docker run --rm -it -v ${PWD}:/workspace larvae-cv
```

GPU container:

```bash
docker build -f Dockerfile.gpu -t larvae-cv-gpu .
docker run --rm --gpus all --shm-size=8g -v ${PWD}:/workspace -w /workspace larvae-cv-gpu python scripts/benchmark_yolo_multiseed.py --models yolov8n.pt yolo11n.pt --seeds 0 1 2 --epochs 20 --train-device 0 --eval-device cuda --workers 2 --batch 16
```

## License

Code in this repository is released under the MIT License. See `LICENSE`.
