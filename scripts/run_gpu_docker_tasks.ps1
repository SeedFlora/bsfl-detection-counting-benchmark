$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$image = "larvae-cv-gpu"

docker build -f "$root\\Dockerfile.gpu" -t $image $root

docker run --rm --gpus all `
  -v "${root}:/workspace" `
  -w /workspace `
  $image python scripts/train_regression_gpu_torch.py --epochs 12 --batch-size 32 --device cuda

docker run --rm --gpus all `
  -v "${root}:/workspace" `
  -w /workspace `
  $image python scripts/train_detection_yolo.py --epochs 20 --device 0 --run-name larvae_counting_gpu

docker run --rm --gpus all `
  -v "${root}:/workspace" `
  -w /workspace `
  $image python scripts/evaluate_yolo_counting_weights.py --weights reports/yolo_runs/larvae_counting_gpu/weights/best.pt --device cuda

docker run --rm --gpus all --shm-size=8g `
  -v "${root}:/workspace" `
  -w /workspace `
  $image python scripts/benchmark_yolo_variants.py --models yolo11n.pt yolo11s.pt --epochs 20 --train-device 0 --eval-device cuda --workers 2 --batch 16
