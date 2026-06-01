param(
  [switch]$BuildImage,
  [switch]$RunTraining,
  [string]$Image = "larvae-cv-gpu",
  [string]$TrainDevice = "0",
  [string]$EvalDevice = "cuda",
  [string]$BenchmarkName = "reviewer_experiments",
  [string[]]$TrainingCases = @("default20", "default50", "minimal_aug20", "robust_aug20"),
  [string[]]$Models = @("yolov8n.pt", "yolo11n.pt"),
  [int[]]$Seeds = @(0, 1, 2),
  [int]$Batch = 16,
  [int]$Workers = 2,
  [string]$HostTrainingRoot = "C:\BSF_reviewer_experiments",
  [string]$DockerTrainingRoot = "/reviewer_runs",
  [switch]$EnableTrainingPlots
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$TrainingCases = @($TrainingCases | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$Models = @($Models | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$Seeds = @($Seeds | ForEach-Object { $_ -split "," } | ForEach-Object { [int]$_ })

if ($BuildImage) {
  docker build -f "$root\Dockerfile.gpu" -t $Image $root
}

Write-Host "Running post-processing sweep with existing weights..."
docker run --rm --gpus all --shm-size=8g `
  -v "${root}:/workspace" `
  -w /workspace `
  $Image python scripts/benchmark_yolo_postprocessing_sweep.py `
    --device $EvalDevice `
    --benchmark-name "${BenchmarkName}_postprocessing" `
    --include-agnostic-nms

Write-Host "Running robustness inference with existing weights..."
docker run --rm --gpus all --shm-size=8g `
  -v "${root}:/workspace" `
  -w /workspace `
  $Image python scripts/benchmark_yolo_robustness.py `
    --device $EvalDevice `
    --benchmark-name "${BenchmarkName}_robustness"

if ($RunTraining) {
  Write-Host "Running reviewer training ablations..."
  New-Item -ItemType Directory -Force -Path $HostTrainingRoot | Out-Null
  $plotArgs = @()
  if ($EnableTrainingPlots) {
    $plotArgs += "--plots"
  }
  docker run --rm --gpus all --shm-size=8g `
    -v "${root}:/workspace" `
    -v "${HostTrainingRoot}:${DockerTrainingRoot}" `
    -w /workspace `
    $Image python scripts/benchmark_yolo_reviewer_training.py `
      --models $Models `
      --seeds $Seeds `
      --cases $TrainingCases `
      --train-device $TrainDevice `
      --eval-device $EvalDevice `
      --batch $Batch `
      --workers $Workers `
      --benchmark-name "${BenchmarkName}_training" `
      --run-project "${DockerTrainingRoot}/yolo_runs" `
      --output-root "${DockerTrainingRoot}/training" `
      $plotArgs
} else {
  Write-Host "Training ablations were skipped. Add -RunTraining when you are ready to train."
}
