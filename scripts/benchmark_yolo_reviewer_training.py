from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from larvae_cv.counting_benchmarks import evaluate_yolo_counting_weights
from larvae_cv.data import read_manifest_csv, remap_path_columns, write_json, write_runtime_detection_yaml
from larvae_cv.paths import PROCESSED_DIR, REPORTS_DIR


TRAINING_CASES: dict[str, dict[str, Any]] = {
    "default20": {
        "epochs": 20,
        "description": "Default Ultralytics augmentation for the original 20-epoch benchmark.",
        "overrides": {},
    },
    "default50": {
        "epochs": 50,
        "description": "Longer default-augmentation training to test whether 20 epochs underfit.",
        "overrides": {},
    },
    "minimal_aug20": {
        "epochs": 20,
        "description": "Reduced augmentation sensitivity check.",
        "overrides": {
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "cutmix": 0.0,
            "copy_paste": 0.0,
            "erasing": 0.0,
        },
    },
    "robust_aug20": {
        "epochs": 20,
        "description": "Stronger illumination/geometric augmentation for field-robustness probing.",
        "overrides": {
            "hsv_h": 0.02,
            "hsv_s": 0.80,
            "hsv_v": 0.55,
            "degrees": 10.0,
            "translate": 0.15,
            "scale": 0.60,
            "shear": 2.0,
            "perspective": 0.0005,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.05,
            "erasing": 0.4,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reviewer-oriented YOLO training ablations for epoch and augmentation sensitivity."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolov8n.pt", "yolo11n.pt"],
        help="Ultralytics model checkpoints to train.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Random seeds for repeated training.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["default20", "default50", "minimal_aug20", "robust_aug20"],
        choices=sorted(TRAINING_CASES),
        help="Training cases to run.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--train-device", default="0", help="Training device, for example 0 or cpu.")
    parser.add_argument("--eval-device", default="cuda", help="Evaluation device, for example cuda or cpu.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers.")
    parser.add_argument(
        "--benchmark-name",
        default="",
        help="Output folder under reports/reviewer_experiments/training. Default uses a timestamp.",
    )
    parser.add_argument(
        "--run-project",
        default=str(REPORTS_DIR / "yolo_runs"),
        help="Directory for Ultralytics training run folders. Use a large external mount when reports/ is space-limited.",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPORTS_DIR / "reviewer_experiments" / "training"),
        help="Directory that will contain the benchmark output folder.",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Ultralytics plot image generation during training. Disabled by default to reduce disk I/O.",
    )
    parser.add_argument(
        "--reuse-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse matching run directories when best.pt or last.pt already exists.",
    )
    return parser.parse_args()


def slugify_model_name(model_name: str) -> str:
    stem = Path(model_name).stem.lower()
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_")


def metric_or_nan(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if value is None:
        return float("nan")
    return float(value)


def load_training_history(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def history_summary(history: pd.DataFrame) -> dict[str, Any]:
    if history.empty:
        return {}
    best_idx = history["metrics/mAP50(B)"].astype(float).idxmax()
    best_row = history.loc[best_idx]
    last_row = history.iloc[-1]
    return {
        "epoch_count": int(len(history)),
        "best_val_epoch": int(best_row["epoch"]),
        "best_val_precision": float(best_row["metrics/precision(B)"]),
        "best_val_recall": float(best_row["metrics/recall(B)"]),
        "best_val_mAP50": float(best_row["metrics/mAP50(B)"]),
        "best_val_mAP50_95": float(best_row["metrics/mAP50-95(B)"]),
        "last_epoch": int(last_row["epoch"]),
        "last_val_mAP50": float(last_row["metrics/mAP50(B)"]),
        "last_val_mAP50_95": float(last_row["metrics/mAP50-95(B)"]),
    }


def train_or_reuse(
    model_name: str,
    dataset_yaml: Path,
    run_name: str,
    seed: int,
    case: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    from ultralytics import YOLO

    run_project = Path(args.run_project)
    run_dir = run_project / run_name
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"
    if args.reuse_existing:
        if best_weights.exists():
            return run_dir, best_weights
        if last_weights.exists():
            return run_dir, last_weights

    train_kwargs = {
        "data": str(dataset_yaml),
        "epochs": int(case["epochs"]),
        "imgsz": args.imgsz,
        "device": args.train_device,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(run_project),
        "name": run_name,
        "exist_ok": True,
        "seed": seed,
        "deterministic": True,
        "plots": args.plots,
    }
    train_kwargs.update(case["overrides"])

    model = YOLO(model_name)
    model.train(**train_kwargs)
    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = save_dir / "weights" / "last.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"YOLO weights not found after training: {save_dir}")
    return save_dir, best_weights


def aggregate_mean_std(frame: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "min", "max"])
    grouped.columns = ["_".join([col, stat]) for col, stat in grouped.columns]
    return grouped.reset_index()


def main() -> None:
    args = parse_args()
    benchmark_name = args.benchmark_name or f"reviewer_training_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir = Path(args.output_root) / benchmark_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_yaml = write_runtime_detection_yaml()
    manifest = read_manifest_csv(PROCESSED_DIR / "detection_counting_manifest.csv")
    manifest = remap_path_columns(manifest, ["source_image_path", "source_label_path"])

    run_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for case_name in args.cases:
        case = TRAINING_CASES[case_name]
        for model_name in args.models:
            variant = slugify_model_name(model_name)
            for seed in args.seeds:
                run_name = f"{benchmark_name}_{case_name}_{variant}_seed{seed}"
                print(f"[train] case={case_name} model={model_name} seed={seed} -> {run_name}")
                run_dir, weights_path = train_or_reuse(model_name, dataset_yaml, run_name, seed, case, args)
                print(f"[eval] {weights_path}")
                summary, predictions = evaluate_yolo_counting_weights(
                    manifest=manifest,
                    weights_path=weights_path,
                    device=args.eval_device,
                    method_name=f"{case_name}_{variant}_seed{seed}",
                )
                history = load_training_history(run_dir)
                hist_summary = history_summary(history)
                test_detection = summary.get("test_detection_metrics", {})
                test_count = summary["test"]
                val_count = summary["val"]

                row = {
                    "case": case_name,
                    "case_description": case["description"],
                    "variant": variant,
                    "model_name": model_name,
                    "seed": seed,
                    "run_name": run_name,
                    "run_dir": str(run_dir.resolve()),
                    "weights_path": str(weights_path.resolve()),
                    "epochs_configured": int(case["epochs"]),
                    "best_confidence_for_counting": float(summary["best_confidence_for_counting"]),
                    "test_detection_precision": metric_or_nan(test_detection, "metrics/precision(B)"),
                    "test_detection_recall": metric_or_nan(test_detection, "metrics/recall(B)"),
                    "test_detection_mAP50": metric_or_nan(test_detection, "metrics/mAP50(B)"),
                    "test_detection_mAP50_95": metric_or_nan(test_detection, "metrics/mAP50-95(B)"),
                    "test_count_mae": float(test_count["mae"]),
                    "test_count_rmse": float(test_count["rmse"]),
                    "test_count_r2": float(test_count["r2"]),
                    "test_count_exact_match_rate": float(test_count["exact_match_rate"]),
                    "val_count_mae": float(val_count["mae"]),
                    "val_count_rmse": float(val_count["rmse"]),
                    "val_count_r2": float(val_count["r2"]),
                }
                row.update(hist_summary)
                run_rows.append(row)

                predictions = predictions.copy()
                predictions["case"] = case_name
                predictions["variant"] = variant
                predictions["model_name"] = model_name
                predictions["seed"] = seed
                predictions["run_name"] = run_name
                predictions["weights_path"] = str(weights_path.resolve())
                prediction_frames.append(predictions)

    run_frame = pd.DataFrame(run_rows).sort_values(["case", "variant", "seed"]).reset_index(drop=True)
    predictions_frame = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    aggregate = aggregate_mean_std(
        run_frame,
        ["case", "variant", "model_name"],
        [
            "test_detection_precision",
            "test_detection_recall",
            "test_detection_mAP50",
            "test_detection_mAP50_95",
            "test_count_mae",
            "test_count_rmse",
            "test_count_r2",
            "test_count_exact_match_rate",
            "val_count_mae",
            "val_count_r2",
        ],
    ).sort_values(["test_count_mae_mean", "test_detection_mAP50_mean"], ascending=[True, False])

    run_frame.to_csv(output_dir / "run_level_results.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_results.csv", index=False)
    if not predictions_frame.empty:
        predictions_frame.to_csv(output_dir / "predictions.csv", index=False)

    summary = {
        "benchmark_name": benchmark_name,
        "models": args.models,
        "seeds": args.seeds,
        "cases": {case_name: TRAINING_CASES[case_name] for case_name in args.cases},
        "imgsz": args.imgsz,
        "train_device": args.train_device,
        "eval_device": args.eval_device,
        "batch": args.batch,
        "workers": args.workers,
        "aggregate": aggregate.to_dict(orient="records"),
        "outputs": {
            "run_level_results": str((output_dir / "run_level_results.csv").resolve()),
            "aggregate_results": str((output_dir / "aggregate_results.csv").resolve()),
            "predictions": str((output_dir / "predictions.csv").resolve()),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(aggregate.to_string(index=False))
    print(f"Saved reviewer training benchmark to {output_dir}")


if __name__ == "__main__":
    main()
