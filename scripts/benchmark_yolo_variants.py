from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from larvae_cv.counting_benchmarks import evaluate_yolo_counting_weights
from larvae_cv.data import read_manifest_csv, remap_path_columns, write_json, write_runtime_detection_yaml
from larvae_cv.paths import PROCESSED_DIR, REPORTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare multiple YOLO variants on larvae detection/counting.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo11n.pt", "yolo11s.pt"],
        help="Ultralytics model checkpoints to benchmark.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs per model.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--train-device", default="0", help="Training device, for example 0 or cpu.")
    parser.add_argument("--eval-device", default="cuda", help="Evaluation device, for example cuda or cpu.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers.")
    parser.add_argument(
        "--benchmark-name",
        default="",
        help="Output folder name under reports/yolo_benchmark. Default uses a timestamp.",
    )
    return parser.parse_args()


def slugify_model_name(model_name: str) -> str:
    stem = Path(model_name).stem.lower()
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_")


def _metric_value(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if value is None:
        return float("nan")
    return float(value)


def train_variant(
    model_name: str,
    dataset_yaml: Path,
    run_name: str,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "ultralytics belum terpasang. Jalankan `pip install -r requirements.txt` atau gunakan Docker GPU image."
        ) from exc

    model = YOLO(model_name)
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.train_device,
        batch=args.batch,
        workers=args.workers,
        project=str(REPORTS_DIR / "yolo_runs"),
        name=run_name,
        exist_ok=True,
    )
    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = save_dir / "weights" / "last.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"YOLO weights tidak ditemukan setelah training: {save_dir}")
    return save_dir, best_weights


def load_training_history(run_dir: Path) -> pd.DataFrame:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return pd.DataFrame()
    return pd.read_csv(results_csv)


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


def flatten_variant_summary(variant_result: dict[str, Any]) -> dict[str, Any]:
    test_detection = variant_result["summary"].get("test_detection_metrics", {})
    test_counting = variant_result["summary"].get("test", {})
    val_counting = variant_result["summary"].get("val", {})
    row = {
        "variant": variant_result["variant"],
        "model_name": variant_result["model_name"],
        "run_name": variant_result["run_name"],
        "run_dir": str(variant_result["run_dir"].resolve()),
        "weights_path": str(variant_result["weights_path"].resolve()),
        "best_confidence_for_counting": float(variant_result["summary"]["best_confidence_for_counting"]),
        "test_detection_precision": _metric_value(test_detection, "metrics/precision(B)"),
        "test_detection_recall": _metric_value(test_detection, "metrics/recall(B)"),
        "test_detection_mAP50": _metric_value(test_detection, "metrics/mAP50(B)"),
        "test_detection_mAP50_95": _metric_value(test_detection, "metrics/mAP50-95(B)"),
        "test_count_mae": float(test_counting["mae"]),
        "test_count_rmse": float(test_counting["rmse"]),
        "test_count_r2": float(test_counting["r2"]),
        "test_count_exact_match_rate": float(test_counting["exact_match_rate"]),
        "val_count_mae": float(val_counting["mae"]),
        "val_count_rmse": float(val_counting["rmse"]),
        "val_count_r2": float(val_counting["r2"]),
    }
    row.update(variant_result["history_summary"])
    return row


def plot_metric_comparison(frame: pd.DataFrame, columns: list[str], title: str, output_path: Path, ylabel: str) -> None:
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4.8))
    if len(columns) == 1:
        axes = [axes]

    for axis, column in zip(axes, columns):
        order = frame.sort_values(column, ascending=False if "r2" in column.lower() or "map" in column.lower() or "precision" in column.lower() or "recall" in column.lower() else True)
        axis.bar(order["variant"], order[column], color="#2f6fed")
        axis.set_title(column.replace("_", "\n"))
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=20)
        for idx, value in enumerate(order[column]):
            axis.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_curve(histories: dict[str, pd.DataFrame], column: str, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant, history in histories.items():
        if history.empty or column not in history.columns:
            continue
        ax.plot(history["epoch"], history[column], marker="o", linewidth=1.8, label=variant)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(column)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_count_scatter(predictions: pd.DataFrame, variant: str, output_path: Path) -> None:
    frame = predictions.loc[predictions["split"] == "test"].copy()
    if frame.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(frame["y_true"], frame["y_pred"], alpha=0.75, color="#2f6fed")
    low = float(min(frame["y_true"].min(), frame["y_pred"].min()))
    high = float(max(frame["y_true"].max(), frame["y_pred"].max()))
    ax.plot([low, high], [low, high], linestyle="--", color="#ef4444", linewidth=1.5)
    ax.set_title(f"Test Count Scatter - {variant}")
    ax.set_xlabel("True count")
    ax.set_ylabel("Predicted count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_count_error_hist(predictions: pd.DataFrame, variant: str, output_path: Path) -> None:
    frame = predictions.loc[predictions["split"] == "test"].copy()
    if frame.empty:
        return

    errors = frame["y_pred"] - frame["y_true"]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.hist(errors, bins=16, color="#2f6fed", alpha=0.85, edgecolor="white")
    ax.axvline(0.0, linestyle="--", color="#ef4444", linewidth=1.5)
    ax.set_title(f"Test Counting Error Histogram - {variant}")
    ax.set_xlabel("Prediction error")
    ax.set_ylabel("Frequency")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    benchmark_name = args.benchmark_name or f"yolo_benchmark_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir = REPORTS_DIR / "yolo_benchmark" / benchmark_name
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset_yaml = write_runtime_detection_yaml()
    manifest = read_manifest_csv(PROCESSED_DIR / "detection_counting_manifest.csv")
    manifest = remap_path_columns(manifest, ["source_image_path", "source_label_path"])

    all_predictions: list[pd.DataFrame] = []
    variant_results: list[dict[str, Any]] = []
    histories: dict[str, pd.DataFrame] = {}

    for model_name in args.models:
        variant = slugify_model_name(model_name)
        run_name = f"{benchmark_name}_{variant}"
        print(f"[train] {model_name} -> {run_name}")
        run_dir, weights_path = train_variant(model_name, dataset_yaml, run_name, args)
        print(f"[eval] {weights_path}")
        summary, predictions = evaluate_yolo_counting_weights(
            manifest=manifest,
            weights_path=weights_path,
            device=args.eval_device,
            method_name=variant,
        )
        history = load_training_history(run_dir)
        histories[variant] = history

        predictions = predictions.copy()
        predictions["variant"] = variant
        predictions["model_name"] = model_name
        predictions["run_name"] = run_name
        all_predictions.append(predictions)

        variant_results.append(
            {
                "variant": variant,
                "model_name": model_name,
                "run_name": run_name,
                "run_dir": run_dir,
                "weights_path": weights_path,
                "history_summary": history_summary(history),
                "summary": summary,
            }
        )

        plot_count_scatter(predictions, variant, plots_dir / f"{variant}_test_count_scatter.png")
        plot_count_error_hist(predictions, variant, plots_dir / f"{variant}_test_count_error_hist.png")

    comparison = pd.DataFrame([flatten_variant_summary(item) for item in variant_results])
    comparison = comparison.sort_values(["test_count_mae", "test_detection_mAP50"], ascending=[True, False]).reset_index(drop=True)
    predictions_frame = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    comparison.to_csv(output_dir / "comparison.csv", index=False)
    if not predictions_frame.empty:
        predictions_frame.to_csv(output_dir / "predictions.csv", index=False)

    plot_metric_comparison(
        comparison,
        ["test_detection_precision", "test_detection_recall", "test_detection_mAP50", "test_detection_mAP50_95"],
        title="YOLO Detection Metrics Comparison",
        output_path=plots_dir / "detection_metrics_comparison.png",
        ylabel="Score",
    )
    plot_metric_comparison(
        comparison,
        ["test_count_mae", "test_count_rmse", "test_count_r2", "test_count_exact_match_rate"],
        title="YOLO Counting Metrics Comparison",
        output_path=plots_dir / "counting_metrics_comparison.png",
        ylabel="Value",
    )
    plot_training_curve(
        histories,
        column="metrics/mAP50(B)",
        title="Validation mAP50 by Epoch",
        output_path=plots_dir / "training_curve_map50.png",
    )
    plot_training_curve(
        histories,
        column="metrics/mAP50-95(B)",
        title="Validation mAP50-95 by Epoch",
        output_path=plots_dir / "training_curve_map50_95.png",
    )
    plot_training_curve(
        histories,
        column="val/box_loss",
        title="Validation Box Loss by Epoch",
        output_path=plots_dir / "training_curve_val_box_loss.png",
    )

    ranking_by_count = comparison.sort_values(["test_count_mae", "test_count_rmse", "test_count_r2"], ascending=[True, True, False])
    ranking_by_detection = comparison.sort_values(["test_detection_mAP50", "test_detection_mAP50_95"], ascending=[False, False])

    summary = {
        "benchmark_name": benchmark_name,
        "models": args.models,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "train_device": args.train_device,
        "eval_device": args.eval_device,
        "batch": args.batch,
        "workers": args.workers,
        "ranking_by_counting": ranking_by_count.to_dict(orient="records"),
        "ranking_by_detection": ranking_by_detection.to_dict(orient="records"),
        "variants": [
            {
                "variant": item["variant"],
                "model_name": item["model_name"],
                "run_name": item["run_name"],
                "run_dir": str(item["run_dir"].resolve()),
                "weights_path": str(item["weights_path"].resolve()),
                "history_summary": item["history_summary"],
                "summary": item["summary"],
            }
            for item in variant_results
        ],
        "plots": {
            "detection_metrics_comparison": str((plots_dir / "detection_metrics_comparison.png").resolve()),
            "counting_metrics_comparison": str((plots_dir / "counting_metrics_comparison.png").resolve()),
            "training_curve_map50": str((plots_dir / "training_curve_map50.png").resolve()),
            "training_curve_map50_95": str((plots_dir / "training_curve_map50_95.png").resolve()),
            "training_curve_val_box_loss": str((plots_dir / "training_curve_val_box_loss.png").resolve()),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(comparison.to_string(index=False))
    print(f"Saved benchmark summary to {output_dir}")


if __name__ == "__main__":
    main()
