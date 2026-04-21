from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

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
    parser = argparse.ArgumentParser(description="Multi-seed YOLO benchmark with density analysis and qualitative figures.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolov8n.pt", "yolo11n.pt"],
        help="Ultralytics model checkpoints to benchmark.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Random seeds for repeated runs.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--train-device", default="0", help="Training device, for example 0 or cpu.")
    parser.add_argument("--eval-device", default="cuda", help="Evaluation device, for example cuda or cpu.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers.")
    parser.add_argument(
        "--benchmark-name",
        default="",
        help="Output folder name under reports/yolo_multiseed. Default uses a timestamp.",
    )
    parser.add_argument(
        "--reuse-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing run directories when matching weights already exist.",
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
    return {
        "epoch_count": int(len(history)),
        "best_val_epoch": int(best_row["epoch"]),
        "best_val_precision": float(best_row["metrics/precision(B)"]),
        "best_val_recall": float(best_row["metrics/recall(B)"]),
        "best_val_mAP50": float(best_row["metrics/mAP50(B)"]),
        "best_val_mAP50_95": float(best_row["metrics/mAP50-95(B)"]),
    }


def count_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "exact_match_rate": float(np.mean(np.rint(y_true) == np.rint(y_pred))),
        "mean_true": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_pred)),
    }


def aggregate_mean_std(frame: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "min", "max"])
    grouped.columns = ["_".join([col, stat]) for col, stat in grouped.columns]
    return grouped.reset_index()


def choose_density_bins(test_manifest: pd.DataFrame) -> pd.DataFrame:
    frame = test_manifest[["image_name", "object_count"]].copy()
    ranked = frame["object_count"].rank(method="first")
    frame["density_bin"] = pd.qcut(ranked, q=3, labels=["low", "medium", "high"])
    return frame


def train_or_reuse(
    model_name: str,
    dataset_yaml: Path,
    run_name: str,
    args: argparse.Namespace,
    seed: int,
) -> tuple[Path, Path]:
    from ultralytics import YOLO

    run_dir = REPORTS_DIR / "yolo_runs" / run_name
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"
    if args.reuse_existing:
        if best_weights.exists():
            return run_dir, best_weights
        if last_weights.exists():
            return run_dir, last_weights

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
        seed=seed,
        deterministic=True,
    )
    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = save_dir / "weights" / "last.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"YOLO weights tidak ditemukan setelah training: {save_dir}")
    return save_dir, best_weights


def plot_aggregate_metric(
    frame: pd.DataFrame,
    metric_prefix: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.bar(
        frame["variant"],
        frame[f"{metric_prefix}_mean"],
        yerr=frame[f"{metric_prefix}_std"].fillna(0.0),
        capsize=6,
        color="#2f6fed",
        alpha=0.9,
    )
    for idx, value in enumerate(frame[f"{metric_prefix}_mean"]):
        ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_density_metric(density_summary: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    order = ["low", "medium", "high"]
    pivot_mean = density_summary.pivot(index="density_bin", columns="variant", values=f"{metric}_mean").reindex(order)
    pivot_std = density_summary.pivot(index="density_bin", columns="variant", values=f"{metric}_std").reindex(order).fillna(0.0)

    bins = list(pivot_mean.index)
    variants = list(pivot_mean.columns)
    x = np.arange(len(bins))
    width = 0.8 / max(1, len(variants))

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for idx, variant in enumerate(variants):
        offset = (idx - (len(variants) - 1) / 2) * width
        values = pivot_mean[variant].to_numpy(dtype=float)
        errors = pivot_std[variant].to_numpy(dtype=float)
        ax.bar(x + offset, values, width=width, yerr=errors, capsize=4, label=variant)

    ax.set_xticks(x)
    ax.set_xticklabels([str(item).title() for item in bins])
    ax.set_title(title)
    ax.set_ylabel(metric.upper() if metric != "mae" else "MAE")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def select_success_examples(frame: pd.DataFrame, count: int = 3) -> pd.DataFrame:
    chosen: list[pd.Series] = []
    used_names: set[str] = set()
    for density in ["low", "medium", "high"]:
        candidates = frame.loc[frame["density_bin"] == density].sort_values(["abs_error", "image_name"])
        if not candidates.empty:
            row = candidates.iloc[0]
            chosen.append(row)
            used_names.add(str(row["image_name"]))
    if len(chosen) < count:
        remaining = frame.loc[~frame["image_name"].isin(used_names)].sort_values(["abs_error", "image_name"])
        for _, row in remaining.iterrows():
            chosen.append(row)
            if len(chosen) >= count:
                break
    return pd.DataFrame(chosen).head(count).reset_index(drop=True)


def select_failure_examples(frame: pd.DataFrame, count: int = 3) -> pd.DataFrame:
    chosen: list[pd.Series] = []
    used_names: set[str] = set()

    over = frame.loc[frame["error"] > 0].sort_values(["abs_error", "image_name"], ascending=[False, True])
    under = frame.loc[frame["error"] < 0].sort_values(["abs_error", "image_name"], ascending=[False, True])

    if not over.empty:
        row = over.iloc[0]
        chosen.append(row)
        used_names.add(str(row["image_name"]))
    if not under.empty:
        row = under.iloc[0]
        if str(row["image_name"]) not in used_names:
            chosen.append(row)
            used_names.add(str(row["image_name"]))

    remaining = frame.loc[~frame["image_name"].isin(used_names)].sort_values(["abs_error", "image_name"], ascending=[False, True])
    for _, row in remaining.iterrows():
        chosen.append(row)
        if len(chosen) >= count:
            break
    return pd.DataFrame(chosen).head(count).reset_index(drop=True)


def annotate_image(model: Any, image_path: Path, conf: float, device: str, imgsz: int) -> np.ndarray:
    result = model.predict(
        source=str(image_path),
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
        max_det=1000,
    )[0]
    plotted = result.plot()
    return cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)


def build_qualitative_montage(
    best_run: pd.Series,
    selected_examples: pd.DataFrame,
    output_path: Path,
    device: str,
    imgsz: int,
) -> None:
    from ultralytics import YOLO

    model = YOLO(str(best_run["weights_path"]))
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for axis, (_, row) in zip(axes, selected_examples.iterrows()):
        image = annotate_image(
            model=model,
            image_path=Path(str(row["source_image_path"])),
            conf=float(best_run["best_confidence_for_counting"]),
            device=device,
            imgsz=imgsz,
        )
        label = "Success" if row["example_type"] == "success" else "Failure"
        axis.imshow(image)
        axis.set_title(
            f"{label} | {str(row['density_bin']).title()}\n"
            f"GT={int(row['y_true'])} Pred={int(row['y_pred'])} Err={int(row['error']):+d}",
            fontsize=10,
        )
        axis.axis("off")

    for axis in axes[len(selected_examples):]:
        axis.axis("off")

    fig.suptitle(
        f"Qualitative Counting Examples - {best_run['variant']} seed {int(best_run['seed'])}",
        fontsize=14,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    benchmark_name = args.benchmark_name or f"yolo_multiseed_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir = REPORTS_DIR / "yolo_multiseed" / benchmark_name
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset_yaml = write_runtime_detection_yaml()
    manifest = read_manifest_csv(PROCESSED_DIR / "detection_counting_manifest.csv")
    manifest = remap_path_columns(manifest, ["source_image_path", "source_label_path"])
    test_manifest = manifest.loc[manifest["split"] == "test"].copy()
    density_bins = choose_density_bins(test_manifest)

    run_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for model_name in args.models:
        variant = slugify_model_name(model_name)
        for seed in args.seeds:
            run_name = f"{benchmark_name}_{variant}_seed{seed}"
            print(f"[train] {model_name} seed={seed} -> {run_name}")
            run_dir, weights_path = train_or_reuse(
                model_name=model_name,
                dataset_yaml=dataset_yaml,
                run_name=run_name,
                args=args,
                seed=seed,
            )
            print(f"[eval] {weights_path}")
            summary, predictions = evaluate_yolo_counting_weights(
                manifest=manifest,
                weights_path=weights_path,
                device=args.eval_device,
                method_name=f"{variant}_seed{seed}",
            )
            history = load_training_history(run_dir)
            hist_summary = history_summary(history)
            test_detection = summary.get("test_detection_metrics", {})
            test_count = summary["test"]
            val_count = summary["val"]

            row = {
                "variant": variant,
                "model_name": model_name,
                "seed": seed,
                "run_name": run_name,
                "run_dir": str(run_dir.resolve()),
                "weights_path": str(weights_path.resolve()),
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
            predictions["variant"] = variant
            predictions["model_name"] = model_name
            predictions["seed"] = seed
            predictions["run_name"] = run_name
            predictions["weights_path"] = str(weights_path.resolve())
            predictions["best_confidence_for_counting"] = float(summary["best_confidence_for_counting"])
            predictions["error"] = predictions["y_pred"] - predictions["y_true"]
            predictions["abs_error"] = predictions["error"].abs()
            prediction_frames.append(predictions)

    run_frame = pd.DataFrame(run_rows).sort_values(["variant", "seed"]).reset_index(drop=True)
    predictions_frame = pd.concat(prediction_frames, ignore_index=True)
    predictions_frame = predictions_frame.merge(
        density_bins.rename(columns={"object_count": "object_count_true"}),
        on="image_name",
        how="left",
    )
    predictions_frame = predictions_frame.merge(
        test_manifest[["image_name", "source_image_path"]],
        on="image_name",
        how="left",
    )

    aggregate = aggregate_mean_std(
        run_frame,
        ["variant", "model_name"],
        [
            "test_detection_precision",
            "test_detection_recall",
            "test_detection_mAP50",
            "test_detection_mAP50_95",
            "test_count_mae",
            "test_count_rmse",
            "test_count_r2",
            "test_count_exact_match_rate",
        ],
    ).sort_values(["test_count_mae_mean", "test_detection_mAP50_mean"], ascending=[True, False]).reset_index(drop=True)

    test_predictions = predictions_frame.loc[predictions_frame["split"] == "test"].copy()
    density_rows: list[dict[str, Any]] = []
    for (variant, model_name, seed, density), group in test_predictions.groupby(["variant", "model_name", "seed", "density_bin"], dropna=False):
        metrics = count_metrics(group["y_true"].to_numpy(dtype=float), group["y_pred"].to_numpy(dtype=float))
        density_rows.append(
            {
                "variant": variant,
                "model_name": model_name,
                "seed": seed,
                "density_bin": str(density),
                "n_images": int(len(group)),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "exact_match_rate": metrics["exact_match_rate"],
                "mean_true": metrics["mean_true"],
                "mean_pred": metrics["mean_pred"],
            }
        )
    density_frame = pd.DataFrame(density_rows)
    density_summary = aggregate_mean_std(
        density_frame,
        ["variant", "model_name", "density_bin"],
        ["mae", "rmse", "r2", "exact_match_rate", "mean_true", "mean_pred", "n_images"],
    )

    best_run = run_frame.sort_values(["test_count_mae", "test_detection_mAP50"], ascending=[True, False]).iloc[0]
    best_predictions = test_predictions.loc[
        (test_predictions["variant"] == best_run["variant"]) & (test_predictions["seed"] == best_run["seed"])
    ].copy()
    success = select_success_examples(best_predictions, count=3)
    success["example_type"] = "success"
    failure = select_failure_examples(best_predictions, count=3)
    failure["example_type"] = "failure"
    selected_examples = pd.concat([success, failure], ignore_index=True)

    run_frame.to_csv(output_dir / "run_level_results.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_results.csv", index=False)
    predictions_frame.to_csv(output_dir / "predictions.csv", index=False)
    density_frame.to_csv(output_dir / "density_run_level.csv", index=False)
    density_summary.to_csv(output_dir / "density_summary.csv", index=False)
    selected_examples.to_csv(output_dir / "qualitative_examples.csv", index=False)

    plot_aggregate_metric(
        aggregate,
        metric_prefix="test_count_mae",
        title="Multi-seed Test Count MAE",
        ylabel="MAE",
        output_path=plots_dir / "multiseed_test_count_mae.png",
    )
    plot_aggregate_metric(
        aggregate,
        metric_prefix="test_count_r2",
        title="Multi-seed Test Count R2",
        ylabel="R2",
        output_path=plots_dir / "multiseed_test_count_r2.png",
    )
    plot_aggregate_metric(
        aggregate,
        metric_prefix="test_detection_mAP50",
        title="Multi-seed Test Detection mAP50",
        ylabel="mAP50",
        output_path=plots_dir / "multiseed_test_detection_map50.png",
    )
    plot_aggregate_metric(
        aggregate,
        metric_prefix="test_detection_mAP50_95",
        title="Multi-seed Test Detection mAP50-95",
        ylabel="mAP50-95",
        output_path=plots_dir / "multiseed_test_detection_map50_95.png",
    )
    plot_density_metric(
        density_summary,
        metric="mae",
        output_path=plots_dir / "density_mae_comparison.png",
        title="Counting MAE by Density Bin",
    )
    plot_density_metric(
        density_summary,
        metric="r2",
        output_path=plots_dir / "density_r2_comparison.png",
        title="Counting R2 by Density Bin",
    )
    build_qualitative_montage(
        best_run=best_run,
        selected_examples=selected_examples,
        output_path=plots_dir / "qualitative_success_failure_montage.png",
        device=args.eval_device,
        imgsz=args.imgsz,
    )

    summary = {
        "benchmark_name": benchmark_name,
        "models": args.models,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "train_device": args.train_device,
        "eval_device": args.eval_device,
        "batch": args.batch,
        "workers": args.workers,
        "best_run": {key: (None if pd.isna(value) else value) for key, value in best_run.to_dict().items()},
        "aggregate_ranking": aggregate.to_dict(orient="records"),
        "density_bins": {
            "labels": ["low", "medium", "high"],
            "test_image_count": int(len(test_manifest)),
        },
        "plots": {
            "multiseed_test_count_mae": str((plots_dir / "multiseed_test_count_mae.png").resolve()),
            "multiseed_test_count_r2": str((plots_dir / "multiseed_test_count_r2.png").resolve()),
            "multiseed_test_detection_map50": str((plots_dir / "multiseed_test_detection_map50.png").resolve()),
            "multiseed_test_detection_map50_95": str((plots_dir / "multiseed_test_detection_map50_95.png").resolve()),
            "density_mae_comparison": str((plots_dir / "density_mae_comparison.png").resolve()),
            "density_r2_comparison": str((plots_dir / "density_r2_comparison.png").resolve()),
            "qualitative_success_failure_montage": str((plots_dir / "qualitative_success_failure_montage.png").resolve()),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(run_frame.to_string(index=False))
    print("\nAggregate:")
    print(aggregate.to_string(index=False))
    print(f"\nSaved multi-seed benchmark to {output_dir}")


if __name__ == "__main__":
    main()
