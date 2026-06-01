from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from larvae_cv.data import read_manifest_csv, remap_path_columns, write_json
from larvae_cv.paths import PROCESSED_DIR, REPORTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate confidence, IoU, and agnostic-NMS post-processing settings for YOLO counting."
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        default=[],
        help="YOLO weight paths. If omitted, best.pt files from the existing multi-seed runs are used.",
    )
    parser.add_argument(
        "--conf-grid",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
        help="Confidence thresholds to sweep.",
    )
    parser.add_argument(
        "--iou-grid",
        nargs="+",
        type=float,
        default=[0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        help="NMS IoU thresholds to sweep.",
    )
    parser.add_argument(
        "--include-agnostic-nms",
        action="store_true",
        help="Also evaluate agnostic_nms=True. Default evaluates only agnostic_nms=False.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="cuda", help="Inference device, for example cuda or cpu.")
    parser.add_argument("--max-det", type=int, default=1000, help="Maximum detections per image.")
    parser.add_argument(
        "--benchmark-name",
        default="",
        help="Output folder under reports/reviewer_experiments/postprocessing. Default uses a timestamp.",
    )
    return parser.parse_args()


def discover_default_weights() -> list[Path]:
    return sorted((REPORTS_DIR / "yolo_runs").glob("yolo_multiseed_strengthening_*_seed*/weights/best.pt"))


def slugify(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")


def run_label(weights_path: Path) -> str:
    parts = weights_path.parts
    if "yolo_runs" in parts:
        run_name = weights_path.parents[1].name
        return slugify(run_name.replace("yolo_multiseed_strengthening_", ""))
    return slugify(weights_path.stem)


def variant_label(label: str) -> str:
    if "yolo11n" in label:
        return "yolo11n"
    if "yolov8n" in label:
        return "yolov8n"
    if "yolo11s" in label:
        return "yolo11s"
    return label.split("_seed")[0]


def count_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "exact_match_rate": float(np.mean(np.rint(y_true) == np.rint(y_pred))),
        "mean_true": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_pred)),
    }


def predict_counts(
    model: Any,
    split_frame: pd.DataFrame,
    conf: float,
    iou: float,
    agnostic_nms: bool,
    args: argparse.Namespace,
) -> np.ndarray:
    outputs = model.predict(
        source=split_frame["source_image_path"].tolist(),
        imgsz=args.imgsz,
        device=args.device,
        conf=conf,
        iou=iou,
        agnostic_nms=agnostic_nms,
        max_det=args.max_det,
        verbose=False,
    )
    return np.array([len(output.boxes) for output in outputs], dtype=float)


def evaluate_weight(
    weights_path: Path,
    manifest: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from ultralytics import YOLO

    label = run_label(weights_path)
    variant = variant_label(label)
    model = YOLO(str(weights_path))
    nms_options = [False, True] if args.include_agnostic_nms else [False]
    result_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for conf in args.conf_grid:
        for iou in args.iou_grid:
            for agnostic_nms in nms_options:
                for split_name in ["val", "test"]:
                    split_frame = manifest.loc[manifest["split"] == split_name].copy()
                    y_true = split_frame["object_count"].to_numpy(dtype=float)
                    y_pred = predict_counts(model, split_frame, conf, iou, agnostic_nms, args)
                    metrics = count_metrics(y_true, y_pred)
                    result_rows.append(
                        {
                            "run_label": label,
                            "variant": variant,
                            "weights_path": str(weights_path.resolve()),
                            "split": split_name,
                            "conf": conf,
                            "iou": iou,
                            "agnostic_nms": agnostic_nms,
                            **metrics,
                        }
                    )
                    for image_name, truth, pred in zip(split_frame["image_name"], y_true, y_pred):
                        prediction_rows.append(
                            {
                                "run_label": label,
                                "variant": variant,
                                "split": split_name,
                                "image_name": image_name,
                                "conf": conf,
                                "iou": iou,
                                "agnostic_nms": agnostic_nms,
                                "y_true": float(truth),
                                "y_pred": float(pred),
                                "error": float(pred - truth),
                                "abs_error": float(abs(pred - truth)),
                            }
                        )

    return pd.DataFrame(result_rows), pd.DataFrame(prediction_rows)


def select_best_configs(sweep: pd.DataFrame) -> pd.DataFrame:
    val = sweep.loc[sweep["split"] == "val"].copy()
    val = val.sort_values(["run_label", "mae", "rmse", "r2"], ascending=[True, True, True, False])
    best_val = val.groupby("run_label", as_index=False).head(1)
    rows: list[pd.Series] = []
    for record in best_val.to_dict(orient="records"):
        match = sweep.loc[
            (sweep["run_label"] == record["run_label"])
            & (sweep["conf"] == record["conf"])
            & (sweep["iou"] == record["iou"])
            & (sweep["agnostic_nms"] == record["agnostic_nms"])
        ].copy()
        match["selected_by_validation"] = True
        rows.extend([row for _, row in match.iterrows()])
    return pd.DataFrame(rows)


def aggregate_selected(selected: pd.DataFrame) -> pd.DataFrame:
    test = selected.loc[selected["split"] == "test"].copy()
    metric_cols = ["mae", "rmse", "r2", "exact_match_rate", "mean_pred"]
    grouped = test.groupby("variant", dropna=False)[metric_cols].agg(["mean", "std", "min", "max"])
    grouped.columns = ["_".join(item) for item in grouped.columns]
    return grouped.reset_index().sort_values("mae_mean")


def main() -> None:
    args = parse_args()
    benchmark_name = args.benchmark_name or f"postprocessing_sweep_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir = REPORTS_DIR / "reviewer_experiments" / "postprocessing" / benchmark_name
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = [Path(item) for item in args.weights] if args.weights else discover_default_weights()
    if not weights:
        raise SystemExit("No weights found. Pass --weights or train/evaluate YOLO models first.")

    manifest = read_manifest_csv(PROCESSED_DIR / "detection_counting_manifest.csv")
    manifest = remap_path_columns(manifest, ["source_image_path", "source_label_path"])

    sweep_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for weights_path in weights:
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        print(f"[sweep] {weights_path}")
        sweep, predictions = evaluate_weight(weights_path, manifest, args)
        sweep_frames.append(sweep)
        prediction_frames.append(predictions)

    sweep_frame = pd.concat(sweep_frames, ignore_index=True)
    prediction_frame = pd.concat(prediction_frames, ignore_index=True)
    selected = select_best_configs(sweep_frame)
    aggregate = aggregate_selected(selected)

    selected_keys = selected.loc[selected["split"] == "test", ["run_label", "conf", "iou", "agnostic_nms"]].drop_duplicates()
    selected_predictions = prediction_frame.merge(
        selected_keys,
        on=["run_label", "conf", "iou", "agnostic_nms"],
        how="inner",
    )

    sweep_frame.to_csv(output_dir / "sweep_results.csv", index=False)
    selected.to_csv(output_dir / "selected_results.csv", index=False)
    selected_predictions.to_csv(output_dir / "selected_predictions.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_selected_test.csv", index=False)

    summary = {
        "benchmark_name": benchmark_name,
        "weights": [str(path.resolve()) for path in weights],
        "conf_grid": args.conf_grid,
        "iou_grid": args.iou_grid,
        "include_agnostic_nms": args.include_agnostic_nms,
        "imgsz": args.imgsz,
        "device": args.device,
        "max_det": args.max_det,
        "selected_test_aggregate": aggregate.to_dict(orient="records"),
        "outputs": {
            "sweep_results": str((output_dir / "sweep_results.csv").resolve()),
            "selected_results": str((output_dir / "selected_results.csv").resolve()),
            "selected_predictions": str((output_dir / "selected_predictions.csv").resolve()),
            "aggregate_selected_test": str((output_dir / "aggregate_selected_test.csv").resolve()),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(aggregate.to_string(index=False))
    print(f"Saved post-processing sweep to {output_dir}")


if __name__ == "__main__":
    main()
