from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from larvae_cv.data import write_json
from larvae_cv.paths import REPORTS_DIR


PAPER_DIR = REPORTS_DIR / "paper_q4_prep"


def load_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sanitize_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def sanitize_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: sanitize_value(value) for key, value in record.items()}


def build_detection_counting_table() -> pd.DataFrame:
    baseline = pd.read_csv(REPORTS_DIR / "detection_counting_benchmark" / "ranking.csv")
    baseline["category"] = baseline["method"].map(
        {
            "random_forest_count_regression": "classical_count_regression",
            "extra_trees_count_regression": "classical_count_regression",
            "hist_gradient_boosting_count_regression": "classical_count_regression",
            "knn_count_regression": "classical_count_regression",
            "watershed_counting": "classical_cv",
            "otsu_connected_components": "classical_cv",
            "adaptive_connected_components": "classical_cv",
            "blob_detector_counting": "classical_cv",
            "distance_peak_counting": "classical_cv",
            "yolo_v8n_smoke": "legacy_smoke",
        }
    ).fillna("other")
    baseline["model_name"] = baseline["method"]
    baseline["test_detection_precision"] = float("nan")
    baseline["test_detection_recall"] = float("nan")
    baseline["test_detection_mAP50"] = float("nan")
    baseline["test_detection_mAP50_95"] = float("nan")
    baseline["epochs"] = float("nan")
    baseline["report_source"] = str((REPORTS_DIR / "detection_counting_benchmark" / "ranking.csv").resolve())

    yolov8 = load_json(REPORTS_DIR / "counting_from_yolo" / "larvae_counting_gpu2" / "summary.json")
    yolov8_row = pd.DataFrame(
        [
            {
                "method": "yolov8n_gpu_20ep",
                "category": "deep_learning_detector",
                "model_name": "yolov8n",
                "test_mae": yolov8["test"]["mae"],
                "test_rmse": yolov8["test"]["rmse"],
                "test_r2": yolov8["test"]["r2"],
                "test_detection_precision": yolov8["test_detection_metrics"]["metrics/precision(B)"],
                "test_detection_recall": yolov8["test_detection_metrics"]["metrics/recall(B)"],
                "test_detection_mAP50": yolov8["test_detection_metrics"]["metrics/mAP50(B)"],
                "test_detection_mAP50_95": yolov8["test_detection_metrics"]["metrics/mAP50-95(B)"],
                "epochs": 20,
                "report_source": str((REPORTS_DIR / "counting_from_yolo" / "larvae_counting_gpu2" / "summary.json").resolve()),
            }
        ]
    )

    yolo11 = pd.read_csv(REPORTS_DIR / "yolo_benchmark" / "yolo11_gpu_benchmark" / "comparison.csv")
    yolo11 = yolo11.rename(
        columns={
            "variant": "method",
            "test_count_mae": "test_mae",
            "test_count_rmse": "test_rmse",
            "test_count_r2": "test_r2",
        }
    )
    yolo11["category"] = "deep_learning_detector"
    yolo11["report_source"] = str((REPORTS_DIR / "yolo_benchmark" / "yolo11_gpu_benchmark" / "comparison.csv").resolve())
    yolo11["epochs"] = yolo11["epoch_count"]
    yolo11["method"] = yolo11["method"].map({"yolo11n": "yolo11n_gpu_20ep", "yolo11s": "yolo11s_gpu_20ep"}).fillna(yolo11["method"])
    yolo11 = yolo11[
        [
            "method",
            "category",
            "model_name",
            "test_mae",
            "test_rmse",
            "test_r2",
            "test_detection_precision",
            "test_detection_recall",
            "test_detection_mAP50",
            "test_detection_mAP50_95",
            "epochs",
            "report_source",
        ]
    ]

    baseline = baseline[
        [
            "method",
            "category",
            "model_name",
            "test_mae",
            "test_rmse",
            "test_r2",
            "test_detection_precision",
            "test_detection_recall",
            "test_detection_mAP50",
            "test_detection_mAP50_95",
            "epochs",
            "report_source",
        ]
    ]

    combined = pd.concat([baseline, yolov8_row, yolo11], ignore_index=True)
    combined = combined.sort_values(["test_mae", "test_rmse", "test_r2"], ascending=[True, True, False]).reset_index(drop=True)
    return combined


def build_key_results_summary(table: pd.DataFrame) -> dict[str, Any]:
    best_counting = sanitize_record(table.iloc[0].to_dict())
    best_detection_frame = table.dropna(subset=["test_detection_mAP50"])
    best_detection = best_detection_frame.sort_values(
        ["test_detection_mAP50", "test_detection_mAP50_95"], ascending=[False, False]
    ).iloc[0].to_dict()
    return {
        "best_counting_method": best_counting,
        "best_detection_method": sanitize_record(best_detection),
    }


def main() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    table = build_detection_counting_table()
    table.to_csv(PAPER_DIR / "results_table_detection_counting.csv", index=False)
    write_json(PAPER_DIR / "key_results.json", build_key_results_summary(table))
    print(table.to_string(index=False))
    print(f"Saved paper-prep artifacts to {PAPER_DIR}")


if __name__ == "__main__":
    main()
