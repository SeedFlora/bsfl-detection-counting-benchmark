from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)

from larvae_cv.data import to_builtin, write_json
from larvae_cv.paths import MODELS_DIR


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("feat_") or column.startswith("px_")]


def _merge_manifest_features(manifest: pd.DataFrame, features: pd.DataFrame, image_col: str) -> pd.DataFrame:
    merged = manifest.merge(features, on=["image_id", image_col], how="inner")
    merged = merged.loc[merged.get("feature_error", "").fillna("") == ""].copy()
    return merged


def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
        "report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def run_regression_baseline(
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    image_col: str,
    target_cols: tuple[str, ...] = ("weight", "length"),
    report_dir: Path | None = None,
) -> dict[str, Any]:
    report_dir = report_dir or Path("reports") / "regression"
    report_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data = _merge_manifest_features(manifest, features, image_col=image_col)
    feature_cols = _feature_columns(data)
    results: dict[str, Any] = {"feature_count": len(feature_cols), "targets": {}}

    predictions: list[dict[str, Any]] = []
    for target in target_cols:
        target_frame = data.loc[data[target].notna()].copy()
        train = target_frame.loc[target_frame["split"] == "train"]
        val = target_frame.loc[target_frame["split"] == "val"]
        test = target_frame.loc[target_frame["split"] == "test"]

        model = ExtraTreesRegressor(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(train[feature_cols], train[target])

        target_result = {}
        for split_name, split_frame in [("val", val), ("test", test)]:
            pred = model.predict(split_frame[feature_cols])
            target_result[split_name] = _regression_metrics(split_frame[target], pred)
            for image_id, truth, guess in zip(split_frame["image_id"], split_frame[target], pred):
                predictions.append(
                    {
                        "task": target,
                        "split": split_name,
                        "image_id": image_id,
                        "y_true": float(truth),
                        "y_pred": float(guess),
                    }
                )

        model_path = MODELS_DIR / f"{target}_regression.joblib"
        joblib.dump(model, model_path)
        target_result["model_path"] = str(model_path.resolve())
        results["targets"][target] = target_result

    pd.DataFrame(predictions).to_csv(report_dir / "predictions.csv", index=False)
    write_json(report_dir / "metrics.json", results)
    return to_builtin(results)


def run_classifier_baseline(
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    image_col: str,
    target_col: str,
    task_name: str,
    report_dir: Path,
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data = _merge_manifest_features(manifest, features, image_col=image_col)
    data = data.loc[data[target_col].notna()].copy()
    feature_cols = _feature_columns(data)
    labels = sorted(data[target_col].astype(str).unique())

    train = data.loc[data["split"] == "train"]
    val = data.loc[data["split"] == "val"]
    test = data.loc[data["split"] == "test"]

    model = ExtraTreesClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(train[feature_cols], train[target_col].astype(str))

    results: dict[str, Any] = {
        "feature_count": len(feature_cols),
        "target": target_col,
        "labels": labels,
    }
    predictions: list[dict[str, Any]] = []
    for split_name, split_frame in [("val", val), ("test", test)]:
        pred = model.predict(split_frame[feature_cols])
        results[split_name] = _classification_metrics(split_frame[target_col].astype(str), pred, labels)
        for image_id, truth, guess in zip(split_frame["image_id"], split_frame[target_col], pred):
            predictions.append(
                {
                    "task": task_name,
                    "split": split_name,
                    "image_id": image_id,
                    "y_true": str(truth),
                    "y_pred": str(guess),
                }
            )

    model_path = MODELS_DIR / f"{task_name}.joblib"
    joblib.dump(model, model_path)
    results["model_path"] = str(model_path.resolve())

    pd.DataFrame(predictions).to_csv(report_dir / "predictions.csv", index=False)
    write_json(report_dir / "metrics.json", results)
    return to_builtin(results)
