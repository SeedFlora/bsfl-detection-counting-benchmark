from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from larvae_cv.data import build_or_load_feature_cache, read_manifest_csv, write_json
from larvae_cv.paths import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR


os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("feat_") or column.startswith("px_")]


def _regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _classification_metrics(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> dict[str, Any]:
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


def _build_feature_table() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    single_manifest = read_manifest_csv(PROCESSED_DIR / "single_larvae_manifest.csv")
    single_manifest = single_manifest.loc[single_manifest["single_image_exists"]].copy()
    features = build_or_load_feature_cache(
        manifest=single_manifest,
        image_col="single_image_path",
        cache_name="single_image_features",
    )
    regression_manifest = read_manifest_csv(PROCESSED_DIR / "regression_manifest.csv")
    size_manifest = read_manifest_csv(PROCESSED_DIR / "size_classification_manifest.csv")
    return features, regression_manifest, size_manifest


def _regression_models() -> dict[str, Pipeline]:
    return {
        "linear_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesRegressor(n_estimators=500, random_state=42, n_jobs=-1)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingRegressor(random_state=42)),
            ]
        ),
    }


def _classification_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced")),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced_subsample")),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(random_state=42)),
            ]
        ),
        "knn": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
    }


def run_regression_and_size_benchmarks() -> dict[str, Any]:
    features, regression_manifest, size_manifest = _build_feature_table()
    report_dir = REPORTS_DIR / "regression_size_benchmark"
    report_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    regression_data = regression_manifest.merge(features, on=["image_id", "single_image_path"], how="inner")
    size_data = size_manifest.merge(features, on=["image_id", "single_image_path"], how="inner")
    regression_feature_cols = _feature_columns(regression_data)
    size_feature_cols = _feature_columns(size_data)

    regression_results: dict[str, Any] = {}
    regression_predictions: list[dict[str, Any]] = []
    regression_ranking: list[dict[str, Any]] = []

    for model_name, model in _regression_models().items():
        per_target: dict[str, Any] = {}
        score_accumulator = []
        for target in ["weight", "length"]:
            target_data = regression_data.loc[regression_data[target].notna()].copy()
            train = target_data.loc[target_data["split"] == "train"]
            val = target_data.loc[target_data["split"] == "val"]
            test = target_data.loc[target_data["split"] == "test"]

            model.fit(train[regression_feature_cols], train[target])
            target_payload: dict[str, Any] = {}
            for split_name, split_frame in [("val", val), ("test", test)]:
                pred = model.predict(split_frame[regression_feature_cols])
                target_payload[split_name] = _regression_metrics(split_frame[target], pred)
                if split_name == "test":
                    score_accumulator.append(target_payload[split_name]["r2"])
                for image_id, truth, guess in zip(split_frame["image_id"], split_frame[target], pred):
                    regression_predictions.append(
                        {
                            "model": model_name,
                            "target": target,
                            "split": split_name,
                            "image_id": image_id,
                            "y_true": float(truth),
                            "y_pred": float(guess),
                        }
                    )

            import joblib

            model_path = MODELS_DIR / f"{model_name}_{target}_benchmark.joblib"
            joblib.dump(model, model_path)
            target_payload["model_path"] = str(model_path.resolve())
            per_target[target] = target_payload

        regression_results[model_name] = per_target
        regression_ranking.append(
            {
                "model": model_name,
                "avg_test_r2": float(sum(score_accumulator) / len(score_accumulator)),
                "weight_test_r2": per_target["weight"]["test"]["r2"],
                "length_test_r2": per_target["length"]["test"]["r2"],
            }
        )

    classification_results: dict[str, Any] = {}
    classification_predictions: list[dict[str, Any]] = []
    classification_ranking: list[dict[str, Any]] = []

    for model_name, model in _classification_models().items():
        per_target: dict[str, Any] = {}
        score_accumulator = []
        for target in ["weight_size_class", "length_size_class"]:
            target_data = size_data.loc[size_data[target].notna()].copy()
            labels = sorted(target_data[target].astype(str).unique())
            train = target_data.loc[target_data["split"] == "train"]
            val = target_data.loc[target_data["split"] == "val"]
            test = target_data.loc[target_data["split"] == "test"]

            model.fit(train[size_feature_cols], train[target].astype(str))
            target_payload: dict[str, Any] = {"labels": labels}
            for split_name, split_frame in [("val", val), ("test", test)]:
                pred = model.predict(split_frame[size_feature_cols])
                target_payload[split_name] = _classification_metrics(split_frame[target].astype(str), pred, labels)
                if split_name == "test":
                    score_accumulator.append(target_payload[split_name]["balanced_accuracy"])
                for image_id, truth, guess in zip(split_frame["image_id"], split_frame[target], pred):
                    classification_predictions.append(
                        {
                            "model": model_name,
                            "target": target,
                            "split": split_name,
                            "image_id": image_id,
                            "y_true": str(truth),
                            "y_pred": str(guess),
                        }
                    )

            import joblib

            model_path = MODELS_DIR / f"{model_name}_{target}_benchmark.joblib"
            joblib.dump(model, model_path)
            target_payload["model_path"] = str(model_path.resolve())
            per_target[target] = target_payload

        classification_results[model_name] = per_target
        classification_ranking.append(
            {
                "model": model_name,
                "avg_test_balanced_accuracy": float(sum(score_accumulator) / len(score_accumulator)),
                "weight_size_test_balanced_accuracy": per_target["weight_size_class"]["test"]["balanced_accuracy"],
                "length_size_test_balanced_accuracy": per_target["length_size_class"]["test"]["balanced_accuracy"],
            }
        )

    regression_ranking_frame = pd.DataFrame(regression_ranking).sort_values("avg_test_r2", ascending=False)
    classification_ranking_frame = pd.DataFrame(classification_ranking).sort_values(
        "avg_test_balanced_accuracy",
        ascending=False,
    )

    pd.DataFrame(regression_predictions).to_csv(report_dir / "regression_predictions.csv", index=False)
    pd.DataFrame(classification_predictions).to_csv(report_dir / "size_classification_predictions.csv", index=False)
    regression_ranking_frame.to_csv(report_dir / "regression_ranking.csv", index=False)
    classification_ranking_frame.to_csv(report_dir / "size_classification_ranking.csv", index=False)

    summary = {
        "regression": {
            "models": regression_results,
            "ranking": regression_ranking_frame.to_dict(orient="records"),
        },
        "size_classification": {
            "models": classification_results,
            "ranking": classification_ranking_frame.to_dict(orient="records"),
        },
    }
    write_json(report_dir / "summary.json", summary)
    return summary
