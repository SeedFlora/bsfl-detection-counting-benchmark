from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from larvae_cv.data import (
    read_manifest_csv,
    remap_path_columns,
    to_builtin,
    write_json,
    write_runtime_detection_yaml,
)
from larvae_cv.paths import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR


CountFn = Callable[[Path], int]


def _count_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "exact_match_rate": float(np.mean(y_true == np.rint(y_pred))),
        "mean_true": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_pred)),
    }


def _read_gray(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    return image


def _pick_foreground_mask(gray: np.ndarray) -> np.ndarray:
    candidates: list[tuple[float, np.ndarray]] = []
    for inverse in [True, False]:
        flag = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
        _, mask = cv2.threshold(gray, 0, 255, flag | cv2.THRESH_OTSU)
        frac = float(mask.mean() / 255.0)
        score = abs(frac - 0.18)
        if not 0.005 <= frac <= 0.85:
            score += 1.0
        candidates.append((score, mask))
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _count_components(mask: np.ndarray, min_area: int = 80, max_area_frac: float = 0.2) -> int:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    max_area = mask.shape[0] * mask.shape[1] * max_area_frac
    count = 0
    for idx in range(1, n_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            count += 1
    return count


def otsu_connected_components(image_path: Path) -> int:
    gray = _read_gray(image_path)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = _pick_foreground_mask(blur)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return _count_components(mask, min_area=90)


def adaptive_connected_components(image_path: Path) -> int:
    gray = _read_gray(image_path)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    inv = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        3,
    )
    normal = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        3,
    )
    candidates = []
    for mask in [inv, normal]:
        frac = float(mask.mean() / 255.0)
        score = abs(frac - 0.18)
        candidates.append((score, mask))
    candidates.sort(key=lambda item: item[0])
    mask = candidates[0][1]
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return _count_components(mask, min_area=60)


def watershed_counting(image_path: Path) -> int:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = _pick_foreground_mask(blur)
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    sure_bg = cv2.dilate(opened, kernel, iterations=2)
    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.23 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image.copy(), markers)

    count = 0
    for marker_id in np.unique(markers):
        if marker_id <= 1:
            continue
        area = int(np.sum(markers == marker_id))
        if 60 <= area <= image.shape[0] * image.shape[1] * 0.2:
            count += 1
    return count


def blob_detector_counting(image_path: Path) -> int:
    gray = _read_gray(image_path)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = _pick_foreground_mask(blur)
    masked = cv2.bitwise_and(255 - blur, 255 - blur, mask=mask)

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 255
    params.thresholdStep = 10
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = max(200.0, float(mask.shape[0] * mask.shape[1] * 0.025))
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.filterByColor = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(masked)
    return int(len(keypoints))


def distance_peak_counting(image_path: Path) -> int:
    gray = _read_gray(image_path)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = _pick_foreground_mask(blur)
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    opened = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    if float(dist.max()) <= 0.0:
        return 0

    dist = cv2.GaussianBlur(dist, (0, 0), 1.1)
    peak_threshold = max(0.35, float(dist.max()) * 0.22)
    dilated = cv2.dilate(dist, np.ones((9, 9), np.float32))
    peaks = ((dist >= dilated - 1e-6) & (dist >= peak_threshold)).astype(np.uint8) * 255
    return _count_components(peaks, min_area=1, max_area_frac=0.01)


def _extract_count_features(image_path: Path) -> dict[str, float]:
    gray = _read_gray(image_path)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = _pick_foreground_mask(blur)
    edge = cv2.Canny(blur, 50, 150)

    features: dict[str, float] = {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "edge_density": float(edge.mean() / 255.0),
        "foreground_fraction": float(mask.mean() / 255.0),
        "laplacian_var": float(cv2.Laplacian(blur, cv2.CV_64F).var()),
        "otsu_component_count_40": float(_count_components(mask, min_area=40)),
        "otsu_component_count_80": float(_count_components(mask, min_area=80)),
        "otsu_component_count_120": float(_count_components(mask, min_area=120)),
    }

    hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
    hist = hist / max(1.0, hist.sum())
    for idx, value in enumerate(hist):
        features[f"hist_{idx:02d}"] = float(value)

    thumb = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    for idx, value in enumerate(thumb.reshape(-1)):
        features[f"px_{idx:02d}"] = float(value)

    return features


def _feature_frame(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in manifest.itertuples(index=False):
        path = Path(row.source_image_path)
        rows.append(
            {
                "image_name": row.image_name,
                **_extract_count_features(path),
            }
        )
    return pd.DataFrame(rows)


def _prepare_feature_dataset(manifest: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = _feature_frame(manifest)
    data = manifest.merge(features, on="image_name", how="inner")
    feature_cols = [
        column
        for column in data.columns
        if column.startswith("gray_")
        or column.startswith("edge_")
        or column.startswith("foreground_")
        or column.startswith("laplacian_")
        or column.startswith("otsu_")
        or column.startswith("hist_")
        or column.startswith("px_")
    ]
    train = data.loc[data["split"] == "train"].copy()
    val = data.loc[data["split"] == "val"].copy()
    test = data.loc[data["split"] == "test"].copy()
    return data, feature_cols, train, val, test


def _evaluate_count_function(method_name: str, manifest: pd.DataFrame, count_fn: CountFn) -> tuple[dict[str, Any], pd.DataFrame]:
    predictions: list[dict[str, Any]] = []
    results: dict[str, Any] = {"method": method_name}

    for split_name in ["val", "test"]:
        split_frame = manifest.loc[manifest["split"] == split_name].copy()
        y_true = split_frame["object_count"].to_numpy(dtype=float)
        y_pred = np.array([count_fn(Path(path)) for path in split_frame["source_image_path"]], dtype=float)
        results[split_name] = _count_metrics(y_true, y_pred)
        for image_name, truth, pred in zip(split_frame["image_name"], y_true, y_pred):
            predictions.append(
                {
                    "method": method_name,
                    "split": split_name,
                    "image_name": image_name,
                    "y_true": float(truth),
                    "y_pred": float(pred),
                }
            )

    return results, pd.DataFrame(predictions)


def _evaluate_rf_regressor(manifest: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    _, feature_cols, train, val, test = _prepare_feature_dataset(manifest)

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(train[feature_cols], train["object_count"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    model_path = MODELS_DIR / "count_random_forest.joblib"
    joblib.dump(model, model_path)

    predictions: list[dict[str, Any]] = []
    results: dict[str, Any] = {
        "method": "random_forest_count_regression",
        "feature_count": len(feature_cols),
        "model_path": str(model_path.resolve()),
    }
    for split_name, split_frame in [("val", val), ("test", test)]:
        y_true = split_frame["object_count"].to_numpy(dtype=float)
        y_pred = model.predict(split_frame[feature_cols])
        results[split_name] = _count_metrics(y_true, y_pred)
        for image_name, truth, pred in zip(split_frame["image_name"], y_true, y_pred):
            predictions.append(
                {
                    "method": "random_forest_count_regression",
                    "split": split_name,
                    "image_name": image_name,
                    "y_true": float(truth),
                    "y_pred": float(pred),
                }
            )

    return results, pd.DataFrame(predictions)


def _evaluate_feature_regressor(
    manifest: pd.DataFrame,
    method_name: str,
    estimator: Any,
    model_filename: str,
    use_scaler: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    _, feature_cols, train, val, test = _prepare_feature_dataset(manifest)

    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    model = Pipeline(steps)
    model.fit(train[feature_cols], train["object_count"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    model_path = MODELS_DIR / model_filename
    joblib.dump(model, model_path)

    predictions: list[dict[str, Any]] = []
    results: dict[str, Any] = {
        "method": method_name,
        "feature_count": len(feature_cols),
        "model_path": str(model_path.resolve()),
    }

    for split_name, split_frame in [("val", val), ("test", test)]:
        y_true = split_frame["object_count"].to_numpy(dtype=float)
        y_pred = model.predict(split_frame[feature_cols])
        results[split_name] = _count_metrics(y_true, y_pred)
        for image_name, truth, pred in zip(split_frame["image_name"], y_true, y_pred):
            predictions.append(
                {
                    "method": method_name,
                    "split": split_name,
                    "image_name": image_name,
                    "y_true": float(truth),
                    "y_pred": float(pred),
                }
            )

    return results, pd.DataFrame(predictions)


def _evaluate_yolo_counting(manifest: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    from ultralytics import YOLO

    weights_path = REPORTS_DIR / "yolo_runs" / "larvae_counting_smoke" / "weights" / "best.pt"
    return evaluate_yolo_counting_weights(
        manifest=manifest,
        weights_path=weights_path,
        device="cpu",
        method_name="yolo_v8n_smoke",
    )


def evaluate_yolo_counting_weights(
    manifest: pd.DataFrame,
    weights_path: Path,
    device: str = "cpu",
    method_name: str = "yolo_detector",
) -> tuple[dict[str, Any], pd.DataFrame]:
    from ultralytics import YOLO

    if not weights_path.exists():
        raise FileNotFoundError(
            f"YOLO weights not found: {weights_path}"
        )

    model = YOLO(str(weights_path))
    predictions: list[dict[str, Any]] = []
    results: dict[str, Any] = {
        "method": method_name,
        "weights_path": str(weights_path.resolve()),
    }

    conf_grid = [0.05, 0.1, 0.2, 0.3, 0.4]
    val_frame = manifest.loc[manifest["split"] == "val"].copy()
    best_conf = conf_grid[0]
    best_mae = float("inf")
    for conf in conf_grid:
        outputs = model.predict(
            source=val_frame["source_image_path"].tolist(),
            imgsz=640,
            device=device,
            verbose=False,
            conf=conf,
            max_det=1000,
        )
        y_true = val_frame["object_count"].to_numpy(dtype=float)
        y_pred = np.array([len(output.boxes) for output in outputs], dtype=float)
        mae = mean_absolute_error(y_true, y_pred)
        if mae < best_mae:
            best_mae = float(mae)
            best_conf = conf

    results["best_confidence_for_counting"] = best_conf

    for split_name in ["val", "test"]:
        split_frame = manifest.loc[manifest["split"] == split_name].copy()
        outputs = model.predict(
            source=split_frame["source_image_path"].tolist(),
            imgsz=640,
            device=device,
            verbose=False,
            conf=best_conf,
            max_det=1000,
        )
        y_true = split_frame["object_count"].to_numpy(dtype=float)
        y_pred = np.array([len(output.boxes) for output in outputs], dtype=float)
        results[split_name] = _count_metrics(y_true, y_pred)
        for image_name, truth, pred in zip(split_frame["image_name"], y_true, y_pred):
            predictions.append(
                {
                    "method": method_name,
                    "split": split_name,
                    "image_name": image_name,
                    "y_true": float(truth),
                    "y_pred": float(pred),
                }
            )

    runtime_yaml = write_runtime_detection_yaml()
    test_metrics = model.val(
        data=str(runtime_yaml),
        split="test",
        device=device,
        verbose=False,
    )
    results["test_detection_metrics"] = to_builtin(test_metrics.results_dict)
    return results, pd.DataFrame(predictions)


def run_detection_counting_benchmarks() -> dict[str, Any]:
    manifest = read_manifest_csv(PROCESSED_DIR / "detection_counting_manifest.csv")
    manifest = remap_path_columns(manifest, ["source_image_path", "source_label_path"])
    report_dir = REPORTS_DIR / "detection_counting_benchmark"
    report_dir.mkdir(parents=True, exist_ok=True)

    methods: list[tuple[str, CountFn]] = [
        ("otsu_connected_components", otsu_connected_components),
        ("adaptive_connected_components", adaptive_connected_components),
        ("watershed_counting", watershed_counting),
        ("blob_detector_counting", blob_detector_counting),
        ("distance_peak_counting", distance_peak_counting),
    ]

    all_results: dict[str, Any] = {}
    prediction_frames: list[pd.DataFrame] = []

    try:
        yolo_result, yolo_predictions = _evaluate_yolo_counting(manifest)
        all_results[yolo_result["method"]] = yolo_result
        prediction_frames.append(yolo_predictions)
    except Exception as exc:  # pragma: no cover
        all_results["yolo_v8n_smoke"] = {"error": f"{type(exc).__name__}: {exc}"}

    for method_name, fn in methods:
        result, prediction_frame = _evaluate_count_function(method_name, manifest, fn)
        all_results[method_name] = result
        prediction_frames.append(prediction_frame)

    rf_result, rf_predictions = _evaluate_rf_regressor(manifest)
    all_results[rf_result["method"]] = rf_result
    prediction_frames.append(rf_predictions)

    extra_trees_result, extra_trees_predictions = _evaluate_feature_regressor(
        manifest=manifest,
        method_name="extra_trees_count_regression",
        estimator=ExtraTreesRegressor(
            n_estimators=600,
            random_state=42,
            n_jobs=-1,
        ),
        model_filename="count_extra_trees.joblib",
    )
    all_results[extra_trees_result["method"]] = extra_trees_result
    prediction_frames.append(extra_trees_predictions)

    knn_result, knn_predictions = _evaluate_feature_regressor(
        manifest=manifest,
        method_name="knn_count_regression",
        estimator=KNeighborsRegressor(
            n_neighbors=7,
            weights="distance",
        ),
        model_filename="count_knn.joblib",
        use_scaler=True,
    )
    all_results[knn_result["method"]] = knn_result
    prediction_frames.append(knn_predictions)

    hgb_result, hgb_predictions = _evaluate_feature_regressor(
        manifest=manifest,
        method_name="hist_gradient_boosting_count_regression",
        estimator=HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=500,
            random_state=42,
        ),
        model_filename="count_hist_gradient_boosting.joblib",
    )
    all_results[hgb_result["method"]] = hgb_result
    prediction_frames.append(hgb_predictions)

    test_ranking = []
    for method_name, payload in all_results.items():
        if "test" not in payload:
            continue
        test_ranking.append(
            {
                "method": method_name,
                "test_mae": payload["test"]["mae"],
                "test_rmse": payload["test"]["rmse"],
                "test_r2": payload["test"]["r2"],
            }
        )
    ranking_frame = pd.DataFrame(test_ranking).sort_values(["test_mae", "test_rmse", "test_r2"], ascending=[True, True, False])

    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(report_dir / "predictions.csv", index=False)
    ranking_frame.to_csv(report_dir / "ranking.csv", index=False)
    write_json(report_dir / "summary.json", {"methods": all_results, "ranking": ranking_frame.to_dict(orient="records")})
    return {"methods": all_results, "ranking": ranking_frame.to_dict(orient="records")}
