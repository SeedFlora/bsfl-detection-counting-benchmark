from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import GroupShuffleSplit

from larvae_cv.paths import (
    BSF_IMAGES_DIR,
    BSF_LABELS_DIR,
    METADATA_XLSX,
    MODELS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    SEX_CROPS_DIR,
    SINGLE_LARVAE_DIR,
)


SIZE_LABELS = ["small", "medium", "large"]


def ensure_project_dirs() -> None:
    for path in [
        PROCESSED_DIR,
        PROCESSED_DIR / "features",
        PROCESSED_DIR / "detection_counting",
        REPORTS_DIR,
        MODELS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def read_manifest_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"image_id": str})


def resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(str(raw_path))
    if path.exists():
        return path

    normalized = str(raw_path).replace("\\", "/")
    parts = [part for part in re.split(r"/+", normalized) if part]
    for index in range(len(parts)):
        candidate = PROJECT_ROOT.joinpath(*parts[index:])
        if candidate.exists() or candidate.parent.exists():
            return candidate
    return path


def remap_path_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column not in frame.columns:
            continue
        frame[column] = frame[column].map(lambda value: str(resolve_project_path(value)) if pd.notna(value) else value)
    return frame


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_builtin(payload), handle, indent=2)


def write_runtime_detection_yaml() -> Path:
    ensure_project_dirs()
    dataset_root = PROCESSED_DIR / "detection_counting"
    runtime_yaml = dataset_root / "dataset.runtime.yaml"
    dataset_yaml = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "larva"},
    }
    with runtime_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dataset_yaml, handle, sort_keys=False)
    return runtime_yaml


def _three_band_labels(series: pd.Series) -> tuple[pd.Series, dict[str, float]]:
    clean = series.dropna().astype(float)
    lower = float(clean.quantile(1 / 3))
    upper = float(clean.quantile(2 / 3))

    if lower >= upper:
        ranked = clean.rank(method="first")
        labels = pd.qcut(ranked, q=3, labels=SIZE_LABELS)
        out = pd.Series(pd.NA, index=series.index, dtype="string")
        out.loc[labels.index] = labels.astype("string")
        return out, {"lower": lower, "upper": upper}

    bins = [-np.inf, lower, upper, np.inf]
    cut = pd.cut(series.astype(float), bins=bins, labels=SIZE_LABELS, include_lowest=True)
    return cut.astype("string"), {"lower": lower, "upper": upper}


def load_metadata() -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    rename_map = {
        "Batch_ID": "batch_id",
        "Date": "date",
        "L_ID": "l_id",
        "Weight": "weight",
        "Length": "length",
        "Width": "width",
        "Sex": "sex",
    }

    frame = pd.read_excel(METADATA_XLSX, sheet_name="Larva Data").rename(columns=rename_map)
    frame = frame.loc[frame["l_id"].notna()].copy()
    frame["l_id"] = frame["l_id"].astype(int)
    frame["image_id"] = frame["l_id"].map("{:04d}".format)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["batch_id"] = frame["batch_id"].fillna("NONE").astype(str)
    frame["sex"] = frame["sex"].astype("string").str.strip()
    frame.loc[frame["sex"] == "", "sex"] = pd.NA

    for column in ["weight", "length", "width"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["single_image_path"] = frame["image_id"].map(lambda image_id: str((SINGLE_LARVAE_DIR / f"{image_id}.jpg").resolve()))
    frame["sex_crop_path"] = frame["image_id"].map(lambda image_id: str((SEX_CROPS_DIR / f"{image_id}.jpg").resolve()))
    frame["single_image_exists"] = frame["single_image_path"].map(lambda value: Path(value).exists())
    frame["sex_crop_exists"] = frame["sex_crop_path"].map(lambda value: Path(value).exists())

    weight_labels, weight_bins = _three_band_labels(frame["weight"])
    length_labels, length_bins = _three_band_labels(frame["length"])
    frame["weight_size_class"] = weight_labels
    frame["length_size_class"] = length_labels

    frame["sex_crop_valid"] = False
    frame["has_any_image"] = frame["single_image_exists"] | frame["sex_crop_exists"]
    frame["preferred_image_path"] = frame["single_image_path"]
    frame["preferred_image_source"] = "single_fullres"
    frame["sex_task_image_path"] = frame["single_image_path"]
    frame["sex_task_image_source"] = "single_fullres"

    return frame, {
        "weight": weight_bins,
        "length": length_bins,
    }


def verify_image_file(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return False


def assign_group_splits(
    frame: pd.DataFrame,
    group_col: str,
    label_columns: list[str] | None = None,
    train_size: float = 0.7,
    random_state: int = 42,
) -> pd.Series:
    label_columns = label_columns or []

    for offset in range(100):
        first = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state + offset)
        train_idx, temp_idx = next(first.split(frame, groups=frame[group_col]))

        temp = frame.iloc[temp_idx]
        if temp[group_col].nunique() < 2:
            continue

        second = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=random_state + 100 + offset)
        val_rel, test_rel = next(second.split(temp, groups=temp[group_col]))
        val_idx = temp.index[val_rel]
        test_idx = temp.index[test_rel]

        split = pd.Series("train", index=frame.index, dtype="string")
        split.loc[val_idx] = "val"
        split.loc[test_idx] = "test"

        if _split_has_required_labels(frame, split, label_columns):
            return split

    raise RuntimeError(f"Unable to build grouped splits for {group_col}.")


def _split_has_required_labels(frame: pd.DataFrame, split: pd.Series, label_columns: list[str]) -> bool:
    for column in label_columns:
        global_values = set(frame[column].dropna().astype(str).unique())
        if not global_values:
            continue
        for split_name in ["train", "val", "test"]:
            split_values = set(frame.loc[split == split_name, column].dropna().astype(str).unique())
            if not global_values.issubset(split_values):
                return False
    return True


def build_single_larva_manifest(verify_crops: bool = True) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    manifest, thresholds = load_metadata()

    if verify_crops:
        valid_flags = []
        for crop_path, crop_exists in zip(manifest["sex_crop_path"], manifest["sex_crop_exists"]):
            valid_flags.append(bool(crop_exists) and verify_image_file(Path(crop_path)))
        manifest["sex_crop_valid"] = valid_flags
    else:
        manifest["sex_crop_valid"] = manifest["sex_crop_exists"]

    manifest["sex_task_image_path"] = np.where(manifest["sex_crop_valid"], manifest["sex_crop_path"], manifest["single_image_path"])
    manifest["sex_task_image_source"] = np.where(manifest["sex_crop_valid"], "sex_crop", "single_fullres")

    manifest["split"] = assign_group_splits(
        manifest,
        group_col="batch_id",
        label_columns=["sex", "weight_size_class", "length_size_class"],
    )

    ordered_columns = [
        "image_id",
        "batch_id",
        "date",
        "weight",
        "length",
        "width",
        "sex",
        "weight_size_class",
        "length_size_class",
        "split",
        "single_image_path",
        "single_image_exists",
        "sex_crop_path",
        "sex_crop_exists",
        "sex_crop_valid",
        "preferred_image_path",
        "preferred_image_source",
        "sex_task_image_path",
        "sex_task_image_source",
        "has_any_image",
    ]
    return manifest[ordered_columns].copy(), thresholds


def prepare_detection_counting_dataset(force: bool = True) -> pd.DataFrame:
    ensure_project_dirs()
    dataset_root = PROCESSED_DIR / "detection_counting"
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"

    rows: list[dict[str, Any]] = []
    for image_path in sorted(BSF_IMAGES_DIR.glob("*.jpg")):
        label_path = BSF_LABELS_DIR / f"{image_path.stem}.txt"
        raw_lines = []
        if label_path.exists():
            raw_lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        base_id = image_path.stem.split(".rf.")[0]
        class_ids = sorted({line.split()[0] for line in raw_lines if line.split()})
        rows.append(
            {
                "image_name": image_path.name,
                "base_id": base_id,
                "source_image_path": str(image_path.resolve()),
                "source_label_path": str(label_path.resolve()),
                "object_count": len(raw_lines),
                "source_class_ids": ",".join(class_ids),
            }
        )

    manifest = pd.DataFrame(rows)
    manifest["split"] = assign_group_splits(manifest, group_col="base_id")

    if force and dataset_root.exists():
        shutil.rmtree(dataset_root)

    for split_name in ["train", "val", "test"]:
        (images_root / split_name).mkdir(parents=True, exist_ok=True)
        (labels_root / split_name).mkdir(parents=True, exist_ok=True)

    for row in manifest.itertuples(index=False):
        source_image = Path(row.source_image_path)
        source_label = Path(row.source_label_path)
        target_image = images_root / row.split / source_image.name
        target_label = labels_root / row.split / source_label.name

        shutil.copy2(source_image, target_image)

        remapped_lines = []
        if source_label.exists():
            for line in source_label.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                remapped_lines.append(" ".join(["0", *parts[1:]]))
        target_label.write_text("\n".join(remapped_lines), encoding="utf-8")

    dataset_yaml = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "larva"},
    }
    with (dataset_root / "dataset.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dataset_yaml, handle, sort_keys=False)

    return manifest


def _largest_contour(mask: np.ndarray) -> tuple[np.ndarray | None, float]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0
    contour = max(contours, key=cv2.contourArea)
    return contour, float(cv2.contourArea(contour))


def _candidate_mask(gray: np.ndarray, inverse: bool) -> tuple[np.ndarray, np.ndarray | None]:
    flag = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, 0, 255, flag | cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contour, _ = _largest_contour(mask)
    return mask, contour


def _best_object_mask(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    best_mask = np.zeros_like(gray)
    best_contour = None
    best_score = -1.0
    image_area = float(gray.shape[0] * gray.shape[1])

    for inverse in [True, False]:
        mask, contour = _candidate_mask(gray, inverse=inverse)
        if contour is None:
            continue
        area = float(cv2.contourArea(contour))
        area_fraction = area / image_area
        if not 0.001 <= area_fraction <= 0.85:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        center_score = 1.0 - (
            abs((x + w / 2) / gray.shape[1] - 0.5) + abs((y + h / 2) / gray.shape[0] - 0.5)
        )
        score = area_fraction + 0.05 * center_score
        if score > best_score:
            best_mask = mask
            best_contour = contour
            best_score = score

    return best_mask, best_contour


def extract_single_larva_features(image_path: Path, thumb_size: int = 12, work_max_side: int = 1024) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    orig_h, orig_w = image.shape[:2]
    scale = min(1.0, work_max_side / float(max(orig_h, orig_w)))
    if scale < 1.0:
        work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        work = image.copy()

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, contour = _best_object_mask(blur)

    features: dict[str, Any] = {
        "image_width": orig_w,
        "image_height": orig_h,
        "feat_scale_used": scale,
        "feat_segmentation_success": int(contour is not None),
        "feat_gray_mean": float(gray.mean()),
        "feat_gray_std": float(gray.std()),
    }

    if contour is None:
        object_pixels = gray.reshape(-1)
        background_pixels = gray.reshape(-1)
        area = 0.0
        perimeter = 0.0
        bbox_w = 0
        bbox_h = 0
        convex_area = 0.0
        extent = 0.0
        solidity = 0.0
        center_x = 0.5
        center_y = 0.5
        major_axis = 0.0
        minor_axis = 0.0
        angle = 0.0
    else:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        x, y, bbox_w, bbox_h = cv2.boundingRect(contour)
        hull = cv2.convexHull(contour)
        convex_area = float(cv2.contourArea(hull))
        extent = area / max(1.0, float(bbox_w * bbox_h))
        solidity = area / max(1.0, convex_area)
        center_x = (x + bbox_w / 2) / gray.shape[1]
        center_y = (y + bbox_h / 2) / gray.shape[0]

        if len(contour) >= 5:
            (_, _), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
            major_axis = float(max(axis_a, axis_b))
            minor_axis = float(min(axis_a, axis_b))
        else:
            rect = cv2.minAreaRect(contour)
            axis_a, axis_b = rect[1]
            angle = float(rect[2])
            major_axis = float(max(axis_a, axis_b))
            minor_axis = float(min(axis_a, axis_b))

        filled = np.zeros_like(gray)
        cv2.drawContours(filled, [contour], contourIdx=-1, color=255, thickness=-1)
        object_pixels = gray[filled > 0]
        background_pixels = gray[filled == 0]

    image_area = float(gray.shape[0] * gray.shape[1])
    features.update(
        {
            "feat_area_fraction": area / image_area,
            "feat_perimeter_norm": perimeter / max(1.0, np.sqrt(image_area)),
            "feat_bbox_width_fraction": bbox_w / max(1.0, float(gray.shape[1])),
            "feat_bbox_height_fraction": bbox_h / max(1.0, float(gray.shape[0])),
            "feat_aspect_ratio": (bbox_w / max(1.0, float(bbox_h))) if bbox_h else 0.0,
            "feat_extent": float(extent),
            "feat_solidity": float(solidity),
            "feat_center_x": float(center_x),
            "feat_center_y": float(center_y),
            "feat_major_axis_fraction": major_axis / max(1.0, float(max(gray.shape))),
            "feat_minor_axis_fraction": minor_axis / max(1.0, float(max(gray.shape))),
            "feat_axis_ratio": major_axis / max(1.0, minor_axis),
            "feat_angle_abs": abs(angle) / 180.0,
            "feat_object_mean": float(object_pixels.mean()) if len(object_pixels) else 0.0,
            "feat_object_std": float(object_pixels.std()) if len(object_pixels) else 0.0,
            "feat_background_mean": float(background_pixels.mean()) if len(background_pixels) else 0.0,
            "feat_background_std": float(background_pixels.std()) if len(background_pixels) else 0.0,
        }
    )

    thumbnail = cv2.resize(gray, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    for index, value in enumerate(thumbnail.reshape(-1)):
        features[f"px_{index:03d}"] = float(value)

    return features


def build_or_load_feature_cache(
    manifest: pd.DataFrame,
    image_col: str,
    cache_name: str,
    force: bool = False,
) -> pd.DataFrame:
    ensure_project_dirs()
    cache_path = PROCESSED_DIR / "features" / f"{cache_name}.csv"
    if cache_path.exists() and not force:
        return read_manifest_csv(cache_path)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for record in manifest.to_dict(orient="records"):
        image_id = str(record["image_id"]).zfill(4)
        image_path = record.get(image_col)
        if pd.isna(image_path) or not image_path:
            continue
        image_path = str(image_path)
        key = (image_id, image_path)
        if key in seen:
            continue
        seen.add(key)

        path = Path(image_path)
        if not path.exists():
            continue

        try:
            features = extract_single_larva_features(path)
            features["feature_error"] = ""
        except Exception as exc:  # pragma: no cover
            features = {"feature_error": f"{type(exc).__name__}: {exc}"}

        rows.append(
            {
                "image_id": image_id,
                image_col: image_path,
                **features,
            }
        )

    feature_frame = pd.DataFrame(rows)
    feature_frame.to_csv(cache_path, index=False)
    return feature_frame


def save_preparation_summary(
    single_manifest: pd.DataFrame,
    detection_manifest: pd.DataFrame,
    thresholds: dict[str, dict[str, float]],
) -> None:
    summary = {
        "project_root": str(PROJECT_ROOT.resolve()),
        "single_larvae": {
            "rows": int(len(single_manifest)),
            "single_images_present": int(single_manifest["single_image_exists"].sum()),
            "valid_sex_crops": int(single_manifest["sex_crop_valid"].sum()),
            "sex_labeled": int(single_manifest["sex"].notna().sum()),
            "regression_ready": int(single_manifest[["weight", "length"]].notna().all(axis=1).sum()),
            "weight_size_distribution": single_manifest["weight_size_class"].value_counts(dropna=False).to_dict(),
            "length_size_distribution": single_manifest["length_size_class"].value_counts(dropna=False).to_dict(),
            "split_distribution": single_manifest["split"].value_counts().to_dict(),
        },
        "detection_counting": {
            "rows": int(len(detection_manifest)),
            "total_objects": int(detection_manifest["object_count"].sum()),
            "split_distribution": detection_manifest["split"].value_counts().to_dict(),
            "split_object_counts": detection_manifest.groupby("split")["object_count"].sum().to_dict(),
        },
        "size_thresholds": thresholds,
    }
    write_json(REPORTS_DIR / "preparation_summary.json", summary)
