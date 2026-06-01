from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from larvae_cv.data import read_manifest_csv, remap_path_columns, write_json
from larvae_cv.paths import PROCESSED_DIR, REPORTS_DIR


ImageTransform = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO counting robustness under deterministic test-time image perturbations."
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        default=[],
        help="YOLO weight paths. If omitted, best.pt files from the existing multi-seed runs are used.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "original",
            "brightness_dark",
            "brightness_bright",
            "contrast_low",
            "contrast_high",
            "gaussian_blur",
            "gaussian_noise",
            "synthetic_occlusion",
        ],
        help="Perturbation conditions to evaluate.",
    )
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Dataset split to perturb and evaluate.")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold for counting.")
    parser.add_argument("--iou", type=float, default=0.70, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="cuda", help="Inference device, for example cuda or cpu.")
    parser.add_argument("--max-det", type=int, default=1000, help="Maximum detections per image.")
    parser.add_argument("--seed", type=int, default=2026, help="Seed for deterministic noise and occlusion.")
    parser.add_argument(
        "--benchmark-name",
        default="",
        help="Output folder under reports/reviewer_experiments/robustness. Default uses a timestamp.",
    )
    parser.add_argument(
        "--reuse-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse already generated perturbed images if present.",
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


def clip_image(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def identity(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return image.copy()


def brightness_dark(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return clip_image(image.astype(np.float32) - 45.0)


def brightness_bright(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return clip_image(image.astype(np.float32) + 45.0)


def contrast_low(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return clip_image((image.astype(np.float32) - 127.5) * 0.70 + 127.5)


def contrast_high(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return clip_image((image.astype(np.float32) - 127.5) * 1.30 + 127.5)


def gaussian_blur(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return cv2.GaussianBlur(image, (5, 5), sigmaX=1.2)


def gaussian_noise(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=12.0, size=image.shape)
    return clip_image(image.astype(np.float32) + noise)


def synthetic_occlusion(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    occ_w = max(8, int(w * 0.12))
    occ_h = max(8, int(h * 0.12))
    x = int(rng.integers(0, max(1, w - occ_w + 1)))
    y = int(rng.integers(0, max(1, h - occ_h + 1)))
    patch_color = np.array([int(out[:, :, channel].mean()) for channel in range(3)], dtype=np.uint8)
    out[y : y + occ_h, x : x + occ_w] = patch_color
    return out


TRANSFORMS: dict[str, ImageTransform] = {
    "original": identity,
    "brightness_dark": brightness_dark,
    "brightness_bright": brightness_bright,
    "contrast_low": contrast_low,
    "contrast_high": contrast_high,
    "gaussian_blur": gaussian_blur,
    "gaussian_noise": gaussian_noise,
    "synthetic_occlusion": synthetic_occlusion,
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


def build_condition_images(
    split_frame: pd.DataFrame,
    condition: str,
    condition_dir: Path,
    args: argparse.Namespace,
) -> list[Path]:
    if condition not in TRANSFORMS:
        raise ValueError(f"Unknown condition '{condition}'. Choices: {sorted(TRANSFORMS)}")

    condition_dir.mkdir(parents=True, exist_ok=True)
    transform = TRANSFORMS[condition]
    paths: list[Path] = []

    for row in split_frame.itertuples(index=False):
        source_path = Path(row.source_image_path)
        if condition == "original":
            paths.append(source_path)
            continue

        target_path = condition_dir / source_path.name
        if target_path.exists() and args.reuse_images:
            paths.append(target_path)
            continue

        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {source_path}")
        seed_payload = f"{args.seed}|{condition}|{source_path.name}".encode("utf-8")
        per_image_seed = int(hashlib.sha256(seed_payload).hexdigest()[:8], 16)
        rng = np.random.default_rng(per_image_seed)
        transformed = transform(image, rng)
        cv2.imwrite(str(target_path), transformed)
        paths.append(target_path)

    return paths


def evaluate_condition(
    model: Any,
    split_frame: pd.DataFrame,
    image_paths: list[Path],
    args: argparse.Namespace,
) -> tuple[dict[str, float], np.ndarray]:
    outputs = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        verbose=False,
    )
    y_true = split_frame["object_count"].to_numpy(dtype=float)
    y_pred = np.array([len(output.boxes) for output in outputs], dtype=float)
    return count_metrics(y_true, y_pred), y_pred


def evaluate_weight(
    weights_path: Path,
    split_frame: pd.DataFrame,
    perturbed_root: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from ultralytics import YOLO

    label = run_label(weights_path)
    variant = variant_label(label)
    model = YOLO(str(weights_path))
    result_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for condition in args.conditions:
        print(f"[robustness] {label} condition={condition}")
        image_paths = build_condition_images(split_frame, condition, perturbed_root / condition, args)
        metrics, y_pred = evaluate_condition(model, split_frame, image_paths, args)
        y_true = split_frame["object_count"].to_numpy(dtype=float)
        result_rows.append(
            {
                "run_label": label,
                "variant": variant,
                "weights_path": str(weights_path.resolve()),
                "split": args.split,
                "condition": condition,
                "conf": args.conf,
                "iou": args.iou,
                **metrics,
            }
        )
        for image_name, path, truth, pred in zip(split_frame["image_name"], image_paths, y_true, y_pred):
            prediction_rows.append(
                {
                    "run_label": label,
                    "variant": variant,
                    "split": args.split,
                    "condition": condition,
                    "image_name": image_name,
                    "image_path": str(Path(path).resolve()),
                    "y_true": float(truth),
                    "y_pred": float(pred),
                    "error": float(pred - truth),
                    "abs_error": float(abs(pred - truth)),
                }
            )

    return pd.DataFrame(result_rows), pd.DataFrame(prediction_rows)


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["mae", "rmse", "r2", "exact_match_rate", "mean_pred"]
    grouped = results.groupby(["variant", "condition"], dropna=False)[metric_cols].agg(["mean", "std", "min", "max"])
    grouped.columns = ["_".join(item) for item in grouped.columns]
    return grouped.reset_index().sort_values(["condition", "mae_mean"])


def main() -> None:
    args = parse_args()
    benchmark_name = args.benchmark_name or f"robustness_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir = REPORTS_DIR / "reviewer_experiments" / "robustness" / benchmark_name
    perturbed_root = output_dir / "perturbed_images" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    unknown = sorted(set(args.conditions) - set(TRANSFORMS))
    if unknown:
        raise SystemExit(f"Unknown conditions: {unknown}. Choices: {sorted(TRANSFORMS)}")

    weights = [Path(item) for item in args.weights] if args.weights else discover_default_weights()
    if not weights:
        raise SystemExit("No weights found. Pass --weights or train/evaluate YOLO models first.")

    manifest = read_manifest_csv(PROCESSED_DIR / "detection_counting_manifest.csv")
    manifest = remap_path_columns(manifest, ["source_image_path", "source_label_path"])
    split_frame = manifest.loc[manifest["split"] == args.split].copy().reset_index(drop=True)

    result_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for weights_path in weights:
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        results, predictions = evaluate_weight(weights_path, split_frame, perturbed_root, args)
        result_frames.append(results)
        prediction_frames.append(predictions)

    results_frame = pd.concat(result_frames, ignore_index=True)
    predictions_frame = pd.concat(prediction_frames, ignore_index=True)
    aggregate = aggregate_results(results_frame)

    results_frame.to_csv(output_dir / "condition_results.csv", index=False)
    predictions_frame.to_csv(output_dir / "predictions.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_condition_results.csv", index=False)

    summary = {
        "benchmark_name": benchmark_name,
        "weights": [str(path.resolve()) for path in weights],
        "conditions": args.conditions,
        "split": args.split,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "device": args.device,
        "max_det": args.max_det,
        "seed": args.seed,
        "aggregate": aggregate.to_dict(orient="records"),
        "outputs": {
            "condition_results": str((output_dir / "condition_results.csv").resolve()),
            "predictions": str((output_dir / "predictions.csv").resolve()),
            "aggregate_condition_results": str((output_dir / "aggregate_condition_results.csv").resolve()),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(aggregate.to_string(index=False))
    print(f"Saved robustness benchmark to {output_dir}")


if __name__ == "__main__":
    main()
