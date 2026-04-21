from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from larvae_cv.data import write_runtime_detection_yaml
from larvae_cv.paths import PROCESSED_DIR, REPORTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on the prepared larvae detection/counting dataset.")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model checkpoint.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--device", default="cpu", help="Training device, for example cpu or 0.")
    parser.add_argument("--run-name", default="larvae_counting", help="Output run name.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    return parser.parse_args()


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "ultralytics belum terpasang. Jalankan `pip install ultralytics` lalu ulangi command ini."
        ) from exc

    args = parse_args()
    dataset_yaml = write_runtime_detection_yaml()
    model = YOLO(args.model)
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        workers=args.workers,
        project=str(REPORTS_DIR / "yolo_runs"),
        name=args.run_name,
    )
    metrics = model.val(data=str(dataset_yaml), split="test", device=args.device)
    print(metrics.results_dict)


if __name__ == "__main__":
    main()
