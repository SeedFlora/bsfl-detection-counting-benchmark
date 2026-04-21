from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from larvae_cv.counting_benchmarks import evaluate_yolo_counting_weights
from larvae_cv.data import read_manifest_csv, remap_path_columns, write_json
from larvae_cv.paths import PROCESSED_DIR, REPORTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate counting quality from trained YOLO weights.")
    parser.add_argument("--weights", required=True, help="Path to YOLO weights file.")
    parser.add_argument("--device", default="cpu", help="Inference device, for example cpu or cuda.")
    parser.add_argument(
        "--method-name",
        default="yolo_detector",
        help="Label stored in the output summary and predictions.",
    )
    parser.add_argument(
        "--output-name",
        default="yolo_counting_eval",
        help="Output folder name under reports/counting_from_yolo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_manifest_csv(PROCESSED_DIR / "detection_counting_manifest.csv")
    manifest = remap_path_columns(manifest, ["source_image_path", "source_label_path"])
    summary, predictions = evaluate_yolo_counting_weights(
        manifest=manifest,
        weights_path=Path(args.weights),
        device=args.device,
        method_name=args.method_name,
    )

    output_dir = REPORTS_DIR / "counting_from_yolo" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    write_json(output_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
