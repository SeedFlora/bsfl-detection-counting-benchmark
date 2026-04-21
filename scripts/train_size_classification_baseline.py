from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from larvae_cv.data import build_or_load_feature_cache, read_manifest_csv, write_json
from larvae_cv.paths import PROCESSED_DIR, REPORTS_DIR
from larvae_cv.train import run_classifier_baseline


def main() -> None:
    manifest = read_manifest_csv(PROCESSED_DIR / "size_classification_manifest.csv")
    features = build_or_load_feature_cache(
        manifest=manifest,
        image_col="single_image_path",
        cache_name="single_image_features",
    )

    results = {
        "weight_size_class": run_classifier_baseline(
            manifest=manifest,
            features=features,
            image_col="single_image_path",
            target_col="weight_size_class",
            task_name="weight_size_classification",
            report_dir=REPORTS_DIR / "size_classification" / "weight",
        ),
        "length_size_class": run_classifier_baseline(
            manifest=manifest,
            features=features,
            image_col="single_image_path",
            target_col="length_size_class",
            task_name="length_size_classification",
            report_dir=REPORTS_DIR / "size_classification" / "length",
        ),
    }

    write_json(REPORTS_DIR / "size_classification" / "summary.json", results)
    print(results)


if __name__ == "__main__":
    main()
