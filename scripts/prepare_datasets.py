from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from larvae_cv.data import (
    build_single_larva_manifest,
    ensure_project_dirs,
    prepare_detection_counting_dataset,
    save_preparation_summary,
)
from larvae_cv.paths import PROCESSED_DIR


def main() -> None:
    ensure_project_dirs()

    single_manifest, thresholds = build_single_larva_manifest(verify_crops=True)
    single_manifest.to_csv(PROCESSED_DIR / "single_larvae_manifest.csv", index=False)

    regression_manifest = single_manifest.loc[
        single_manifest["single_image_exists"] & single_manifest["weight"].notna() & single_manifest["length"].notna()
    ].copy()
    regression_manifest.to_csv(PROCESSED_DIR / "regression_manifest.csv", index=False)

    size_manifest = single_manifest.loc[
        single_manifest["single_image_exists"]
        & single_manifest["weight_size_class"].notna()
        & single_manifest["length_size_class"].notna()
    ].copy()
    size_manifest.to_csv(PROCESSED_DIR / "size_classification_manifest.csv", index=False)

    sex_manifest = single_manifest.loc[
        single_manifest["has_any_image"] & single_manifest["sex"].notna()
    ].copy()
    sex_manifest.to_csv(PROCESSED_DIR / "sex_classification_manifest.csv", index=False)

    detection_manifest = prepare_detection_counting_dataset(force=True)
    detection_manifest.to_csv(PROCESSED_DIR / "detection_counting_manifest.csv", index=False)

    save_preparation_summary(single_manifest, detection_manifest, thresholds)

    print("Saved:")
    print(PROCESSED_DIR / "single_larvae_manifest.csv")
    print(PROCESSED_DIR / "regression_manifest.csv")
    print(PROCESSED_DIR / "size_classification_manifest.csv")
    print(PROCESSED_DIR / "sex_classification_manifest.csv")
    print(PROCESSED_DIR / "detection_counting_manifest.csv")


if __name__ == "__main__":
    main()
