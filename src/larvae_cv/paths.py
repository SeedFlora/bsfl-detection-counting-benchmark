from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

SINGLE_LARVAE_DIR = RAW_DIR / "single_larvae_data" / "Single Larvae data" / "single_larvae_renamed_IDs"
SEX_CROPS_DIR = RAW_DIR / "larva_sex_classification" / "Larva_sex_classification"
BSF_IMAGES_DIR = RAW_DIR / "bsf_larvae_v1" / "BSF_Larvae_v1" / "Images"
BSF_LABELS_DIR = RAW_DIR / "bsf_larvae_v1" / "BSF_Larvae_v1" / "Labels"
METADATA_XLSX = RAW_DIR / "Larvae--size-data.xlsx"
