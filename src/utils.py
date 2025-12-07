from pathlib import Path


# -----------------------------
# Base Project Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Raw files
CRIME_RAW = RAW_DIR / "crime_raw.csv"
IMD_RAW = RAW_DIR / "imd_2019_raw.xlsx"

# Processed files
CRIME_CLEAN = PROCESSED_DIR / "crime_clean.csv"
IMD_CLEAN = PROCESSED_DIR / "imd_2019_clean.csv"
CRIME_IMD_MERGED = PROCESSED_DIR / "crime_with_imd.csv"
MODELLING_DATA = PROCESSED_DIR / "modelling_dataset.csv"


# -----------------------------
# Ensure folders exist
# -----------------------------
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
