import pandas as pd
from utils import (
    CRIME_RAW,
    CRIME_CLEAN,
    IMD_RAW,
    IMD_CLEAN
)

# ============================================
# CRIME CLEANING
# ============================================

def load_raw_crime():
    """Load the raw LSOA-level crime CSV."""
    if not CRIME_RAW.exists():
        raise FileNotFoundError(f"Raw crime file not found: {CRIME_RAW}")

    df = pd.read_csv(CRIME_RAW)
    print(f"Loaded crime data: {df.shape}")
    return df


def identify_columns(df):
    """Separate ID columns from month columns."""
    id_cols = ["LSOA Code", "LSOA Name", "Borough", "Major Category", "Minor Category"]

    # Month columns = everything that is not an ID column
    month_cols = [col for col in df.columns if col not in id_cols]

    print(f"Found {len(month_cols)} month columns.")
    return id_cols, month_cols


def reshape_to_long(df, id_cols, month_cols):
    """Convert wide-format month columns into long format."""
    df_long = df.melt(
        id_vars=id_cols,
        value_vars=month_cols,
        var_name="Month",
        value_name="Crime Count"
    )
    print(f"Reshaped to long format: {df_long.shape}")
    return df_long


def clean_types(df_long):
    """Convert YYYYMM → datetime and ensure crime counts are numeric."""
    df_long["Month"] = pd.to_datetime(df_long["Month"], format="%Y%m")

    df_long["Crime Count"] = (
        pd.to_numeric(df_long["Crime Count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    return df_long


def clean_crime():
    """Full cleaning pipeline for crime data."""
    df = load_raw_crime()
    id_cols, month_cols = identify_columns(df)
    df_long = reshape_to_long(df, id_cols, month_cols)
    df_clean = clean_types(df_long)
    df_clean.to_csv(CRIME_CLEAN, index=False)
    print(f"Saved cleaned crime data → {CRIME_CLEAN}")
    return df_clean


# ============================================
# IMD CLEANING
# ============================================

def clean_imd():
    """Load and clean IMD 2019 dataset."""
    if not IMD_RAW.exists():
        raise FileNotFoundError(f"IMD file not found: {IMD_RAW}")

    # Load IMD with correct header
    df = pd.read_excel(IMD_RAW, sheet_name="IMD 2019")
    print("Loaded IMD:", df.shape)

    # Rename to clean and consistent names
    rename_map = {
        "LSOA code (2011)": "LSOA Code",
        "LSOA name (2011)": "LSOA Name",
        "Index of Multiple Deprivation (IMD) Score": "IMD Score",
        "Income Score (rate)": "Income Score",
        "Employment Score (rate)": "Employment Score",
        "Education, Skills and Training Score": "Education and Skills Score",
        "Health Deprivation and Disability Score": "Health Score",
        "Crime Score": "Crime Score",
        "Barriers to Housing and Services Score": "Barriers to Housing Score",
        "Living Environment Score": "Living Environment Score",
    }

    df = df.rename(columns=rename_map)

    # Now select only the needed columns
    keep_cols = [
        "LSOA Code",
        "LSOA Name",
        "IMD Score",
        "Income Score",
        "Employment Score",
        "Education and Skills Score",
        "Health Score",
        "Crime Score",
        "Barriers to Housing Score",
        "Living Environment Score",
    ]

    df_clean = df[keep_cols].copy()

    df_clean.to_csv(IMD_CLEAN, index=False)
    print(f"Saved cleaned IMD → {IMD_CLEAN}")

    return df_clean



# ============================================
# MASTER EXECUTION BLOCK
# ============================================

if __name__ == "__main__":
    print("\n=== Cleaning crime dataset ===")
    clean_crime()

    print("\n=== Cleaning IMD dataset ===")
    clean_imd()

    print("\n=== All cleaning complete ===")
