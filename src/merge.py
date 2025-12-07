import pandas as pd
from utils import (
    CRIME_CLEAN,
    IMD_CLEAN,
    CRIME_IMD_MERGED,
    MODELLING_DATA
)

def load_clean_data():
    crime = pd.read_csv(CRIME_CLEAN)
    imd = pd.read_csv(IMD_CLEAN)

    print("Crime clean:", crime.shape)
    print("IMD clean:", imd.shape)
    
    return crime, imd


def merge_crime_imd(crime, imd):
    merged = crime.merge(imd, on="LSOA Code", how="left")
    print("Merged dataset:", merged.shape)

    # Fix duplicate LSOA Name columns
    if "LSOA Name_x" in merged.columns:
        merged = merged.drop(columns=["LSOA Name_x"])
    if "LSOA Name_y" in merged.columns:
        merged = merged.rename(columns={"LSOA Name_y": "LSOA Name"})

    merged.to_csv(CRIME_IMD_MERGED, index=False)
    print(f"Saved merged dataset → {CRIME_IMD_MERGED}")

    return merged


def create_modelling_dataset(merged):
    """
    Aggregate crime by month + borough + IMD attributes
    to create features for forecasting and ML.
    """
    agg = (
        merged.groupby(["Month", "Borough"])
        .agg({
            "Crime Count": "sum",
            "IMD Score": "mean",
            "Income Score": "mean",
            "Employment Score": "mean",
            "Education and Skills Score": "mean",
            "Health Score": "mean",
            "Crime Score": "mean",
            "Barriers to Housing Score": "mean",
            "Living Environment Score": "mean",
        })
        .reset_index()
    )

    print("Modelling dataset:", agg.shape)

    agg.to_csv(MODELLING_DATA, index=False)
    print(f"Saved modelling dataset → {MODELLING_DATA}")

    return agg



if __name__ == "__main__":
    crime, imd = load_clean_data()
    merged = merge_crime_imd(crime, imd)
    create_modelling_dataset(merged)
