"""
Raw EDA Module for Crime Forecasting London
Produces a small, clean set of visuals showing:
- Crime distribution
- Temporal patterns
- IMD relationships
- Key deprivation drivers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from utils import PROCESSED_DIR

sns.set(style="whitegrid")

#---------------------------------------------------------
# Output Directory
#---------------------------------------------------------

RAW_DASH_DIR = Path(__file__).resolve().parents[1] / "dashboards" / "raw"
RAW_DASH_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name):
    path = RAW_DASH_DIR / name
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


#---------------------------------------------------------
# Borough Name Mapping (for readability)
#---------------------------------------------------------

BOROUGH_MAP = {
    "E09000001": "City of London", "E09000002": "Barking & Dagenham",
    "E09000003": "Barnet", "E09000004": "Bexley", "E09000005": "Brent",
    "E09000006": "Bromley", "E09000007": "Camden", "E09000008": "Croydon",
    "E09000009": "Ealing", "E09000010": "Enfield", "E09000011": "Greenwich",
    "E09000012": "Hackney", "E09000013": "Hammersmith & Fulham",
    "E09000014": "Haringey", "E09000015": "Harrow", "E09000016": "Havering",
    "E09000017": "Hillingdon", "E09000018": "Hounslow", "E09000019": "Islington",
    "E09000020": "Kensington & Chelsea", "E09000021": "Kingston",
    "E09000022": "Lambeth", "E09000023": "Lewisham", "E09000024": "Merton",
    "E09000025": "Newham", "E09000026": "Redbridge", "E09000027": "Richmond",
    "E09000028": "Southwark", "E09000029": "Sutton", "E09000030": "Tower Hamlets",
    "E09000031": "Waltham Forest", "E09000032": "Wandsworth",
    "E09000033": "Westminster"
}


#---------------------------------------------------------
# Load the raw cleaned + IMD merged data
#---------------------------------------------------------

def load_data():
    df = pd.read_csv(
        PROCESSED_DIR / "crime_with_imd.csv",
        parse_dates=["Month"]
    )
    df["Borough_Name"] = df["Borough"].map(BOROUGH_MAP)
    return df


#---------------------------------------------------------
# EDA Visuals
#---------------------------------------------------------

# 1) Crime by Borough
def plot_crime_by_borough(df):
    totals = df.groupby("Borough_Name")["Crime Count"].sum().sort_values()

    plt.figure(figsize=(12, 10))
    totals.plot(kind="barh", color="steelblue")
    plt.title("Total Crime by Borough")
    plt.xlabel("Crime Count")
    savefig("01_crime_by_borough.png")


# 2) Crime Over Time
def plot_crime_over_time(df):
    monthly = df.groupby("Month")["Crime Count"].sum()

    plt.figure(figsize=(14, 6))
    plt.plot(monthly.index, monthly.values, marker="o")
    plt.title("Crime Over Time (All London)")
    plt.ylabel("Crime Count")
    savefig("02_crime_over_time.png")


# 3) IMD Score vs Crime (Regression)
def plot_imd_vs_crime(df):
    summary = df.groupby("Borough_Name").agg({
        "Crime Count": "mean",
        "IMD Score": "mean"
    }).reset_index()

    plt.figure(figsize=(8, 6))
    sns.regplot(data=summary, x="IMD Score", y="Crime Count", scatter_kws={"alpha":0.7})
    plt.title("IMD Score vs Average Crime (by Borough)")
    savefig("03_imd_vs_crime_regression.png")

    corr = summary["IMD Score"].corr(summary["Crime Count"])
    print(f"IMD Score correlation with crime: {corr:.3f}")


# 4) IMD Domain Correlations
def plot_imd_domain_correlations(df):
    """
    Computes correlation between Crime Count and IMD socio-economic domains.
    Excludes the IMD Crime Domain and the overall IMD Index because they contain
    overlapping components that cause double-counting.
    """

    # Clean readable names
    domain_map = {
        "Income Score": "Income Deprivation",
        "Employment Score": "Employment Deprivation",
        "Education and Skills Score": "Education, Skills & Training Deprivation",
        "Health Score": "Health & Disability Deprivation",
        "Barriers to Housing Score": "Housing Access & Affordability Issues",
        "Living Environment Score": "Living Environment Deprivation"
    }

    selected = list(domain_map.keys())

    # Compute correlations with Crime Count
    corr_values = df[selected + ["Crime Count"]].corr()["Crime Count"].drop("Crime Count")

    # Apply readable labels
    corr_values.index = [domain_map[col] for col in corr_values.index]

    # Plot
    plt.figure(figsize=(12, 6))
    corr_values.sort_values().plot(kind="barh", color="royalblue")
    plt.title("Correlation of Crime with Socio-Economic Deprivation Factors")
    plt.xlabel("Correlation with Crime Count")
    savefig("04_imd_domain_correlations.png")



# 5) Heatmap (borough × month) — FIXED VERSION
def plot_heatmap(df):
    # Aggregate to borough-month totals
    monthly = df.groupby(["Borough_Name", "Month"])["Crime Count"].sum().reset_index()

    pivot = monthly.pivot(index="Borough_Name", columns="Month", values="Crime Count")
    pivot.columns = pivot.columns.strftime("%Y-%m")  # shorter labels

    # Normalise rows
    norm = (pivot - pivot.min(axis=1).values[:, None]) / (
        pivot.max(axis=1).values[:, None] - pivot.min(axis=1).values[:, None]
    )

    plt.figure(figsize=(14, 10))
    sns.heatmap(norm, cmap="magma", linewidths=0.2)
    plt.title("Normalised Crime Heatmap (Borough × Month)")
    savefig("05_crime_heatmap.png")


#---------------------------------------------------------
# Main Runner
#---------------------------------------------------------

def run_raw_eda():
    df = load_data()

    print("\n=== Running Raw EDA (simple + IMD-focused) ===")
    plot_crime_by_borough(df)
    plot_crime_over_time(df)
    plot_imd_vs_crime(df)
    plot_imd_domain_correlations(df)
    plot_heatmap(df)
    print("\n=== Raw EDA Complete. Files saved in dashboards/raw/ ===")


if __name__ == "__main__":
    run_raw_eda()
