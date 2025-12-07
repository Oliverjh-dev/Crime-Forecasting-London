"""
Clean, portfolio-ready evaluation suite for Crime Forecasting London.
Interpretable GLM diagnostics, percentage-based residuals,
and coefficients expressed as incidence rate ratios (IRR).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from pathlib import Path
from utils import PROCESSED_DIR

sns.set(style="whitegrid")

# Output folder
EVAL_DIR = Path(__file__).resolve().parents[1] / "dashboards" / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

def savefig(name):
    path = EVAL_DIR / name
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ============================================================
# 1. LOAD PREDICTIONS AND MODEL COMPARISON
# ============================================================

def load_data():
    preds = pd.read_csv(PROCESSED_DIR / "model_predictions.csv", parse_dates=["Month"])

    # Standardise expected naming
    if "Crime Count" in preds.columns:
        preds = preds.rename(columns={"Crime Count": "Actual"})

    comp = pd.read_csv(PROCESSED_DIR / "model_comparison.csv")
    return preds, comp


# ============================================================
# 2. MODEL PERFORMANCE BAR CHART
# ============================================================

def plot_model_comparison(comp):
    plt.figure(figsize=(10, 6))

    comp_melt = comp.melt(id_vars="Model", value_vars=["MAE", "RMSE"])

    sns.barplot(
        data=comp_melt,
        x="Model",
        y="value",
        hue="variable",
        palette=["#4C72B0", "#DD8452"]
    )

    plt.title("Model Performance (Test Set)")
    plt.ylabel("Error")
    plt.xlabel("Model")
    savefig("01_model_comparison.png")


# ============================================================
# 3. TOTAL LONDON ACTUAL vs PREDICTED
# ============================================================

def plot_total_london(preds):
    total = preds.groupby("Month")[["Actual", "GLM_Predict", "MixedLM_Predict"]].sum()

    plt.figure(figsize=(14, 6))
    plt.plot(total.index, total["Actual"], marker="o", label="Actual")
    plt.plot(total.index, total["GLM_Predict"], marker="o", label="GLM Predicted")
    plt.plot(total.index, total["MixedLM_Predict"], marker="o", label="MixedLM Predicted")

    plt.title("Total London Crime – Actual vs Predicted")
    plt.ylabel("Crime Count")
    plt.legend()
    savefig("02_total_london_actual_vs_pred.png")


# ============================================================
# 4. BOROUGH-LEVEL GLM ERROR (MAE)
# ============================================================

def plot_borough_mae(preds):
    borough_mae = preds.groupby("Borough").apply(
        lambda d: np.mean(np.abs(d["Actual"] - d["GLM_Predict"]))
    ).sort_values()

    plt.figure(figsize=(12, 10))
    sns.barplot(x=borough_mae.values, y=borough_mae.index)
    plt.title("GLM Error by Borough (Test Set)")
    plt.xlabel("Mean Absolute Error")
    plt.ylabel("Borough")
    savefig("03_borough_mae_glm.png")


# ============================================================
# 5. RESIDUAL ANALYSIS (percentage-based)
# ============================================================

def plot_residuals(preds):
    preds["Residual"] = preds["Actual"] - preds["GLM_Predict"]
    preds["Pct_Residual"] = (preds["Residual"] / preds["Actual"]) * 100

    # Histogram
    plt.figure(figsize=(12, 6))
    sns.histplot(preds["Pct_Residual"], bins=30, kde=True)
    plt.title("GLM Percentage Residuals")
    plt.xlabel("Residual (% of Actual)")
    savefig("04_glm_residual_hist_pct.png")

    # Residual vs fitted (percentage)
    plt.figure(figsize=(12, 6))
    plt.scatter(preds["GLM_Predict"], preds["Pct_Residual"], alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.xscale("log")
    plt.xlabel("Fitted Value (log scale)")
    plt.ylabel("Residual (% of Actual)")
    plt.title("Residuals vs Fitted (Percentage)")
    savefig("05_glm_residual_vs_fitted_pct.png")


# ============================================================
# 6. COEFFICIENTS AS HUMAN-READABLE EFFECTS (IRR)
# ============================================================

def plot_coefficients():
    pickle_path = PROCESSED_DIR / "glm_results.pickle"

    # -------- Safety: skip if model pickle doesn't exist --------
    if not pickle_path.exists():
        print("⚠️ glm_results.pickle not found — skipping coefficient plot.")
        return

    glm = sm.load(pickle_path)

    coefs = glm.params.drop("const")
    irr = np.exp(coefs) - 1     # convert log-coefs → % effect on rate

    irr_df = irr.sort_values()

    plt.figure(figsize=(10, 14))
    sns.barplot(x=irr_df.values * 100, y=irr_df.index)
    plt.xlabel("Effect on Crime (%)")
    plt.title("GLM Feature Impact (Incidence Rate Ratios)")
    savefig("06_glm_coefficients_irr.png")

    irr.to_csv(PROCESSED_DIR / "glm_coefficients_irr.csv")


# ============================================================
# MASTER RUNNER
# ============================================================

def run_evaluation():
    print("Loading predictions and comparison metrics...")
    preds, comp = load_data()

    print("Generating model comparison plot...")
    plot_model_comparison(comp)

    print("Generating total London actual vs predicted...")
    plot_total_london(preds)

    print("Generating borough MAE plot...")
    plot_borough_mae(preds)

    print("Generating GLM residual diagnostics...")
    plot_residuals(preds)

    print("Generating GLM coefficient plot (IRR)...")
    plot_coefficients()

    print("\nEvaluation complete ✓")


if __name__ == "__main__":
    run_evaluation()
