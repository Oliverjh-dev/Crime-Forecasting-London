"""
Unified Modelling Module for Crime Forecasting London

Models:
 - Negative Binomial GLM with IMD + seasonal Fourier terms + borough dummies
 - Hierarchical Mixed-Effects (MixedLM) with borough random intercepts

Outputs:
 - model_comparison.csv
 - model_predictions.csv
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from pathlib import Path

from utils import MODELLING_DATA, PROCESSED_DIR


# =========================================================
# 1. Fourier Seasonal Features
# =========================================================

def add_fourier_terms(df: pd.DataFrame, K: int = 2) -> pd.DataFrame:
    """
    Add K Fourier harmonics (sin/cos) for monthly seasonality.
    """
    month = df["Month"].dt.month
    w = 2 * np.pi / 12

    for k in range(1, K + 1):
        df[f"Fourier_sin_{k}"] = np.sin(k * w * month)
        df[f"Fourier_cos_{k}"] = np.cos(k * w * month)

    return df


# =========================================================
# 2. Load + Prepare Data
# =========================================================

def load_and_prepare():
    """
    Load modelling dataset and create:
      - Time_Index
      - Month_sin / Month_cos
      - Fourier terms
      - Clean IMD Score
      - Borough one-hot dummies
    """
    df = pd.read_csv(MODELLING_DATA, parse_dates=["Month"])

    # Sort for sanity
    df = df.sort_values(["Borough", "Month"]).reset_index(drop=True)

    # Time index in months since start
    df["Time_Index"] = (df["Month"] - df["Month"].min()).dt.days / 30.0

    # Basic seasonal terms
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"].dt.month / 12.0)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"].dt.month / 12.0)

    # Fourier harmonics
    df = add_fourier_terms(df, K=2)

    # Clean up IMD in case it has commas or weird formatting
    df["IMD Score"] = (
        df["IMD Score"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace("nan", np.nan)
        .astype(float)
    )

    # Borough dummies
    borough_dummies = pd.get_dummies(df["Borough"], prefix="B", dtype=float)
    df = pd.concat([df, borough_dummies], axis=1)

    borough_cols = borough_dummies.columns.tolist()
    return df, borough_cols


# =========================================================
# 3. Feature Matrix Builders
# =========================================================

GLM_FEATURES = [
    "Time_Index",
    "Month_sin", "Month_cos",
    "Fourier_sin_1", "Fourier_cos_1",
    "Fourier_sin_2", "Fourier_cos_2",
    "IMD Score",
]


def enforce_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """
    Force ALL columns to float64 (no bools, no objects).
    """
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")
    return X


def build_glm_matrix(df: pd.DataFrame, borough_cols) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build design matrix X and target y for GLM.
    Ensures:
      - float64 features
      - intercept column 'const'
    """
    cols = GLM_FEATURES + borough_cols
    X = df[cols].copy()

    X = enforce_numeric(X)
    X = X.fillna(X.mean(numeric_only=True))

    # Add intercept as float64
    X = sm.add_constant(X)
    X["const"] = X["const"].astype("float64")

    y = pd.to_numeric(df["Crime Count"], errors="coerce").astype(float).values

    return X, y


def build_mixedlm_matrix(df: pd.DataFrame):
    """
    Build X, y, groups for MixedLM (no borough dummies, borough as group).
    """
    X = df[GLM_FEATURES].copy()
    X = enforce_numeric(X)
    X = X.fillna(X.mean(numeric_only=True))
    X = sm.add_constant(X)
    X["const"] = X["const"].astype("float64")

    y = pd.to_numeric(df["Crime Count"], errors="coerce").astype(float).values
    groups = df["Borough"]

    return X, y, groups


# =========================================================
# 4. Models
# =========================================================

def fit_glm(X: pd.DataFrame, y: np.ndarray):
    """
    Fit a Negative Binomial GLM, forcing NumPy float64 arrays for exog/endog.
    This avoids any Pandas object-dtype issues.
    """
    X_np = np.asarray(X.values, dtype="float64")
    y_np = np.asarray(y, dtype="float64")

    model = sm.GLM(y_np, X_np, family=sm.families.NegativeBinomial())
    result = model.fit(maxiter=300)
    return result


def fit_mixedlm(X: pd.DataFrame, y: np.ndarray, groups: pd.Series):
    """
    Fit a mixed-effects model with borough random intercepts.
    """
    model = MixedLM(endog=y, exog=X, groups=groups)
    result = model.fit(reml=True)
    return result


# =========================================================
# 5. Evaluation
# =========================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return mae, rmse


# =========================================================
# 6. Orchestration
# =========================================================

def run_modelling():
    print("\nLoading modelling dataset...")
    df, borough_cols = load_and_prepare()

    # -------------------------
    # Train/Test Split (last 6 months as test)
    # -------------------------
    unique_months = df["Month"].sort_values().unique()
    split_point = unique_months[-6]

    train = df[df["Month"] < split_point].copy()
    test = df[df["Month"] >= split_point].copy()

    print(f"\nTrain rows: {len(train)}")
    print(f"Test rows : {len(test)}")

    # -------------------------
    # Build design matrices
    # -------------------------
    print("\n=== Building GLM matrix ===")
    X_train_glm, y_train = build_glm_matrix(train, borough_cols)
    X_test_glm, y_test = build_glm_matrix(test, borough_cols)
    print("GLM matrix built ✔")

    print("\n=== Building MixedLM matrix ===")
    X_train_m, y_train_m, groups_train = build_mixedlm_matrix(train)
    X_test_m, _, _ = build_mixedlm_matrix(test)
    print("MixedLM matrix built ✔")

    # -------------------------
    # Fit Models
    # -------------------------
    print("\nFitting Negative Binomial GLM...")
    glm_res = fit_glm(X_train_glm, y_train)

    print("\nFitting Hierarchical MixedLM (borough random intercept)...")
    mix_res = fit_mixedlm(X_train_m, y_train_m, groups_train)

    # -------------------------
    # Predictions
    # -------------------------
    glm_pred = glm_res.predict(np.asarray(X_test_glm.values, dtype="float64"))
    mix_pred = mix_res.predict(X_test_m)

    # -------------------------
    # Evaluation
    # -------------------------
    glm_mae, glm_rmse = evaluate(y_test, glm_pred)
    mix_mae, mix_rmse = evaluate(y_test, mix_pred)

    print("\n=== MODEL PERFORMANCE (TEST SET) ===")
    print(f"GLM NegBin     →  MAE {glm_mae:.2f}, RMSE {glm_rmse:.2f}")
    print(f"MixedLM (REML) →  MAE {mix_mae:.2f}, RMSE {mix_rmse:.2f}")

    # -------------------------
    # Save comparison
    # -------------------------
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    comparison = pd.DataFrame({
        "Model": ["GLM_NegBin", "MixedLM"],
        "MAE": [glm_mae, mix_mae],
        "RMSE": [glm_rmse, mix_rmse],
    })
    comp_path = PROCESSED_DIR / "model_comparison.csv"
    comparison.to_csv(comp_path, index=False)
    print(f"\nSaved model comparison → {comp_path}")

    # -------------------------
    # Save predictions
    # -------------------------
    out = test[["Month", "Borough", "Crime Count"]].copy()
    out["GLM_Predict"] = glm_pred
    out["MixedLM_Predict"] = mix_pred

    pred_path = PROCESSED_DIR / "model_predictions.csv"
    out.to_csv(pred_path, index=False)
    print(f"Saved predictions → {pred_path}")

    print("\nModelling complete ✓")


# =========================================================
# Script entry point
# =========================================================

if __name__ == "__main__":
    run_modelling()
