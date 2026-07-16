"""
model_trainer.py — ML model training, evaluation, and persistence.

Responsibilities:
- Train Linear Regression, Random Forest, XGBoost per sector × horizon.
- Evaluate each model on validation set (MAE, RMSE, R²).
- Select and save the best model per combination.
- Write metadata.json with training results.
"""

import os
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config import (
    SECTOR_TICKERS,
    FEATURE_DATASET_PATH,
    MODELS_DIR,
    METADATA_PATH,
    METADATA_DIR,
    HORIZONS,
    TARGET_TEMPLATE,
    SECTOR_MODEL_STEM,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature columns used for training (must match feature_engineering.py)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "Daily_Return", "Weekly_Return", "Monthly_Return",
    "MA5", "MA10", "MA20", "EMA", "RSI", "MACD",
    "BB_High", "BB_Low", "BB_Mid",
    "Rolling_Volatility", "Avg_Daily_Range",
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _build_models() -> dict:
    """Return fresh model instances for each algorithm, wrapped with scaling to prevent overfitting."""
    return {
        "LinearRegression": make_pipeline(
            StandardScaler(),
            LinearRegression()
        ),
        "RandomForest": make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=50,
                max_depth=3,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )
        ),
        "XGBoost": make_pipeline(
            StandardScaler(),
            XGBRegressor(
                n_estimators=50,
                max_depth=2,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
            )
        ),
    }


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"mae": round(float(mae), 4), "rmse": round(float(rmse), 4), "r2": round(float(r2), 4)}


def train_all_models() -> list[dict]:
    """
    Train models for every sector × horizon combination.

    For each combination:
    1. Filter training rows by sector.
    2. Train 3 models.
    3. Evaluate on validation set (same sector).
    4. Pick best by R².
    5. Save model to models/<stem>_<horizon>d.joblib.
    6. Record metadata entry.

    Returns
    -------
    list[dict]  metadata entries for all 12 combinations.
    """
    # Load feature dataset
    if not os.path.exists(FEATURE_DATASET_PATH):
        raise FileNotFoundError(f"Feature dataset not found: {FEATURE_DATASET_PATH}")

    feature_df = pd.read_csv(FEATURE_DATASET_PATH, parse_dates=["Date"])

    _ensure_dir(MODELS_DIR)
    _ensure_dir(METADATA_DIR)

    all_metadata = []

    for sector in SECTOR_TICKERS.keys():
        stem = SECTOR_MODEL_STEM[sector]

        sector_df = feature_df[feature_df["Sector"] == sector].copy()

        if sector_df.empty:
            logger.warning(f"No data for sector: {sector}")
            continue

        for horizon in HORIZONS:
            target_col = TARGET_TEMPLATE.format(horizon=horizon)

            # Drop rows where target or features are NaN for THIS specific horizon
            df_h = sector_df.dropna(subset=FEATURE_COLS + [target_col]).copy()

            if df_h.empty or len(df_h) < 10:
                logger.warning(f"Insufficient data for {sector} horizon {horizon}d — skipping.")
                continue

            # Chronological 80/20 split on the valid rows
            df_h = df_h.sort_values("Date")
            split_idx = int(len(df_h) * 0.8)
            t_df = df_h.iloc[:split_idx]
            v_df = df_h.iloc[split_idx:]

            if t_df.empty or v_df.empty:
                logger.warning(f"Insufficient split data for {sector} horizon {horizon}d — skipping.")
                continue

            X_train = t_df[FEATURE_COLS].values
            y_train = t_df[target_col].values
            X_val   = v_df[FEATURE_COLS].values
            y_val   = v_df[target_col].values

            best_model      = None
            best_model_name = None
            best_metrics    = None
            best_r2         = -np.inf

            models = _build_models()

            for model_name, model in models.items():
                logger.info(f"  Training {model_name} | {sector} | {horizon}d ...")
                try:
                    model.fit(X_train, y_train)
                    y_pred   = model.predict(X_val)
                    metrics  = _evaluate(y_val, y_pred)

                    logger.info(
                        f"    MAE={metrics['mae']:.4f}  "
                        f"RMSE={metrics['rmse']:.4f}  "
                        f"R²={metrics['r2']:.4f}"
                    )

                    if metrics["r2"] > best_r2:
                        best_r2         = metrics["r2"]
                        best_model      = model
                        best_model_name = model_name
                        best_metrics    = metrics

                except Exception as exc:
                    logger.error(f"  {model_name} failed for {sector}/{horizon}d: {exc}")
                    continue

            if best_model is None:
                logger.error(f"All models failed for {sector} | {horizon}d")
                continue

            # Save model
            model_path = os.path.join(MODELS_DIR, f"{stem}_{horizon}d.joblib")
            joblib.dump(best_model, model_path)
            logger.info(
                f"  ✓ Best: {best_model_name} | R²={best_r2:.4f} → {model_path}"
            )

            # Build metadata entry
            entry = {
                "sector":          sector,
                "horizon":         horizon,
                "selected_model":  best_model_name,
                "training_start":  TRAIN_START_DATE,
                "training_end":    TRAIN_END_DATE,
                "mae":             best_metrics["mae"],
                "rmse":            best_metrics["rmse"],
                "r2":              best_metrics["r2"],
                "trained_at":      datetime.now().isoformat(),
                "model_path":      model_path,
                "train_samples":   int(len(t_df)),
                "val_samples":     int(len(v_df)),
            }
            all_metadata.append(entry)

    # Save metadata JSON
    with open(METADATA_PATH, "w") as f:
        json.dump(all_metadata, f, indent=2)
    logger.info(f"Metadata saved: {METADATA_PATH}")

    return all_metadata


def load_metadata() -> list[dict]:
    """Load and return the metadata JSON. Returns empty list if not found."""
    if not os.path.exists(METADATA_PATH):
        return []
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


def get_model_metadata(sector: str, horizon: int) -> dict | None:
    """Return the metadata entry for a specific sector + horizon, or None."""
    for entry in load_metadata():
        if entry["sector"] == sector and entry["horizon"] == horizon:
            return entry
    return None
