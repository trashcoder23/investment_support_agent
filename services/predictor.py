"""
predictor.py — Prediction engine.

Responsibilities:
- Load saved joblib model for sector + horizon.
- Fetch recent OHLCV data from Yahoo Finance.
- Compute live features.
- Return predicted return % and confidence for each company in the sector.
- Log every prediction to prediction_history.csv.
"""

import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config import (
    SECTOR_TICKERS,
    MODELS_DIR,
    SECTOR_MODEL_STEM,
    PREDICTION_HISTORY_PATH,
    PREDICTIONS_DIR,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
)
from services.yahoo_service import fetch_recent_data
from services.feature_engineering import compute_live_features
from services.model_trainer import get_model_metadata

logger = logging.getLogger(__name__)

# Feature columns must match training order exactly
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "Daily_Return", "Weekly_Return", "Monthly_Return",
    "MA5", "MA10", "MA20", "EMA", "RSI", "MACD",
    "BB_High", "BB_Low", "BB_Mid",
    "Rolling_Volatility", "Avg_Daily_Range",
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _confidence_label(r2: float) -> str:
    """Convert R² value to human-readable confidence label."""
    if r2 >= CONFIDENCE_HIGH:
        return "High"
    elif r2 >= CONFIDENCE_MEDIUM:
        return "Medium"
    return "Low"


def load_model(sector: str, horizon: int):
    """
    Load the saved joblib model for a sector + horizon combination.

    Raises
    ------
    FileNotFoundError  if the model file does not exist.
    """
    stem       = SECTOR_MODEL_STEM[sector]
    model_path = os.path.join(MODELS_DIR, f"{stem}_{horizon}d.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Please run `python train.py` to train the models first."
        )

    return joblib.load(model_path)


def predict_sector(sector: str, horizon: int, budget: float) -> list[dict]:
    """
    Generate predictions for all companies in a sector.

    Parameters
    ----------
    sector  : str    One of "IT", "Banking", "Pharma", "Automobile"
    horizon : int    30, 60, or 90
    budget  : float  Investment amount in ₹

    Returns
    -------
    list[dict]  One dict per company:
        {
            "ticker":           str,
            "company":          str,
            "sector":           str,
            "predicted_return": float (% over horizon),
            "confidence":       str ("High" / "Medium" / "Low"),
            "r2":               float,
            "volatility":       float,  (rolling std of daily returns)
            "current_price":    float,
        }
    """
    model    = load_model(sector, horizon)
    metadata = get_model_metadata(sector, horizon)
    r2       = metadata["r2"] if metadata else 0.0

    tickers   = SECTOR_TICKERS[sector]
    results   = []

    for ticker, display_name in tickers.items():
        try:
            logger.info(f"Fetching live data for {ticker} ...")
            raw_df = fetch_recent_data(ticker, lookback_days=120)

            feature_df = compute_live_features(raw_df)

            # Use the most recent row for prediction
            latest_features = feature_df[FEATURE_COLS].iloc[[-1]].values
            predicted_return = float(model.predict(latest_features)[0])

            # Volatility: rolling std of daily returns from live data
            volatility = float(
                feature_df["Rolling_Volatility"].dropna().iloc[-1]
                if not feature_df["Rolling_Volatility"].dropna().empty
                else np.nan
            )
            current_price = float(feature_df["Close"].iloc[-1])

            results.append({
                "ticker":           ticker,
                "company":          display_name,
                "sector":           sector,
                "predicted_return": round(predicted_return, 2),
                "confidence":       _confidence_label(r2),
                "r2":               round(r2, 4),
                "volatility":       round(volatility, 4) if not np.isnan(volatility) else None,
                "current_price":    round(current_price, 2),
            })

        except Exception as exc:
            logger.error(f"Prediction failed for {ticker}: {exc}")
            results.append({
                "ticker":           ticker,
                "company":          display_name,
                "sector":           sector,
                "predicted_return": None,
                "confidence":       "Low",
                "r2":               r2,
                "volatility":       None,
                "current_price":    None,
                "error":            str(exc),
            })

    # Log predictions to history
    _log_predictions(results, budget, sector, horizon)

    return results


def _log_predictions(
    predictions: list[dict],
    budget: float,
    sector: str,
    horizon: int,
) -> None:
    """Append a summary row to prediction_history.csv."""
    _ensure_dir(PREDICTIONS_DIR)

    timestamp = datetime.now().isoformat()

    companies    = [p["company"]          for p in predictions]
    pred_returns = [p["predicted_return"] for p in predictions]
    confidences  = [p["confidence"]       for p in predictions]

    row = {
        "timestamp":        timestamp,
        "budget":           budget,
        "sector":           sector,
        "horizon":          horizon,
        "companies":        "|".join(companies),
        "predicted_returns":"|".join(str(r) for r in pred_returns),
        "confidences":      "|".join(confidences),
    }

    history_df = pd.DataFrame([row])

    if os.path.exists(PREDICTION_HISTORY_PATH):
        history_df.to_csv(PREDICTION_HISTORY_PATH, mode="a", header=False, index=False)
    else:
        history_df.to_csv(PREDICTION_HISTORY_PATH, index=False)

    logger.info(f"Prediction logged to {PREDICTION_HISTORY_PATH}")


def load_prediction_history() -> pd.DataFrame:
    """Load prediction history CSV. Returns empty DataFrame if not found."""
    if not os.path.exists(PREDICTION_HISTORY_PATH):
        return pd.DataFrame()
    return pd.read_csv(PREDICTION_HISTORY_PATH)
