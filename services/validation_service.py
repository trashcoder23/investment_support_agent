"""
validation_service.py — Model validation against real post-training market data.

Simulation scenario
-------------------
  - Training window  : Oct 2025 → Mar 2026
  - Simulation date  : April 1, 2026  ("the day after training ended")
  - What we do       : Run the trained model on March 31 market features
                       → get predicted % return for 30 / 60 / 90 days
  - Validate against : Actual Yahoo Finance prices for April / May / June 2026
                       (all fully available as of July 2026)

Expected result dates
---------------------
  30-day horizon → ~April 30 / May 1, 2026
  60-day horizon → ~May 31 / June 1, 2026
  90-day horizon → ~June 30 / July 1, 2026
"""

import os
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config import (
    SECTOR_TICKERS,
    SECTOR_MODEL_STEM,
    MODELS_DIR,
    VALIDATION_RESULTS_PATH,
    VALIDATION_DIR,
    VALIDATION_SIMULATION_DATE,
)
from services.yahoo_service import fetch_ohlcv_range, fetch_actual_returns
from services.feature_engineering import compute_live_features

logger = logging.getLogger(__name__)

# Must match training feature order exactly
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "Daily_Return", "Weekly_Return", "Monthly_Return",
    "MA5", "MA10", "MA20", "EMA", "RSI", "MACD",
    "BB_High", "BB_Low", "BB_Mid",
    "Rolling_Volatility", "Avg_Daily_Range",
]

# How many calendar days of historical OHLCV to load for feature computation
# (need at least ~60 days for all indicators to warm up)
FEATURE_LOOKBACK_DAYS = 180


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_model_validation(sector: str, horizon: int) -> dict:
    """
    Validate the trained model by simulating a prediction made on April 1, 2026
    and comparing against actual market data for Apr / May / Jun 2026.

    Parameters
    ----------
    sector  : str  One of "IT", "Banking", "Pharma", "Automobile"
    horizon : int  30, 60, or 90

    Returns
    -------
    dict with:
        simulation_date : str   — Fixed prediction reference date (Apr 1 2026)
        target_date     : str   — Expected outcome date (Apr 1 + horizon trading days)
        metrics         : dict  — { mae, rmse, r2 }
        chart_data      : list  — Per-company dicts with prices + returns
        error           : str   — Only present if validation failed completely
    """
    simulation_date = VALIDATION_SIMULATION_DATE  # "2026-04-01"
    sim_dt = pd.to_datetime(simulation_date)

    # Window for feature computation: 180 calendar days before simulation date
    feature_start = (sim_dt - timedelta(days=FEATURE_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    # Fetch up to simulation date (exclusive in yfinance, so add 1 day)
    feature_end   = (sim_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    # Validation window: simulation date → horizon + buffer trading days
    # Add extra buffer to account for weekends/holidays
    actual_end = (sim_dt + timedelta(days=horizon + 25)).strftime("%Y-%m-%d")

    # ── Load model ───────────────────────────────────────────────────────────
    stem       = SECTOR_MODEL_STEM[sector]
    model_path = os.path.join(MODELS_DIR, f"{stem}_{horizon}d.joblib")

    if not os.path.exists(model_path):
        return {"error": f"Model not found: {model_path}. Run python train.py first."}

    model = joblib.load(model_path)

    # ── Per-company validation ───────────────────────────────────────────────
    tickers_map   = SECTOR_TICKERS[sector]
    chart_data    = []
    y_pred_list   = []
    y_actual_list = []
    target_date_str = ""

    for ticker, display_name in tickers_map.items():
        logger.info(f"Validating {display_name} ({ticker}) ...")

        # ── Step 1: Fetch pre-simulation OHLCV and compute features ──────────
        try:
            hist_df = fetch_ohlcv_range(ticker, feature_start, feature_end)
        except RuntimeError as exc:
            logger.warning(f"Could not fetch pre-simulation data for {ticker}: {exc}")
            chart_data.append({
                "company":  display_name,
                "predicted": None,
                "actual":    None,
                "note":      f"Feature data unavailable: {exc}",
            })
            continue

        if len(hist_df) < 30:
            logger.warning(f"Too few rows for {ticker} ({len(hist_df)}) — skipping.")
            chart_data.append({
                "company":  display_name,
                "predicted": None,
                "actual":    None,
                "note":      f"Only {len(hist_df)} rows of pre-simulation data (need ≥ 30)",
            })
            continue

        try:
            feature_df = compute_live_features(hist_df)
        except Exception as exc:
            logger.warning(f"Feature computation failed for {ticker}: {exc}")
            chart_data.append({
                "company":  display_name,
                "predicted": None,
                "actual":    None,
                "note":      f"Feature computation failed: {exc}",
            })
            continue

        # Last available row ≤ simulation date = our prediction baseline
        last_row      = feature_df.iloc[-1]
        entry_price   = round(float(last_row["Close"]), 2)
        feature_date  = pd.to_datetime(last_row["Date"]).strftime("%Y-%m-%d") \
                        if "Date" in last_row.index else simulation_date

        features      = last_row[FEATURE_COLS].values.reshape(1, -1)
        predicted_ret = round(float(model.predict(features)[0]), 2)
        predicted_end_price = round(entry_price * (1 + predicted_ret / 100), 2)

        # ── Step 2: Fetch actual prices for the horizon window ───────────────
        try:
            actual_df = fetch_actual_returns(ticker, simulation_date, actual_end)
        except RuntimeError as exc:
            logger.warning(f"Could not fetch actual data for {ticker}: {exc}")
            chart_data.append({
                "company":            display_name,
                "simulation_date":    simulation_date,
                "feature_date":       feature_date,
                "predicted":          predicted_ret,
                "actual":             None,
                "entry_price":        entry_price,
                "predicted_end_price":predicted_end_price,
                "note":               f"Actual data unavailable: {exc}",
            })
            continue

        if len(actual_df) < 2:
            chart_data.append({
                "company":            display_name,
                "simulation_date":    simulation_date,
                "feature_date":       feature_date,
                "predicted":          predicted_ret,
                "actual":             None,
                "entry_price":        entry_price,
                "predicted_end_price":predicted_end_price,
                "note":               f"Only {len(actual_df)} actual trading days available (horizon not yet elapsed)",
            })
            continue

        # Anchor on the first available trading day on/after simulation date
        actual_start_price = round(float(actual_df["Close"].iloc[0]), 2)
        actual_start_date  = str(actual_df["Date"].iloc[0])[:10]

        # Closest row to `horizon` trading days out
        end_idx      = min(horizon, len(actual_df) - 1)
        actual_end_price = round(float(actual_df["Close"].iloc[end_idx]), 2)
        actual_end_date  = str(actual_df["Date"].iloc[end_idx])[:10]
        if not target_date_str:
            target_date_str = actual_end_date

        actual_ret  = round(((actual_end_price - actual_start_price) / actual_start_price) * 100, 2)
        price_change_actual    = round(actual_end_price    - actual_start_price,  2)
        price_change_predicted = round(predicted_end_price - entry_price,          2)

        chart_data.append({
            "company":               display_name,
            "simulation_date":       simulation_date,
            "feature_date":          feature_date,
            "target_date":           actual_end_date,
            # Returns
            "predicted":             predicted_ret,
            "actual":                actual_ret,
            # Prices
            "entry_price":           entry_price,
            "actual_start_price":    actual_start_price,
            "actual_start_date":     actual_start_date,
            "predicted_end_price":   predicted_end_price,
            "actual_end_price":      actual_end_price,
            # ₹ changes
            "price_change_predicted":price_change_predicted,
            "price_change_actual":   price_change_actual,
        })

        y_pred_list.append(predicted_ret)
        y_actual_list.append(actual_ret)

    # ── Compute metrics ──────────────────────────────────────────────────────
    comparable = [d for d in chart_data if d.get("actual") is not None]

    if not comparable:
        return {
            "error": (
                "Could not compute actual returns for any company in this sector. "
                "Check internet connection or try a shorter horizon."
            ),
            "chart_data":       chart_data,
            "simulation_date":  simulation_date,
            "target_date":      target_date_str,
        }

    y_pred   = np.array(y_pred_list)
    y_actual = np.array(y_actual_list)

    metrics = {
        "mae":  round(float(mean_absolute_error(y_actual, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_actual, y_pred))), 4),
        "r2":   round(float(r2_score(y_actual, y_pred)), 4) if len(y_actual) > 1 else None,
    }

    _save_validation_result(sector, horizon, simulation_date, target_date_str, metrics, comparable)

    return {
        "simulation_date": simulation_date,
        "target_date":     target_date_str,
        "metrics":         metrics,
        "chart_data":      chart_data,
    }


def _save_validation_result(
    sector: str,
    horizon: int,
    simulation_date: str,
    target_date: str,
    metrics: dict,
    chart_data: list[dict],
) -> None:
    """Append a validation result row to validation_results.csv."""
    _ensure_dir(VALIDATION_DIR)

    row = {
        "validated_at":      datetime.now().isoformat(),
        "sector":            sector,
        "horizon":           horizon,
        "simulation_date":   simulation_date,
        "target_date":       target_date,
        "mae":               metrics.get("mae"),
        "rmse":              metrics.get("rmse"),
        "r2":                metrics.get("r2"),
        "companies":         "|".join(d["company"] for d in chart_data),
        "predicted_returns": "|".join(str(d.get("predicted", "")) for d in chart_data),
        "actual_returns":    "|".join(str(d.get("actual", "")) for d in chart_data),
        "entry_prices":      "|".join(str(d.get("entry_price", "")) for d in chart_data),
        "actual_end_prices": "|".join(str(d.get("actual_end_price", "")) for d in chart_data),
    }

    result_df = pd.DataFrame([row])

    if os.path.exists(VALIDATION_RESULTS_PATH):
        try:
            existing = pd.read_csv(VALIDATION_RESULTS_PATH, on_bad_lines="skip")
            # If column sets match → append; otherwise overwrite to avoid parse errors
            if set(existing.columns) == set(result_df.columns):
                result_df.to_csv(VALIDATION_RESULTS_PATH, mode="a", header=False, index=False)
            else:
                logger.warning(
                    "validation_results.csv schema mismatch — overwriting with new schema."
                )
                result_df.to_csv(VALIDATION_RESULTS_PATH, index=False)
        except Exception:
            # Corrupted file — overwrite cleanly
            result_df.to_csv(VALIDATION_RESULTS_PATH, index=False)
    else:
        result_df.to_csv(VALIDATION_RESULTS_PATH, index=False)

    logger.info(f"Validation result saved: {VALIDATION_RESULTS_PATH}")


def load_validation_results() -> pd.DataFrame:
    """
    Load validation results CSV.
    Returns empty DataFrame if not found or if parsing fails.
    Uses on_bad_lines='skip' to tolerate schema-mismatched rows from old service versions.
    """
    if not os.path.exists(VALIDATION_RESULTS_PATH):
        return pd.DataFrame()
    try:
        df = pd.read_csv(VALIDATION_RESULTS_PATH, on_bad_lines="skip")
        return df
    except Exception as exc:
        logger.warning(f"Could not parse validation_results.csv ({exc}) — returning empty.")
        return pd.DataFrame()

