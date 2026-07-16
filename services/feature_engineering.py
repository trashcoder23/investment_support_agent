"""
feature_engineering.py — Feature engineering pipeline.

Responsibilities:
1. merge_raw_data()     → combined_dataset.csv
2. engineer_features()  → feature_dataset.csv
3. create_train_val_split() → train_dataset.csv + validation_dataset.csv

All transformations are chronological. No data leakage.
Target variable: forward % return for each horizon.
"""

import os
import logging

import numpy as np
import pandas as pd
import ta

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config import (
    SECTOR_TICKERS,
    TICKER_FILE_STEMS,
    TICKER_TO_NAME,
    RAW_DIR,
    COMBINED_DATASET_PATH,
    FEATURE_DATASET_PATH,
    TRAIN_DATASET_PATH,
    VALIDATION_DATASET_PATH,
    PROCESSED_DIR,
    HORIZONS,
    TARGET_TEMPLATE,
    TRAIN_RATIO,
)

logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Stage 1 → Stage 2: Merge raw CSVs
# ---------------------------------------------------------------------------

def merge_raw_data() -> pd.DataFrame:
    """
    Read all raw sector CSVs, add Company and Sector columns,
    concatenate into one DataFrame, and save as combined_dataset.csv.

    Returns
    -------
    pd.DataFrame  combined dataset
    """
    frames = []

    for sector, tickers in SECTOR_TICKERS.items():
        for ticker, display_name in tickers.items():
            file_stem = TICKER_FILE_STEMS[ticker]
            csv_path  = os.path.join(RAW_DIR, sector, f"{file_stem}.csv")

            if not os.path.exists(csv_path):
                logger.warning(f"Raw file not found: {csv_path} — skipping.")
                continue

            df = pd.read_csv(csv_path)
            df["Ticker"]  = ticker
            df["Company"] = display_name
            df["Sector"]  = sector
            frames.append(df)
            logger.info(f"  Loaded {len(df)} rows for {display_name}")

    if not frames:
        raise FileNotFoundError(
            "No raw CSV files found. Run download_raw_data() first."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    combined.sort_values(["Ticker", "Date"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    _ensure_dir(PROCESSED_DIR)
    combined.to_csv(COMBINED_DATASET_PATH, index=False)
    logger.info(f"Combined dataset saved: {COMBINED_DATASET_PATH} ({len(combined)} rows)")
    return combined


# ---------------------------------------------------------------------------
# Stage 2 → Stage 3: Feature Engineering
# ---------------------------------------------------------------------------

def _compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute forward-looking % return targets per horizon.
    Must be computed per ticker group to avoid cross-company leakage.
    """
    for horizon in HORIZONS:
        col = TARGET_TEMPLATE.format(horizon=horizon)
        df[col] = (
            df.groupby("Ticker")["Close"]
            .transform(lambda s: s.shift(-horizon) / s - 1) * 100
        )
    return df


def _add_technical_indicators(group: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators for a single ticker group.
    Uses the `ta` library + manual pandas rolling calculations.
    """
    close  = group["Close"]
    high   = group["High"]
    low    = group["Low"]
    volume = group["Volume"]

    # --- Returns ---
    group["Daily_Return"]   = close.pct_change() * 100
    group["Weekly_Return"]  = close.pct_change(periods=5) * 100
    group["Monthly_Return"] = close.pct_change(periods=21) * 100

    # --- Moving Averages ---
    group["MA5"]  = close.rolling(5).mean()
    group["MA10"] = close.rolling(10).mean()
    group["MA20"] = close.rolling(20).mean()

    # --- EMA ---
    group["EMA"] = ta.trend.ema_indicator(close, window=20)

    # --- RSI ---
    group["RSI"] = ta.momentum.rsi(close, window=14)

    # --- MACD ---
    macd = ta.trend.MACD(close)
    group["MACD"] = macd.macd()

    # --- Bollinger Bands ---
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    group["BB_High"] = bb.bollinger_hband()
    group["BB_Low"]  = bb.bollinger_lband()
    group["BB_Mid"]  = bb.bollinger_mavg()

    # --- Rolling Volatility (20-day std of daily returns) ---
    group["Rolling_Volatility"] = group["Daily_Return"].rolling(20).std()

    # --- Average Daily Range ---
    group["Avg_Daily_Range"] = (high - low).rolling(14).mean()

    return group


def engineer_features(combined: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Apply feature engineering to the combined dataset.
    Loads from disk if combined is not provided.

    Steps:
    - Compute technical indicators per ticker.
    - Compute forward return targets per horizon.
    - Drop rows with NaN (indicators need warm-up periods).
    - Save feature_dataset.csv.

    Returns
    -------
    pd.DataFrame  feature dataset
    """
    if combined is None:
        if not os.path.exists(COMBINED_DATASET_PATH):
            raise FileNotFoundError(f"Combined dataset not found at {COMBINED_DATASET_PATH}")
        combined = pd.read_csv(COMBINED_DATASET_PATH, parse_dates=["Date"])

    logger.info(f"Engineering features for {len(combined)} rows ...")

    # Apply indicators per ticker using an explicit loop.
    indicator_frames = []
    for ticker, group in combined.groupby("Ticker"):
        indicator_frames.append(_add_technical_indicators(group.copy()))
    feature_df = pd.concat(indicator_frames, ignore_index=True)

    # Compute forward-looking return targets (per ticker)
    feature_df = _compute_targets(feature_df)

    # Drop rows with any NaN in feature or target columns
    target_cols  = [TARGET_TEMPLATE.format(horizon=h) for h in HORIZONS]
    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "Daily_Return", "Weekly_Return", "Monthly_Return",
        "MA5", "MA10", "MA20", "EMA", "RSI", "MACD",
        "BB_High", "BB_Low", "BB_Mid",
        "Rolling_Volatility", "Avg_Daily_Range",
    ]

    before = len(feature_df)
    # Only drop rows where the FEATURES (technical indicators) are missing.
    feature_df.dropna(subset=feature_cols, inplace=True)
    after  = len(feature_df)
    logger.info(f"Dropped {before - after} rows with NaN features (warm-up periods).")

    feature_df.reset_index(drop=True, inplace=True)
    _ensure_dir(PROCESSED_DIR)
    feature_df.to_csv(FEATURE_DATASET_PATH, index=False)
    logger.info(f"Feature dataset saved: {FEATURE_DATASET_PATH} ({after} rows)")
    return feature_df


# ---------------------------------------------------------------------------
# Stage 3 → Stage 4: Chronological train/val split
# ---------------------------------------------------------------------------

def create_train_val_split(feature_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a chronological 80/20 split — NO shuffling.
    Split is performed per-ticker so each company retains temporal ordering.

    Returns
    -------
    (train_df, val_df)
    """
    if feature_df is None:
        if not os.path.exists(FEATURE_DATASET_PATH):
            raise FileNotFoundError(f"Feature dataset not found at {FEATURE_DATASET_PATH}")
        feature_df = pd.read_csv(FEATURE_DATASET_PATH, parse_dates=["Date"])

    train_frames = []
    val_frames   = []

    for ticker, group in feature_df.groupby("Ticker"):
        group_sorted = group.sort_values("Date")
        split_idx    = int(len(group_sorted) * TRAIN_RATIO)
        train_frames.append(group_sorted.iloc[:split_idx])
        val_frames.append(group_sorted.iloc[split_idx:])

    train_df = pd.concat(train_frames, ignore_index=True)
    val_df   = pd.concat(val_frames,   ignore_index=True)

    train_df.to_csv(TRAIN_DATASET_PATH,      index=False)
    val_df.to_csv(VALIDATION_DATASET_PATH,   index=False)

    logger.info(f"Train set saved : {TRAIN_DATASET_PATH} ({len(train_df)} rows)")
    logger.info(f"Val set saved   : {VALIDATION_DATASET_PATH} ({len(val_df)} rows)")

    return train_df, val_df


# ---------------------------------------------------------------------------
# Live feature computation (for predictions on fresh data)
# ---------------------------------------------------------------------------

def compute_live_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering pipeline to a fresh OHLCV DataFrame
    downloaded for a single ticker at prediction time.

    Targets are NOT computed (no future data available).
    Returns the last row of features (most recent data point).
    """
    df = df.copy()
    df = _add_technical_indicators(df)

    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "Daily_Return", "Weekly_Return", "Monthly_Return",
        "MA5", "MA10", "MA20", "EMA", "RSI", "MACD",
        "BB_High", "BB_Low", "BB_Mid",
        "Rolling_Volatility", "Avg_Daily_Range",
    ]

    df = df.dropna(subset=feature_cols)
    if df.empty:
        raise ValueError("Not enough data to compute live features (need at least 30 rows).")

    return df
