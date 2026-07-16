"""
config.py — Central configuration for the Investment Support Agent.

All constants, paths, ticker mappings, feature lists, and thresholds
are defined here. Import from this module throughout the project.
"""

import os

# ---------------------------------------------------------------------------
# Base directories
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR         = os.path.join(BASE_DIR, "data")
RAW_DIR          = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR    = os.path.join(DATA_DIR, "processed")
PREDICTIONS_DIR  = os.path.join(DATA_DIR, "predictions")
VALIDATION_DIR   = os.path.join(DATA_DIR, "validation")
METADATA_DIR     = os.path.join(DATA_DIR, "metadata")
MODELS_DIR       = os.path.join(BASE_DIR, "models")

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
COMBINED_DATASET_PATH    = os.path.join(PROCESSED_DIR, "combined_dataset.csv")
FEATURE_DATASET_PATH     = os.path.join(PROCESSED_DIR, "feature_dataset.csv")
TRAIN_DATASET_PATH       = os.path.join(PROCESSED_DIR, "train_dataset.csv")
VALIDATION_DATASET_PATH  = os.path.join(PROCESSED_DIR, "validation_dataset.csv")
PREDICTION_HISTORY_PATH  = os.path.join(PREDICTIONS_DIR, "prediction_history.csv")
VALIDATION_RESULTS_PATH  = os.path.join(VALIDATION_DIR, "validation_results.csv")
METADATA_PATH            = os.path.join(METADATA_DIR, "metadata.json")

# ---------------------------------------------------------------------------
# Investment universe
# ---------------------------------------------------------------------------
# Ticker symbol → Display name
SECTOR_TICKERS: dict[str, dict[str, str]] = {
    "IT": {
        "TCS.NS":      "TCS",
        "INFY.NS":     "Infosys",
        "HCLTECH.NS":  "HCLTech",
        "WIPRO.NS":    "Wipro",
    },
    "Banking": {
        "HDFCBANK.NS": "HDFC Bank",
        "ICICIBANK.NS": "ICICI Bank",
        "SBIN.NS":     "SBI",
        "AXISBANK.NS": "Axis Bank",
    },
    "Pharma": {
        "SUNPHARMA.NS": "Sun Pharma",
        "CIPLA.NS":     "Cipla",
        "DIVISLAB.NS":  "Divi's Labs",
        "DRREDDY.NS":   "Dr. Reddy's",
    },
    "Automobile": {
        "HEROMOTOCO.NS": "Hero MotoCorp",
        "MARUTI.NS":     "Maruti Suzuki",
        "M&M.NS":        "Mahindra & Mahindra",
        "BAJAJ-AUTO.NS": "Bajaj Auto",
    },
}

# Reverse map: ticker → sector
TICKER_TO_SECTOR: dict[str, str] = {
    ticker: sector
    for sector, tickers in SECTOR_TICKERS.items()
    for ticker in tickers
}

# Reverse map: ticker → display name
TICKER_TO_NAME: dict[str, str] = {
    ticker: name
    for tickers in SECTOR_TICKERS.values()
    for ticker, name in tickers.items()
}

# Sector-wise CSV file stem (used for raw file naming)
TICKER_FILE_STEMS: dict[str, str] = {
    "TCS.NS":        "TCS",
    "INFY.NS":       "INFY",
    "HCLTECH.NS":    "HCLTECH",
    "WIPRO.NS":      "WIPRO",
    "HDFCBANK.NS":   "HDFCBANK",
    "ICICIBANK.NS":  "ICICIBANK",
    "SBIN.NS":       "SBIN",
    "AXISBANK.NS":   "AXISBANK",
    "SUNPHARMA.NS":  "SUNPHARMA",
    "CIPLA.NS":      "CIPLA",
    "DIVISLAB.NS":   "DIVISLAB",
    "DRREDDY.NS":    "DRREDDY",
    "HEROMOTOCO.NS": "HEROMOTOCO",
    "MARUTI.NS":     "MARUTI",
    "M&M.NS":        "MM",
    "BAJAJ-AUTO.NS": "BAJAJAUTO",
}

# ---------------------------------------------------------------------------
# Training data window
# ---------------------------------------------------------------------------
TRAIN_START_DATE = "2025-10-01"
TRAIN_END_DATE   = "2026-03-31"

# Simulated "today" for validation.
# Model was trained up to Mar 31 → we simulate making predictions on Apr 1
# and validate against actual Apr/May/Jun/Jul 2026 data (all available now).
VALIDATION_SIMULATION_DATE = "2026-04-01"


# ---------------------------------------------------------------------------
# Prediction horizons (days)
# ---------------------------------------------------------------------------
HORIZONS = [30, 60, 90]

# ---------------------------------------------------------------------------
# Feature column names
# ---------------------------------------------------------------------------
RAW_FEATURES = ["Open", "High", "Low", "Close", "Volume"]

ENGINEERED_FEATURES = [
    "Daily_Return",
    "Weekly_Return",
    "Monthly_Return",
    "MA5",
    "MA10",
    "MA20",
    "EMA",
    "RSI",
    "MACD",
    "BB_High",
    "BB_Low",
    "BB_Mid",
    "Rolling_Volatility",
    "Avg_Daily_Range",
]

ALL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES

# Target column template — formatted with horizon int
TARGET_TEMPLATE = "Return_{horizon}d"

# ---------------------------------------------------------------------------
# Train / validation split ratio
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.80

# ---------------------------------------------------------------------------
# Confidence thresholds (based on R²)
# ---------------------------------------------------------------------------
CONFIDENCE_HIGH   = 0.70
CONFIDENCE_MEDIUM = 0.40
# Below CONFIDENCE_MEDIUM → Low

# ---------------------------------------------------------------------------
# Model names (for logging / metadata)
# ---------------------------------------------------------------------------
MODEL_NAMES = ["LinearRegression", "RandomForest", "XGBoost"]

# ---------------------------------------------------------------------------
# Sector → model file stem mapping
# ---------------------------------------------------------------------------
SECTOR_MODEL_STEM: dict[str, str] = {
    "IT":         "it",
    "Banking":    "banking",
    "Pharma":     "pharma",
    "Automobile": "automobile",
}
