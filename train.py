"""
train.py — One-time offline training CLI script.

Run this ONCE before launching the Streamlit application:
    python train.py

This script will:
1. Download raw OHLCV data from Yahoo Finance.
2. Merge into combined_dataset.csv.
3. Engineer all features → feature_dataset.csv.
4. Create chronological train/val split.
5. Train 3 ML models per sector × horizon.
6. Save the best model per combination as .joblib.
7. Write metadata.json.
"""

import logging
import sys
import os

# Configure logging before any imports that use it
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.yahoo_service import download_raw_data
from services.feature_engineering import merge_raw_data, engineer_features, create_train_val_split
from services.model_trainer import train_all_models


def main():
    print("\n" + "=" * 60)
    print("  Investment Support Agent — Training Pipeline")
    print("=" * 60 + "\n")

    # -----------------------------------------------------------------------
    # Stage 1: Download raw data
    # -----------------------------------------------------------------------
    print("📥  Stage 1/5 — Downloading historical stock data ...")
    results = download_raw_data()

    successes = [t for t, s in results.items() if s == "ok"]
    failures  = [t for t, s in results.items() if s != "ok"]

    print(f"  ✓ Downloaded: {len(successes)} tickers")
    if failures:
        print(f"  ⚠ Failed    : {len(failures)} tickers — {failures}")
        if len(failures) >= len(results):
            print("\n❌ All downloads failed. Check your internet connection.")
            sys.exit(1)

    # -----------------------------------------------------------------------
    # Stage 2: Merge raw data
    # -----------------------------------------------------------------------
    print("\n🔗  Stage 2/5 — Merging raw datasets ...")
    combined = merge_raw_data()
    print(f"  ✓ Combined dataset: {len(combined):,} rows")

    # -----------------------------------------------------------------------
    # Stage 3: Feature engineering
    # -----------------------------------------------------------------------
    print("\n⚙️   Stage 3/5 — Engineering features ...")
    features = engineer_features(combined)
    print(f"  ✓ Feature dataset : {len(features):,} rows | {len(features.columns)} columns")

    # -----------------------------------------------------------------------
    # Stage 4: Train / validation split
    # -----------------------------------------------------------------------
    print("\n✂️   Stage 4/5 — Creating chronological train/val split ...")
    train_df, val_df = create_train_val_split(features)
    print(f"  ✓ Train set       : {len(train_df):,} rows")
    print(f"  ✓ Validation set  : {len(val_df):,} rows")

    # -----------------------------------------------------------------------
    # Stage 5: Train models
    # -----------------------------------------------------------------------
    print("\n🤖  Stage 5/5 — Training ML models (this may take a few minutes) ...")
    metadata = train_all_models()
    print(f"\n  ✓ Trained {len(metadata)} sector × horizon model combinations")

    print("\n" + "=" * 60)
    print("  Training Summary")
    print("=" * 60)
    for entry in metadata:
        print(
            f"  [{entry['sector']:12s}] {entry['horizon']:2d}d | "
            f"{entry['selected_model']:20s} | "
            f"R²={entry['r2']:+.4f}  MAE={entry['mae']:.4f}  RMSE={entry['rmse']:.4f}"
        )

    print("\n✅  Training complete!")
    print("   Run `streamlit run ui/app.py` to launch the application.\n")


if __name__ == "__main__":
    main()
