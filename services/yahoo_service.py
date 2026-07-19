"""
yahoo_service.py — Data collection service.

Responsibilities:
- Download historical OHLCV data from Yahoo Finance for all tickers.
- Save raw CSVs into data/raw/<Sector>/<STEM>.csv
- Provide a helper to fetch recent data for live predictions.
"""

import os
import logging
from datetime import datetime, timedelta

import json
import pandas as pd
import yfinance as yf
import ta

# Ensure utils is importable when running from project root
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.cache_service import cache_manager

from utils.config import (
    SECTOR_TICKERS,
    TICKER_FILE_STEMS,
    RAW_DIR
)

logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> None:
    """Create directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def download_raw_data(start: str = "2025-10-01", end: str = "2026-03-31") -> dict:
    """
    Download historical OHLCV data for every ticker and save to raw CSVs.

    Parameters
    ----------
    start : str  ISO date, e.g. "2025-10-01"
    end   : str  ISO date, e.g. "2026-03-31"

    Returns
    -------
    dict  { ticker: status }  where status is "ok" or an error message.
    """
    results = {}

    for sector, tickers in SECTOR_TICKERS.items():
        sector_dir = os.path.join(RAW_DIR, sector)
        _ensure_dir(sector_dir)

        for ticker, display_name in tickers.items():
            file_stem = TICKER_FILE_STEMS[ticker]
            save_path = os.path.join(sector_dir, f"{file_stem}.csv")

            try:
                logger.info(f"Downloading {display_name} ({ticker}) [{start} → {end}]")
                df = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                )

                if df.empty:
                    msg = f"No data returned for {ticker}"
                    logger.warning(msg)
                    results[ticker] = msg
                    continue

                # Flatten MultiIndex columns if present (yfinance v0.2+)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Reset index so Date becomes a column
                df.reset_index(inplace=True)
                df.to_csv(save_path, index=False)
                logger.info(f"  Saved {len(df)} rows → {save_path}")
                results[ticker] = "ok"

            except Exception as exc:
                msg = f"Error downloading {ticker}: {exc}"
                logger.error(msg)
                results[ticker] = msg

    return results


def fetch_recent_data(ticker: str, lookback_days: int = 120) -> pd.DataFrame:
    """
    Fetch recent OHLCV data for a single ticker (used at prediction time).

    Parameters
    ----------
    ticker        : str  Yahoo Finance ticker symbol (e.g. "TCS.NS")
    lookback_days : int  Number of calendar days to look back from today

    Returns
    -------
    pd.DataFrame  with columns: Date, Open, High, Low, Close, Volume
    Raises RuntimeError if download fails or is empty.
    """
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=lookback_days)

    try:
        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to download recent data for {ticker}: {exc}") from exc

    if df.empty:
        raise RuntimeError(f"No recent data available for {ticker}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


from services.technical_indicator_service import add_technical_indicators

def get_financial_metrics(ticker: str) -> dict:
    """
    Fetch structured financial metrics for the LLM to use.
    Includes current price, volume, market cap, P/E ratio, and recent return,
    as well as technical indicators (RSI, MACD, MA, Volatility, etc.).
    Returns a dictionary.
    """
    cache_key = f"yahoo_{ticker}"
    cached = cache_manager.get(cache_key)
    if cached:
        return cached

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Current Price
        current_price = info.get("currentPrice", info.get("regularMarketPrice"))
        
        # Market Cap
        market_cap = info.get("marketCap")
        
        # Volume
        volume = info.get("volume")
        
        # Financial Ratios
        pe_ratio = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        
        # Technical indicators (using 6 months of history)
        hist = stock.history(period="6mo")
        technicals = {}
        
        if not hist.empty and len(hist) > 30:
            hist = add_technical_indicators(hist)
            latest = hist.iloc[-1]
            
            technicals = {
                "Daily_Return_Pct": round(latest.get("Daily_Return", 0), 2),
                "Monthly_Return_Pct": round(latest.get("Monthly_Return", 0), 2),
                "MA5": round(latest.get("MA5", 0), 2),
                "MA10": round(latest.get("MA10", 0), 2),
                "MA20": round(latest.get("MA20", 0), 2),
                "MA50": round(latest.get("MA50", 0), 2),
                "EMA": round(latest.get("EMA", 0), 2),
                "RSI": round(latest.get("RSI", 0), 2),
                "MACD": round(latest.get("MACD", 0), 2),
                "BB_High": round(latest.get("BB_High", 0), 2),
                "BB_Low": round(latest.get("BB_Low", 0), 2),
                "BB_Mid": round(latest.get("BB_Mid", 0), 2),
                "Rolling_Volatility_Pct": round(latest.get("Rolling_Volatility", 0), 2),
                "Avg_Daily_Range": round(latest.get("Avg_Daily_Range", 0), 2),
                "ATR": round(latest.get("ATR", 0), 2),
                "Momentum": round(latest.get("Momentum", 0), 2)
            }
            
        metrics = {
            "Ticker": ticker,
            "Current_Price_INR": current_price if current_price else None,
            "Volume": volume if volume else None,
            "Market_Cap_INR": market_cap if market_cap else None,
            "Trailing_PE": round(pe_ratio, 2) if pe_ratio else None,
            "Forward_PE": round(forward_pe, 2) if forward_pe else None,
            "Technical_Indicators": technicals
        }
        
        cache_manager.set(cache_key, metrics)
        return metrics
    except Exception as exc:
        logger.error(f"Failed to fetch metrics for {ticker}: {exc}")
        return {"error": f"Error fetching financial metrics for {ticker}: {exc}"}
def fetch_actual_returns(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download actual closing prices for a ticker between two dates.
    Used by the validation service.

    Returns
    -------
    pd.DataFrame with columns: Date, Close
    """
    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch actual data for {ticker}: {exc}") from exc

    if df.empty:
        raise RuntimeError(f"No data available for {ticker} in window {start_date}–{end_date}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    return df[["Date", "Close"]]


def fetch_ohlcv_range(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download full OHLCV data for a ticker between two specific dates.
    Used by validation to compute features as of a fixed simulation date.

    Returns
    -------
    pd.DataFrame  with columns: Date, Open, High, Low, Close, Volume
    Raises RuntimeError if download fails or is empty.
    """
    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch OHLCV for {ticker} [{start_date}→{end_date}]: {exc}") from exc

    if df.empty:
        raise RuntimeError(f"No data for {ticker} in window {start_date}–{end_date}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

