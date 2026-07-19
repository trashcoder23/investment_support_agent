import pandas as pd
from services.yahoo_service import fetch_ohlcv_range, fetch_actual_returns

def get_historical_market_data(ticker: str, start_date: str, cutoff_date: str) -> pd.DataFrame:
    """
    Fetch market data restricted strictly to the evaluation period (start_date to cutoff_date).
    This guarantees no future leakage.
    """
    return fetch_ohlcv_range(ticker, start_date, cutoff_date)

def get_validation_market_data(ticker: str, cutoff_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch the actual market returns for the validation period (cutoff_date to end_date).
    Used ONLY by the evaluation metrics, NEVER seen by the AI Agent.
    """
    return fetch_actual_returns(ticker, cutoff_date, end_date)
