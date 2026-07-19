import pandas as pd
import ta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical indicators for a given DataFrame containing Close, High, Low, Volume.
    """
    df = df.copy()
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # --- Returns ---
    df["Daily_Return"]   = close.pct_change() * 100
    df["Weekly_Return"]  = close.pct_change(periods=5) * 100
    df["Monthly_Return"] = close.pct_change(periods=21) * 100

    # --- Moving Averages ---
    df["MA5"]  = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()

    # --- EMA ---
    df["EMA"] = ta.trend.ema_indicator(close, window=20)

    # --- RSI ---
    df["RSI"] = ta.momentum.rsi(close, window=14)

    # --- MACD ---
    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()

    # --- Bollinger Bands ---
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"]  = bb.bollinger_lband()
    df["BB_Mid"]  = bb.bollinger_mavg()

    # --- Rolling Volatility (20-day std of daily returns) ---
    df["Rolling_Volatility"] = df["Daily_Return"].rolling(20).std()

    # --- Average Daily Range (ATR equivalent/helper) ---
    df["Avg_Daily_Range"] = (high - low).rolling(14).mean()
    
    # --- ATR ---
    df["ATR"] = ta.volatility.average_true_range(high, low, close, window=14)
    
    # --- Momentum (10-day rate of change) ---
    df["Momentum"] = ta.momentum.roc(close, window=10)

    return df
