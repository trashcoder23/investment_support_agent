import pandas as pd
from services.technical_indicator_service import add_technical_indicators
from backtesting.historical_data_loader import get_historical_market_data

def get_historical_financial_metrics(ticker: str, start_date: str, cutoff_date: str) -> dict:
    """
    Calculates technical indicators strictly on data available up to cutoff_date.
    Never uses yf.Ticker().info because that leaks today's data.
    """
    try:
        hist = get_historical_market_data(ticker, start_date, cutoff_date)
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
            
            current_price = round(latest["Close"], 2)
            volume = latest["Volume"]
        else:
            current_price = None
            volume = None

        return {
            "Ticker": ticker,
            "Current_Price_INR": current_price,
            "Volume": volume,
            "Market_Cap_INR": "N/A (Historical)",
            "Trailing_PE": "N/A (Historical)",
            "Forward_PE": "N/A (Historical)",
            "Technical_Indicators": technicals
        }
    except Exception as exc:
        return {"error": f"Error fetching historical metrics for {ticker}: {exc}"}
