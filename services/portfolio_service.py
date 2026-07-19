import logging

logger = logging.getLogger(__name__)

def calculate_allocations(company_metrics: dict, budget: float) -> dict:
    """
    Computes portfolio allocation recommendations based on budget and company technicals.
    Returns Focused and Diversified allocation proposals to be used as LLM context.
    """
    price = company_metrics.get("Current_Price_INR")
    volatility = company_metrics.get("Technical_Indicators", {}).get("Rolling_Volatility_Pct", 1.5)
    ticker = company_metrics.get("Ticker", "Unknown")
    
    if not price or price <= 0:
        return {
            "focused": "Unavailable (missing current price)",
            "diversified": "Unavailable (missing current price)"
        }
        
    # Strategy A: Focused (All-in)
    shares_focused = int(budget / price)
    allocated_focused = shares_focused * price
    cash_left_focused = budget - allocated_focused
    
    focused_desc = (
        f"Strategy A (Focused): Invest 100% of budget in {ticker}. "
        f"Buy {shares_focused} shares at ₹{price:.2f} each. "
        f"Total Allocated: ₹{allocated_focused:.2f}, Cash Remaining: ₹{cash_left_focused:.2f}."
    )
    
    # Strategy B: Diversified (Allocation split between company and risk-free cash based on volatility)
    # Higher volatility -> more cash buffer
    vol_factor = max(0.2, min(1.0, 1.5 / (volatility if volatility > 0 else 1.0)))
    allocated_diversified_amount = budget * vol_factor
    shares_diversified = int(allocated_diversified_amount / price)
    allocated_diversified = shares_diversified * price
    cash_left_diversified = budget - allocated_diversified
    
    diversified_desc = (
        f"Strategy B (Risk-Adjusted Diversified): Volatility-adjusted allocation (volatility: {volatility}%). "
        f"Allocated {vol_factor*100:.1f}% of budget to equities. "
        f"Buy {shares_diversified} shares of {ticker} at ₹{price:.2f} each. "
        f"Total Allocated: ₹{allocated_diversified:.2f}, Cash Buffer/Remaining: ₹{cash_left_diversified:.2f}."
    )
    
    return {
        "focused": focused_desc,
        "diversified": diversified_desc
    }
