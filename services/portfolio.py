"""
portfolio.py — Portfolio generation service.

Responsibilities:
- Strategy A: Invest 100% in the single highest predicted return company.
- Strategy B: Risk-adjusted diversified allocation using
              Risk Score = Predicted Return / Historical Volatility.
              Filters out companies with predicted return ≤ 0.
              Normalises scores and allocates budget proportionally.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def strategy_a(predictions: list[dict], budget: float) -> dict:
    """
    Strategy A — All-in on the top performer.

    Parameters
    ----------
    predictions : list[dict]  Output from predictor.predict_sector()
    budget      : float       Total investment amount in ₹

    Returns
    -------
    dict with keys:
        strategy, company, ticker, predicted_return, confidence,
        current_price, allocation_pct, investment_amount, shares_approx
    """
    # Filter out predictions with errors
    valid = [p for p in predictions if p.get("predicted_return") is not None]

    if not valid:
        return {"error": "No valid predictions available for Strategy A."}

    best = max(valid, key=lambda p: p["predicted_return"])

    shares_approx = (
        int(budget / best["current_price"])
        if best.get("current_price") and best["current_price"] > 0
        else None
    )

    return {
        "strategy":          "A",
        "company":           best["company"],
        "ticker":            best["ticker"],
        "predicted_return":  best["predicted_return"],
        "confidence":        best["confidence"],
        "r2":                best["r2"],
        "current_price":     best.get("current_price"),
        "allocation_pct":    100.0,
        "investment_amount": round(budget, 2),
        "shares_approx":     shares_approx,
    }


def strategy_b(predictions: list[dict], budget: float) -> list[dict]:
    """
    Strategy B — Diversified risk-adjusted portfolio.

    Risk Score = Predicted Return / Historical Volatility
    - Companies with predicted return ≤ 0 are excluded.
    - Scores are normalised.
    - Budget is allocated proportionally.

    Parameters
    ----------
    predictions : list[dict]  Output from predictor.predict_sector()
    budget      : float       Total investment amount in ₹

    Returns
    -------
    list[dict]  Per-company allocation dicts, sorted by allocation descending.
    Each dict has keys:
        company, ticker, predicted_return, volatility, risk_score,
        allocation_pct, investment_amount, confidence, current_price, shares_approx
    """
    # Filter: must have valid return AND volatility AND return > 0
    candidates = [
        p for p in predictions
        if (
            p.get("predicted_return") is not None
            and p["predicted_return"] > 0
            and p.get("volatility") is not None
            and p["volatility"] > 0
        )
    ]

    if not candidates:
        logger.warning(
            "Strategy B: No candidates with positive predicted return + valid volatility. "
            "Falling back to Strategy A allocation."
        )
        # Fallback: show best company with 100% allocation (mirror strategy A)
        valid = [p for p in predictions if p.get("predicted_return") is not None]
        if not valid:
            return [{"error": "No valid predictions for Strategy B."}]

        best = max(valid, key=lambda p: p["predicted_return"])
        return [{
            "company":           best["company"],
            "ticker":            best["ticker"],
            "predicted_return":  best["predicted_return"],
            "volatility":        best.get("volatility"),
            "risk_score":        None,
            "allocation_pct":    100.0,
            "investment_amount": round(budget, 2),
            "confidence":        best["confidence"],
            "current_price":     best.get("current_price"),
            "shares_approx":     None,
        }]

    # Compute risk scores
    for candidate in candidates:
        candidate["risk_score"] = candidate["predicted_return"] / candidate["volatility"]

    total_score = sum(c["risk_score"] for c in candidates)

    # Normalise and allocate
    allocations = []
    for c in candidates:
        alloc_pct    = (c["risk_score"] / total_score) * 100
        alloc_amount = budget * (c["risk_score"] / total_score)

        shares_approx = (
            int(alloc_amount / c["current_price"])
            if c.get("current_price") and c["current_price"] > 0
            else None
        )

        allocations.append({
            "company":           c["company"],
            "ticker":            c["ticker"],
            "predicted_return":  c["predicted_return"],
            "volatility":        round(c["volatility"], 4),
            "risk_score":        round(c["risk_score"], 4),
            "allocation_pct":    round(alloc_pct, 2),
            "investment_amount": round(alloc_amount, 2),
            "confidence":        c["confidence"],
            "current_price":     c.get("current_price"),
            "shares_approx":     shares_approx,
        })

    # Sort highest allocation first
    allocations.sort(key=lambda x: x["allocation_pct"], reverse=True)
    return allocations
