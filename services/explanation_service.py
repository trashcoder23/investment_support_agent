"""
explanation_service.py — AI-powered explanations via Google Gemini.

Responsibilities:
- Build a structured prompt from prediction data.
- Call the Gemini API.
- Return 2–3 sentence beginner-friendly explanations per company.
- Fail gracefully — returns None if the API call fails.
"""

import os
import logging

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)

# Attempt to load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _get_gemini_client():
    """Initialise and return the Gemini GenerativeModel, or None if unavailable."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — AI explanations will be unavailable.")
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        return model
    except Exception as exc:
        logger.error(f"Failed to initialise Gemini client: {exc}")
        return None


def _build_prompt(predictions: list[dict], sector: str, horizon: int, budget: float) -> str:
    """
    Build the prompt for Gemini.

    predictions: list of dicts with company, predicted_return, volatility,
                 risk_score (optional), confidence.
    """
    lines = [
        f"You are a friendly investment educator. Explain the following stock "
        f"predictions to a beginner investor in very simple language. "
        f"The investor has ₹{budget:,.0f} to invest in the {sector} sector "
        f"over the next {horizon} days.\n",
        "Here are the ML model predictions:",
    ]

    for p in predictions:
        if p.get("predicted_return") is None:
            continue
        vol_str = "N/A" if p.get("volatility") is None else f"{p['volatility']:.2f}%"
        lines.append(
            f"\n- **{p['company']}**: "
            f"Predicted return = {p['predicted_return']:.2f}%, "
            f"Confidence = {p['confidence']}, "
            f"Historical Volatility = {vol_str}"
        )
        if p.get("risk_score") is not None:
            lines.append(f"  Risk Score = {p['risk_score']:.4f}")


    lines.append(
        "\nFor each company, write exactly 2–3 sentences explaining in simple terms:\n"
        "1. What the prediction means for a beginner.\n"
        "2. Whether it looks risky or safe based on volatility.\n"
        "3. Any helpful context about this company or sector.\n"
        "Keep it encouraging, clear, and avoid jargon. "
        "Do NOT give specific buy/sell advice."
    )

    return "\n".join(lines)


def get_explanations(
    predictions: list[dict],
    sector: str,
    horizon: int,
    budget: float,
) -> dict[str, str] | None:
    """
    Generate Gemini explanations for all companies in the prediction list.

    Parameters
    ----------
    predictions : list[dict]  Company prediction dicts from predictor.py
    sector      : str
    horizon     : int
    budget      : float

    Returns
    -------
    dict[company_name -> explanation_text]  or  None if Gemini is unavailable.
    """
    model = _get_gemini_client()
    if model is None:
        return None

    prompt = _build_prompt(predictions, sector, horizon, budget)

    try:
        response = model.generate_content(prompt)
        raw_text = response.text

        # Parse the response into per-company sections
        # We'll do a second targeted prompt to get structured per-company output
        result = {}
        for p in predictions:
            if p.get("predicted_return") is None:
                continue
            result[p["company"]] = _extract_company_explanation(raw_text, p["company"])

        return result

    except Exception as exc:
        logger.error(f"Gemini API call failed: {exc}")
        return None


def get_single_explanation(
    company: str,
    predicted_return: float,
    volatility: float | None,
    risk_score: float | None,
    confidence: str,
    sector: str,
    horizon: int,
) -> str | None:
    """
    Generate a focused 2–3 sentence explanation for a single company.
    Returns None on failure.
    """
    model = _get_gemini_client()
    if model is None:
        return None

    vol_str   = f"{volatility:.2f}%" if volatility is not None else "unknown"
    score_str = f"{risk_score:.4f}"  if risk_score is not None  else "N/A"

    prompt = (
        f"You are explaining a stock prediction to a complete beginner investor.\n\n"
        f"Company: {company}\n"
        f"Sector: {sector}\n"
        f"Prediction window: {horizon} days\n"
        f"Predicted return: {predicted_return:.2f}%\n"
        f"Model confidence: {confidence}\n"
        f"Historical volatility: {vol_str}\n"
        f"Risk score: {score_str}\n\n"
        f"Write exactly 2–3 sentences:\n"
        f"1. What this prediction means in plain English.\n"
        f"2. Whether this looks risky or relatively stable.\n"
        f"3. One encouraging or cautionary note for a first-time investor.\n"
        f"No jargon, no specific buy/sell advice."
    )

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as exc:
        logger.error(f"Gemini failed for {company}: {exc}")
        return None


def _extract_company_explanation(full_text: str, company_name: str) -> str:
    """
    Attempt to extract the paragraph relevant to a company from a bulk response.
    Falls back to the full response if extraction fails.
    """
    lines = full_text.split("\n")
    collecting = False
    collected  = []

    for line in lines:
        if company_name.lower() in line.lower():
            collecting = True
        if collecting:
            collected.append(line)
            # Stop after collecting ~3 non-empty lines
            non_empty = [l for l in collected if l.strip()]
            if len(non_empty) >= 3:
                break

    if collected:
        return " ".join(l.strip() for l in collected if l.strip())
    return full_text.strip()
