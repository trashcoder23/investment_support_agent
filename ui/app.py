"""
app.py — Investment Support Agent — Streamlit UI.

Pages:
  🏠 Home        — Overview and instructions
  📊 Predict     — Generate investment predictions + portfolio strategies
  🔍 Validate    — Compare past predictions against actual market returns
  ℹ️  About      — Project details

The UI layer ONLY calls service functions — no ML or data logic here.
"""

import os
import sys
import logging
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config import SECTOR_TICKERS, HORIZONS, MODELS_DIR, SECTOR_MODEL_STEM
from services.predictor import predict_sector, load_prediction_history
from services.portfolio import strategy_a, strategy_b
from services.explanation_service import get_single_explanation
from services.validation_service import run_model_validation, load_validation_results

logging.basicConfig(level=logging.WARNING)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISA — Investment Support Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Dark glass card ── */
    .card {
        background: linear-gradient(135deg, rgba(255,255,255,0.07) 0%, rgba(255,255,255,0.03) 100%);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    .card-highlight {
        background: linear-gradient(135deg, rgba(99,179,237,0.15) 0%, rgba(128,90,213,0.10) 100%);
        border: 1px solid rgba(99,179,237,0.35);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }

    /* ── Metric pills ── */
    .metric-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 100px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .pill-high   { background: rgba(72,199,142,0.20); color: #48C78E; border: 1px solid #48C78E55; }
    .pill-medium { background: rgba(255,183,77,0.20);  color: #FFB74D; border: 1px solid #FFB74D55; }
    .pill-low    { background: rgba(255,100,100,0.20); color: #FF6464; border: 1px solid #FF646455; }

    /* ── Page title ── */
    .page-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #63B3ED, #B794F4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .page-sub {
        color: rgba(255,255,255,0.55);
        font-size: 0.95rem;
        margin-bottom: 1.6rem;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #63B3ED;
        margin: 1.4rem 0 0.8rem;
        padding-bottom: 4px;
        border-bottom: 1px solid rgba(99,179,237,0.3);
    }

    /* ── Strategy badge ── */
    .strategy-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 2px 10px;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-right: 8px;
    }
    .badge-a { background: rgba(72,199,142,0.20); color: #48C78E; }
    .badge-b { background: rgba(99,179,237,0.20);  color: #63B3ED; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b27 100%);
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    /* ── Nav buttons ── */
    .nav-btn {
        display: flex;
        align-items: center;
        gap: 10px;
        width: 100%;
        padding: 0.65rem 1rem;
        border-radius: 10px;
        border: 1px solid transparent;
        background: transparent;
        color: rgba(255,255,255,0.6);
        font-size: 0.9rem;
        font-weight: 500;
        cursor: pointer;
        margin-bottom: 4px;
        transition: all 0.18s ease;
        text-decoration: none;
    }
    .nav-btn:hover {
        background: rgba(99,179,237,0.12);
        color: #63B3ED;
        border-color: rgba(99,179,237,0.25);
    }
    .nav-btn.active {
        background: linear-gradient(135deg,rgba(99,179,237,0.20),rgba(183,148,244,0.15));
        color: #63B3ED;
        border-color: rgba(99,179,237,0.40);
        font-weight: 600;
    }

    /* ── Streamlit overrides ── */
    .stButton > button {
        background: linear-gradient(135deg, #63B3ED, #B794F4);
        color: #0d1117;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.4rem;
        transition: opacity 0.2s;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.88; color: #0d1117; }

    div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
    div[data-testid="stMetricLabel"] { font-size: 0.78rem; color: rgba(255,255,255,0.5); }

    .disclaimer {
        background: rgba(255,183,77,0.08);
        border: 1px solid rgba(255,183,77,0.25);
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        color: #FFB74D;
        font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _confidence_colour(level: str) -> str:
    return {"High": "pill-high", "Medium": "pill-medium", "Low": "pill-low"}.get(level, "pill-low")


def _models_ready() -> bool:
    """Check whether at least some joblib models exist."""
    if not os.path.exists(MODELS_DIR):
        return False
    return any(f.endswith(".joblib") for f in os.listdir(MODELS_DIR))


def _fmt_inr(amount: float) -> str:
    """Format a number as Indian Rupees."""
    if amount >= 1_00_00_000:
        return f"₹{amount / 1_00_00_000:.2f} Cr"
    elif amount >= 1_00_000:
        return f"₹{amount / 1_00_000:.2f} L"
    return f"₹{amount:,.0f}"


def _return_colour(val: float | None) -> str:
    if val is None: return "rgba(255,255,255,0.6)"
    return "#48C78E" if val >= 0 else "#FF6464"


# ── Sidebar navigation ────────────────────────────────────────────────────────

# Initialise page in session state so nav buttons can control it
if "page" not in st.session_state:
    st.session_state["page"] = "🏠 Home"

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 1.2rem 0 1.6rem;">
          <div style="font-size:2.4rem;">📈</div>
          <div style="font-size:1.15rem; font-weight:700; color:#63B3ED;">ISA</div>
          <div style="font-size:0.72rem; color:rgba(255,255,255,0.45); margin-top:2px;">
            Investment Support Agent
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_items = [
        ("🏠", "Home",     "🏠 Home"),
        ("📊", "Predict",  "📊 Predict"),
        ("🔍", "Validate", "🔍 Validate"),
        ("ℹ️", "About",    "ℹ️ About"),
    ]
    for icon, label, key in nav_items:
        active = "active" if st.session_state["page"] == key else ""
        if st.button(
            f"{icon}  {label}",
            key=f"nav_{label}",
            use_container_width=True,
        ):
            st.session_state["page"] = key
            # Clear stale result when navigating away from validate
            if key != "🔍 Validate":
                st.session_state.pop("val_result", None)
            st.rerun()

    page = st.session_state["page"]

    st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:0.8rem 0;'>", unsafe_allow_html=True)

    # Last prediction badge
    lp_sector  = st.session_state.get("last_pred_sector")
    lp_horizon = st.session_state.get("last_pred_horizon")
    if lp_sector and lp_horizon:
        st.markdown(
            f'<div style="background:rgba(99,179,237,0.10);border:1px solid rgba(99,179,237,0.25);'
            f'border-radius:8px;padding:0.55rem 0.8rem;font-size:0.78rem;">'
            f'<div style="color:rgba(255,255,255,0.4);font-size:0.68rem;margin-bottom:3px;">LAST PREDICTION</div>'
            f'<strong style="color:#63B3ED;">{lp_sector}</strong> · {lp_horizon} days</div>',
            unsafe_allow_html=True,
        )

    # Model status
    st.markdown("<div style='margin-top:0.5rem'>", unsafe_allow_html=True)
    if _models_ready():
        st.markdown(
            '<div style="text-align:center;color:#48C78E;font-size:0.78rem;">✓ Models ready</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="text-align:center;color:#FF6464;font-size:0.78rem;">'
            "⚠ Models not found<br>"
            '<span style="color:rgba(255,255,255,0.4);">Run <code>python train.py</code></span>'
            "</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div style="margin-top:1.5rem;font-size:0.68rem;color:rgba(255,255,255,0.22);text-align:center;">'
        'Not financial advice. For educational use only.</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Home":
    st.markdown('<div class="page-title">Investment Support Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">ML-powered predictions for Indian equity markets • Educational tool for beginner investors</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="disclaimer">⚠️ <strong>Disclaimer:</strong> ISA is NOT a financial advisor. '
        "Predictions are based on historical ML models and should NOT be used as sole investment guidance. "
        "Always consult a SEBI-registered financial advisor before investing.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sectors Covered", "4")
    with col2:
        st.metric("Companies", "16")
    with col3:
        st.metric("ML Models", "12")
    with col4:
        st.metric("Prediction Horizons", "3")

    st.markdown("---")

    # How it works
    st.markdown('<div class="section-header">🔄 How It Works</div>', unsafe_allow_html=True)

    steps = [
        ("1", "Data Collection", "Historical OHLCV data downloaded from Yahoo Finance (Oct 2025 – Mar 2026)"),
        ("2", "Feature Engineering", "12+ technical indicators computed: RSI, MACD, Bollinger Bands, Moving Averages, and more"),
        ("3", "Model Training", "3 ML models trained per sector × horizon: Linear Regression, Random Forest, XGBoost"),
        ("4", "Best Model Selection", "Model with highest R² on validation set selected for each combination"),
        ("5", "Prediction", "Latest market data fetched, features engineered, prediction generated"),
        ("6", "Portfolio Strategy", "Strategy A (focused) and Strategy B (diversified) portfolios constructed"),
        ("7", "AI Explanation", "Google Gemini explains predictions in beginner-friendly language"),
    ]

    for num, title, desc in steps:
        st.markdown(
            f'<div class="card"><strong style="color:#63B3ED;">{num}. {title}</strong><br>'
            f'<span style="color:rgba(255,255,255,0.65);font-size:0.88rem;">{desc}</span></div>',
            unsafe_allow_html=True,
        )

    # Investment universe
    st.markdown('<div class="section-header">🏢 Investment Universe</div>', unsafe_allow_html=True)
    for sector, tickers in SECTOR_TICKERS.items():
        companies = ", ".join(tickers.values())
        st.markdown(
            f'<div class="card"><strong style="color:#B794F4;">{sector}</strong><br>'
            f'<span style="color:rgba(255,255,255,0.7);font-size:0.9rem;">{companies}</span></div>',
            unsafe_allow_html=True,
        )

    # Setup instructions
    if not _models_ready():
        st.markdown('<div class="section-header">⚙️ First-Time Setup</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-highlight">'
            "<strong>Models not yet trained.</strong> Run the following command in your terminal to set up ISA:<br><br>"
            "<code>python train.py</code><br><br>"
            "This will download historical data, engineer features, and train all 12 ML models. "
            "Expected time: 5–15 minutes."
            "</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Predict":
    st.markdown('<div class="page-title">Generate Predictions</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Enter your investment parameters to receive ML-powered recommendations</div>',
        unsafe_allow_html=True,
    )

    if not _models_ready():
        st.error(
            "⚠️ **Models not found.** Please run `python train.py` first to train the ML models.",
            icon="🤖",
        )
        st.stop()

    # ── Input form ──────────────────────────────────────────────────────────
    with st.form("predict_form"):
        col_a, col_b, col_c = st.columns([2, 1.5, 1.5])

        with col_a:
            budget = st.number_input(
                "💰 Investment Budget (₹)",
                min_value=1_000.0,
                max_value=1_00_00_000.0,
                value=50_000.0,
                step=1_000.0,
                format="%.0f",
                help="Total amount you wish to invest across this sector",
            )
        with col_b:
            sector = st.selectbox(
                "🏢 Sector",
                options=list(SECTOR_TICKERS.keys()),
                help="Select the sector to analyse",
            )
        with col_c:
            horizon = st.selectbox(
                "📅 Prediction Horizon",
                options=HORIZONS,
                format_func=lambda h: f"{h} Days",
                help="How far ahead to predict returns",
            )

        submitted = st.form_submit_button("🚀 Generate Predictions", use_container_width=True)

    if submitted:
        with st.spinner(f"Fetching live data and running predictions for {sector} ({horizon} days) ..."):
            try:
                predictions = predict_sector(sector, horizon, budget)
            except Exception as exc:
                st.error(f"❌ Prediction failed: {exc}")
                st.stop()

        if not predictions:
            st.warning("No predictions returned. Check internet connection and model files.")
            st.stop()

        # Store what was predicted so Validate page can lock to it
        st.session_state["last_pred_sector"]  = sector
        st.session_state["last_pred_horizon"] = horizon
        # Clear any stale validation result so Validate page shows fresh
        st.session_state.pop("val_result", None)

        # ── Per-company prediction cards ─────────────────────────────────
        st.markdown('<div class="section-header">📊 Company Predictions</div>', unsafe_allow_html=True)

        valid_preds = [p for p in predictions if p.get("predicted_return") is not None]

        if not valid_preds:
            st.error("Could not generate predictions for any company. Check your internet connection.")
            st.stop()

        num_cols = min(len(valid_preds), 4)
        cols = st.columns(num_cols)

        for i, pred in enumerate(valid_preds):
            ret_val  = pred["predicted_return"]
            ret_sign = "+" if ret_val >= 0 else ""
            ret_col  = _return_colour(ret_val)
            conf_cls = _confidence_colour(pred["confidence"])

            with cols[i % num_cols]:
                st.markdown(
                    f"""
                    <div class="card" style="text-align:center;">
                      <div style="font-size:0.8rem;color:rgba(255,255,255,0.5);margin-bottom:4px;">{pred['sector']}</div>
                      <div style="font-size:1.05rem;font-weight:700;margin-bottom:6px;">{pred['company']}</div>
                      <div style="font-size:2rem;font-weight:700;color:{ret_col};">
                        {ret_sign}{ret_val:.2f}%
                      </div>
                      <div style="font-size:0.75rem;color:rgba(255,255,255,0.4);margin-top:2px;">
                        Predicted {horizon}-day return
                      </div>
                      <div style="margin-top:10px;">
                        <span class="metric-pill {conf_cls}">{pred['confidence']} Confidence</span>
                      </div>
                      {"" if pred.get("current_price") is None else
                       f'<div style="font-size:0.78rem;color:rgba(255,255,255,0.35);margin-top:8px;">CMP ₹{pred["current_price"]:,.2f}</div>'}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── Strategy A ──────────────────────────────────────────────────────
        st.markdown(
            '<div class="section-header">🎯 Strategy A — Focused Investment</div>',
            unsafe_allow_html=True,
        )

        strat_a = strategy_a(predictions, budget)

        if "error" in strat_a:
            st.warning(strat_a["error"])
        else:
            ret_val  = strat_a["predicted_return"]
            ret_col  = _return_colour(ret_val)
            ret_sign = "+" if ret_val >= 0 else ""
            conf_cls = _confidence_colour(strat_a["confidence"])

            st.markdown(
                f"""
                <div class="card-highlight">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:1rem;">
                    <div>
                      <span class="strategy-badge badge-a">Strategy A</span>
                      <span style="font-size:1.3rem;font-weight:700;">{strat_a['company']}</span>
                      <div style="color:rgba(255,255,255,0.5);font-size:0.82rem;margin-top:4px;">
                        100% of budget allocated to the top-predicted company
                      </div>
                    </div>
                    <div style="text-align:right;">
                      <div style="font-size:2rem;font-weight:700;color:{ret_col};">{ret_sign}{ret_val:.2f}%</div>
                      <div style="font-size:0.75rem;color:rgba(255,255,255,0.4);">Predicted {horizon}-day return</div>
                    </div>
                  </div>
                  <hr style="border-color:rgba(255,255,255,0.08);margin:12px 0;">
                  <div style="display:flex;gap:2rem;flex-wrap:wrap;">
                    <div>
                      <div style="font-size:0.75rem;color:rgba(255,255,255,0.4);">Investment Amount</div>
                      <div style="font-size:1.1rem;font-weight:600;">{_fmt_inr(strat_a['investment_amount'])}</div>
                    </div>
                    {"" if strat_a.get("shares_approx") is None else
                     f'<div><div style="font-size:0.75rem;color:rgba(255,255,255,0.4);">Approx. Shares</div>'
                     f'<div style="font-size:1.1rem;font-weight:600;">{strat_a["shares_approx"]}</div></div>'}
                    <div>
                      <div style="font-size:0.75rem;color:rgba(255,255,255,0.4);">Confidence</div>
                      <div style="margin-top:4px;"><span class="metric-pill {conf_cls}">{strat_a['confidence']}</span></div>
                    </div>
                    <div>
                      <div style="font-size:0.75rem;color:rgba(255,255,255,0.4);">Model R²</div>
                      <div style="font-size:1.1rem;font-weight:600;">{strat_a['r2']:.4f}</div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Strategy B ──────────────────────────────────────────────────────
        st.markdown(
            '<div class="section-header">📐 Strategy B — Diversified Portfolio</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="color:rgba(255,255,255,0.45);font-size:0.82rem;margin-bottom:0.8rem;">'
            "Allocation based on Risk Score = Predicted Return / Historical Volatility. "
            "Companies with non-positive predicted returns are excluded."
            "</div>",
            unsafe_allow_html=True,
        )

        strat_b = strategy_b(predictions, budget)

        if strat_b and "error" in strat_b[0]:
            st.warning(strat_b[0]["error"])
        else:
            # Summary bar chart
            if strat_b:
                fig, ax = plt.subplots(figsize=(8, 3.2))
                fig.patch.set_alpha(0)
                ax.set_facecolor("#0d1117")

                companies_b = [s["company"] for s in strat_b]
                alloc_pcts  = [s["allocation_pct"] for s in strat_b]
                colours     = ["#63B3ED", "#B794F4", "#48C78E", "#FFB74D"]

                bars = ax.barh(companies_b, alloc_pcts, color=colours[:len(strat_b)], height=0.5, edgecolor="none")
                for bar, pct in zip(bars, alloc_pcts):
                    ax.text(
                        pct + 0.5, bar.get_y() + bar.get_height() / 2,
                        f"{pct:.1f}%",
                        va="center", ha="left",
                        color="white", fontsize=9, fontweight="bold",
                    )
                ax.set_xlabel("Allocation %", color=(1,1,1,0.5), fontsize=9)
                ax.tick_params(colors=(1,1,1,0.6), labelsize=9)
                ax.spines[["top", "right", "bottom"]].set_visible(False)
                ax.spines["left"].set_color((1,1,1,0.1))
                ax.set_xlim(0, max(alloc_pcts) * 1.18)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            # Detailed table
            table_rows = []
            for s in strat_b:
                table_rows.append({
                    "Company":            s["company"],
                    "Predicted Return %": f"{s['predicted_return']:+.2f}%",
                    "Volatility":         f"{s['volatility']:.2f}%" if s.get("volatility") else "N/A",
                    "Risk Score":         f"{s['risk_score']:.4f}"  if s.get("risk_score") else "N/A",
                    "Allocation %":       f"{s['allocation_pct']:.1f}%",
                    "Amount":             _fmt_inr(s["investment_amount"]),
                    "Confidence":         s["confidence"],
                })

            st.dataframe(
                pd.DataFrame(table_rows),
                use_container_width=True,
                hide_index=True,
            )

        # ── AI Explanations ─────────────────────────────────────────────────
        st.markdown(
            '<div class="section-header">🤖 AI Explanations (powered by Gemini)</div>',
            unsafe_allow_html=True,
        )

        for pred in valid_preds:
            risk_score = None
            for s in (strat_b if isinstance(strat_b, list) else []):
                if s.get("company") == pred["company"]:
                    risk_score = s.get("risk_score")
                    break

            with st.spinner(f"Getting explanation for {pred['company']} ..."):
                explanation = get_single_explanation(
                    company          = pred["company"],
                    predicted_return = pred["predicted_return"],
                    volatility       = pred.get("volatility"),
                    risk_score       = risk_score,
                    confidence       = pred["confidence"],
                    sector           = sector,
                    horizon          = horizon,
                )

            if explanation:
                conf_cls = _confidence_colour(pred["confidence"])
                st.markdown(
                    f"""
                    <div class="card">
                      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                        <strong>{pred['company']}</strong>
                        <span class="metric-pill {conf_cls}">{pred['confidence']}</span>
                      </div>
                      <div style="color:rgba(255,255,255,0.7);font-size:0.9rem;line-height:1.6;">
                        {explanation}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="card"><strong>{pred["company"]}</strong>'
                    f'<span style="color:rgba(255,255,255,0.35);font-size:0.85rem;"> — AI explanation unavailable '
                    f"(check GEMINI_API_KEY)</span></div>",
                    unsafe_allow_html=True,
                )

        st.success("✅ Prediction logged to history. Visit the **Validate** tab later to check accuracy.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: VALIDATE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Validate":
    st.markdown('<div class="page-title">Validate Model Accuracy</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Test how accurately the ML model predicted real post-training market returns</div>',
        unsafe_allow_html=True,
    )

    if not _models_ready():
        st.error("⚠️ **Models not found.** Please run `python train.py` first.", icon="🤖")
        st.stop()

    # ── Simulation scenario card ─────────────────────────────────────────────
    st.markdown(
        '<div class="card-highlight">'
        '<strong style="color:#63B3ED;">🗓️ Validation Scenario</strong><br>'
        '<div style="display:flex;gap:2rem;flex-wrap:wrap;margin-top:10px;">'

        '<div style="flex:1;min-width:180px;">'
        '<div style="font-size:0.72rem;color:rgba(255,255,255,0.4);margin-bottom:2px;">TRAINING DATA</div>'
        '<div style="font-weight:600;">Oct 2025 → Mar 2026</div>'
        '</div>'

        '<div style="flex:1;min-width:180px;">'
        '<div style="font-size:0.72rem;color:rgba(255,255,255,0.4);margin-bottom:2px;">SIMULATED PREDICTION DATE</div>'
        '<div style="font-weight:600;color:#FFB74D;">April 1, 2026</div>'
        '<div style="font-size:0.75rem;color:rgba(255,255,255,0.45);">Market features as of Mar 31</div>'
        '</div>'

        '<div style="flex:1;min-width:200px;">'
        '<div style="font-size:0.72rem;color:rgba(255,255,255,0.4);margin-bottom:2px;">HORIZON TARGETS</div>'
        '<div style="font-size:0.85rem;">'
        '30 days → <strong>~Apr 30, 2026</strong><br>'
        '60 days → <strong>~May 31, 2026</strong><br>'
        '90 days → <strong>~Jun 30, 2026</strong>'
        '</div>'
        '</div>'

        '<div style="flex:1;min-width:180px;">'
        '<div style="font-size:0.72rem;color:rgba(255,255,255,0.4);margin-bottom:2px;">ACTUAL DATA AVAILABLE</div>'
        '<div style="font-weight:600;color:#48C78E;">✓ Apr / May / Jun / Jul 2026</div>'
        '<div style="font-size:0.75rem;color:rgba(255,255,255,0.45);">All horizons can be validated</div>'
        '</div>'

        '</div></div>',
        unsafe_allow_html=True,
    )

    # ── Enforce prediction-first workflow ─────────────────────────────────────
    lp_sector = st.session_state.get("last_pred_sector")
    lp_horizon = st.session_state.get("last_pred_horizon")

    if not lp_sector or not lp_horizon:
        st.info("ℹ️ Please go to the **Predict** tab and generate a prediction first to validate it.")
        st.stop()

    st.markdown(
        f'<div style="margin-bottom:1.5rem; padding: 1rem; border: 1px solid rgba(99,179,237,0.3); border-radius: 8px; background: rgba(99,179,237,0.05);">'
        f'<div style="font-size:0.85rem; color: rgba(255,255,255,0.6); margin-bottom: 4px;">TARGET TO VALIDATE</div>'
        f'<div style="font-size:1.1rem;"><strong>{lp_sector}</strong> model at <strong>{lp_horizon}-day</strong> horizon</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    if st.button("🔄 Run Validation for Last Prediction", use_container_width=True):
        with st.spinner(
            f"Fetching Apr 2026 market features → running {lp_sector} model → "
            f"downloading actual prices {lp_horizon} days later ..."
        ):
            result = run_model_validation(lp_sector, lp_horizon)
        # Persist so the display survives Streamlit reruns / file-change hot reloads
        st.session_state["val_result"]  = result
        st.session_state["val_sector"]  = lp_sector
        st.session_state["val_horizon"] = lp_horizon

    # ── Display result (from session state so it survives reruns) ────────────
    if "val_result" in st.session_state:
        result      = st.session_state["val_result"]
        val_sector  = st.session_state["val_sector"]
        val_horizon = st.session_state["val_horizon"]


        chart_data = result.get("chart_data", [])

        if "error" in result and not chart_data:
            st.error(f"❌ {result['error']}")
            st.stop()

        if "error" in result and chart_data:
            st.warning(f"⚠️ {result['error']}")

        sim_date    = result.get("simulation_date", "Apr 1, 2026")
        target_date = result.get("target_date", "")

        st.markdown(
            f'<div style="color:rgba(255,255,255,0.45);font-size:0.82rem;margin-bottom:0.8rem;">'
            f'📅 Simulated prediction on: <strong style="color:#FFB74D;">{sim_date}</strong>'
            + (f' &nbsp;→&nbsp; Actual prices measured at: <strong style="color:#48C78E;">{target_date}</strong>' if target_date else "")
            + f' &nbsp;|&nbsp; Horizon: <strong>{val_horizon} trading days</strong></div>',
            unsafe_allow_html=True,
        )



        # ── Metrics ──────────────────────────────────────────────────────────
        if "metrics" in result:
            metrics = result["metrics"]
            st.markdown('<div class="section-header">📏 Validation Metrics</div>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("MAE",  f"{metrics['mae']:.4f}%",  help="Mean Absolute Error — average prediction error in %")
            with m2:
                st.metric("RMSE", f"{metrics['rmse']:.4f}%", help="Root Mean Squared Error — penalises large errors more")
            with m3:
                r2_val = metrics.get("r2")
                st.metric(
                    "R²",
                    f"{r2_val:.4f}" if r2_val is not None else "N/A",
                    help="R² score — 1.0 = perfect, 0.0 = no better than mean, negative = worse than mean",
                )

        # ── Chart ────────────────────────────────────────────────────────────
        if chart_data:
            st.markdown(
                '<div class="section-header">📈 Predicted vs Actual Returns</div>',
                unsafe_allow_html=True,
            )

            comparable = [d for d in chart_data if d.get("actual") is not None]
            unavailable = [d for d in chart_data if d.get("actual") is None]

            if comparable:
                companies_c = [d["company"]  for d in comparable]
                predicted_c = [d["predicted"] for d in comparable]
                actual_c    = [d["actual"]    for d in comparable]
                x           = np.arange(len(companies_c))
                width       = 0.35

                fig, ax = plt.subplots(figsize=(9, 4.5))
                fig.patch.set_alpha(0)
                ax.set_facecolor("#0d1117")

                bars1 = ax.bar(x - width / 2, predicted_c, width, label="Predicted (%)", color="#63B3ED", alpha=0.85, edgecolor="none")
                bars2 = ax.bar(x + width / 2, actual_c,    width, label="Actual (%)",    color="#48C78E", alpha=0.85, edgecolor="none")

                for bar in list(bars1) + list(bars2):
                    h = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + (0.15 if h >= 0 else -0.6),
                        f"{h:+.2f}",
                        ha="center", va="bottom",
                        fontsize=8, color="white",
                    )

                ax.axhline(0, color=(1,1,1,0.2), linewidth=0.8, linestyle="--")
                ax.set_xticks(x)
                ax.set_xticklabels(companies_c, color=(1,1,1,0.7), fontsize=9)
                ax.set_ylabel("Return %", color=(1,1,1,0.5), fontsize=9)
                ax.tick_params(colors=(1,1,1,0.5))
                ax.spines[["top", "right"]].set_visible(False)
                ax.spines[["left", "bottom"]].set_color((1,1,1,0.1))
                ax.legend(framealpha=0.15, labelcolor="white", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                # ── Return % chart (existing) ──────────────────────────────
                st.markdown(
                    '<div style="font-size:0.82rem;color:rgba(255,255,255,0.45);margin-bottom:6px;">'
                    "Return % comparison"
                    "</div>",
                    unsafe_allow_html=True,
                )

                # ── Stock Price chart ────────────────────────────────────────
                st.markdown(
                    '<div class="section-header">💹 Stock Prices — Entry vs Predicted vs Actual</div>',
                    unsafe_allow_html=True,
                )

                if all(d.get("entry_price") for d in comparable):
                    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
                    fig2.patch.set_alpha(0)
                    ax2.set_facecolor("#0d1117")

                    entry_prices    = [d["entry_price"]           for d in comparable]
                    pred_prices     = [d["predicted_end_price"]   for d in comparable]
                    actual_prices   = [d["actual_end_price"]      for d in comparable]
                    x3              = np.arange(len(companies_c))
                    w               = 0.25

                    b1 = ax2.bar(x3 - w,     entry_prices,  w, label="Entry Price (₹)",      color="#6B7280", alpha=0.85, edgecolor="none")
                    b2 = ax2.bar(x3,          pred_prices,   w, label="Predicted Price (₹)",   color="#63B3ED", alpha=0.85, edgecolor="none")
                    b3 = ax2.bar(x3 + w,      actual_prices, w, label="Actual Price (₹)",      color="#48C78E", alpha=0.85, edgecolor="none")

                    for bars in [b1, b2, b3]:
                        for bar in bars:
                            h = bar.get_height()
                            ax2.text(
                                bar.get_x() + bar.get_width() / 2,
                                h + (h * 0.005),
                                f"₹{h:,.0f}",
                                ha="center", va="bottom",
                                fontsize=7, color="white",
                            )

                    ax2.set_xticks(x3)
                    ax2.set_xticklabels(companies_c, color=(1,1,1,0.7), fontsize=9)
                    ax2.set_ylabel("Price (₹)", color=(1,1,1,0.5), fontsize=9)
                    ax2.tick_params(colors=(1,1,1,0.5))
                    ax2.spines[["top", "right"]].set_visible(False)
                    ax2.spines[["left", "bottom"]].set_color((1,1,1,0.1))
                    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"₹{v:,.0f}"))
                    ax2.legend(framealpha=0.15, labelcolor="white", fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)

                # ── Detailed price + profit table ────────────────────────────
                st.markdown(
                    '<div class="section-header">📋 Detailed Price & Profit Analysis</div>',
                    unsafe_allow_html=True,
                )
                table_rows = []
                for d in comparable:
                    err       = abs(d["predicted"] - d["actual"])
                    direction = "✅ Correct" if (d["predicted"] >= 0) == (d["actual"] >= 0) else "❌ Wrong"
                    sp        = d.get("entry_price")
                    pe        = d.get("predicted_end_price")
                    ae        = d.get("actual_end_price")
                    pc_pred   = d.get("price_change_predicted")
                    pc_act    = d.get("price_change_actual")
                    asp       = d.get("actual_start_price")
                    asd       = d.get("actual_start_date", "")

                    table_rows.append({
                        "Company":               d["company"],
                        "Sim. Date":             d.get("simulation_date", ""),
                        "Actual Start Date":     asd,
                        "Target Date":           d.get("target_date", ""),
                        "Entry Price (₹)":       f"₹{sp:,.2f}"  if sp  is not None else "—",
                        "Entry (Apr 1) Price(₹)":f"₹{asp:,.2f}" if asp is not None else "—",
                        "Predicted Price (₹)":   f"₹{pe:,.2f}"  if pe  is not None else "—",
                        "Actual Price (₹)":      f"₹{ae:,.2f}"  if ae  is not None else "—",
                        "Predicted ΔPrice":      (f"+₹{pc_pred:,.2f}" if pc_pred >= 0 else f"-₹{abs(pc_pred):,.2f}")
                                                 if pc_pred is not None else "—",
                        "Actual ΔPrice":         (f"+₹{pc_act:,.2f}"  if pc_act  >= 0 else f"-₹{abs(pc_act):,.2f}")
                                                 if pc_act  is not None else "—",
                        "Predicted Return":      f"{d['predicted']:+.2f}%",
                        "Actual Return":         f"{d['actual']:+.2f}%",
                        "Error":                 f"{err:.2f}%",
                        "Direction":             direction,
                    })
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)



            if unavailable:
                st.warning(
                    f"⚠️ Could not fetch actual data for: "
                    + ", ".join(d["company"] for d in unavailable)
                    + ". These companies are excluded from metrics."
                )

            st.success("✅ Validation result saved to validation_results.csv")

    # ── Past validation results ──────────────────────────────────────────────
    val_results = load_validation_results()
    if not val_results.empty:
        st.markdown(
            '<div class="section-header">📋 Past Validation Results</div>',
            unsafe_allow_html=True,
        )
        rename_map = {
            "validated_at":    "Validated At",
            "sector":          "Sector",
            "horizon":         "Horizon (days)",
            "simulation_date": "Prediction Date",
            "target_date":     "Actual Prices Date",
            "mae":             "MAE %",
            "rmse":            "RMSE %",
            "r2":              "R²",
        }
        preferred = list(rename_map.keys())
        available = [c for c in preferred if c in val_results.columns]
        st.dataframe(
            val_results[available].tail(20).rename(columns=rename_map),
            use_container_width=True,
            hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ About":
    st.markdown('<div class="page-title">About ISA</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Investment Support Agent — Technical Details</div>',
        unsafe_allow_html=True,
    )

    # Model metadata table
    from services.model_trainer import load_metadata
    metadata = load_metadata()

    if metadata:
        st.markdown('<div class="section-header">🤖 Trained Models</div>', unsafe_allow_html=True)

        rows = []
        for m in metadata:
            rows.append({
                "Sector":         m["sector"],
                "Horizon":        f"{m['horizon']}d",
                "Best Model":     m["selected_model"],
                "R²":             f"{m['r2']:.4f}",
                "MAE":            f"{m['mae']:.4f}",
                "RMSE":           f"{m['rmse']:.4f}",
                "Train Samples":  m.get("train_samples", "—"),
                "Val Samples":    m.get("val_samples", "—"),
                "Trained At":     m.get("trained_at", "—")[:16],
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Models not yet trained. Run `python train.py` to populate this section.")

    st.markdown('<div class="section-header">🛠️ Technology Stack</div>', unsafe_allow_html=True)
    tech_items = [
        ("Python 3",              "Core language"),
        ("Streamlit",             "Web application framework"),
        ("yfinance",              "Yahoo Finance data downloader"),
        ("pandas / numpy",        "Data processing"),
        ("ta (Technical Analysis Library)", "RSI, MACD, Bollinger Bands, EMA"),
        ("scikit-learn",          "Linear Regression, evaluation metrics"),
        ("XGBoost",               "Gradient Boosted Trees"),
        ("joblib",                "Model serialisation"),
        ("matplotlib",            "Charts and visualisations"),
        ("Google Gemini API",     "AI-powered explanations"),
    ]
    for tech, desc in tech_items:
        st.markdown(
            f'<div class="card" style="padding:0.8rem 1.2rem;">'
            f'<strong style="color:#63B3ED;">{tech}</strong> '
            f'<span style="color:rgba(255,255,255,0.5);font-size:0.85rem;">— {desc}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">📋 Features Engineered</div>', unsafe_allow_html=True)
    features = [
        "Open, High, Low, Close, Volume (raw OHLCV)",
        "Daily Return, Weekly Return, Monthly Return",
        "Moving Average 5 / 10 / 20 (MA5, MA10, MA20)",
        "Exponential Moving Average (EMA-20)",
        "Relative Strength Index (RSI-14)",
        "MACD (12/26/9)",
        "Bollinger Bands (Upper, Lower, Mid — 20-period, 2σ)",
        "Rolling Volatility (20-day std of daily returns)",
        "Average Daily Range (14-day average of High–Low)",
    ]
    for feat in features:
        st.markdown(f"- {feat}")

    st.markdown('<div class="section-header">⚠️ Disclaimer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="disclaimer">'
        "This application is built for <strong>educational purposes only</strong>. "
        "It does NOT constitute financial advice, investment advice, trading advice, or any other advice. "
        "Past performance of ML models on historical data does not guarantee future returns. "
        "Always conduct your own research and consult a SEBI-registered financial advisor "
        "before making investment decisions."
        "</div>",
        unsafe_allow_html=True,
    )
