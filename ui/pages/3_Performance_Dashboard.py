import os
import sys
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backtesting.historical_report import generate_performance_metrics
from ui.theme import apply_theme

st.set_page_config(page_title="Performance Dashboard", layout="wide")
apply_theme()
st.title("📊 Performance Dashboard")

st.markdown("""
Aggregate metrics from Historical Backtesting to evaluate the AI Agent's portfolio performance over the validation window.
""")

history = st.session_state.get("backtest_history", [])

if not history:
    st.info("No backtests run yet. Go to 'Historical Backtesting' in the sidebar to evaluate some stocks, then return here.")
else:
    metrics = generate_performance_metrics(history)
    
    if "error" in metrics:
        st.error(metrics["error"])
    else:
        st.markdown("### 🏆 Overall Portfolio Performance")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Evaluations", metrics.get("total_evaluated", 0))
        c2.metric("Win Rate / Accuracy", metrics.get("win_rate", "0%"))
        c3.metric("Average Actual Return", metrics.get("average_actual_return", "0%"))
        c4.metric("Portfolio Return", metrics.get("portfolio_return", "0%"))
        
        # --- RAG IMPACT METRICS ---
        st.markdown("---")
        st.subheader("🧠 RAG Impact (Why Unstructured Knowledge Matters)")
        st.caption("Demonstrating the contribution of unstructured knowledge over structured technical analysis alone.")
        rag = metrics.get("rag_impact", {})
        
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Companies Evaluated", rag.get("companies_evaluated", 0))
        r2.metric("Recommendations Changed", rag.get("recommendations_changed", 0))
        r3.metric("Avg Confidence Improvement", rag.get("avg_confidence_improvement", "0%"))
        r4.metric("Avg Bullishness Change", rag.get("avg_bullishness_change", "0"))
        
        st.markdown("#### Direction Accuracy")
        a1, a2 = st.columns(2)
        a1.metric("Technical Only (Structured)", rag.get("accuracy_technical_only", "0%"))
        a2.metric("Structured + RAG", rag.get("accuracy_structured_rag", "0%"))
        
        st.markdown("---")
        st.subheader("Distribution of Recommendations")
        dist = metrics.get("recommendation_distribution", {})
        
        c_buy, c_hold, c_sell = st.columns(3)
        c_buy.metric("BUY", dist.get("BUY", 0))
        c_hold.metric("HOLD", dist.get("HOLD", 0))
        c_sell.metric("SELL", dist.get("SELL", 0))
        
        st.markdown("---")
        st.subheader("Evaluation History Logs")
        st.dataframe(history)
