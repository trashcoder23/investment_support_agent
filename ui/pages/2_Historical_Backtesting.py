import os
import sys
import pandas as pd
from dateutil.relativedelta import relativedelta
import streamlit as st

# Ensure root path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backtesting.historical_runner import run_ablation_study
from backtesting.historical_report import evaluate_recommendation

st.set_page_config(page_title="Historical Analyst", page_icon="⏳", layout="wide")

st.info(f"""
### Historical Evaluation
**Historical Mode Enabled**
✓ Future data blocked  
✓ Historical news only  
✓ Historical technical indicators only  
✓ Historical RAG enabled
""")

st.title("⏳ Time-Aware Historical AI Analyst")
st.markdown("Evaluate the AI Agent as an analyst frozen in time. Does unstructured knowledge improve the decision?")

col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Ticker", value="HDFCBANK.NS")
with col2:
    cutoff_date = st.date_input("Evaluation Date (Cutoff)", value=pd.to_datetime("2026-03-31"))
with col3:
    horizon = st.selectbox("Prediction Horizon", ["1 Month", "3 Months", "6 Months"], index=1)

start_date = cutoff_date - relativedelta(months=6)
months_to_add = int(horizon.split()[0])
end_date = cutoff_date + relativedelta(months=months_to_add)

st.caption(f"**Training Window:** {start_date.strftime('%d %b %Y')} → {cutoff_date.strftime('%d %b %Y')}  |  **Prediction Window:** {cutoff_date.strftime('%d %b %Y')} → {end_date.strftime('%d %b %Y')}")

if st.button("Run Historical Backtest", type="primary"):
    with st.spinner(f"Simulating Analyst at {cutoff_date}..."):
        query = f"Analyze {ticker} and recommend a portfolio allocation."
        
        try:
            results = run_ablation_study(query, str(start_date), str(cutoff_date))
            version_a = results.get("Version_A", {})
            version_b = results.get("Version_B", {})
            telemetry_b = version_b.get("_telemetry", {})
            retrieved_docs = version_b.get("_retrieved_docs", [])
            
            # Evaluate Performance
            eval_results = evaluate_recommendation(ticker, version_b.get("recommendation", "HOLD"), str(cutoff_date), str(end_date))
            
            # Save for Dashboard
            if "backtest_history" not in st.session_state:
                st.session_state["backtest_history"] = []
                
            eval_record = eval_results.copy()
            eval_record["ticker"] = ticker
            eval_record["evaluation_date"] = str(cutoff_date)
            eval_record["recommendation_a"] = version_a.get("overall_outlook", "HOLD")
            eval_record["recommendation_b"] = version_b.get("overall_outlook", "HOLD")
            eval_record["confidence_a"] = version_a.get("confidence", "N/A")
            eval_record["confidence_b"] = version_b.get("confidence", "N/A")
            eval_record["bullishness_a"] = version_a.get("bullishness_score", 50)
            eval_record["bullishness_b"] = version_b.get("bullishness_score", 50)
            st.session_state["backtest_history"].append(eval_record)
            
            st.success("Historical Evaluation Complete!")
            
            # --- TELEMETRY ---
            st.subheader("🕵️‍♂️ Analyst Data Integrity Logs")
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Historical Docs Retrieved", telemetry_b.get("Historical Documents Retrieved", 0))
            t2.metric("Future Docs REJECTED", telemetry_b.get("Future Documents Rejected", 0))
            t3.metric("Docs Passed to LLM", telemetry_b.get("Documents Passed to LLM", 0))
            t4.metric("Structured Data End", str(cutoff_date))
            
            if telemetry_b.get("Future Documents Rejected", 0) > 0:
                st.warning("⚠️ Future documents were detected and strictly blocked to prevent data leakage.")

            # --- DECISION DELTA ---
            st.markdown("---")
            st.subheader("⚖️ Decision Delta (Ablation Study)")
            
            delta_data = {
                "Metric": ["Recommendation", "Bullishness", "Confidence"],
                "Structured Only": [
                    version_a.get("overall_outlook", "N/A"),
                    version_a.get("bullishness_score", "N/A"),
                    version_a.get("confidence", "N/A")
                ],
                "Structured + RAG": [
                    version_b.get("overall_outlook", "N/A"),
                    version_b.get("bullishness_score", "N/A"),
                    version_b.get("confidence", "N/A")
                ]
            }
            st.table(pd.DataFrame(delta_data).set_index("Metric"))
            
            st.info(f"**Decision Explanation:** {version_b.get('decision_explanation', 'No explanation provided.')}")
            
            # --- PORTFOLIO PROJECTION ---
            st.markdown("---")
            st.subheader("💰 Portfolio Projection (₹100,000 Budget)")
            proj = version_b.get("portfolio_projection", {})
            if proj:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Investment", proj.get("Investment", "₹100,000"))
                c2.metric("Bear Case", proj.get("Bear Case", "N/A"))
                c3.metric("Base Case", proj.get("Base Case", "N/A"))
                c4.metric("Bull Case", proj.get("Bull Case", "N/A"))
                st.caption(f"**Expected Return:** {proj.get('Expected Return', version_b.get('expected_return_range', 'N/A'))}")
            else:
                st.write("No portfolio projection generated.")
                
            # --- RETRIEVED HISTORICAL EVIDENCE ---
            st.markdown("---")
            st.subheader("📰 Historical Documents Used")
            if retrieved_docs:
                for doc in retrieved_docs:
                    st.markdown(f"**✓ {doc.get('document_type', 'Document')} - {doc.get('title', 'Unknown Title')}**")
                    st.caption(f"{doc.get('published_date', 'Unknown Date')} | Source: {doc.get('source', 'Unknown')}")
            else:
                st.write("No historical documents were available before the cutoff date.")
                
            # --- ACTUAL PERFORMANCE ---
            st.markdown("---")
            st.subheader(f"📈 Actual Validation Performance ({cutoff_date.strftime('%b %Y')} → {end_date.strftime('%b %Y')})")
            if "error" in eval_results:
                st.error(eval_results["error"])
            else:
                v1, v2, v3, v4 = st.columns(4)
                v1.metric("Predicted Direction", eval_results["direction_predicted"])
                v2.metric("Actual Return", f"{eval_results['actual_return_pct']}%")
                v3.metric("Direction Correct?", "✅ Yes" if eval_results["is_correct"] else "❌ No")
                
                # Mocking portfolio value based on actual return
                inv = 100000
                act_port = inv * (1 + eval_results['actual_return_pct']/100)
                v4.metric("Actual Portfolio Value", f"₹{act_port:,.2f}")
                
        except Exception as e:
            st.error(f"Backtesting failed: {e}")
