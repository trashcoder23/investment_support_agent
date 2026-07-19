"""
app.py — Investment Support Agent — Streamlit Chat UI.

This interface interacts directly with the custom LangGraph StateGraph Agent,
rendering a visual reasoning pipeline and a highly professional report.
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.agent import run_agent

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISA v2 — Investment Support Agent",
    page_icon="📈",
    layout="wide",
)

st.title("📈 ISA v2 — Investment Support Agent")
st.markdown(
    "Analyze companies or sectors using a production-grade, single-pass StateGraph Agent "
    "powered by Grok, Gemini, or OpenAI."
)

# ── Sidebar Configurations ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Agent Settings")
    provider = os.getenv("LLM_PROVIDER", "grok").upper()
    st.info(f"**LLM Provider:** {provider}")
    
    st.markdown("---")
    st.subheader("Supported Companies")
    st.markdown(
        "- **IT:** TCS, Infosys, HCLTech, Wipro\n"
        "- **Banking:** HDFC Bank, ICICI Bank, SBI, Axis Bank\n"
        "- **Pharma:** Sun Pharma, Cipla, Divi's Labs, Dr. Reddy's\n"
        "- **Auto:** Hero MotoCorp, Maruti Suzuki, M&M, Bajaj Auto"
    )

# ── Initialize Chat History ───────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "type": "text",
            "content": "Hello! I am your StateGraph Investment Support Agent. Ask me about a company (e.g. 'Should I buy TCS with ₹50,000?') or a sector."
        }
    ]

# ── Display Chat Messages ─────────────────────────────────────────────────────
def render_json_report(state: dict):
    data = state.get("response", {})
    if "error" in data:
        st.error(data["error"])
        return

    company = data.get("company", "Unknown")
    st.subheader(f"📊 Investment Analysis Report: {company}")
    
    # 1. Pipeline execution logs & telemetry
    col_logs, col_telemetry = st.columns([2, 1])
    with col_logs:
        st.markdown("### ⚙️ Agent Pipeline Checklist")
        for log in state.get("logs", []):
            st.markdown(log)
            
    with col_telemetry:
        st.markdown("### ⏱️ Telemetry")
        telemetry = state.get("telemetry", {})
        for node, exec_time in telemetry.items():
            st.metric(node, f"{exec_time}s")
            
    st.markdown("---")
    
    # 2. Main Report
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Outlook", data.get("overall_outlook", "N/A"))
    with col2:
        st.metric("Query Type", data.get("query_type", "Analyze"))
    with col3:
        st.metric("Confidence Level", data.get("confidence", "N/A"))

    st.markdown("### 📝 Executive Summary")
    st.write(data.get("recommendation", ""))
    
    with st.expander("📊 Technical Analysis", expanded=True):
        st.write(data.get("technical_analysis", "No technical details provided."))
        
    with st.expander("📰 Recent News & RAG Findings", expanded=True):
        st.write(data.get("news_analysis", "No news details provided."))
        
    with st.expander("⚠️ Risk Analysis", expanded=False):
        st.write(data.get("risk_analysis", "No risks specified."))
        
    with st.expander("💼 Portfolio Allocation Recommendation", expanded=False):
        st.write(data.get("portfolio_allocation", "No portfolio allocation details available."))
        
    if data.get("supporting_evidence"):
        st.markdown("### 🔍 Supporting Evidence Citations")
        for evidence in data["supporting_evidence"]:
            st.markdown(f"- {evidence}")
            
    st.caption(f"Disclaimer: {data.get('disclaimer', '')}")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("type") == "json":
            render_json_report(message["content"])
        else:
            st.markdown(message["content"])

# ── Handle User Input ─────────────────────────────────────────────────────────
if prompt := st.chat_input("E.g., Analyze TCS and allocate Rs 75000"):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Get agent response
    with st.chat_message("assistant"):
        with st.status("Initializing StateGraph Agent...") as status:
            try:
                # Execute graph workflow
                final_state = run_agent(prompt)
                status.update(label="StateGraph Workflow Completed", state="complete", expanded=False)
                
                # Check for LLM / provider errors
                if isinstance(final_state, dict) and "error" in final_state.get("response", {}):
                    err = final_state["response"]["error"]
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "type": "text", "content": err})
                elif "error" in final_state:
                    err = final_state["error"]
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "type": "text", "content": err})
                else:
                    render_json_report(final_state)
                    st.session_state.messages.append({"role": "assistant", "type": "json", "content": final_state})
                    
            except Exception as e:
                status.update(label="Workflow Failed", state="error", expanded=True)
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": error_msg})
