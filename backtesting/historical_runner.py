import os
import sys
from typing import Dict, Any
from unittest.mock import patch

# Ensure paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.agent import run_agent
from agent.nodes import INVESTMENT_SYSTEM_PROMPT
from backtesting.historical_metrics import get_historical_financial_metrics
from backtesting.historical_rag import get_historical_relevant_unstructured_data, RAG_TELEMETRY

def run_ablation_study(query: str, start_date: str, cutoff_date: str) -> Dict[str, Any]:
    """
    Runs the historical evaluation for both Version A (Structured Only) and Version B (Structured + RAG).
    Returns both results and a decision delta.
    """
    results = {}
    
    # Custom system prompt for historical evaluation
    historical_system_prompt = (
        f"{INVESTMENT_SYSTEM_PROMPT}\n\n"
        f"IMPORTANT HISTORICAL CONSTRAINT:\n"
        f"Current Evaluation Date: {cutoff_date}\n"
        f"You are performing historical investment analysis. Only use information available on or before this date. "
        f"Do not assume knowledge of future events. Ignore any information published after this date.\n"
        f"You MUST fill out bullishness_score, expected_return_range, top_positive_factors, top_negative_factors, decision_explanation, and portfolio_projection. "
        f"For portfolio_projection, given a ₹100,000 investment, provide 'Investment', 'Bear Case', 'Base Case', 'Bull Case', and 'Expected Return'."
    )
    
    # ---------------------------------------------------------
    # VERSION A: Structured Data Only
    # ---------------------------------------------------------
    with patch("agent.nodes.get_financial_metrics") as mock_metrics:
        with patch("agent.nodes.get_relevant_unstructured_data") as mock_rag:
            with patch("agent.nodes.INVESTMENT_SYSTEM_PROMPT", historical_system_prompt):
                # 1. Patch metrics to use historical bounds
                mock_metrics.side_effect = lambda ticker: get_historical_financial_metrics(ticker, start_date, cutoff_date)
                
                # 2. Patch RAG to return nothing (Structured Only)
                mock_rag.return_value = "No unstructured data retrieved. Analysis is based on structured metrics only."
                
                print(f"\n[Historical Runner] Executing Version A (Structured Only) for cutoff {cutoff_date}")
                state_a = run_agent(query)
                results["Version_A"] = state_a.get("response", {})
                
    # ---------------------------------------------------------
    # VERSION B: Structured + RAG
    # ---------------------------------------------------------
    with patch("agent.nodes.get_financial_metrics") as mock_metrics:
        with patch("agent.nodes.get_relevant_unstructured_data") as mock_rag:
            with patch("agent.nodes.INVESTMENT_SYSTEM_PROMPT", historical_system_prompt):
                # 1. Patch metrics to use historical bounds
                mock_metrics.side_effect = lambda ticker: get_historical_financial_metrics(ticker, start_date, cutoff_date)
                
                # 2. Patch RAG to use historical filtering
                mock_rag.side_effect = lambda comp, q, top_k=5, ticker=None: get_historical_relevant_unstructured_data(comp, q, cutoff_date, top_k, ticker)
                
                print(f"\n[Historical Runner] Executing Version B (Structured + RAG) for cutoff {cutoff_date}")
                state_b = run_agent(query)
                results["Version_B"] = state_b.get("response", {})
                
                telemetry_b = state_b.get("telemetry", {})
                telemetry_b["Evaluation Date"] = cutoff_date
                telemetry_b["Historical Documents Retrieved"] = RAG_TELEMETRY.get("Historical Documents Retrieved", 0)
                telemetry_b["Future Documents Rejected"] = RAG_TELEMETRY.get("Future Documents Rejected", 0)
                telemetry_b["Documents Passed to LLM"] = RAG_TELEMETRY.get("Documents Passed to LLM", 0)
                
                results["Version_B"]["_telemetry"] = telemetry_b
                results["Version_B"]["_retrieved_docs"] = RAG_TELEMETRY.get("Retrieved Documents", [])
                
    return results
