import os
import sys
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

# Ensure python can find the modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtesting.historical_runner import run_ablation_study
from backtesting.historical_report import evaluate_recommendation

def run_evals():
    print("==================================================")
    print("INITIATING AUTOMATED LLM EVALUATION PIPELINE")
    print("==================================================\n")

    # Define our test scenarios
    eval_scenarios = [
        {
            "query": "Analyze TCS (Ticker: TCS.NS) and recommend a portfolio allocation for 50000 INR",
            "ticker": "TCS.NS",
            "start_date": "2025-09-30",
            "cutoff_date": "2026-03-31",
            "horizon_days": 90
        },
        {
            "query": "Analyze HDFC Bank (Ticker: HDFCBANK.NS) for a 3 month investment",
            "ticker": "HDFCBANK.NS",
            "start_date": "2025-09-30",
            "cutoff_date": "2026-03-31",
            "horizon_days": 90
        }
    ]

    for i, scenario in enumerate(eval_scenarios):
        print(f"\nRUNNING EVAL {i+1}: {scenario['ticker']}")
        print(f"  Cutoff Date (AI Knowledge Boundary): {scenario['cutoff_date']}")
        
        # Run Ablation Study (Structured vs Structured+RAG)
        results = run_ablation_study(scenario["query"], scenario["start_date"], scenario["cutoff_date"])
        
        version_a = results.get("Version_A", {})
        version_b = results.get("Version_B", {})
        telemetry = version_b.get("_telemetry", {})
        
        # Evaluate Ground Truth Performance
        cutoff_dt = datetime.strptime(scenario["cutoff_date"], "%Y-%m-%d")
        end_dt = cutoff_dt + timedelta(days=scenario["horizon_days"])
        end_date_str = end_dt.strftime("%Y-%m-%d")
        
        eval_metrics = evaluate_recommendation(
            ticker=scenario["ticker"],
            recommendation_str=str(version_b.get("recommendation", "HOLD")),
            cutoff_date=scenario["cutoff_date"],
            end_date=end_date_str
        )

        print("\n--- ABLATION EVALUATION (RAG IMPACT) ---")
        print(f"  Historical News Docs Injected: {telemetry.get('Documents Passed to LLM', 0)}")
        print(f"  Version A (No RAG) Bullishness: {version_a.get('bullishness_score', 'N/A')}")
        print(f"  Version B (RAG) Bullishness:    {version_b.get('bullishness_score', 'N/A')}")
        
        print("\n--- GROUND-TRUTH FINANCIAL EVALUATION ---")
        if "error" in eval_metrics:
            print(f"  Error calculating ground truth: {eval_metrics['error']}")
        else:
            print(f"  AI Predicted Direction:   {eval_metrics.get('predicted_direction')}")
            print(f"  Actual Market Direction:  {eval_metrics.get('actual_direction')}")
            print(f"  Direction Correct?        {eval_metrics.get('direction_correct')}")
            print(f"  Actual ROI:               {eval_metrics.get('actual_return_pct', 0):.2f}%")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    run_evals()
