import pandas as pd
import re
from typing import Dict, Any, List
from backtesting.historical_data_loader import get_validation_market_data

def evaluate_recommendation(ticker: str, recommendation_str: str, cutoff_date: str, end_date: str) -> Dict[str, Any]:
    """
    Evaluates a single recommendation against actual market performance in the validation window.
    """
    try:
        actual_df = get_validation_market_data(ticker, cutoff_date, end_date)
        
        if actual_df.empty or len(actual_df) < 2:
            return {"error": "Insufficient validation data"}
            
        start_price = actual_df.iloc[0]["Close"]
        end_price = actual_df.iloc[-1]["Close"]
        
        actual_return_pct = ((end_price - start_price) / start_price) * 100
        
        rec_upper = recommendation_str.upper()
        if "BUY" in rec_upper:
            direction = "BUY"
            is_correct = actual_return_pct > 0
        elif "SELL" in rec_upper:
            direction = "SELL"
            is_correct = actual_return_pct < 0
        else:
            direction = "HOLD"
            is_correct = -5 <= actual_return_pct <= 5
            
        return {
            "actual_return_pct": round(actual_return_pct, 2),
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "direction_predicted": direction,
            "is_correct": is_correct
        }
    except Exception as e:
        return {"error": str(e)}

def _extract_number(conf_str) -> float:
    try:
        if isinstance(conf_str, (int, float)):
            return float(conf_str)
        matches = re.findall(r'\d+', str(conf_str))
        if matches:
            return float(matches[0])
        return 50.0
    except:
        return 50.0

def _check_direction(rec_str, actual_return):
    rec_upper = str(rec_str).upper()
    if "BUY" in rec_upper: return actual_return > 0
    if "SELL" in rec_upper: return actual_return < 0
    return -5 <= actual_return <= 5

def generate_performance_metrics(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates evaluations to compute portfolio-level metrics and RAG Impact.
    """
    valid_evals = [e for e in evaluations if "error" not in e]
    total = len(valid_evals)
    
    if total == 0:
        return {"error": "No valid evaluations to compute metrics"}
        
    avg_actual_return = sum(e["actual_return_pct"] for e in valid_evals) / total
    
    # RAG Impact Metrics
    changed_count = 0
    conf_diff_sum = 0
    bull_diff_sum = 0
    
    correct_a = 0
    correct_b = 0
    
    rec_distribution = {"BUY": 0, "HOLD": 0, "SELL": 0}
    
    for e in valid_evals:
        rec_a = e.get("recommendation_a", "HOLD")
        rec_b = e.get("recommendation_b", "HOLD")
        
        act_ret = e["actual_return_pct"]
        
        if _check_direction(rec_a, act_ret): correct_a += 1
        if _check_direction(rec_b, act_ret): correct_b += 1
        
        if str(rec_a).upper() != str(rec_b).upper():
            changed_count += 1
            
        conf_a = _extract_number(e.get("confidence_a", 50))
        conf_b = _extract_number(e.get("confidence_b", 50))
        conf_diff_sum += (conf_b - conf_a)
        
        bull_a = _extract_number(e.get("bullishness_a", 50))
        bull_b = _extract_number(e.get("bullishness_b", 50))
        bull_diff_sum += (bull_b - bull_a)
        
        dir_pred = e["direction_predicted"]
        if dir_pred in rec_distribution:
            rec_distribution[dir_pred] += 1
            
    win_rate_a = (correct_a / total) * 100
    win_rate_b = (correct_b / total) * 100
    
    avg_conf_imp = conf_diff_sum / total
    avg_bull_change = bull_diff_sum / total
            
    return {
        "total_evaluated": total,
        "win_rate": f"{round(win_rate_b, 2)}%",
        "average_actual_return": f"{round(avg_actual_return, 2)}%",
        "portfolio_return": f"{round(avg_actual_return, 2)}%",
        "recommendation_distribution": rec_distribution,
        "rag_impact": {
            "companies_evaluated": total,
            "recommendations_changed": changed_count,
            "avg_confidence_improvement": f"{'+' if avg_conf_imp > 0 else ''}{round(avg_conf_imp, 2)}%",
            "avg_bullishness_change": f"{'+' if avg_bull_change > 0 else ''}{round(avg_bull_change, 2)}%",
            "accuracy_technical_only": f"{round(win_rate_a, 2)}%",
            "accuracy_structured_rag": f"{round(win_rate_b, 2)}%"
        }
    }
