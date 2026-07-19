import time
import json
import re
import logging
from typing import Dict, Any

from agent.state import AgentState
from llm import get_llm_model
from services.yahoo_service import get_financial_metrics
from services.rag_service import get_relevant_unstructured_data
from services.portfolio_service import calculate_allocations
from prompts.investment_prompt import INVESTMENT_SYSTEM_PROMPT
from utils.json_validator import validate_and_repair_json
from utils.cache_service import extract_tickers_from_query
from utils.config import TICKER_TO_NAME, TICKER_TO_SECTOR
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List

logger = logging.getLogger(__name__)

class InvestmentReportSchema(BaseModel):
    company: str = Field(description="Name of the company being analyzed")
    query_type: str = Field(description="Analyze / Portfolio Allocation / Future Outlook / Comparison")
    overall_outlook: str = Field(description="Qualitative outlook (e.g. Bullish, Bearish, Neutral)")
    confidence: str = Field(description="Evidence-based confidence description and level")
    technical_analysis: str = Field(description="Plain English interpretation of technical indicators")
    news_analysis: str = Field(description="Synthesis of recent news, filings, and transcripts")
    risk_analysis: str = Field(description="Summary of key risks")
    recommendation: str = Field(description="Final investment recommendation")
    portfolio_allocation: str = Field(description="Allocation strategy details referencing computed metrics")
    supporting_evidence: List[str] = Field(description="List of supporting evidence citations from retrieved facts")
    disclaimer: str = Field(description="AI disclaimer")
    
    # Historical / Extended fields
    bullishness_score: int = Field(default=50, description="Score from 0-100 indicating overall bullishness")
    expected_return_range: str = Field(default="N/A", description="Expected percentage return range (e.g. '5% to 10%')")
    top_positive_factors: List[str] = Field(default_factory=list, description="Top positive drivers")
    top_negative_factors: List[str] = Field(default_factory=list, description="Top negative drivers")
    decision_explanation: str = Field(default="N/A", description="Explanation of how qualitative data shifted the decision")
    portfolio_projection: Dict[str, str] = Field(default_factory=dict, description="Keys: 'Investment', 'Bear Case', 'Base Case', 'Bull Case', 'Expected Return'")

def intent_parsing_node(state: AgentState) -> Dict[str, Any]:
    print("\n--- [NODE: INTENT PARSING] Started ---")
    start_time = time.time()
    query = state.get("query", "")
    print(f"[Intent Parsing] Processing Query: '{query}'")
    
    # Deterministic Extraction
    tickers = extract_tickers_from_query(query)
    ticker = tickers[0] if tickers else None
    company = TICKER_TO_NAME.get(ticker, "Unknown") if ticker else "Unknown"
    sector = TICKER_TO_SECTOR.get(ticker, "Unknown") if ticker else "Unknown"
    
    print(f"[Intent Parsing] Extracted Ticker: {ticker}, Company: {company}, Sector: {sector}")
    
    # Budget Extraction
    budget = None
    budget_match = re.search(r'(?:₹|INR|Rs\.?)\s*(\d+(?:,\d+)*(?:\.\d+)?)', query, re.IGNORECASE)
    if not budget_match:
        budget_match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:INR|Rs|rupees)', query, re.IGNORECASE)
    if budget_match:
        try:
            budget_str = budget_match.group(1).replace(",", "")
            budget = float(budget_str)
            print(f"[Intent Parsing] Extracted Budget: ₹{budget}")
        except Exception as e:
            print(f"[Intent Parsing] Error parsing budget: {e}")
            
    # Investment Horizon Extraction
    horizon = "Not specified"
    if "month" in query.lower():
        horizon = "Medium Term (Months)"
    elif "year" in query.lower() or "long term" in query.lower():
        horizon = "Long Term (Years)"
    elif "short term" in query.lower():
        horizon = "Short Term"
    print(f"[Intent Parsing] Extracted Horizon: {horizon}")
        
    # Query Type Extraction
    query_type = "Analyze"
    query_lower = query.lower()
    if "portfolio" in query_lower or "allocate" in query_lower or "budget" in query_lower:
        query_type = "Portfolio Allocation"
    elif "predict" in query_lower or "future" in query_lower:
        query_type = "Future Outlook"
    elif "compare" in query_lower:
        query_type = "Comparison"
    print(f"[Intent Parsing] Extracted Query Type: {query_type}")
        
    intent = {
        "ticker": ticker,
        "company": company,
        "sector": sector,
        "budget": budget,
        "investment_horizon": horizon,
        "query_type": query_type
    }
    
    execution_time = round(time.time() - start_time, 2)
    
    logs = list(state.get("logs", []))
    logs.append(f"✓ Intent Identified ({execution_time}s)")
    
    telemetry = dict(state.get("telemetry", {}))
    telemetry["Intent Parsing"] = execution_time
    
    print(f"--- [NODE: INTENT PARSING] Completed in {execution_time}s ---")
    return {
        "intent": intent,
        "logs": logs,
        "telemetry": telemetry
    }

def structured_context_node(state: AgentState) -> Dict[str, Any]:
    print("\n--- [NODE: STRUCTURED CONTEXT] Started ---")
    start_time = time.time()
    intent = state.get("intent", {})
    ticker = intent.get("ticker")
    budget = intent.get("budget", 50000.0) or 50000.0  # Default budget fallback
    
    print(f"[Structured Context] Target Ticker: {ticker}, Budget for allocation: ₹{budget}")
    
    logs = list(state.get("logs", []))
    telemetry = dict(state.get("telemetry", {}))
    
    if not ticker:
        print("[Structured Context] ✗ Error: No company ticker identified in query.")
        execution_time = round(time.time() - start_time, 2)
        logs.append(f"✗ Fetching Market Data failed: No Ticker ({execution_time}s)")
        telemetry["Structured Context"] = execution_time
        return {
            "structured_context": {"error": "No company ticker identified in query."},
            "logs": logs,
            "telemetry": telemetry
        }
        
    # Fetch structured metrics
    print(f"[Structured Context] Fetching financial metrics for {ticker}...")
    metrics = get_financial_metrics(ticker)
    print(f"[Structured Context] Fetched {len(metrics)} metric groups.")
    
    # Calculate portfolio allocations
    print(f"[Structured Context] Calculating portfolio allocations...")
    allocations = calculate_allocations(metrics, budget)
    
    structured_context = {
        "metrics": metrics,
        "allocations": allocations
    }
    
    execution_time = round(time.time() - start_time, 2)
    
    logs = [l for l in logs if "Fetching Market Data..." not in l and "Computing Technical Indicators..." not in l]
    logs.append(f"✓ Fetching Market Data ({execution_time}s)")
    logs.append(f"✓ Computing Technical Indicators ({execution_time}s)")
    
    telemetry["Structured Context"] = execution_time
    
    print(f"--- [NODE: STRUCTURED CONTEXT] Completed in {execution_time}s ---")
    return {
        "structured_context": structured_context,
        "logs": logs,
        "telemetry": telemetry
    }

def rag_context_node(state: AgentState) -> Dict[str, Any]:
    print("\n--- [NODE: RAG CONTEXT] Started ---")
    start_time = time.time()
    intent = state.get("intent", {})
    company = intent.get("company", "Unknown")
    query = state.get("query", "")
    
    print(f"[RAG Context] Searching unstructured data for Company: '{company}' and Query: '{query}'")
    
    logs = list(state.get("logs", []))
    telemetry = dict(state.get("telemetry", {}))
    
    if company == "Unknown":
        print("[RAG Context] ✗ Skipping vector database search (unknown company).")
        execution_time = round(time.time() - start_time, 2)
        logs.append(f"✗ Searching Vector Database skipped ({execution_time}s)")
        telemetry["RAG Context"] = execution_time
        return {
            "rag_context": "No unstructured data retrieved (unknown company).",
            "logs": logs,
            "telemetry": telemetry
        }
        
    print("[RAG Context] Executing similarity search in local FAISS vectorstore...")
    unstructured_text = get_relevant_unstructured_data(company, query, top_k=5)
    
    if isinstance(unstructured_text, list):
        print(f"[RAG Context] Retrieved {len(unstructured_text)} snippets.")
    else:
        print(f"[RAG Context] Retrieved text block (length={len(str(unstructured_text))}).")
        
    execution_time = round(time.time() - start_time, 2)
    
    logs = [l for l in logs if "Retrieving News..." not in l and "Searching Vector Database..." not in l]
    logs.append(f"✓ Retrieving News ({execution_time}s)")
    logs.append(f"✓ Searching Vector Database ({execution_time}s)")
    
    telemetry["RAG Context"] = execution_time
    
    print(f"--- [NODE: RAG CONTEXT] Completed in {execution_time}s ---")
    return {
        "rag_context": unstructured_text,
        "logs": logs,
        "telemetry": telemetry
    }

def format_structured_context(ctx: Any) -> str:
    if not isinstance(ctx, dict):
        return "No structured context available."
        
    metrics = ctx.get("metrics", {})
    allocations = ctx.get("allocations", {})
    
    if not isinstance(metrics, dict):
        return "No financial metrics available."
        
    lines = []
    lines.append(f"Company Ticker: {metrics.get('Ticker', 'N/A')}")
    lines.append(f"Current Price: ₹{metrics.get('Current_Price_INR', 'N/A')}")
    lines.append(f"Volume: {metrics.get('Volume', 'N/A')}")
    lines.append(f"Market Cap: ₹{metrics.get('Market_Cap_INR', 'N/A')}")
    lines.append(f"Trailing P/E: {metrics.get('Trailing_PE', 'N/A')}")
    lines.append(f"Forward P/E: {metrics.get('Forward_PE', 'N/A')}")
    
    tech = metrics.get("Technical_Indicators", {})
    if isinstance(tech, dict) and tech:
        lines.append("\nTechnical Indicators:")
        lines.append(f"- Daily Return: {tech.get('Daily_Return_Pct', 'N/A')}%")
        lines.append(f"- Monthly Return: {tech.get('Monthly_Return_Pct', 'N/A')}%")
        lines.append(f"- 20-Day Moving Average (MA20): {tech.get('MA20', 'N/A')}")
        lines.append(f"- 50-Day Moving Average (MA50): {tech.get('MA50', 'N/A')}")
        lines.append(f"- EMA (20): {tech.get('EMA', 'N/A')}")
        lines.append(f"- RSI (14): {tech.get('RSI', 'N/A')}")
        lines.append(f"- MACD: {tech.get('MACD', 'N/A')}")
        lines.append(f"- Bollinger Bands: High={tech.get('BB_High', 'N/A')}, Low={tech.get('BB_Low', 'N/A')}, Mid={tech.get('BB_Mid', 'N/A')}")
        lines.append(f"- Rolling Volatility: {tech.get('Rolling_Volatility_Pct', 'N/A')}%")
        lines.append(f"- ATR: {tech.get('ATR', 'N/A')}")
        lines.append(f"- Momentum: {tech.get('Momentum', 'N/A')}")
        
    if isinstance(allocations, dict) and allocations:
        lines.append("\nPortfolio Allocation Proposals:")
        lines.append(f"- Focused: {allocations.get('focused', 'N/A')}")
        lines.append(f"- Diversified: {allocations.get('diversified', 'N/A')}")
        
    return "\n".join(lines)

def format_rag_context(rag: Any) -> str:
    if not rag:
        return "No relevant news retrieved."
        
    if isinstance(rag, str):
        cleaned = rag.strip()
        return cleaned if cleaned else "No relevant news retrieved."
        
    if isinstance(rag, list):
        lines = []
        for i, item in enumerate(rag):
            lines.append(f"News {i+1}:")
            if hasattr(item, "page_content"):
                lines.append(item.page_content.strip())
            elif isinstance(item, dict) and "content" in item:
                lines.append(item["content"].strip())
            else:
                lines.append(str(item).strip())
            lines.append("")
        return "\n".join(lines).strip()
        
    return str(rag).strip()

def _execute_structured_llm_call(llm, messages) -> dict:
    """Strategy 1: Using native structured output (with_structured_output)."""
    print("[LLM Reasoning] Strategy 1: Attempting native structured output.")
    structured_llm = llm.with_structured_output(InvestmentReportSchema)
    response = structured_llm.invoke(messages)
    
    if hasattr(response, "model_dump"):
        return response.model_dump()
    elif hasattr(response, "dict"):
        return response.dict()
    return response

def _execute_raw_llm_call_with_repair(llm, messages, company: str, query_type: str) -> dict:
    """Strategy 2/3: Raw generation and text parsing."""
    print("[LLM Reasoning] Attempting raw generation + text parsing fallback.")
    raw_response = llm.invoke(messages)
    output_text = raw_response.content
    
    if isinstance(output_text, list):
        text_parts = []
        for part in output_text:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
            else:
                if hasattr(part, "text"):
                    text_parts.append(part.text)
                elif hasattr(part, "content"):
                    text_parts.append(part.content)
                else:
                    text_parts.append(str(part))
        output_text = "".join(text_parts)
    elif not isinstance(output_text, str):
        output_text = str(output_text)
        
    print(f"[LLM Reasoning] Raw output received (len={len(output_text)}). Validating/repairing...")
    response_json = validate_and_repair_json(output_text, company, query_type)
    
    if response_json.get("confidence") == "Low (parsing failure)":
        print("[LLM Reasoning] ✗ Raw output failed schema validation.")
        raise ValueError("Raw JSON output was truncated or failed schema parsing.")
        
    return response_json

def llm_reasoning_node(state: AgentState) -> Dict[str, Any]:
    print("\n--- [NODE: LLM REASONING] Started ---")
    start_time = time.time()
    logs = list(state.get("logs", []))
    telemetry = dict(state.get("telemetry", {}))
    
    intent = state.get("intent", {})
    structured_context = state.get("structured_context", {})
    rag_context = state.get("rag_context", "")
    query = state.get("query", "")
    
    company = intent.get("company", "Unknown")
    ticker = intent.get("ticker", "Unknown")
    query_type = intent.get("query_type", "Analyze")
    
    print(f"[LLM Reasoning] Analyzing: {company} ({ticker}) | Query Type: {query_type}")
    
    structured_context_str = format_structured_context(structured_context)
    rag_context_str = format_rag_context(rag_context)
    
    context = (
        f"COMPANY: {company} ({ticker})\n"
        f"INVESTMENT BUDGET: ₹{intent.get('budget', 'Not specified')}\n"
        f"INVESTMENT HORIZON: {intent.get('investment_horizon', 'Not specified')}\n"
        f"QUERY TYPE: {query_type}\n\n"
        f"--- STRUCTURED FINANCIALS & TECHNICAL INDICATORS ---\n"
        f"{structured_context_str}\n\n"
        f"--- RETRIEVED UNSTRUCTURED DOCUMENTS (RAG) ---\n"
        f"{rag_context_str}\n\n"
        f"USER QUERY: {query}\n"
    )
    
    messages = [
        SystemMessage(content=INVESTMENT_SYSTEM_PROMPT),
        HumanMessage(content=context)
    ]
    
    response_json = None
    llm = get_llm_model()
    
    try:
        response_json = _execute_structured_llm_call(llm, messages)
        print("[LLM Reasoning] ✓ Native structured output succeeded.")
    except Exception as e:
        print(f"[LLM Reasoning] ✗ Native structured output failed: {e}")
        try:
            response_json = _execute_raw_llm_call_with_repair(llm, messages, company, query_type)
            print("[LLM Reasoning] ✓ Raw generation fallback succeeded.")
        except Exception as retry_err:
            print(f"[LLM Reasoning] ✗ Raw generation fallback failed: {retry_err}. Executing Final Rescue Prompt...")
            try:
                repair_messages = messages + [
                    HumanMessage(content="The previous response was not valid JSON or was cut off. Return ONLY valid JSON matching the schema. Do not include markdown or explanations. Keep fields concise.")
                ]
                response_json = _execute_raw_llm_call_with_repair(llm, repair_messages, company, query_type)
                print("[LLM Reasoning] ✓ Final Rescue Prompt succeeded.")
            except Exception as final_err:
                print(f"[LLM Reasoning] ✗ All LLM strategies failed. Error: {final_err}")
                response_json = {
                    "company": company,
                    "query_type": query_type,
                    "overall_outlook": "N/A",
                    "confidence": "Low (LLM reasoning failed)",
                    "technical_analysis": "Error retrieving details.",
                    "news_analysis": "Error retrieving details.",
                    "risk_analysis": "Error retrieving details.",
                    "recommendation": f"An error occurred during LLM processing: {str(final_err)}",
                    "portfolio_allocation": "Unavailable",
                    "supporting_evidence": [],
                    "disclaimer": "AI Reasoning Node failed to complete cleanly."
                }
                
    execution_time = round(time.time() - start_time, 2)
    
    logs = [l for l in logs if "AI Reasoning..." not in l]
    logs.append(f"✓ AI Reasoning ({execution_time}s)")
    logs.append(f"✓ Recommendation Generated ({execution_time}s)")
    
    telemetry["LLM Reasoning"] = execution_time
    
    print(f"--- [NODE: LLM REASONING] Completed in {execution_time}s ---")
    return {
        "response": response_json,
        "logs": logs,
        "telemetry": telemetry
    }
