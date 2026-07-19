import json
import re
import logging

logger = logging.getLogger(__name__)

def extract_json_object(text: str) -> str:
    """
    Locates the first '{' and counts matching braces to extract exactly the JSON payload,
    properly ignoring brackets inside string literals.
    """
    start_idx = text.find('{')
    if start_idx == -1:
        print("[JSON Validator] ✗ No opening brace '{' found in text.")
        return ""
    
    brace_count = 0
    in_string = False
    escape_char = False
    
    for idx in range(start_idx, len(text)):
        char = text[idx]
        
        if escape_char:
            escape_char = False
            continue
            
        if char == '\\':
            escape_char = True
            continue
            
        if char == '"':
            in_string = not in_string
            continue
            
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    extracted = text[start_idx:idx+1]
                    print(f"[JSON Validator] ✓ Successfully extracted JSON object (length={len(extracted)}).")
                    return extracted
                    
    print("[JSON Validator] ✗ Could not find a matching closing brace '}'.")
    return ""

def validate_and_repair_json(text: str, company_name: str, query_type: str) -> dict:
    """
    Extracts the JSON payload from raw LLM output, validates fields,
    and returns a structured dict conforming to the schema.
    """
    print(f"\n--- [JSON Validator] Normalizing response (total length {len(text)} chars) ---")
    
    # 1. Clean markdown fences or wrappers
    cleaned = text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = cleaned.strip()
    
    # 2. Extract matching brace block
    json_str = extract_json_object(cleaned)
    if not json_str:
        print("[JSON Validator] ⚠️ No JSON object braces detected in cleaned text. Falling back to raw text.")
        json_str = cleaned

    required_keys = [
        "company", "query_type", "overall_outlook", "confidence",
        "technical_analysis", "news_analysis", "risk_analysis",
        "recommendation", "portfolio_allocation", "supporting_evidence",
        "disclaimer", "bullishness_score", "expected_return_range",
        "top_positive_factors", "top_negative_factors", "decision_explanation",
        "portfolio_projection"
    ]
    
    try:
        # Pre-process escaping for any common raw control characters inside text strings
        print("[JSON Validator] Parsing JSON string...")
        processed_str = re.sub(r'\n(?=[^"]*"[^"]*(?:"[^"]*"[^"]*)*$)', '\\n', json_str)
        data = json.loads(processed_str)
        
        # Ensure all required keys exist
        missing_keys = []
        for key in required_keys:
            if key not in data:
                missing_keys.append(key)
                data[key] = "Not specified" if key != "supporting_evidence" else []
                
        if missing_keys:
            print(f"[JSON Validator] ⚠️ Missing keys padded with defaults: {missing_keys}")
                
        print("[JSON Validator] ✓ JSON validation/repair succeeded.")
        print("--- [JSON Validator] Completed ---\n")
        return data
        
    except Exception as e:
        print(f"[JSON Validator] ✗ JSON validation failed: {e}")
        print(f"[JSON Validator] Raw extracted JSON preview: {json_str[:300]}...")
        
        # Emergency fallback repair
        print("[JSON Validator] Applying emergency fallback default dictionary.")
        fallback = {
            "company": company_name,
            "query_type": query_type,
            "overall_outlook": "Neutral (LLM format error)",
            "confidence": "Low (parsing failure)",
            "technical_analysis": "Unavailable due to formatting issues.",
            "news_analysis": "Unavailable due to formatting issues.",
            "risk_analysis": "Formatting error encountered.",
            "recommendation": f"Raw AI reasoning content: {text[:500]}...",
            "portfolio_allocation": "Unavailable",
            "supporting_evidence": ["Raw reasoning output retrieved"],
            "disclaimer": "AI response was auto-repaired due to structural JSON errors.",
            "bullishness_score": 50,
            "expected_return_range": "N/A",
            "top_positive_factors": [],
            "top_negative_factors": [],
            "decision_explanation": "Unavailable due to parsing error.",
            "portfolio_projection": {}
        }
        print("--- [JSON Validator] Completed with Errors ---\n")
        return fallback
