from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    query: str
    intent: Dict[str, Any]
    structured_context: Dict[str, Any]
    rag_context: str
    response: Dict[str, Any]
    logs: List[str]
    telemetry: Dict[str, float]
