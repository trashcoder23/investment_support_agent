from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes import (
    intent_parsing_node,
    structured_context_node,
    rag_context_node,
    llm_reasoning_node
)

def create_agent_graph() -> StateGraph:
    """
    Assembles and compiles the StateGraph workflow.
    """
    workflow = StateGraph(AgentState)
    
    # Register Nodes
    workflow.add_node("intent_parsing", intent_parsing_node)
    workflow.add_node("structured_context", structured_context_node)
    workflow.add_node("rag_context", rag_context_node)
    workflow.add_node("llm_reasoning", llm_reasoning_node)
    
    # Configure Connections
    workflow.add_edge(START, "intent_parsing")
    workflow.add_edge("intent_parsing", "structured_context")
    workflow.add_edge("structured_context", "rag_context")
    workflow.add_edge("rag_context", "llm_reasoning")
    workflow.add_edge("llm_reasoning", END)
    
    # Compile
    return workflow.compile()

# Single-point run method for backward compatibility/simplicity
def run_agent(query: str, chat_history: list = None) -> dict:
    """
    Runs thecompiled StateGraph workflow for a given user query.
    """
    app = create_agent_graph()
    
    initial_state = {
        "query": query,
        "intent": {},
        "structured_context": {},
        "rag_context": "",
        "response": {},
        "logs": [],
        "telemetry": {}
    }
    
    # Run the graph
    final_state = app.invoke(initial_state)
    return final_state
