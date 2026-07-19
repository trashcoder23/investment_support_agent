from agent.graph import run_agent as graph_run_agent

def run_agent(query: str, chat_history: list = None) -> dict:
    """
    Exposes the StateGraph runner.
    """
    return graph_run_agent(query, chat_history)
