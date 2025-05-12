import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from state import State
from langchain_core.messages import HumanMessage

def user_query(state: dict) -> dict:

    """taking query from user"""
    state.setdefault("messages", [])
    query = state.get("query", "").strip()

    if not query:
        print("No query found")
    else:
        print(f"Query received...: {query}")
        state["messages"].append(HumanMessage(content=query))

    return state