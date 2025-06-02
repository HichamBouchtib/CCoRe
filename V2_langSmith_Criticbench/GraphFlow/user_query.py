import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from state import State
from langchain_core.messages import HumanMessage

# def user_query(state: dict) -> dict:
def user_query(state: State):

    """taking query from user"""
    
    state.setdefault("messages", [])
    
    query = state.get("query", "").strip()

    if not query:
        print("No query found")
        query = next(
            (m.content for m in state["messages"] if isinstance(m, HumanMessage)),
            "").strip()
        state["query"] = query
        print(f"Query received from messages: {query}")
    
    # else:
    #     # print(f"Query received...: {query}")
    #     state["messages"].append(HumanMessage(content=query))
    # print("🧾 Messages in state:")
    # for m in state.get("messages", []):
    #     print(f"- {type(m)} → {getattr(m, 'content', '')}")
        
    return state