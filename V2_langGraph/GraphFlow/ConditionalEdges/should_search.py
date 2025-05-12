# route_search_vote.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from state import State

def should_search(state: State) -> str:
    context = state["context"]
    if context.awaiting_search and context.current_search:
        source = context.current_search.source.lower()
        if source == "web":
            return "search_web"
        elif source == "wikipedia":
            return "search_wikipedia"
        else:
            raise ValueError(f"Unknown search source: {source}")
    else:
        return "generate_answers"