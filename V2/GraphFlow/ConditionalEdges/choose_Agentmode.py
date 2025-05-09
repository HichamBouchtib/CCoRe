import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from state import State

def choose_AgentMode(state: State) -> str:
    """Route based on whether agent answered directly or not."""
    response = state.get("response", None)
    if response and response.lower() != "none":
        print("Question requires no Task Graph planning")
        return "answer_user"
    else:
        print("Question requires some Task Graph planning")
        return "generate_TG"