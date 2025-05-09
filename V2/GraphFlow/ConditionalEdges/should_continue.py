import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from state import State

def should_continue(state: State):
    if state.get("human_wiseragent_feedback") and not state.get("feedback_handled"):
        return "create_wiseragents"
    else:
        return "agent_mode"