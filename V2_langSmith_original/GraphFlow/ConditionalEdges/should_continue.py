import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from state import State

def should_continue(state: dict) -> str:
    feedback = state.get("human_wiseragent_feedback", "").strip()
    handled = state.get("feedback_handled", False)

    if feedback and not handled:
        print("🔁 Feedback needs handling → regenerate wiseragents")
        return "create_wiseragents"
    print("✅ Feedback handled or empty → go to agent_mode")
    return "user_query"