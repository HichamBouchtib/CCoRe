from state import State
from langchain_core.messages import HumanMessage, AIMessage
import json

def human_feedback(state: dict) -> dict:
    state.setdefault("messages", [])
    feedback = state.get("human_wiseragent_feedback", "").strip()

    if feedback:
        print("✅ Feedback received. Will regenerate.")
        return {**state, "feedback_handled": False}

    # Show "No feedback" in chat
    state["messages"] = [
        m for m in state["messages"]
        if not (isinstance(m, HumanMessage) and m.content == "No feedback")
    ]
    state["messages"].append(HumanMessage(content="No feedback"))
    print("🟡 No feedback. Proceed to agent_mode.")

    return {**state, "human_wiseragent_feedback": "", "feedback_handled": True}