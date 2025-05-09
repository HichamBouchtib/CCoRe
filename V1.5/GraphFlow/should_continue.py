from state import State

def should_continue(state):
    if state.get("human_wiseragent_feedback") and not state.get("feedback_handled"):
        return "create_wiseragents"
    else:
        return "generate_TG"