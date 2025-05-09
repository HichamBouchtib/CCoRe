from state import State

def human_feedback(state: State):
    if state["feedback_handled"]:
        print("✅ Skipping Human Feedback.")
        return {
            "human_wiseragent_feedback": "",
            "feedback_handled": True
        }
    feedback = state.get("human_wiseragent_feedback", "").strip()

    # If feedback already exists and hasn't been handled, prompt again
    if not feedback or not state.get("feedback_handled", True):
        print(">> Awaiting human feedback on WiserAgent output...")
        feedback = input("Enter your human feedback (if any, else press Enter): ").strip()

        if feedback:
            print(">> Feedback received. Looping back.")
            return {
                "human_wiseragent_feedback": feedback,
                "feedback_handled": False
            }
    
    # If no feedback is provided or it's already handled, proceed
    print(">> No feedback entered or already handled. Proceeding.")
    return {
        "human_wiseragent_feedback": "",
        "feedback_handled": True
    }


