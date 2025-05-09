from state import State

def human_feedback(state: State) -> dict:
    print(">> Awaiting human feedback on WiserAgent output...")
    human_feedback = input("Enter your human feedback (if any, else press Enter): ")

    if human_feedback.strip():
        print(">> Feedback received. Passing to create_wiseragents.")
        return {
            "human_wiseragent_feedback": human_feedback,
            "feedback_handled": False
        }
    else:
        print(">> No feedback entered. Proceeding to generate_TG.")
        return {
            "human_wiseragent_feedback": None,
            "feedback_handled": True
        }
