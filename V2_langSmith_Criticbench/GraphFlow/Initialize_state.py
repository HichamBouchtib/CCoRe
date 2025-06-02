from langchain_core.messages import AIMessage, HumanMessage
from state import State


# def initialize_state(state: dict) -> dict:
def initialize_state(state: State):
    messages = state.get("messages", [])

    if messages:
        last = messages[-1]

        if isinstance(last, HumanMessage):
            cleaned = last.content.strip()

            # Case 1: If meaningless input, reply with greeting and halt
            if cleaned in ["", ".", "?", "start"]:
                state.setdefault("messages", []).append(
                    AIMessage(content="💬 Hello! What’s your topic?")
                )
                print("✅ returning with __end__ from initialize_state")
                return {
                    **state,
                    "__end__": True
                }

            # Case 2: If meaningful input, store it in topic
            else:
                state["topic"] = cleaned
                print("➡️ valid topic set:", cleaned)

    return state  # proceed to next step
