from langchain_core.messages import RemoveMessage
from state import State

# decreasing the token usage in the long running conversations
def filter_messages(state: State): 
    # Delete all but the 10 most recent messages
    interview_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-10]]
    return {"interview": interview_messages}
    # messages = trim_messages(
    #         state["messages"],
    #         max_tokens=100,
    #         strategy="last",
    #         token_counter=ChatOpenAI(model="gpt-4o"),
    #         allow_partial=False,
    #     )