from langchain_core.messages import get_buffer_string
from state import State

def save_interview(state: State):
    
    """ Save interviews """

    # Get messages
    interview = state["interview"]
    
    # Convert interview to a string
    interview = get_buffer_string(interview)
    
    # Save to interviews key
    return {"interview": interview}