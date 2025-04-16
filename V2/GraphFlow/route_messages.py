from langchain_core.messages import get_buffer_string
from langchain_core.messages import AIMessage
from state import State

def route_messages(state: State, name: str = "TGWiserAgent"):

    """ Route between question and answer """
    
    # Get messages
    interview = state["interview"]
    max_num_turns = state.get('max_num_turns',3)

    # Check the number of TGWiserAgent answers 
    num_responses = len(
        [m for m in interview if isinstance(m, AIMessage) and m.name == name]
    )

    # End if TGWiserAgent has answered more than the max turns
    if num_responses >= max_num_turns:
        return 'save_interview'

    # This router is run after each question - answer pair 
    # Get the last question asked to check if it signals the end of discussion
    last_question = interview[-2]
    
    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'
    return "ask_question"
