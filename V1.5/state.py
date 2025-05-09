from typing import Dict, List, Tuple
from agents.wiseragent import WiserAgent
from langgraph.graph import MessagesState
from typing import  Annotated
import operator
from interview.Interview import Interview 
from TG.task_graph import TaskGraph


class State(MessagesState):
    topic: str
    query: str
    human_wiseragent_feedback: str
    feedback_handled: bool = False # Flag to check if feedback is handled
    WS: int # Wisdom Score
    wiseragents: List[WiserAgent]
    max_num_turns: int # Number turns of conversation allowed            
    interview: Interview 
    questions: List # List of questions of interviewers
    context: Annotated[list, operator.add] # Source docs
    tg_candidates = List[Tuple[WiserAgent, TaskGraph]]
    tg_chosen: dict = {}
    tg_response: str = "" # the aggregated placeholder-styled response

# Step 2: Define a function to instantiate and return the current state
def get_current_state() -> State:
    # Ask the user to input the topic dynamically
    topic = input("Please enter the topic: ")

    # You can populate other fields dynamically or load from a configuration file
    return State(
        topic=topic,
        query="",  # Start with an empty query
        human_wiseragent_feedback="",  # Initially empty feedback
        feedback_handled=False,
        WS=50,  # Example wisdom score
        wiseragents=[],  # Start with an empty list of wiseragents
        max_num_turns=5,  # Set the max number of turns
        interview=Interview(),  
        context=[],  # Initialize with empty context
        tg_candidates=[],  # Empty list of tg candidates
        tg_chosen={},  # Empty dict for tg choices
        tg_response=""  # Empty response
    )


