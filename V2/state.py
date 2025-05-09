from typing import List, Annotated, Optional
from agents.wiseragent import WiserAgent
from langgraph.graph import MessagesState
from interview.Interview import Interview 
from TG.task_graph import TaskGraph
import pprint
from tools.context import Context
from pydantic import BaseModel

# class State(BaseModel):
class State(MessagesState):
    topic: str
    last_topic: str = "" # for conversation persistence
    query: str
    human_wiseragent_feedback: str
    feedback_handled: bool = False
    WS: int
    wiseragents: List[WiserAgent]
    tg_candidates: List[TaskGraph]          
    interview: Annotated[Interview, "list"]
    max_num_turns: int # i'll use that later
    context: Context # retrieval
    tg_chosen: TaskGraph
    response: str = ""

    def print_state(self):
        """Pretty-print the current state."""
        print("\n📌 Current State Snapshot:")
        pprint.pprint(self.__dict__)

def get_current_state(topic: str, human_wiseragent_feedback: str = "") -> State:
    """Initialize and return the current state."""
    return State(
        topic=topic,
        last_topic="",
        query="",
        human_wiseragent_feedback=human_wiseragent_feedback,
        feedback_handled=False,
        WS=50,
        wiseragents=[],
        tg_candidates=[],
        # interview=[Interview()],
        interview=[],
        max_num_turns=3,
        context=Context(),
        tg_chosen=None,
        response=""
    )


