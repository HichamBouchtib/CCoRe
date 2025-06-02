from typing import List, Annotated, Optional
from agents.wiseragent import WiserAgent
from langgraph.graph import MessagesState
from interview.Interview import Interview 
from TG.task_graph import TaskGraph
import pprint
from tools.context import Context

class State(MessagesState):
    topic: Optional[str]
    query: str
    human_wiseragent_feedback: str
    feedback_handled: bool = False
    WS: int
    wiseragents: List[WiserAgent]
    tg_candidates: List[TaskGraph]          
    # interview: Annotated[Interview, "list"]
    interview: Interview
    max_num_turns: int # i'll use that later
    context: Context # retrieval
    tg_chosen: TaskGraph
    g_response: str = ""
    q_response: str = ""
    c_response: str = ""
    subfolder: str = ""

    
    def print_state(self):
        """Pretty-print the current state."""
        print("\n Current State Snapshot:")
        pprint.pprint(self.__dict__)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

