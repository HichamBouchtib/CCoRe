from typing import Dict, List
from agents.workeragent import WorkerAgent
from agents.masteragent import MasterAgent
from agents.wiseragent import WiserAgent
from langgraph.graph import MessagesState
from typing import  Annotated
import operator

class State(MessagesState):
    topic: str
    query: str
    human_wiseragent_feedback: str
    WS: int
    wiseragents: List[WiserAgent]
    master_agents: Dict[MasterAgent]
    Worker_agents: Dict[str, WorkerAgent]
    max_num_turns: int # Number turns of conversation allowed            
    interview: str  # transcript of the interview
    questions: List # List of questions of interviewers
    context: Annotated[list, operator.add] # Source docs
    tg_candidates: dict
    tg_chosen: dict
    tg_response: str # the aggregated placeholder-styled response
