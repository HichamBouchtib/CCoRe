import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from state import State

def agent_mode_router(state: State) -> State:

    """This node does nothing — it's only used to allow conditional routing after logic execution"""
    print("📍 agent_mode_router called, preparing to route...")
    return state
