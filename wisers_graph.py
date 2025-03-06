from typing_extensions import TypedDict
from typing import List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Import WiserAgent node functions from wiser_agent.py
# e.g. create_wiser_agents, human_feedback_node, should_continue, etc.
from agents.wiser_agent import (
    create_wiser_agents,
    human_feedback_node,
    should_continue
)

def generate_task_graphs_node(state):
    """
    Placeholder logic: For each WiserAgent, generate a JSON-based Task Graph.
    """
    wiser_agents = state.get("wiser_agents", [])
    task_graphs = []

    for wa in wiser_agents:
        # e.g. call an LLM or the WiserAgent's method
        # For demo, we just create a dummy TG
        dummy_tg = {
            "agent_id": wa.get("agent_id", "wiser_undefined"),
            "plan": f"Plan for {wa.get('domain_expertise', 'NoDomain')}",
            "roles": ["Role A", "Role B"]
        }
        task_graphs.append(dummy_tg)

    state["task_graphs"] = task_graphs
    return {"task_graphs": task_graphs}


###############################################################################
# STATE DEFINITION for the Wiser graph
###############################################################################
class WiserState(TypedDict):
    """
    State that flows through the WiserAgent pipeline.
    """
    topic: str
    max_wiser_agents: int
    human_wiser_feedback: str
    wiser_agents: list     # List of WiserAgent profiles
    task_graphs: list      # List of Task Graphs produced by WiserAgents


###############################################################################
# BUILD THE WISER GRAPH
###############################################################################
def build_wiser_graph():
    """
    1) Creates WiserAgents
    2) (Interrupt for human feedback)
    3) Generate Task Graphs
    4) End
    """
    builder = StateGraph(WiserState)

    # Add nodes
    builder.add_node("create_wiser_agents", create_wiser_agents)
    builder.add_node("human_feedback", human_feedback_node)
    builder.add_node("generate_task_graphs", generate_task_graphs_node)

    # Edges
    builder.add_edge(START, "create_wiser_agents")
    builder.add_edge("create_wiser_agents", "human_feedback")
    builder.add_conditional_edges(
        "human_feedback",
        should_continue,
        ["create_wiser_agents", "generate_task_graphs"]
    )
    builder.add_edge("generate_task_graphs", END)

    # Compile
    memory = MemorySaver()
    graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)
    return graph
