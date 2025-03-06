from typing_extensions import TypedDict
from typing import List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Import or define your MasterAgent node functions:
#   vote_on_tgs_node: gathers candidate TGs, votes on them
#   structure_output_node: transforms chosen TG into placeholder-styled output
from agents.master_agent import (
    vote_on_tgs_node,
    structure_output_node
)

# Import the two tool-based node functions from your single `tool.py`
# that uses TavilySearch and WikipediaLoader
from tools.tool import (
    search_web,
    search_wikipedia
)

class MasterState(TypedDict):
    """
    State that flows through the MasterAgent pipeline.
    """
    task_graphs: list      # The Task Graphs from WiserAgents
    chosen_tg: dict        # The TG selected by MasterAgent
    structured_plan: dict  # The final placeholder-based plan
    messages: list         # For searching or Wikipedia lookups
    context: list          # Search results (if needed)

def build_master_graph():
    """
    1) vote_on_tgs_node
    2) structure_output_node
    3) search_web
    4) search_wikipedia
    5) End
    """
    builder = StateGraph(MasterState)

    # MasterAgent nodes
    builder.add_node("vote_on_tgs", vote_on_tgs_node)
    builder.add_node("structure_output", structure_output_node)

    # Tool nodes
    builder.add_node("search_web", search_web)
    builder.add_node("search_wikipedia", search_wikipedia)

    # Edges
    builder.add_edge(START, "vote_on_tgs")
    builder.add_edge("vote_on_tgs", "structure_output")
    builder.add_edge("structure_output", "search_web")
    builder.add_edge("search_web", "search_wikipedia")
    builder.add_edge("search_wikipedia", END)

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph
