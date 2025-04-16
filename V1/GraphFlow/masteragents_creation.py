from agents.masteragent import MasterAgent
from V1.state import State

def instantiate_masteragents(state: State):
    """
    Converts multiple TaskGraphs from tg_candidates into a dictionary of MasterAgent instances.

    :param state: The current state containing multiple TaskGraph candidates.
    :return: A dictionary of MasterAgent instances.
    """

    master_agents = {}

    tg_candidates = state.tg_candidates
    if not tg_candidates:
        raise ValueError("No TaskGraph candidates found in state.")

    for tg_candidate in tg_candidates:
        task_graph = tg_candidate.get("task_graph")
        if not task_graph:
            continue  # Skip if no valid task graph

        masteragents = task_graph.get("masteragents", [])
        tasks = task_graph.get("tasks", {})

        # Iterate over each master agent in the task graph
        for masteragent_name in masteragents:
            # Get the task description from the 'tasks' section
            task_description = tasks.get(masteragent_name, "No description available")

            # Ensure unique MasterAgent names
            unique_name = masteragent_name
            counter = 1
            while unique_name in master_agents:
                unique_name = f"{masteragent_name}_{counter}"
                counter += 1

            # Instantiate MasterAgent with appropriate values
            master_agents[unique_name] = MasterAgent(
                name=unique_name,
                task=task_description,
                preferred_llm="llama3.2",  # Default preferred LLM
                WS=20  # Default Wisdom Score
            )

    return {"master_agents": master_agents}
