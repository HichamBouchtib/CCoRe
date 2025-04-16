from V1.state import State
from llm import llm

def execute_task_graph(state: State):
    """
    Executes the chosen TaskGraph using the corresponding MasterAgents and aggregates the responses.

    :param state: MasterAgentState containing instantiated master agents.
    :return: The aggregated Task Graph Resolution.
    """
    master_agents = {agent.name: agent for agent in state["master_agents"]}
    chosen_tg = state["chosen_tg"]
    agents = chosen_tg["agents"]
    orders = chosen_tg["orders"]

    task_outputs = {}

    # Iterate only over master agents that exist in the task graph
    for agent_name in master_agents.keys() & agents.keys():
        
        task_outputs[agent_name] = master_agents[agent_name].generate_response()

    # Aggregate the responses into a flow-based execution trace
    execution_trace = []
    for order in orders:
        src, dest, condition = order["from"], order["to"], order["condition"]
        if src in task_outputs:
            execution_trace.append(f"{src} executed: {task_outputs[src]}\n → Condition: {condition} → Next: {dest}")

    return {"tg_response": "\n".join(execution_trace)}


# example usage
# Port Scanner executed: Scan completed. Open ports found.
#  → Condition: success → Next: Logfile Analyzer

# Logfile Analyzer executed: Threats detected in logs.
#  → Condition: threats_found → Next: Vulnerability Protector

# Vulnerability Protector executed: Threats mitigated. System secured.
#  → Condition: resolved → Next: End
