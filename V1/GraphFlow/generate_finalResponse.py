from llm import llm
from langchain_core.messages import SystemMessage
from V1.state import State

def generate_final_response(state: State):
    """
    Aggregates all WorkerAgent responses and formulates the final answer using the LLM.
    
    :param state: The WorkerAgentState containing responses from different WorkerAgents.
    :return: A coherent, structured final response.
    """
    
    worker_agents = state["worker_agents"]
    query = state["query"]

    if not worker_agents:
        raise ValueError("No WorkerAgents found in state.")

    # Collect responses from WorkerAgents
    agent_responses = []
    for agent_name, agent in worker_agents.items():
        if hasattr(agent, "execute_subtask") and callable(agent.execute_subtask):
            # Execute subtask and store response
            agent_response = agent.execute_subtask()
            agent.response = agent_response  # Store response in the agent
            agent_responses.append(f"{agent_name}: {agent_response}")

    if not agent_responses:
        raise ValueError("No valid responses collected from WorkerAgents.")

    # Construct the aggregation prompt using query and responses
    aggregation_prompt = f"""
    You are an AI assistant responsible for compiling and structuring responses from multiple WorkerAgents
    to provide a final answer to the following query:

    **Query:** {query}

    Below are the responses from the WorkerAgents:
    
    {chr(10).join(agent_responses)}

    ### Instructions:
    - Ensure the responses are well-structured and coherent.
    - Merge relevant responses into a single, accurate answer.
    - If there are dependencies or missing values, highlight them appropriately.
    - Maintain a clear and professional tone.

    Generate the final response accordingly.
    """

    # Invoke LLM to generate the final response
    response = llm.invoke([SystemMessage(content=aggregation_prompt)])

    return response.content
