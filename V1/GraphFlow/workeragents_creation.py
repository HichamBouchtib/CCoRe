from agents.workeragent import WorkerAgent
from llm import llm
from langchain_core.messages import SystemMessage
from tools.mcp_tool_manager import get_mcp_tool
from tools.generate_tool import generate_tool_function
from V1.state import State

def instantiate_workeragents(state: State):
    """
    Instantiate a dictionary of WorkerAgent instances based on TaskGraph response placeholders.
    If no MCP tool is found, the LLM generates a new Python function for the WorkerAgent.

    :param state: The current state containing the TaskGraph response with placeholders.
    :return: A dictionary of WorkerAgent instances.
    """

    Worker_agents = {}
    
    tg_response = state.tg_response
    if not tg_response:
        raise ValueError("No TaskGraph response found in state.")

    # Structured instruction for LLM to extract WorkerAgent details and generate tools if necessary
    worker_instanciate_instructions = f"""
    You are an AI expert in task decomposition and agent creation.
    
    **TaskGraph Response:**  
    {tg_response}

    **Objective:**  
    Identify all placeholders (`[ ]`) in the response and extract the following details for each WorkerAgent:
    - **Name**: Unique identifier for the agent.
    - **Task Description**: A concise but detailed description of the subtask.
    - **Tool**: The function that executes the subtask. Use the MCP (Model Context Protocol) if available.
      - If an MCP tool is available, select it.
      - If no MCP tool is found, generate a **Python function** to execute the task.

    **Expected JSON Output Format:**
    ```json
    {{
        "worker_agents": {{
            "UniqueAgentName1": {{
                "task": "Task description here...",
                "tool": "MCP_TOOL_NAME or function definition"
            }},
            "UniqueAgentName2": {{
                "task": "Another task...",
                "tool": "MCP_TOOL_NAME or function definition"
            }}
        }}
    }}
    ```
    """

    # Invoke LLM to process the TaskGraph response and generate WorkerAgent details
    system_message = SystemMessage(content=worker_instanciate_instructions)
    response = llm.invoke([system_message])

    try:
        worker_data = response.content if hasattr(response, 'content') else str(response)
        worker_agents_dict = eval(worker_data)["worker_agents"]  # Convert JSON-like string to dictionary
    except Exception as e:
        raise ValueError(f"Failed to parse WorkerAgents response: {str(e)}")

    # Instantiate WorkerAgents
    for unique_name, details in worker_agents_dict.items():
        tool_name = details["tool"]

        if "MCP_" not in tool_name:  # If no MCP tool is found, generate a Python function
            tool_function = generate_tool_function(tool_name, details["task"])
        else:
            tool_function = get_mcp_tool(tool_name)  # Use MCP tool directly

        Worker_agents[unique_name] = WorkerAgent(
            name=unique_name,
            task=details["task"],
            tool=tool_function  # Either an MCP tool name or a generated function
        )

    return {"Worker_agents": Worker_agents}
