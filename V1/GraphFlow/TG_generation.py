# TG generation

from TG.task_graph import TaskGraph
from V1.state import State
from llm import llm
from langchain_core.messages import SystemMessage

# Updated TG generation instructions for each WiserAgent
task_graph_instructions = """You are a WiserAgent tasked with generating a Task Graph (TG) to precisely solve the user's question.
Your goal is to design a clear, actionable plan that includes:
1. **Masteragents**:
   - A list of **masteragents** involved in the task graph. Each agent corresponds to a distinct role or task that will be executed as part of the overall plan 
   Example: ["Port Scanner", "Logfile Analyzer", "Vulnerability Protector"]).

3. **Tasks**:
   - A dictionary where each key corresponds to a **masteragent** and the value is a description of the task it performs. 
   Example:
     - `"Port Scanner": "Scan the network for open ports using predefined tools."`
     - `"Logfile Analyzer": "Analyze the logs for any suspicious activities and detect potential threats."`
     - `"Vulnerability Protector": "Mitigate detected vulnerabilities by applying security patches or blocking malicious activities."`

4. **Orders**:
   - A list of transitions between tasks, where each transition is represented by an object with three properties:
     - `from`: The task or agent where the transition originates.
     - `to`: The task or agent where the transition leads.
     - `condition`: The condition under which the transition occurs. This can specify the result of the task (e.g., "success", "failure", "threats_found", etc.).
   Example:
    `{"from": "Port Scanner", "to": "Logfile Analyzer", "condition": "success"}`
    `{"from": "Port Scanner", "to": "Vulnerability Protector", "condition": "failure"}`
    `{"from": "Logfile Analyzer", "to": "Vulnerability Protector", "condition": "threats_found"}`
    `{"from": "Logfile Analyzer", "to": "End", "condition": "no_threats"}`
    `{"from": "Vulnerability Protector", "to": "End", "condition": "resolved"}`
    
3. The TG structure must reflect the task dependencies:
   - Use a sequential structure if tasks must occur one after another.
   - Use a parallel structure if tasks can run concurrently.
   - Use a hierarchical or reactive structure if tasks depend on dynamic adjustments.
4. Collaborate with fellow WiserAgents:
   - If you believe you can best generate the TG, propose it.
   - Alternatively, act as a validator or coordinator to help refine the TG.
Here is the user query to consider:
User Query: {query}
Generate the Task Graph accordingly.
"""

def generate_task_graphs(state: State):
    query = state["query"]
    tg_candidates = []
    
    # Each WiserAgent can decide to generate a TG proposal.
    for agent in state["wiseragents"]:
        
        system_message = task_graph_instructions.format(query=query)
        
        tg_output = llm.invoke([SystemMessage(content=system_message)])
        
        task_graph = TaskGraph.from_json(tg_output)
        
        tg_candidates.append({"agent": agent.name, "task_graph": task_graph})
    
    return {"tg_candidates": tg_candidates}
