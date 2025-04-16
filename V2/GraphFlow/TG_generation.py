# TG generation
import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TG.task_graph import TaskGraph, TGList
from agents.wiseragent import WiserAgent
from state import State
from llm import llm
from langchain_core.messages import SystemMessage, HumanMessage


# Updated TG generation instructions for each WiserAgent
task_graph_instructions = """You are a WiserAgent called {agent} tasked with generating a Task Graph (TG) to precisely solve the user's question.
Your goal is to design a clear, actionable plan that includes:
1. **Agent**: The owner of the TG, which is you. This owner_agent field must be a full structured dictionary corresponding to the WiserAgent who is creating the TG.
1. **Tasks**: A dictionary where each key corresponds to a key task and the value is a description of the task.
2. **Orders**: A list of transitions between tasks, where each transition is represented by an object with three properties:
     - `from`: The task or agent where the transition originates.
     - `to`: The task or agent where the transition leads.
     - `condition`: The condition under which the transition occurs. This can specify the result of the task (e.g., "success", "failure", "threats_found", etc.).
Example:
{{
  "owner_agent": {{
    "name": "Malware Analyst",
    "domain_expertise": "AI-Driven Malware Analysis",
    "description": "Specializes in analyzing malware using advanced AI techniques to identify new threats and vulnerabilities.",
    "WS": 50,
    "preferred_llm": "qwen2.5:latest"
    }}
  "tasks": {{
    "Port Scanning": "Scanning the network for open ports using predefined tools.",
    "Logfile Analyzing": "Analyzing the logs for any suspicious activities and detect potential threats.",
    "Vulnerability Protectoring": "Mitigating detected vulnerabilities by applying security patches or blocking malicious activities."}},
  "orders": [
    `{{"from": "Port Scanning", "to": "Logfile Analyzing", "condition": "success"}}`
    `{{"from": "Port Scanning", "to": "Vulnerability Protectoring", "condition": "failure"}}`
    `{{"from": "Logfile Analyzing", "to": "Vulnerability Protectoring", "condition": "threats_found"}}`
    `{{"from": "Logfile Analyzing", "to": "End", "condition": "no_threats"}}`
    `{{"from": "Vulnerability Protectoring", "to": "End", "condition": "resolved"}}`
    ]
}}
3. The TG structure must reflect the task dependencies:
   - Use a sequential structure if tasks must occur one after another.
   - Use a parallel structure if tasks can run concurrently.
   - Use a hierarchical or reactive structure if tasks depend on dynamic adjustments.
4. Collaborate with fellow WiserAgents:
   - If you believe you can best generate the TG, propose it.
   - Alternatively, act as a validator or coordinator to help refine the TG.
Here is the user query to consider:
User Query: **{query}**
Generate the Task Graph accordingly.
"""

def generate_task_graphs(state: State):
    query = input("Now whats your query ?: ")
    state['query'] = query
    tg_candidates = []

    for agent in state["wiseragents"]:
        system_message = task_graph_instructions.format(query=query, agent=agent.name)

        message = [
            SystemMessage(content=system_message),
            HumanMessage(content="Generate the most appropriate task graph.")
        ]

        # Unpack if agent is a tuple (common pattern from LangGraph outputs)
        if isinstance(agent, tuple):
            agent = agent[0]
        try:
            # Get dict output from LLM
            structured_llm = llm.with_structured_output(TaskGraph)
            tg = structured_llm.invoke(message)
            tg.save_to_file()
            tg_candidates.append(tg)

        except Exception as e:
            print(f"Error processing TG from {agent.name}: {e}")
            continue       
    return {"tg_candidates": tg_candidates}


# Mock test state based on working graph execution output
# test_state = State(**{
#     "query": "How can AI be used to prevent and mitigate cyberattacks?",
#     "wiseragents": [
#         WiserAgent(
#             name="Malware Analyst",
#             domain_expertise="AI-Driven Malware Analysis",
#             description="Specializes in analyzing malware using advanced AI techniques to identify new threats and vulnerabilities.",
#             WS=50,
#             preferred_llm="qwen2.5:latest"
#         ),
#         WiserAgent(
#             name="Vulnerability Scout",
#             domain_expertise="Automated Vulnerability Detection Using AI",
#             description="Focuses on automating the process of detecting software vulnerabilities through AI-driven methods.",
#             WS=50,
#             preferred_llm="qwen2.5:latest"
#         ),
#         WiserAgent(
#             name="Intrusion Sentinel",
#             domain_expertise="AI-Powered Intrusion Detection Systems (IDS)",
#             description="Develops and deploys AI-based intrusion detection systems to protect networks from cyber attacks.",
#             WS=50,
#             preferred_llm="qwen2.5:latest"
#         ),
#         WiserAgent(
#             name="Phish Defender",
#             domain_expertise="AI in Phishing Attacks and Countermeasures",
#             description="Specializes in detecting phishing attempts using AI, providing real-time protection against social engineering attacks.",
#             WS=50,
#             preferred_llm="qwen2.5:latest"
#         )
#     ]
# })

# Run the function
# result = generate_task_graphs(test_state)

# print("\nTask Graph Candidates:")
# for tg in result["tg_candidates"]:
#     agent = tg.owner_agent.name  # Or just `tg.owner_agent` if it's a string
#     tg_dict = {
#         "owner_agent": agent,
#         "tasks": tg.tasks,
#         "orders": tg.orders
#     }

#     print(f"\nProposed by: {agent}")
#     print(json.dumps(tg_dict, indent=2))
#     print("-" * 40)



