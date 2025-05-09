import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from TG.task_graph import TaskGraph
from state import State
from llm import llm
from langchain_core.messages import SystemMessage, HumanMessage
from IPython.display import Image, display

task_graph_instructions = """You are a WiserAgent named {agent} tasked with generating a Task Graph (TG) to precisely solve the user's question below.
User Query: {query}

Always Return the structured output as a TaskGraph object, following this defined schema:
{{
  'owner_agent': {{ ... }},
  'tasks': {{ ... }},
  'orders': [ ... ],
  'refused': false/true,
  'answer': null
}}

## RULES: Self-Assessment:
- If you refuse to generate a taskgraph because it's not your speciality, set the 'refused' field to true.
- If you cannot directly solve it alone, generate a TaskGraph object with 'answer' field set to null and 'refused' to false:
this is a taskgraph example generated for a this query : "what AI agents should i use to help protect my website against cyber attacks such as phising":
{{
  'owner_agent': {{
    'name': 'Malware Analyst',
    'domain_expertise': 'AI-Driven Malware Analysis',
    'description': 'Specializes in analyzing malware using advanced AI techniques to identify new threats and vulnerabilities.',
    'WS': 50,
    'preferred_llm': 'qwen2.5:latest'
  }},
  'tasks': {{
    'Port Scanning': 'Scanning the network for open ports using predefined tools.',
    'Logfile Analyzing': 'Analyzing the logs for any suspicious activities and detect potential threats.',
    'Vulnerability Protectoring': 'Mitigating detected vulnerabilities by applying security patches or blocking malicious activities.'
  }},
  'orders': [
    {{'from': 'Port Scanning', 'to': 'Logfile Analyzing', 'condition': 'success'}},
    {{'from': 'Port Scanning', 'to': 'Vulnerability Protectoring', 'condition': 'failure'}},
    {{'from': 'Logfile Analyzing', 'to': 'Vulnerability Protectoring', 'condition': 'threats_found'}},
    {{'from': 'Logfile Analyzing', 'to': 'End', 'condition': 'no_threats'}},
    {{'from': 'Vulnerability Protectoring', 'to': 'End', 'condition': 'resolved'}}
  ],
  'refused': false,
  'answer': null
}}
### Task Graph STRUCTURE GUIDELINES:
- Use sequential dependencies if tasks must happen in order.
- Use parallel structure if tasks can be run concurrently.
- Use conditional transitions to reflect dynamic or reactive flows.
- Your Wisdom Score (WS) will decrease if you generate a task graph that is not optimal or not well-structured and will still the same if you choose to not generate, it will increase if your TG is the chosen one by the end of the execution.
"""

def generate_task_graphs(state: State):
    print("Generating Task Graph for the user query...")
    query = state['query']
    agents = state["wiseragents"]
    tg_candidates = []
    
    for agent in agents:
        system_message = task_graph_instructions.format(query=query, agent=agent.name)

        message = [
            SystemMessage(content=system_message),
            HumanMessage(content="Generate the most appropriate task graph.")
        ]

        if isinstance(agent, tuple):
            agent = agent[0]

        try:
            structured_llm = llm.with_structured_output(TaskGraph)
            tg = structured_llm.invoke(message)
            
            if tg is None:
                raise ValueError("Structured output was None. LLM failed to produce a valid TaskGraph object.")

            if tg.refused:
                print(f"Agent {agent.name} refused to generate TG.")
                continue
            # tg.pretty_print()
            # visualize the task graph
            tg.visualize_with_graphviz(output_path=f"TG/saved_TGs/visuals/TG_{agent.name}", view=False)
            img_path = f"TG/saved_TGs/visuals/TG_{agent.name}.png"
            display(Image(filename=img_path, width=300, height=400))

            tg.save_to_file()
            tg_candidates.append(tg)

        except Exception as e:
            print(f"Error processing {agent.name}: {e}")
            continue
    print("Task Graphs generated.\n")

    return State(**{
        **state,
        "tg_candidates": tg_candidates
    })

# test_state = State(**{
#     "query": "",
#     "wiseragents": [
#         WiserAgent(
#             name="Malware Analyst",
#             domain_expertise="AI-Driven Malware Analysis",
#             description="Specializes in analyzing malware using advanced AI techniques to identify new threats and vulnerabilities.",
#             WS=50,
#             preferred_llm="qwen2.5:latest"
#         )
#     ]
# })

# # Run the TG generation test
# result = generate_task_graphs(test_state)