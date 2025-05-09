import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm import llm
from langchain_core.messages import SystemMessage, HumanMessage
from state import State

agentmode_prompt = """You are {name}, an expert in {expertise}.
Your task is to check if the user question is easy and simple and within your expertise.
If yes, answer it confidently in 2-3 sentences, output it.
If not, output 'None'
User question: {query}
"""

def agent_mode(state: State):
    """Check if the agent can answer the query directly without TG."""
    query = input("Now what's your query?: ")
    # query = "How can I design a multi-layered cybersecurity defense for my website using a team of specialized AI agents, each responsible for different types of threats such as phishing, malware, bot attacks, and insider threats, while ensuring low false positives and scalable real-time protection?"
    print("Checking if question requires a Taskgraph...")
    state["response"] = None
    state["query"] = query
    agents = state["wiseragents"]
    for agent in agents:

        system_message = agentmode_prompt.format(name=agent.name, expertise=agent.domain_expertise, query=query)
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content="Please answer the question if you can, if not respond with None.")
        ]

        try:
            answer = llm.invoke(messages)
            AgentAnswer = answer.content.strip()
            if AgentAnswer and AgentAnswer.lower() != "none":
                state["response"] = AgentAnswer
                break
            else:
                state["response"] = None
        except Exception as e:
            print(f"Error in agent_mode: {e}")

    return state
