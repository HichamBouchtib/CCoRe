import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm import llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from state import State
import sys
import logging

agentmode_prompt = """You are {name}, an expert in {expertise}.
Your task is to check if the user question is easy and simple and within your expertise.
If yes, answer it confidently in 2-3 sentences, output it.
If not, output 'None'
User question: {query}
"""

def agent_mode(state: State) -> dict:
    # print(state)
    # logging.warning("🔥 agent_mode triggered")

    state.setdefault("messages", [])
    query = state.get("query", "").strip()
    
    
    # state["messages"].append(HumanMessage(content=query))
    state["response"] = None

    agents = state.get("wiseragents", [])
    for agent in agents:
        system_prompt = agentmode_prompt.format(
            name=agent.name,
            expertise=agent.domain_expertise,
            query=query
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Please answer the question if you can, if not respond with None.")
        ]
        try:
            answer = llm.invoke(messages)
            AgentAnswer = answer.content.strip()
            if AgentAnswer.lower() != "none":
                state["response"] = AgentAnswer
                state["messages"].append(AIMessage(content="Routing to Single Agent Mode..."))
                break
        except Exception as e:
            print(f"❌ LLM invoke error: {e}")

    if state["response"] is None:
        state["messages"].append(AIMessage(content="Routing to Multi Agent Mode..."))

    # print(f"✅ Final response: {state['response']}")
    return state
