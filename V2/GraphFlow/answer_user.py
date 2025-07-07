import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from state import State, WiserAgent
from langchain_core.messages import SystemMessage
from llm import llm


answer_instructions = """
You are an advanced AI Agent.

A Task Graph (TG) has been selected to solve the following user query:
"{query}"

You must execute the TG step by step to produce a direct and informative answer to the query.

TG:
{tg_chosen}

Do not include any introductions, confirmations, or conversational phrases.
Start directly with the solution steps and provide the final answer at the end.
"""

def answer_user(state: State):
    query = state["query"]
    agents = state["wiseragents"]
    tg_chosen = state.get("tg_chosen", None)

    # Case 1: Already answered earlier in the graph
    if state["response"] is not None:
    # if state["response"] is not None and not tg_chosen:
        print(f"The answer from SA mode: {state['response']}")

    # Case 2: a TG was chosen
    if tg_chosen:
        # WS update
        chosen_agent = tg_chosen.owner_agent
        submitted_agents = [
            tg.owner_agent for tg in state.get("tg_candidates", [])
            if tg.owner_agent.name != chosen_agent.name
        ]

        updated_agents = []
        for agent in agents:
            if agent.name == chosen_agent.name:
                updated_agents.append(WiserAgent(
                    name=agent.name,
                    domain_expertise=agent.domain_expertise,
                    description=agent.description,
                    WS=agent.WS + 1,
                    preferred_llm=agent.preferred_llm
                ))
            elif any(agent.name == submitted.name for submitted in submitted_agents):
                updated_agents.append(WiserAgent(
                    name=agent.name,
                    domain_expertise=agent.domain_expertise,
                    description=agent.description,
                    WS=agent.WS - 1,
                    preferred_llm=agent.preferred_llm
                ))
            else:
                updated_agents.append(agent)

        state["wiseragents"] = updated_agents

        print("\nUpdated WS: ")
        for agent in updated_agents:
            print(f"{agent.name} → WS = {agent.WS}")

        # TG results
        message = answer_instructions.format(
                query=query,
                tg_chosen=tg_chosen.to_json_string()
            )
        system_message = SystemMessage(content=message)
        try:
            result = llm.invoke([system_message])
            state["response"] = result.content
            print("-----------------------------------------------------------------------------------------------")
            print(f"The answer from MA mode: {state['response']}")
        except Exception as e:
            print(f"\nError executing TG: {e}")

    return state