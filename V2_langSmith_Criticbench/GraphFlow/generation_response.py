import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm import llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from state import State
import sys

agentmode_prompt = """You are {name}, an expert in {expertise}.

Given the following question, reason step by step based on your domain knowledge and provide a complete, final answer.

Do not ask follow-up questions.
Do not offer further help or optimizations.
Do not include polite closings like 'let me know...'.
Do not say 'hope this helps' or 'if needed'.

Respond with only the technical content required to fully solve the problem.

Question: {query}
"""

def Generation_response(state: State):
    """G response for criticbench dataset"""
    state.setdefault("messages", [])
    query = state.get("query", "").strip()
    g_response = None

    agents = state.get("wiseragents", [])
    for agent in agents:
        # print(agents)
        if g_response is None:
            system_prompt = agentmode_prompt.format(
                name=agent.name,
                expertise=agent.domain_expertise,
                query=query
            )
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content="Please answer the question step by step.")
            ]
            try:
                answer = llm.invoke(messages)
                AgentAnswer = answer.content.strip()
                if AgentAnswer.lower() == "none":
                    print(f"{agent.name} skipped (out of domain)")
                else:
                    print(f"{agent.name} answered")    
                    state["G_response"] = AgentAnswer
                    g_response = AgentAnswer
                    state["messages"].append(AIMessage(content=f"(G) response: {g_response}"))
                    break
            except Exception as e: 
                print(f"❌ LLM invoke error: {e}")

    print("(G) response :", g_response)
    state["g_response"] = g_response

    return state
    # return State(**{
    #     **state,
    #     "g_response": g_response,
    #     "query": query
    # })
