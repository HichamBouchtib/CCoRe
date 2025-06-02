import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from langchain_core.messages import SystemMessage, AIMessage
from llm import llm
from state import State
from tools.context import Context

critique_prompt = """You are WiserAgent {TG_Owner}, the creator of this task graph (TG): {tg}. It was designed to solve the initial user query:

"{query}"

Several other WiserAgents reviewed your TG and asked questions. You already answered these questions during the interview:

{qa_pairs}

Now, your task is to deliver a final expert response to the initial query, using your TG as a reasoning framework and incorporating the insights and clarifications from the interview.

Do not restate the interview questions or the TG design. Do not make suggestions for improving the TG. Instead, focus on delivering a complete and confident answer to the original query, integrating all validated reasoning and clarifications.

Respond directly, as if presenting the final answer to the user."""


def Critique_response(state: State):
    query = state["query"]
    interviews = state["interview"]
    criticSuspect_response = []

    for interview in interviews:
        for entry in interview.entries:
            qa_blocks = []
            for q, a in zip(entry.Questions, entry.Answers):
                qa_blocks.append(f"Q ({q.from_.name}): {q.content}\nA ({a.to_.name}): {a.content}")

            qa_combined = "\n\n".join(qa_blocks)

            prompt = critique_prompt.format(
                TG_Owner=entry.TG_Owner.name,
                tg=entry.task_graph.to_json_string(),
                query=query,
                qa_pairs=qa_combined
            )

            response = llm.invoke([SystemMessage(content=prompt)])
            # print(f"Critique response from {entry.TG_Owner.name} TG:\n{response.content}\n")
            criticSuspect_response.append(response.content)
        # print("critique prompt: ", prompt)
            # print(f"suspect response from {entry.TG_Owner.name} TG:\n{response.content}\n")
    
    answer = llm.invoke([SystemMessage(content="evalaute these answers against each other and choose the best one then directly re-output it. dont add, delete nor change anything :\n" + "\n".join(criticSuspect_response))])
    
    q_response = answer.content
    
    state["messages"].append(AIMessage(content=f"(Q) response: {q_response}"))
    print(f"(Q) response: {q_response}")
    print("-----Critique response done-----")
    
    # return State(**{
    #     **state,
    #     "q_response": q_response
    # })

    state["q_response"] = q_response

    return state