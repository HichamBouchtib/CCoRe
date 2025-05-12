import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from llm import llm
from langchain_core.messages import SystemMessage
from state import State
from interview.Interview import InterviewEntry
from interview.Interview import Interview
from langchain_core.messages import AIMessage


tg_interview_instructions = """You are a technical reviewer WiserAgent with expertise in {expertise}. 
You are reviewing a Task Graph (TG) created to solve the following query:
"{query}"

Here is the TG:
{tg_details}

Here are the questions already asked about this TG:
{already_asked}

Your task is to:
- Suggest **a new and non-redundant** technical question, if something is **missing or unclear**.
- If all the main aspects are covered, simply respond: "No question, Thanks"

Focus technicaly only on:
- Logical dependencies
- Missing or unnecessary tasks
- Incomplete specifications

Do not add greetings or generic commentary. Return only the final question (or the sentence above).
"""


def generate_questions(state: State):
    print("- Deep Monologue: Started")

    tgs = state["tg_candidates"]
    agents = state["wiseragents"]
    query = state["query"]
    interview_obj = Interview()

    for tg in tgs:
        owner_agent = tg.owner_agent
        tg_text = json.dumps(tg.to_json(), indent=2)
        entry = InterviewEntry(TG_Owner=owner_agent, task_graph=tg)

        # print(owner_agent.name, ": Do you have any questions for me?")

        asked_questions = []

        for agent in agents:
            if agent.name == owner_agent.name:
                continue  # Skip self-questioning

            # Join previous questions to give context
            previous_qst_block = "\n".join([f"- {q}" for q in asked_questions]) or "None"

            system_message = tg_interview_instructions.format(
                query=query,
                expertise=agent.domain_expertise,
                tg_details=tg_text,
                already_asked=previous_qst_block
            )

            question_msg = llm.invoke([SystemMessage(content=system_message)])
            question_content = question_msg.content if hasattr(question_msg, 'content') else str(question_msg)

            # Add only if it's not a duplicate
            if question_content not in asked_questions:
                asked_questions.append(question_content)
                entry.add_question(from_agent=agent, question=question_content)
                # print(agent.name, ":", question_content)
            else:
                entry.add_question(from_agent=agent, question="No question, Thanks")
                # print(agent.name, ":", "No question, Thanks")
        interview_obj.add_entry(entry)
    interview_obj.save_qsts_to_file()
    # display_interview_cards([interview_obj])
    state["messages"].append(AIMessage(content="DeepMonologue Started..."))

    return {"interview": [interview_obj]}

