import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'GraphFlow')))
from llm import llm
from langchain_core.messages import SystemMessage
from state import State
from interview.Interview import InterviewEntry
from interview.Interview import Interview
from langchain_core.messages import AIMessage
from wiseragents_creation  import append_or_update_AImessage2

# tg_interview_instructions = """You are a technical reviewer WiserAgent with expertise in {expertise}. 
# You are reviewing a Task Graph (TG) created to be an executable plan to solve the following specificaly the this user query:
# "{query}"

# Here is the TG:
# {tg_details}

# Here are the questions already asked about this TG:
# {already_asked}

# Your task is to:
# - Suggest a new and non-redundant technical question, if something is missing or unclear.
# - If all the main aspects are covered, simply respond: "No question, Thanks"

# You can highlight:
# - Logical dependencies
# - Missing or unnecessary tasks
# - Incomplete specifications
# - etc...

# Do not add greetings or generic commentary. Return only the final question (or the sentence above).
# """

DeepmonologueQsts_instructions = """You are a technical reviewer WiserAgent with expertise in {expertise}. 
You are reviewing a Task Graph (TG) created to be an executable plan to solve the following specificaly the this user query:
"{query}"

Here is the TG details:
{tg_details}

Here are the questions already asked about this TG:
{already_asked}

Your task is to:
- Carefully analyze the TG and suggest a **new and non-redundant** technical question.
- Focus on **what can be clarified, improved, or challenged** in the TG.
- Only if you truly believe the TG is entirely complete, with no ambiguities, reply: "No question, Thanks"

You can point out:
- Logical errors or unclear dependencies
- Unnecessary or missing tasks
- Misalignment with the initial user query
- Anything that could be improved or clarified

Avoid greetings or generic commentary.
Return only your question (or the sentence above).
"""

def generate_questions(state: State):
    print("- Deep Monologue: Started")

    tgs = state["tg_candidates"]
    agents = state["wiseragents"]
    query = state["query"]
    interview = Interview()

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

            system_message = DeepmonologueQsts_instructions.format(
                query=query,
                expertise=agent.domain_expertise,
                tg_details=tg_text,
                already_asked=previous_qst_block
            )
            # print("system_message: ", system_message)
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
        interview.add_entry(entry)
        
    interview.save_qsts_to_file()
    

    append_or_update_AImessage2(state, "DeepMonologue Started...")

    # print("type of interview within the end of the generate_interv_qsts node :", type(interview))
    
    state["interview"] = [interview]

    # return {"interview": [interview]}
    return state