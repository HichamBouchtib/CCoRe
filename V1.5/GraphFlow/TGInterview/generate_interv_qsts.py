import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from llm import llm
from langchain_core.messages import SystemMessage
from state import State

# Interview instructions for TG validation (for multiple agents).
tg_interview_instructions = """You are a WiserAgent with expertise in {expertise} tasked with critically reviewing a proposed Task Graph (TG) to solve a user query.
The TG details are as follows:
{tg_details}
Your goal is to ask an insightful question about this Task Graph (only if you think its not comprehensive) that will help the TG owner WiserAgent to refine its TG.
You can address any of the following aspects or any other you find relevant:
- The completeness of the TG.
- The overall flow.
- Clarity and specificity of each task.
- Logic behind the orders.
- Any potential gaps or assumptions in the TG structure.
Introduce yourself according to your expertise, then ask your question and invite further clarification.
"""

from llm import llm
from langchain_core.messages import SystemMessage
from state import State
from interview.Interview import InterviewEntry
from TG.task_graph import TaskGraph

def generate_questions(state: State):
    tgs = state["tg_candidates"]
    agents = state["wiseragents"]
    interview_obj = state.get("Interview")

    for owner_agent, tg in tgs:
        print("Storing in state:", agent.name, type(tg), isinstance(tg, TaskGraph))
        # Create a readable TG text
        tg_text = json.dumps(tg.to_json(), indent=2)

        # Create a new InterviewEntry
        entry = InterviewEntry(
            TG_Owner=tg.owner_agent,
            task_graph=tg
        )

        for agent in agents:
            if agent == tg.owner_agent:
                continue 
            system_message = tg_interview_instructions.format(
                expertise=agent.domain_expertise,
                tg_details=tg_text
            )

            question_msg = llm.invoke([SystemMessage(content=system_message)])
            question_content = question_msg.content if hasattr(question_msg, 'content') else str(question_msg)

            entry.add_question(from_agent=agent, question=question_content)

        interview_obj.add_entry(entry)

    interview_obj.save_to_file()
    return {"interview": interview_obj.to_json()}
