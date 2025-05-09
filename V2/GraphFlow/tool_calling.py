import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.context import SearchRequest
from langchain_core.messages import SystemMessage
from llm import llm
from state import State

tool_calling_instructions = """You're a smart tool selector helping WiserAgent {TG_Owner} to solve technical interviews.
Analyze the following question: {question}
Respond ONLY with one word:
- 'web' if general information from the web is needed
- 'wikipedia' if a structured or encyclopedic definition is needed
- 'none' if no external information is needed
Be extremely concise: reply only with 'web', 'wikipedia', or 'none'. No explanations.
"""

def tool_calling(state: State):
    """Node to verify for each question whether we need to call a tool or not."""
    # print("Analyzing questions to decide if tools are needed...\n")
    interviews_list = state["interview"]
    context = state["context"]

    for interview in interviews_list:
        for entry_idx, entry in enumerate(interview.entries):
            for q_idx, qst in enumerate(entry.Questions):
                agent_name = qst.from_.name
                question_text = qst.content

                if context.get_web_context(agent_name, question_text) or context.get_wikipedia_context(agent_name, question_text):
                    continue

                system_message = tool_calling_instructions.format(
                    TG_Owner=entry.TG_Owner.name,
                    question=question_text
                )

                decision = llm.invoke([SystemMessage(content=system_message)])
                decision_text = decision.content.strip().lower()

                if decision_text in ["web", "wikipedia"]:
                    req = SearchRequest(
                        source=decision_text,
                        content=question_text,
                        from_agent=agent_name,
                        entry_idx=entry_idx,
                        question_idx=q_idx
                    )
                    context.add_search_request(req)
                    print(f"{decision_text} search requested by {agent_name} in {entry.TG_Owner.name} interview")

    if context.pending_searches:
        context.current_search = context.pending_searches[0]
        context.awaiting_search = True
    else:
        context.awaiting_search = False

    return {"context": context}

