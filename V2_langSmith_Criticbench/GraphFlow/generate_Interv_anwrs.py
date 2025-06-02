import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'GraphFlow')))
from langchain_core.messages import SystemMessage
from llm import llm
from state import State
from tools.context import Context
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from wiseragents_creation  import append_or_update_AImessage1

answer_instructions = """
You're WiserAgent {TG_Owner}, the original creator of this task graph (TG): {tg}, which was designed to solve this initial query: {query}. Now, other domain-specialized WiserAgents are interviewing you technically, one by one, to evaluate and improve your TG if needed. Respond to its question: {question}.
Rely on the retrieved information from web or wikipedia if available:
- Web: {web}
- Wikipedia: {wikipedia}
If no web or wikipedia info is provided above, answer directly based on your own technical expertise.
Avoid greetings, too much justification, or suggestions unless explicitly asked.
"""

def generate_answers(state: State):

    """Node to generate FINAL responses after tool_calling pre-augmented the context."""
    
    query = state["query"]
    interviews = state["interview"]
    context = state.get("context", Context())

    for interview in interviews:
        # print("type of interview in generate_interv_anwrs at the begining:", type(interview))
        # print("type of interviews in generate_interv_anwrs at the begining:", type(interviews))
        for entry_idx, entry in enumerate(interview.entries):
            for q_idx, qst in enumerate(entry.Questions):
                # Skip if already answered
                if any(ans.to_.name == qst.from_.name for ans in entry.Answers):
                    continue

                agent_name = qst.from_.name
                question_text = qst.content
                web = context.get_web_context(agent_name, question_text) or ""
                wikipedia = context.get_wikipedia_context(agent_name, question_text) or ""

                system_message = answer_instructions.format(
                    question=question_text,
                    tg=entry.task_graph.to_json_string(),
                    TG_Owner=entry.TG_Owner.name,
                    query=query,
                    web=web,
                    wikipedia=wikipedia
                )
                # skip no-questions
                if question_text == "No question, Thanks":
                    entry.add_answer(to_agent=qst.from_, answer="You are welcome!")
                    continue
                answer = llm.invoke([SystemMessage(content=system_message)])
                entry.add_answer(to_agent=qst.from_, answer=answer.content)
                # print(f"Answer from {entry.TG_Owner.name} to {agent_name}: \n {answer.content}")

    print("-----Deep Monologue: Ended-----")
    for interview in interviews:
        interview.save_answers_to_file()
    
    # reset context
    context.pending_searches = []
    context.awaiting_search = False

    tool_call_id = f"DeepMonologue_display_{len(state['messages'])}"
    interview_payload = []
    for interview in interviews:
        for entry in interview.entries:
            entry_data = {
                "owner": entry.TG_Owner.name,
                "questions": [
                    {
                        "from": q.from_.name,
                        "content": q.content
                    }
                    for q in entry.Questions
                ],
                "answers": [
                    {
                        "to": a.to_.name,
                        "content": a.content
                    }
                    for a in entry.Answers
                ]
            }
            interview_payload.append(entry_data)
    append_or_update_AImessage1(state, "DeepMonologue Started...")
    state["messages"].append(
        AIMessage(
            content="Full DeepMonolgue Q&A:",
            tool_calls=[
                {
                    "name": "DeepMonologue_summary",
                    "args": {
                        "interviews": interview_payload
                    },
                    "id": tool_call_id
                }
            ]
        )
    )

    state["interview"] = interviews
    # print("type of interviews in generate_interv_anwrs at the end:", type(interviews))
    
    # return {"interview": interviews}
    return state


