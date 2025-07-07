from llm import llm
from langchain_core.messages import SystemMessage
from state import State

# Interview instructions for TG validation (for multiple agents).
tg_interview_instructions = """You are a WiserAgent with expertise in {expertise} tasked with critically reviewing a proposed Task Graph (TG) to solve a user query.
The TG details are as follows:
{tg_details}
Your goal is to ask an insightful question that will help the TG owner to refine it by addressing:
- The overall flow.
- Clarity and specificity of each mastergant's task.
- Logic behind the orders.
- Any potential gaps or assumptions in the TG structure.
Introduce yourself according to your expertise, then ask your question and invite further clarification.
"""

def generate_TGinterview_questions(state: State):
    # Extract the candidate TG details.
    tg = state["tg_candidates"]
    generated_questions = []
    
    for agent in state["wiseragents"]:

        system_message = tg_interview_instructions.format(
            expertise=agent.domain_expertise,
            tg_details=tg
        )
        
        # generate an interview question.
        question = llm.invoke([SystemMessage(content=system_message)])
        
        generated_questions.append(f"{agent.name}: {question}")

        # Update the interview transcript
        if state["interview"]:
            state["interview"] += "\n" + "\n".join(generated_questions)
        else:
            state["interview"] = "\n".join(generated_questions)

    # Return the updated transcript and list of generated questions.
    return {"interview": state["interview"]}