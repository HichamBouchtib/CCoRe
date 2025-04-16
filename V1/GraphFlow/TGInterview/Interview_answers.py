from langchain_core.messages import SystemMessage
from llm import llm
from V1.state import State

answer_instructions = """You are a WiserAgent who has previously generated one of these Task Graphs (TGs) : {tg_candidates} to solve a given problem.  
Now, other WiserAgents specializing in different domains are interviewing you to validate and refine your TG.  

rely on the data you retrieved from the internet (web + wikipedia) to help you adress the questions : {context}

Your goal is to respond to their questions with well-structured and domain-specific explanations while maintaining clarity and logical coherence.

### **Guidelines for Answering Questions:**
1. **Stay Within the generated TG Context:**  

2. **Defend Your Design Choices if necessarly:**  
   - Justify why you structured the TG the way you did.  
   - Explain the logic behind node transitions, dependencies, and task assignments.

3. **Address Potential Gaps or Conflicts:**  
   - If a question highlights inconsistencies, clarify or propose adjustments.  
   - Suggest refinements where necessary to enhance task execution flow.

5. **Maintain a Cooperative & Constructive Tone:**  
   - Engage with interviewers as fellow WiserAgents seeking to refine the TG collaboratively.  
   - If the question is unclear or lacks sufficient detail, ask for clarification before responding.

At the end of your answer, summarize the key takeaways in 1-2 sentences and say : "Thank you so much for your help".
"""

def generate_answer(state: State):
    """Node to generate responses to TG validation questions."""

    # Extract state information
    interview = state["interview"]
    context = state["context"]
    tg_candidates = state.get("tg_candidates", {})

    # Answer question
    system_message = answer_instructions.format(
        context=context,
        tg_candidates=tg_candidates
    )
    answer = llm.invoke([SystemMessage(content=system_message)] + interview)
            
    # Name the message as coming from the WiserAgent expert
    answer.name = "TGWiserAgent"
    
    # Append answer to state
    return {"interview": interview + [answer]}
