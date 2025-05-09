from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from state import State
from llm import llm

scoring_instructions = """ You are an expert evaluator in your speciality of Task Graph proposals.
Evaluate the following Task Graph proposals and provide a score from 0 to 5 to each one of them based on:
1. Clarity of task definitions.
2. Logical flow and dependency between tasks.
3. Completeness and feasibility of the plan.
Return your output as a JSON object with a single field "score" that contains a number between 0 and 5.
Here is the Task Graph proposal:
{tg_candidates}
"""

def vote_TG(state: State):
    tg_candidates = state.get("tg_candidates", [])
    if not tg_candidates:
        print("No TG candidates available for voting.")
        return {"tg_chosen": None}
    
    aggregated_scores = []
    
    # Iterate over each TG candidate.
    for candidate in tg_candidates:
        candidate_scores = []
        tg_json = candidate["task_graph"].to_json()  # Get the JSON dict representation.
        
        # Let each wiseragent score this candidate using its own LLM.
        for agent in state["wiseragents"]:
            
            prompt = scoring_instructions.format(tg_proposal=tg_json)
            system_message = SystemMessage(content=prompt)
            
            # Use the agent's LLM to get a structured JSON output.
            try:
                result = llm.with_structured_output(dict).invoke([system_message])
                score = float(result.get("score", 0))
            except Exception as e:
                print(f"Error scoring candidate from agent {agent.name}: {e}")
                score = 0
            
            candidate_scores.append(score)
        
        # Compute the average score for the candidate.
        avg_score = sum(candidate_scores) / len(candidate_scores)
        aggregated_scores.append({
            "agent": candidate["agent"],
            "avg_score": avg_score,
            "task_graph": candidate["task_graph"]
        })
    
    # Select the candidate with the highest average score.
    best_candidate = max(aggregated_scores, key=lambda x: x["avg_score"])
    state["tg_chosen"] = best_candidate["task_graph"]
    
    print(f"Chosen TG from agent: {best_candidate['agent']} with average score: {best_candidate['avg_score']}")
    return {"tg_chosen": best_candidate["task_graph"]}
