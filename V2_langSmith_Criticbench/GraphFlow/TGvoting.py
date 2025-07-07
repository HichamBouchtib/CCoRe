from copy import deepcopy
import sys
import os
from typing import Dict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from TG.task_graph import TaskGraph
from agents.wiseragent import WiserAgent
from langchain_core.messages import SystemMessage, AIMessage
from state import State
from llm import llm
from pydantic import BaseModel
from wiseragents_creation import append_or_update_AImessage2

class ScoreOutput(BaseModel):
    scores: Dict[str, float]  # key = TG owner name, value = score

scoring_instructions = """
You are {agent_name}, a WiserAgent with expertise in {domain}.
You are tasked with reviewing multiple Task Graphs (TGs) proposed to solve the following user query:
"{query}"

Here are the TGs evaluate them against each other:
{all_tgs}

For each TG, evaluate critically based on:
- Clarity and precision of task definitions
- Logical flow and dependency correctness
- Completeness and feasibility of solving the query

Score Guide: 
- 1: Very Poor
- 2: Poor
- 3: Fair
- 4: Good
- 5: Excellent

Don't score them all the same
Be strict: Deduct points for unclear tasks, missing steps, illogical flows, or redundancy.
Return your output as a JSON object structured like:
{{
  "TG Owner Name 1": score1,
  "TG Owner Name 2": score2,
  ...
}}
If you cannot score a TG, return an empty JSON object {{}} for that TG."
"""

def vote_TG(state: State):
    query = state["query"]
    context = state["context"]
    context.pending_searches = []
    context.awaiting_search = False

    tg_candidates = state.get("tg_candidates", [])
    agents = state["wiseragents"]
    
    if not tg_candidates:
        print("No TG candidates available for voting.")
        return {"tg_chosen": None}
    
    # print("\n--- Individual Scores ---\n")
    aggregated_scores = {tg.owner_agent.name: [] for tg in tg_candidates}
    individual_scores = {agent.name: {} for agent in agents}

    # Prepare all TGs nicely
    all_tgs_text = ""
    for candidate in tg_candidates:
        all_tgs_text += f"\n--- TG proposed by {candidate.owner_agent.name} ---\n{candidate.to_json()}\n"

    for agent in agents:
        # print(f"\nAgent {agent.name} is scoring...") 
        prompt = scoring_instructions.format(
            agent_name=agent.name,
            domain=agent.domain_expertise,
            query=query,
            all_tgs=all_tgs_text
        )
        system_message = SystemMessage(content=prompt)
        
        try:
            scores = llm.with_structured_output(ScoreOutput).invoke([system_message])
            # print("Scores received:", scores)
            for tg_owner, score in scores.scores.items():
                if tg_owner != agent.name:  # Skip self-evaluation
                    aggregated_scores[tg_owner].append(score)
                    individual_scores[agent.name][tg_owner] = score
                    # print(f"- {tg_owner} TG score : {score}")
        except Exception as e:
            print(f"- Error during scoring by {agent.name}: {e}")

    # Compute averages
    final_scores = []
    for owner_name, score_list in aggregated_scores.items():
        if score_list:
            avg_score = sum(score_list) / len(score_list)
            candidate = next(c for c in tg_candidates if c.owner_agent.name == owner_name)
            final_scores.append({"agent": candidate.owner_agent, "avg_score": avg_score, "TG": candidate})
            print(f"\n --> Average score for {owner_name}: {avg_score:.2f}")

    # Choose best TG
    best_candidate = max(final_scores, key=lambda x: x["avg_score"])
    state["tg_chosen"] = best_candidate["TG"]

    print(f"\n>>> Chosen TG from agent: {best_candidate['agent'].name}")

    # Tool call ID
    tool_call_id = f"tg_voting_{len(state['messages'])}"
    # Prepare voting results payload
    voting_payload = [
        {
            "agent": fs["agent"].name,
            "avg_score": fs["avg_score"]
        }
        for fs in final_scores
    ]
    
    state["messages"].append(
        AIMessage(
            content="🗳️ Voting Results :",
            tool_calls=[
                {
                    "name": "voting_results_summary",
                    "args": {
                        "results": voting_payload,
                        "individual_votes": individual_scores
                    },
                    "id": tool_call_id
                }
            ]
        )
    )  
    append_or_update_AImessage2(state, f"""🏆 Winning TG owner: `{best_candidate['agent'].name}`""")
    
    state["tg_chosen"] = best_candidate["TG"]
    
    # return {
    #      "tg_chosen": best_candidate["TG"]
    # }
    return state