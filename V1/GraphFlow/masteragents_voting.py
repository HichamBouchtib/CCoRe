from masteragents_creation import instantiate_masteragents
from llm import llm
import json
from state import State
from langchain_core.messages import SystemMessage

def evaluate_task_graphs(state: State):
    """
    Evaluates multiple TaskGraphs using MasterAgents and selects the best one.
    
    :param state: MasterAgentState containing task graphs, master agents, and the query.
    :return: The best TaskGraph instance based on MasterAgent evaluations.
    """
    tg_candidates = state["tg_candidates"]
    master_agents = state["master_agents"]
    query = state["query"]

    # Convert task graphs to JSON format for structured evaluation
    tg_json_candidates = [tg.to_json() for tg in tg_candidates]
    
    # Each MasterAgent evaluates the TaskGraphs
    scores = []
    
    for agent in master_agents:
        # Generate the scoring prompt specific to each MasterAgent
        agent_prompt = f"""
        You are {agent.name}, an expert MasterAgent in {agent.task}. 
        Your task is to evaluate TaskGraphs. Each TaskGraph consists of tasks and orders.
        Please assess each TaskGraph and assign a score from 0 to 10 based on its effectiveness and structure to resolve the user's query: "{query}".

        TaskGraphs (in JSON format):
        {json.dumps(tg_json_candidates, indent=2)}

        Output your response as a JSON list of dictionaries with 'task_graph' and 'score' keys.
        """

        system_message = SystemMessage(content=agent_prompt)
        
        # Invoke LLM to get scores from this MasterAgent's perspective
        scoring_response = llm.invoke([system_message])
        
        try:
            # Extract scores from LLM response
            agent_scores = json.loads(scoring_response.content)
            if not isinstance(agent_scores, list):
                raise ValueError(f"Invalid JSON format received from {agent.name}")

            scores.extend(agent_scores)  # Collect all scores from different MasterAgents

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error processing LLM response from {agent.name}: {e}")

    # Aggregate scores by averaging them per TaskGraph
    tg_score_map = {tg: [] for tg in tg_json_candidates}

    for score_entry in scores:
        tg_json = score_entry["task_graph"]
        score = score_entry["score"]
        tg_score_map[tg_json].append(score)

    # Compute the average score for each TaskGraph
    tg_avg_scores = {
        tg: sum(scores) / len(scores) if scores else 0  # Avoid division by zero
        for tg, scores in tg_score_map.items()
    }

    # Select the TaskGraph with the highest average score
    best_tg_json = max(tg_avg_scores, key=tg_avg_scores.get)
    best_tg_index = tg_json_candidates.index(best_tg_json)

    # Return the chosen TaskGraph
    return {"tg_chosen": tg_candidates[best_tg_index]}

