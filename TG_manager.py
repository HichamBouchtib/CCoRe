from typing import List, Dict, Any, Optional
import uuid

class WiserAgent:
    def __init__(self, agent_id: str, domain_expertise: str):
        self.agent_id = agent_id
        self.domain_expertise = domain_expertise
        self.wisdom_score = 50  # starts at 50

    def generate_task_graph(self, user_query: str) -> Optional[Dict[str, Any]]:
        """
        Possibly calls an LLM to produce a JSON-based Task Graph.
        If the agent decides to skip (not confident), return None.
        """
        # Pseudocode:
        #  - Evaluate if domain_expertise matches user_query
        #  - If mismatch, skip
        #  - Otherwise generate some structured plan
        if self.domain_expertise.lower() not in user_query.lower():
            # e.g. agent chooses to skip
            return None

        # Otherwise, produce a dummy plan
        return {
            "plan_id": str(uuid.uuid4())[:8],
            "agent_id": self.agent_id,
            "plan": f"Plan for {user_query}",
            "roles": ["Port Scanner", "Log Analyzer"],
        }


class MasterAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.wisdom_score = 20  # starts at 20

    def vote_on_tg(self, tg_content: Dict[str, Any], user_query: str) -> int:
        """
        Return 1 if this TG is judged good, 0 if not.
        For example, check if the TG references relevant roles or matches the query.
        """
        # Simple placeholder logic:
        plan_text = tg_content.get("plan", "").lower()
        if "scan" in plan_text or "port" in plan_text:
            return 1
        return 0


class WorkerAgent:
    def __init__(self, agent_id: str, tool_name: str):
        self.agent_id = agent_id
        self.tool_name = tool_name
        # No wisdom score for WorkerAgents

    def do_work(self, query: str) -> str:
        """
        Each WorkerAgent has exactly one tool, e.g., web_search or wikipedia_search.
        For demonstration, we just return a placeholder string.
        """
        return f"[{self.tool_name} result for query='{query}']"

# A small helper class to represent a TG candidate from a WiserAgent
class TaskGraphCandidate:
    def __init__(self, plan_id: str, content: Dict[str, Any], agent_id: str):
        self.plan_id = plan_id
        self.content = content   # e.g. { 'plan': '...', 'roles': [...] }
        self.agent_id = agent_id # which WiserAgent produced it

class TaskGraphManager:
    """
    1) gather_task_graphs(...)    -> WiserAgents each produce a TG if confident
    2) master_agents_vote(...)    -> MasterAgents each vote
    3) select_winning_tg(...)     -> Pick the TG with highest vote tally
    4) update_wisdom_scores(...)  -> +1 to chosen WiserAgent, -1 to non-chosen
    5) run_task_graph(...)        -> Map-Reduce with WorkerAgents (placeholders)
    """

    def __init__(self, 
                 wiser_agents: List[WiserAgent], 
                 master_agents: List[MasterAgent]):
        self.wiser_agents = wiser_agents
        self.master_agents = master_agents

    def gather_task_graphs(self, user_query: str) -> List[TaskGraphCandidate]:
        """
        Each WiserAgent tries to produce a TG for user_query or skip (None).
        """
        candidates = []
        for wa in self.wiser_agents:
            tg = wa.generate_task_graph(user_query)
            if tg is None:
                # WiserAgent chose not to respond (avoid penalty).
                continue

            candidate = TaskGraphCandidate(
                plan_id=tg["plan_id"],
                content=tg,
                agent_id=wa.agent_id
            )
            candidates.append(candidate)

        return candidates

    def master_agents_vote(self, 
                           tgs: List[TaskGraphCandidate], 
                           user_query: str) -> Dict[str, int]:
        """
        Each MasterAgent sees all candidate TGs and votes 1 or 0.
        Return a dictionary {plan_id: vote_count}.
        """
        tally = {}
        for tg in tgs:
            # Each MasterAgent votes
            votes_for_tg = 0
            for ma in self.master_agents:
                vote = ma.vote_on_tg(tg.content, user_query)
                votes_for_tg += vote

            tally[tg.plan_id] = tally.get(tg.plan_id, 0) + votes_for_tg

        return tally

    def select_winning_tg(self, 
                          tgs: List[TaskGraphCandidate], 
                          tally: Dict[str, int]) -> Optional[TaskGraphCandidate]:
        """
        Pick the TG with the highest tally. In case of tie, pick arbitrarily.
        """
        if not tally:
            return None
        # plan_id with highest vote
        best_plan_id = max(tally, key=tally.get)
        for c in tgs:
            if c.plan_id == best_plan_id:
                return c
        return None

    def update_wisdom_scores(self, 
                             chosen_tg: TaskGraphCandidate, 
                             all_candidates: List[TaskGraphCandidate]):
        """
        +1 WS to the WiserAgent whose TG is chosen, -1 to the others that responded.
        WiserAgents that returned None (skipped) get no penalty.
        """
        if chosen_tg is None:
            return  # no updates

        chosen_agent_id = chosen_tg.agent_id

        # Create a set of all agent_ids who responded
        responded_agents = {c.agent_id for c in all_candidates}

        for wa in self.wiser_agents:
            if wa.agent_id in responded_agents:
                if wa.agent_id == chosen_agent_id:
                    wa.wisdom_score += 1
                else:
                    wa.wisdom_score -= 1

    def run_task_graph(self, 
                       chosen_tg: TaskGraphCandidate, 
                       worker_agents: List[WorkerAgent]) -> Dict[str, Any]:
        """
        1. Identify placeholders in the chosen TG's plan
        2. Each WorkerAgent is assigned a query
        3. Combine results (Map-Reduce) into a final plan
        """
        if chosen_tg is None:
            return {"error": "No TG chosen"}

        plan_content = chosen_tg.content  # e.g. { 'plan': '...', 'roles': [...] }
        plan_str = plan_content.get("plan", "")

        # For demonstration, let's assume the placeholders are <SOMETHING> in the roles
        roles = plan_content.get("roles", [])
        placeholders = []
        for role in roles:
            # e.g. "Scan for open ports <PORT_SCANNER_OUTPUT>"
            # This is a naive parse
            if "<" in role:
                placeholders.append(role)

        # We'll pretend each placeholder is a separate query
        # Then we map each WorkerAgent to one placeholder
        results = {}
        for i, placeholder in enumerate(placeholders):
            if i < len(worker_agents):
                worker = worker_agents[i]
                query = f"Fill {placeholder} with real data"
                worker_result = worker.do_work(query)
                results[placeholder] = worker_result
            else:
                results[placeholder] = "No worker assigned"

        # Reduce step: replace placeholders in the plan
        # In a real scenario, you'd parse the roles, not just do string replacement
        final_plan_str = plan_str
        for ph, val in results.items():
            final_plan_str = final_plan_str.replace(ph, val)

        return {
            "chosen_plan_id": chosen_tg.plan_id,
            "final_plan": final_plan_str,
            "raw_roles": roles,
            "worker_results": results
        }
