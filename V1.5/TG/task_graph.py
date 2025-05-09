import sys
import os
import uuid
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import glob
from typing import List, Dict
from pydantic import BaseModel, Field
from agents.wiseragent import WiserAgent


class TaskGraph(BaseModel):
    owner_agent: WiserAgent = Field(..., description="The WiserAgent that owns this task graph")
    tasks: Dict[str, str] = Field(..., description="Dictionary of task names and their descriptions")
    orders: List[Dict[str, str]] = Field(..., description="List of task transitions with from, to, condition") # V1 change here

    def update_tasks(self, tasks: Dict[str, str]) -> None:
        """Sets the tasks dictionary."""
        self.tasks = tasks

    def update_orders(self, orders: List[Dict[str, str]]) -> None:
        """
        Sets the task transitions (orders). Each order must have 'from', 'to', and 'condition'.
        """
        validated_orders = []
        for order in orders:
            validated_orders.append({
                "from": order.get("from", ""),
                "to": order.get("to", ""),
                "condition": order.get("condition", "")
            })
        self.orders = validated_orders

    def to_json(self) -> Dict:
        """Convert the TaskGraph instance into a JSON-compatible dict."""
        return {
            "owner_agent": self.owner_agent.name,
            "tasks": self.tasks,
            "orders": self.orders
        }

    def to_json_string(self, indent=4) -> str:
        """Return the TaskGraph as a JSON-formatted string."""
        return json.dumps(self.to_json(), indent=indent)

    # def save_to_file(self, folder="TG", filename: str = None) -> None:
    #     """Save the TaskGraph to a JSON file. If filename is None, auto-increment."""
    #     os.makedirs(folder, exist_ok=True)

    #     if filename is None:
    #         existing_files = glob.glob(os.path.join(folder, "TG_*.json"))
    #         next_index = len(existing_files) + 1
    #         filename = os.path.join(folder, f"TG_{next_index}.json")
    #     else:
    #         # Only add folder if it's not already part of the filename
    #         if not os.path.isabs(filename) and not filename.startswith(folder):
    #             filename = os.path.join(folder, filename)

    #     with open(filename, "w", encoding="utf-8") as f:
    #         json.dump(self.to_json(), f, indent=4)

    #     print(f"Task graph saved to {filename}")
    def save_to_file(self, folder="TG", filename: str = None) -> None:
        os.makedirs(folder, exist_ok=True)

        if filename is None:
            unique_id = uuid.uuid4().hex[:8]  # Shorter version
            filename = os.path.join(folder, f"TG_{unique_id}.json")
        else:
            if not os.path.isabs(filename) and not filename.startswith(folder):
                filename = os.path.join(folder, filename)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=4)

        print(f"Task graph saved to {filename}")

    @staticmethod
    def from_json(json_data: Dict, agent_registry: Dict[str, WiserAgent]) -> "TaskGraph":
        """Create a TaskGraph instance from a JSON-like dictionary."""
        owner_name = json_data["owner_agent"]
        owner_agent = agent_registry.get(owner_name)
        if owner_agent is None:
            raise ValueError(f"Unknown agent name: {owner_agent}")
        return TaskGraph(
            owner_agent=owner_agent,
            tasks=json_data["tasks"],
            orders=json_data["orders"]
        )

    @staticmethod
    def load_from_file(filename: str, agent_registry: Dict[str, WiserAgent]) -> "TaskGraph":
        """Load a TaskGraph instance from a JSON file."""
        if not os.path.exists(filename):
            print(f"Error: File {filename} does not exist.")
            return None
        with open(filename, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        return TaskGraph.from_json(json_data, agent_registry)

    @staticmethod
    def list_available_graphs(folder="TG") -> List[str]:
        """List all task graph JSON files in the TG folder."""
        os.makedirs(folder, exist_ok=True)
        files = glob.glob(os.path.join(folder, "TG_*.json"))
        return files if files else ["No task graphs found."]

class TGList(BaseModel):
    taskgraphs: List[TaskGraph] = Field(
        description="Full list of candidate task graphs.",
    )

# # Create dummy agents
# agent1 = WiserAgent(
#             name="AgentAlpha",
#             domain_expertise="AI-Driven Malware Analysis",
#             description="Specializes in analyzing malware using advanced AI techniques to identify new threats and vulnerabilities.",
#             WS=50,
#             preferred_llm="qwen2.5:latest"
#         )
# agent_registry = {"AgentAlpha": agent1}

# # Step 1: Create a TaskGraph
# tg = TaskGraph(
#     owner_agent=agent1,
#     tasks={
#         "start": "Begin the process",
#         "verify": "Verify input data",
#         "end": "Finish the task"
#     },
#     orders=[
#         {"from": "start", "to": "verify", "condition": "input_valid"},
#         {"from": "verify", "to": "end", "condition": "verified_true"}
#     ]
# )

# print("🛠 Created TaskGraph object:")
# print(tg.to_json_string())

# # Step 2: Save it
# tg.save_to_file()

# # Step 3: List existing graphs
# print("\n Available TaskGraph files:")
# for file in TaskGraph.list_available_graphs():
#     print(f" - {file}")

# # Step 4: Load the most recent one
# import glob
# import os
# files = sorted(glob.glob("TG/TG_*.json"))
# latest_file = files[-1] if files else None

# if latest_file:
#     loaded_tg = TaskGraph.load_from_file(latest_file, agent_registry)
#     print("\nLoaded TaskGraph from file:")
#     print(loaded_tg.to_json_string())

#     # Step 5: Update tasks
#     loaded_tg.update_tasks({
#         "init": "Initialize system",
#         "process": "Run main logic",
#         "cleanup": "Wrap up and exit"
#     })
#     print("\n Updated tasks:")
#     print(loaded_tg.to_json_string())

#     # Save again to check overwrite safety
#     loaded_tg.save_to_file(filename=latest_file)

# # Step 6: Create TGList
# tg_list = TGList(taskgraphs=[tg])
# print("\n TGList example:")
# print(tg_list.model_dump_json(indent=4))