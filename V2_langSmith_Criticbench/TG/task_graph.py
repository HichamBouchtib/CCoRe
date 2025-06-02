import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import uuid
import json
import glob
import re
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from agents.wiseragent import WiserAgent
from graphviz import Digraph
from ipywidgets import VBox, HTML
from IPython.display import display

class TaskGraph(BaseModel):
    owner_agent: WiserAgent = Field(..., description="The WiserAgent that owns this task graph")
    tasks: Dict[str, str] = Field(..., description="Dictionary of task names and their descriptions")
    orders: List[Dict[str, str]] = Field(..., description="List of task transitions with from, to, condition") 
    refused: Optional[bool] = False
    answer: Optional[str] = Field(..., description="The userquery final answer")

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
            "orders": self.orders,
            "refused": self.refused,
            "answer": self.answer
        }

    def to_json_string(self, indent=4) -> str:
        """Return the TaskGraph as a JSON-formatted string."""
        return json.dumps(self.to_json(), indent=indent)

    # def save_to_file(self, folder="TG/saved_TGs", filename: str = None) -> None:
    #     os.makedirs(folder, exist_ok=True)

    #     if filename is None:
    #         unique_id = uuid.uuid4().hex[:8]  # Shorter version
    #         filename = os.path.join(folder, f"TG_{unique_id}.json")
    #     else:
    #         if not os.path.isabs(filename) and not filename.startswith(folder):
    #             filename = os.path.join(folder, filename)

    #     with open(filename, "w", encoding="utf-8") as f:
    #         json.dump(self.to_json(), f, indent=4)

    #     print(f"Task graph saved to {filename}")
    

    def save_to_file(self, subfolder: str = "query_1", base_folder: str = "TG/saved_TGs", filename: str = None) -> None:
        folder_path = os.path.join(base_folder, subfolder)
        os.makedirs(folder_path, exist_ok=True)

        if filename is None:
            unique_id = uuid.uuid4().hex[:8]
            filename = os.path.join(folder_path, f"TG_{unique_id}.json")
        else:
            if not os.path.isabs(filename) and not filename.startswith(folder_path):
                filename = os.path.join(folder_path, filename)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=4)

        print(f"Task graph saved to {filename}")

    def pretty_print(self) -> None:
        """Prints a nicely formatted view of the TaskGraph."""
        print(f"\nProposed by: {self.owner_agent.name}")

        tg_dict = {
            "tasks": self.tasks if hasattr(self, "tasks") else {},
            "orders": self.orders if hasattr(self, "orders") else [],
            "refused": self.refused,
            "answer": self.answer
        }

        print(json.dumps(tg_dict, indent=2))
    
    # visualization methods
    def visualize_with_graphviz(self, output_path: str = None, view: bool = False) -> Digraph:
        dot = Digraph(comment=f"Task Graph by {self.owner_agent.name}")

        # 'dot' for hierarchical
        dot.engine = 'dot'

        # Control spacing between nodes and layers
        dot.attr(rankdir="TB", nodesep="0.4", ranksep="0.4")

        # Node styling
        dot.attr(
            'node',
            shape='circle',
            style='filled',
            fillcolor='lightgrey',
            fontname='Arial',
            fontsize='18',
            width='1.2',
            height='1.2'
        )

        # Add all tasks as nodes (only names)
        for task_name in self.tasks.keys():
            dot.node(task_name)

        # Edge label font styling
        edge_font_color = "darkblue"

        for order in self.orders:
            from_task = order.get("from", "")
            to_task = order.get("to", "")
            condition = order.get("condition", "")

            dot.edge(
                from_task,
                to_task,
                label=condition,
                fontcolor=edge_font_color,
                fontsize='12',
                fontname='Arial'
            )

        dot.graph_attr.update(splines="line", overlap="false")

        if "Start" not in self.tasks:
            dot.node("Start", shape="plaintext", label="Start", fontsize="14")
            first_task = next(iter(self.tasks))
            dot.edge("Start", first_task)
        if "End" not in self.tasks:
            dot.node("End", shape="plaintext", label="End", fontsize="14")

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            dot.render(output_path, format='png', view=view)
            print(f"Graphviz visualization saved to {output_path}.png")
        elif view:
            dot.view()

        return dot

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
            orders=json_data["orders"],
            refused=json_data["refused"],
            answer=json_data["answer"]
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
    def list_available_graphs(folder="TG/saved_TGs") -> List[str]:
        """List all task graph JSON files in the TG folder."""
        os.makedirs(folder, exist_ok=True)
        files = glob.glob(os.path.join(folder, "TG_*.json"))
        return files if files else ["No task graphs found."]

class TGList(BaseModel):
    taskgraphs: List[TaskGraph] = Field(
        description="Full list of candidate task graphs.",
    )

# Vizualization function for TGs 
def score_to_stars(score: float) -> str:
    full_stars = int(score)
    half_star = score - full_stars >= 0.5
    return "⭐" * full_stars + ("✰" if half_star else "")

def get_next_query_folder(base_folder="TG/saved_TGs") -> str:
    os.makedirs(base_folder, exist_ok=True)
    existing = [
        int(m.group(1)) for d in os.listdir(base_folder)
        if (m := re.match(r"query_(\d+)", d)) and os.path.isdir(os.path.join(base_folder, d))
    ]
    next_index = max(existing, default=0) + 1
    subfolder_name = f"query_{next_index}"
    full_path = os.path.join(base_folder, subfolder_name)
    os.makedirs(full_path, exist_ok=True)
    return subfolder_name
