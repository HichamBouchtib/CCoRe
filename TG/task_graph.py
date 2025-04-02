import json
import os
import glob

class TaskGraph:
    def __init__(self, masteragents=None, tasks=None, orders=None):
        """
        Initializes a Task Graph.
        :param masteragents: List of master agent names (tasks).
        :param tasks: Dictionary of tasks with their descriptions.
        :param orders: List of task transitions based on conditions.
        """
        self.masteragents = masteragents or []
        self.tasks = tasks if tasks else {}
        self.orders = orders if orders else []

    def set_tasks(self, tasks):
        """ Sets the tasks after extracting them from the LLM output. """
        self.tasks = tasks

    def set_orders(self, orders):
        """ 
        Sets the task transitions (orders) after extracting them from the LLM output.
        Each order must have 'from', 'to', and 'condition' elements.
        """
        self.orders = [
            {
                "from": order.get("from", ""),
                "to": order.get("to", ""),
                "condition": order.get("condition", "")
            }
            for order in orders
        ]
    
    def to_json(self):
        """ Convert the TaskGraph instance into a structured JSON format. """
        return {
            "masteragents": self.masteragents,
            "tasks": self.tasks,
            "orders": self.orders
        }

    def save_to_file(self, folder="TG"):
        """ Save the TaskGraph instance as a JSON file in the TG folder with a unique name. """
        os.makedirs(folder, exist_ok=True)  # Ensure the folder exists
        
        # Get a list of existing TG files
        existing_files = glob.glob(os.path.join(folder, "TG_*.json"))
        next_index = len(existing_files) + 1  # Determine the next file index
        
        filename = os.path.join(folder, f"TG_{next_index}.json")
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=4)
        
        print(f"Task graph saved to {filename}")

    @staticmethod
    def from_json(json_data):
        """ Create a TaskGraph instance from a JSON-like dictionary. """
        masteragents = json_data.get("masteragents", [])
        tasks = json_data.get("tasks", {})
        orders = json_data.get("orders", [])
        return TaskGraph(masteragents=masteragents, tasks=tasks, orders=orders)

    @staticmethod
    def load_from_file(filename):
        """ Load a TaskGraph instance from a JSON file. """
        if not os.path.exists(filename):
            print(f"Error: File {filename} does not exist.")
            return None

        with open(filename, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        return TaskGraph.from_json(json_data)

    @staticmethod
    def list_available_graphs(folder="TG"):
        """ List all task graph JSON files in the TG folder. """
        os.makedirs(folder, exist_ok=True)  # Ensure the folder exists
        files = glob.glob(os.path.join(folder, "TG_*.json"))
        return files if files else ["No task graphs found."]

# class TGState(MessagesState):
#     query: str      
#     wiseragents: List[WiserAgent]
#     max_num_turns: int # Number turns of conversation allowed            
#     interview: str  # transcript of the interview
#     questions: List
#     tg_candidates: dict
#     context: Annotated[list, operator.add] # Source docs