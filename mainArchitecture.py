import os
import json
import requests
import torch
from datetime import datetime
from llama_index import GPTListIndex, ServiceContext, LLMPredictor
from llama_index.llms import CustomLLM
import os
from agents.agent import Agent
from ollama_llm import OllamaLLM
from SMP import SharedMessagePool
from agents.wise_agent import WiseAgent
from agents.master_agent import MasterAgent
from agents.worker_agent import WorkerAgent

ollama_llm = OllamaLLM(server_url="http://127.0.0.1:5000/generate")
llm_predictor = LLMPredictor(llm=ollama_llm)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# UTILITY FUNCTIONS
def log_history(agent_name: str, query: str, decision: str, history_file: str = "history.json"):
    """
    Record the decisions or outcomes of an agent.
    """
    record = {
        "agent": agent_name,
        "query": query,
        "decision": decision,
        "timestamp": datetime.now().isoformat()
    }
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
    else:
        history = []
    history.append(record)
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)

def ensure_dir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    ensure_dir("logs")
    ensure_dir("datasets")

    # Initialize agents with varying wisdom levels (for WiseAgents) and roles
    wise_agents = [WiseAgent(name=f"WiseAgent_{i}", wisdom_score=90 + i) for i in range(3)]
    master_agents = [MasterAgent(name=f"MasterAgent_{i}") for i in range(2)]
    worker_agents = [WorkerAgent(name=f"WorkerAgent_{i}") for i in range(5)]
    shared_message_pool = SharedMessagePool()

    # 1. Query reformulation
    query = "Optimize logistics operations for supply chain."
    print(f"Original Query: {query}")
    reformulate_prompt = f"Reformulate the following query to be more context-rich and detailed: {query}"
    reformulated_query = llm_predictor.predict(reformulate_prompt)
    print(f"Reformulated Query: {reformulated_query}")

    # 2. WiseAgents generate TGs for the query
    task_graphs = []
    for agent in wise_agents:
        tg = agent.generate_task_graph(reformulated_query)
        task_graphs.append(tg)
        print(f"{agent.name} generated Task Graph with score {tg['score']}")

    # 3. MasterAgents vote on the best Task Graph (simulate with the first MasterAgent)
    selected_graph = master_agents[0].select_task_graph(task_graphs)
    print(f"Selected Task Graph from {selected_graph['agent']}: {selected_graph['graph']}")

    # 4. MasterAgents produce output with placeholders (simulate by asking for output with markers)
    placeholder_prompt = f"Generate an output with placeholders for missing sub-task details from the following Task Graph: {selected_graph['graph']}"
    master_output = master_agents[0].execute_task(placeholder_prompt)
    print(f"MasterAgent output with placeholders: {master_output}")

    # 5. WorkerAgents execute their assigned sub-tasks (simulate by executing parts of the task graph)
    for worker in worker_agents:
        subtask_prompt = f"Execute sub-task from the following Task Graph: {selected_graph['graph']}"
        task_output = worker.execute_task(subtask_prompt)
        shared_message_pool.add_message({"worker": worker.name, "output": task_output})
        print(f"{worker.name} executed sub-task with output: {task_output}")

    # 6. Optionally MasterAgent creates a custom dataset for fine-tuning
    dataset_data = json.dumps({
        "query": reformulated_query,
        "selected_graph": selected_graph,
        "worker_outputs": shared_message_pool.get_messages()
    }, indent=4)
    dataset_path = master_agents[0].create_custom_dataset(dataset_data)
    print(f"Custom dataset created at: {dataset_path}")

    # 7. Display shared message pool (collective brain)
    print("\nShared Message Pool Contents:")
    for msg in shared_message_pool.get_messages():
        print(msg)

if __name__ == "__main__":
    main()
