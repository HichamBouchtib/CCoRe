from langchain_core.messages import HumanMessage

# 1) Import the two graphs
from wiser_graph import build_wiser_graph, WiserState
from master_graph import build_master_graph, MasterState

def main():
    """
    Demonstrates how to:
      1) Run the Wiser Graph (generate WiserAgents, produce Task Graphs)
      2) Then run the Master Graph (vote on TGs, structure outputs, do web/Wikipedia search)
    """

    wiser_graph = build_wiser_graph()
    thread_wiser = {"configurable": {"thread_id": "wiser_session"}}

    # Initial WiserState
    init_wiser_state: WiserState = {
        "topic": "Cybersecurity for IoT devices",
        "max_wiser_agents": 2,
        "human_wiser_feedback": None,
        "wiser_agents": [],
        "task_graphs": []
    }

    # Run until the first interruption (human_feedback)
    for event in wiser_graph.stream(init_wiser_state, thread_wiser, stream_mode="values"):
        pass

    # Provide some human feedback to refine WiserAgents
    new_feedback = "Add a WiserAgent focusing on hardware-level IoT security."
    wiser_graph.update_state(
        thread_wiser,
        {"human_wiser_feedback": new_feedback},
        as_node="human_feedback"
    )

    # Continue until the Wiser Graph ends
    for event in wiser_graph.stream(None, thread_wiser, stream_mode="values"):
        pass

    # Retrieve final Wiser state
    final_wiser_state = wiser_graph.get_state(thread_wiser)
    print("===== Final Wiser State =====")
    print(final_wiser_state.values)

    master_graph = build_master_graph()
    thread_master = {"configurable": {"thread_id": "master_session"}}

    # Prepare the MasterState:
    #  - We take 'task_graphs' from the WiserState
    #  - Provide 'messages' for search_web / search_wikipedia
    init_master_state: MasterState = {
        "task_graphs": final_wiser_state.values.get("task_graphs", []),
        "chosen_tg": {},
        "structured_plan": {},
        "messages": [
            HumanMessage(content="Which ports are most commonly attacked?")
        ],
        "context": []
    }

    # Run the Master Graph to completion
    for event in master_graph.stream(init_master_state, thread_master, stream_mode="values"):
        pass

    # Retrieve final Master state
    final_master_state = master_graph.get_state(thread_master)
    print("===== Final Master State =====")
    print(final_master_state.values)

if __name__ == "__main__":
    main()


from flask import Flask, request, jsonify
from main import main as run_pipeline

app = Flask(__name__)

@app.route("/run_pipeline", methods=["POST"])
def run_pipeline_endpoint():
    # Potentially parse input from request.json
    # Then call run_pipeline() or embed that logic here
    result = run_pipeline()
    return jsonify({"result": result})
