from dotenv import load_dotenv
load_dotenv()

from state import State, Context
from graph import graph
from langsmith import Client, traceable

@traceable(name="langgraph_run", metadata={"llm": "qwen2.5"})
def target(inputs: dict) -> dict:
    # topic = "AI in cybersecurity"  # this mimics the initial user input step
    initial_state = State(
        # topic=topic,
        topic=inputs["topic"], # from the dataset
        query=inputs["query"], 
        WS=50,
        wiseragents=[],
        tg_candidates=[],
        tg_chosen=None,
        context=Context(),
        interview=[],
        human_wiseragent_feedback="",
        feedback_handled=True,
        max_num_turns=3,
        response="",
        messages=[]
    )
    print("Evaluating with query:", inputs["query"])
    final_state = graph.invoke(initial_state)

    return {
        "answer": final_state["response"] or "N/A"
    }

# def accuracy_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
#     prediction = outputs.get("answer", "").strip().lower()
#     expected = reference_outputs.get("answer", "").strip().lower()

#     score = float(prediction == expected)
#     return {
#         "key": "exact_match",
#         "score": score,
#         "comment": "✔️ Match" if score else f"❌ Expected: '{expected}' but got: '{prediction}'"
#     }

def accuracy_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    prediction = outputs.get("answer", "").strip().lower()
    expected = reference_outputs.get("answer", "").strip().lower()

    score = float(prediction == expected)
    return {
        "key": "Generation_Correction(accuracy)",
        "score": score,
        "comment": "✔️ Match" if score else f"❌ Expected: '{expected}' but got: '{prediction}'"
    }

client = Client()

# experiment_results = client.evaluate(
#     target,
#     data="MAS Agent QA Benchmark",  # must match your dataset name
#     evaluators=[exact_match_evaluator],
#     experiment_prefix="mas-eval-exact-match",
#     max_concurrency=2
# )


experiment_results = client.evaluate(
    target,  # your traced MAS function
    # data="CriticBench",
    data=client.list_examples(dataset_name="CriticBench", limit=5),
    evaluators=[accuracy_evaluator],
    experiment_prefix="mas-vs-criticbench",
    max_concurrency=2,
)


# results = client.evaluate(
#         # The target is a tuple of the experiment IDs to compare
#         target=(
#             "12345678-1234-1234-1234-123456789012",
#             "98765432-1234-1234-1234-123456789012",
#         ),
#         evaluators=[accuracy],
#         summary_evaluators=[precision],
# )

# client.export_results(
#     experiment_name="mas-eval-exact-match-6446e2c1",
#     format="csv",
#     output_path="./results.csv"
# )