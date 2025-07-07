from dotenv import load_dotenv
load_dotenv()
from state import State, Context
from graph import graph
from langsmith import Client, EvaluationResult, traceable
from pydantic import BaseModel, Field, field_validator
from llm import llm
from langsmith import Client
client = Client()
gold_label = """
You are a JSON-only judge.

Your task is to return a JSON object with a single key `score`, with value `true` (correct) or `false` (wrong) — the boolean values, not strings.

Do not explain your answer.
Do not return anything except JSON.
Do not return null, None, or undefined.

Response: {G}
Reference: {expected}

Return ONLY:
{{"score": true}}   # if correct
or
{{"score": false}}  # if wrong
"""
critique_label = """
Your job is to return only a valid JSON object with a single key: `score`, which must be `true` or `false` (the boolean type, not a string)

Instructions:
- If the Critique contains at least one valid flaw or improvement related to the Generation, respond with: {{"score": true}}
- If the Critique is not helpful or fails to correctly criticize the Generation, respond with: {{"score": false}}

Do not explain your answer. Do not output anything else. Do not include null, None, or strings.

Critique:
{Q}

Generation:
{G}
"""


class BinaryScore(BaseModel):
    score: bool = Field(description="True if the response is correct, False if it is wrong.")


def f1score_summary_evaluator(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    TP = 0
    FP = 0
    FN = 0

    for output_dict, reference_output_dict in zip(outputs, reference_outputs):
        Q = output_dict.get("Q_answer", "").strip().lower()
        G = output_dict.get("G_answer", "").strip().lower()
        expected = reference_output_dict.get("answer", "").strip().lower()

        # Gold label: is G wrong?
        structured_llm = llm.with_structured_output(BinaryScore)
        gold_label_instructions = gold_label.format(G=G, expected=expected)

        is_G_wrong = None
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = structured_llm.invoke(gold_label_instructions)
                if response is not None and hasattr(response, "score") and response.score is not None:
                    is_G_wrong = response.score
                    break
            except Exception as e:
                print(f"[GOLD] Attempt {attempt + 1} failed: {e}")
        if is_G_wrong is None:
            print("⚠️ Could not determine gold label, skipping this sample.")
            continue

        # is_G_wrong = structured_llm.invoke(gold_label_instructions).score

        if is_G_wrong in ["True", "true", True]:
            is_G_wrong = 1
        elif is_G_wrong in ["Wrong", "wrong", False]:
            is_G_wrong = 0

        # Critique: did Q correctly identify G as wrong?
        structured_llm = llm.with_structured_output(BinaryScore)
        critique_label_instructions = critique_label.format(G=G, Q=Q)

        Q_identified_wrong = None
        max_retries = 5

        for attempt in range(max_retries):
            try:
                response = structured_llm.invoke(critique_label_instructions)
                if response is not None and hasattr(response, "score") and response.score is not None:
                    Q_identified_wrong = response.score
                    break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")

        if Q_identified_wrong in ["True", "true", True]:
            Q_identified_wrong = 1
        elif Q_identified_wrong in ["Wrong", "wrong", False]:
            Q_identified_wrong = 0

        # Count metrics
        if Q_identified_wrong and is_G_wrong:
            TP += 1
        elif Q_identified_wrong and not is_G_wrong:
            FP += 1
        elif not Q_identified_wrong and is_G_wrong:
            FN += 1

    # Compute precision, recall, f1
    if TP + FP == 0 or TP + FN == 0:
        return {"key": "Q (Critique)", "score": 0.0}

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return {
        "key": "Q (Critique)",
        "score": f1
        # "precision": precision,
        # "recall": recall
        # "TP": TP,
        # "FP": FP,
        # "FN": FN
    }


runs = list(client.list_runs(
    project_name="CoopCompLLMMAS_CriticBench-01b05f70",
    # execution_order="asc", 
    limit=2775  # or however many you logged before the crash
))
print(runs)
outputs = []
reference_outputs = []

import time

for run in runs:
    if run.run_type != "chain":
        continue

    try:
        time.sleep(0.3)  # Add delay to avoid 429
        full_run = client.read_run(run.id)
        outputs_dict = full_run.outputs or {}
        reference_dict = full_run.reference_outputs or {}

        outputs.append({
            "Q_answer": outputs_dict.get("Q_answer", ""),
            "G_answer": outputs_dict.get("G_answer", "")
        })

        reference_outputs.append({
            "answer": reference_dict.get("answer", "")
        })

    except Exception as e:
        print(f"⚠️ Skipping run {run.id} due to error: {e}")




# Now apply your evaluator manually
result = f1score_summary_evaluator(outputs, reference_outputs)
print(result)
