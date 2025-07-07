from dotenv import load_dotenv
load_dotenv()
from state import State, Context
from graph import graph
from langsmith import Client, EvaluationResult, traceable
from pydantic import BaseModel, Field, field_validator
from llm import llm

GC_scoring_instructions = """"
You are an expert in evaluating AI-generated responses as false or true.
Your task is to verify whether the response is correct in compraison with a reference response.
Return only a valid JSON object in the following format:
{{"score": true}}   # if the agent response is correct
{{"score": false}}  # if the agent response is incorrect

Agent response: {response}
Reference response: {reference_response}
"""

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

# Global counters
G_accuracy = 0
C_accuracy = 0
TP = 0
FP = 0
FN = 0

class BinaryScore(BaseModel):
    score: bool = Field(description="True if the response is correct, False if it is wrong.")

# @traceable(name="criticbench", metadata={"llm": "qwen2.5"})
# def target(inputs: dict) -> dict:
#     initial_state = State(
#         # topic=topic,
#         topic=inputs["topic"], # from the dataset
#         query=inputs["query"], 
#         WS=50,
#         wiseragents=[],
#         tg_candidates=[],
#         tg_chosen=None,
#         context=Context(),
#         interview=[],
#         human_wiseragent_feedback="",
#         feedback_handled=True,
#         max_num_turns=3,
#         G_response="",
#         Q_response="",
#         C_response="",
#         messages=[]
#     )
    
#     print("Evaluating with query:\n", inputs["query"])
#     final_state = graph.invoke(initial_state)

#     return {
#         "G_answer": final_state["g_response"] or "N/A",
#         "Q_answer": final_state["q_response"] or "N/A",
#         "C_answer": final_state["c_response"] or "N/A",
#     }

def target(inputs: dict) -> dict:
    from langsmith import traceable

    topic = inputs.get("topic", "General")

    @traceable(name="criticbench", metadata={"llm": "qwen2.5", "topic": topic})
    def _inner_target(inputs):
        initial_state = State(
            # topic=inputs["topic"],
            topic=topic,
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
            G_response="",
            Q_response="",
            C_response="",
            messages=[]
        )
        print("Evaluating with query:\n", inputs["query"])
        final_state = graph.invoke(initial_state)

        return {
            "G_answer": final_state["g_response"] or "N/A",
            "Q_answer": final_state["q_response"] or "N/A",
            "C_answer": final_state["c_response"] or "N/A",
        }

    return _inner_target(inputs)

def accuracy(outputs: dict, reference_outputs: dict):
    
    G = outputs.get("G_answer", "").strip().lower()
    C = outputs.get("C_answer", "").strip().lower()
    expected = reference_outputs.get("answer", "").strip().lower()

    structured_llm = llm.with_structured_output(BinaryScore)
    
    G_scoring_instructions = GC_scoring_instructions.format(response=G, reference_response=expected)
    C_scoring_instructions = GC_scoring_instructions.format(response=C, reference_response=expected)

    Gscore = structured_llm.invoke(G_scoring_instructions)
    if Gscore.score in ["True", "true", True]:
        G_score = 1
    if Gscore.score in["Wrong", "wrong", False]:
        G_score = 0

    Cscore = structured_llm.invoke(C_scoring_instructions)
    if Cscore.score in ["True", "true", True]:
        C_score = 1
    if Cscore.score in["Wrong", "wrong", False]:
        C_score = 0

    global G_accuracy, C_accuracy
    G_accuracy += G_score
    C_accuracy += C_score

    # return {
    #     # "G": G,
    #     # "C": C,
    #     "G_score": G_score,
    #     "C_score": C_score,
    #     "Expected": expected
    # }
    return [
        {"key": "G (Generation)", "score": G_score},
        {"key": "C (Correction)", "score": C_score}
        ]

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

client = Client()


all_examples = list(client.list_examples(dataset_name="CriticBench_dataset", limit=3825))
resume_from = 2775
remaining_examples = all_examples[resume_from:]
experiment_results = client.evaluate(
    target,
    # data="CriticBench",
    data=remaining_examples,
    evaluators=[accuracy],
    summary_evaluators=[f1score_summary_evaluator],
    experiment_prefix="CoopCompLLMMAS_CriticBench",
    max_concurrency=2
    # metadata={"topic": example.inputs["topic"]}
)
# experiment_results = client.evaluate(
#     target,
#     # data="CriticBench",
#     data=client.list_examples(dataset_name="CriticBench_dataset", limit=3825),
#     evaluators=[accuracy],
#     summary_evaluators=[f1score_summary_evaluator],
#     experiment_prefix="CoopCompLLMMAS_CriticBench",
#     max_concurrency=2
#     # metadata={"topic": example.inputs["topic"]}
# )


# correctness
# from openevals.llm import create_llm_as_judge
# from openevals.prompts import CORRECTNESS_PROMPT

# def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
#     evaluator = create_llm_as_judge(
#         prompt=CORRECTNESS_PROMPT,
#         model="openai:o3-mini",
#         feedback_key="correctness",
#     )
#     eval_result = evaluator(
#         inputs=inputs,
#         outputs=outputs,
#         reference_outputs=reference_outputs
#     )
#     return eval_result


# from collections import defaultdict
# from langsmith import Client
# client = Client()
# # Load all examples from the full dataset
# all_examples = client.list_examples(dataset_name="CriticBench_dataset")
# # Group by topic
# topic_groups = defaultdict(list)
# for ex in all_examples:
#     topic = ex.inputs.get("topic", "General")
#     topic_groups[topic].append(ex)
# # Loop over each topic and evaluate separately as a new experiment
# for topic, examples in topic_groups.items():
#     experiment_name = f"CoopCompLLMMAS_{topic.replace(' ', '_')}"
#     print(f"\n🔍 Starting experiment for topic: {topic} ({len(examples)} examples)")
#     client.evaluate(
#         target,
#         data=examples,
#         evaluators=[accuracy],
#         summary_evaluators=[f1score_summary_evaluator],
#         experiment_prefix=experiment_name,
#         max_concurrency=2
#     )
#     print(f"✅ Finished experiment for topic: {topic}")




# split_limits = {
#     "CriticBench_Commonsense_Reasoning": 1129,
#     "CriticBench_Mathematical_Reasoning": 1304,
#     "CriticBench_Code_Generation": 464,
#     "CriticBench_Symbolic_Reasoning": 646,
#     "CriticBench_Algorithmic_task": 282
# }
# for split, limit in split_limits.items():
#     print(f"Processing split: {split}")
#     experiment_results = client.evaluate(
#         target,
#         # data="CriticBench",
#         data=client.list_examples(dataset_name=split, limit=limit),
#         evaluators=[accuracy],
#         summary_evaluators=[f1score_summary_evaluator],
#         experiment_prefix=f"CoopCompLLMMAS_{split}",
#         max_concurrency=2
#     )




