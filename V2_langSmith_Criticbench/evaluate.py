from dotenv import load_dotenv
load_dotenv()
from state import State, Context
from graph import graph
from langsmith import Client, EvaluationResult, traceable
from pydantic import BaseModel, Field, field_validator
from llm import llm

GC_scoring_instructions = """"
You are an expert in evaluating AI-generated responses as "Wrong" or "True".
Your task is to verify whether the response is correct in compraison with a reference response.
the agent response: {response}
the reference response: {reference_response}
"""

gold_label = """
You are a judge. Is the following response correct?
Response: {G}
Reference: {expected}
Answer only with True (correct) or False (wrong).
"""

critique_label = """
You are a judge. Your job is answer with True or False.

If the Critique contains at least one valid flaw or improvement related to the Generation, answer: True

If the Critique is not helpful or fails to correctly criticize the Generation, answer: False

Critique:
{Q}

Generation:
{G}

Respond Only with True or False 
Never respond with null, None, or any other value.
"""

# Global counters
G_accuracy = 0
C_accuracy = 0
TP = 0
FP = 0
FN = 0

class BinaryScore(BaseModel):
    score: bool = Field(description="True if the response is correct, False if it is wrong.")

@traceable(name="criticbench", metadata={"llm": "qwen2.5"})
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

def accuracy(inputs: dict, outputs: dict, reference_outputs: dict):
    
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
        {"key": "G_score", "score": G_score},
        {"key": "C_score", "score": C_score}
        ]

def f1score(inputs: dict, outputs: dict, reference_outputs: dict):
    Q = outputs.get("Q_answer", "").strip().lower()
    G = outputs.get("G_answer", "").strip().lower()
    expected = reference_outputs.get("answer", "").strip().lower()

    structured_llm = llm.with_structured_output(BinaryScore)
    gold_label_instructions = gold_label.format(G=G, expected=expected)
    # Use LLM to judge if G is wrong (gold label)
    is_G_wrong = structured_llm.invoke(gold_label_instructions)
    is_G_wrong = is_G_wrong.score
    print("is_G_wrong :", is_G_wrong)
    if is_G_wrong in ["True", "true", True]:
        is_G_wrong = 1
    if is_G_wrong in["Wrong", "wrong", False]:
        is_G_wrong = 0


    structured_llm = llm.with_structured_output(BinaryScore)
    critique_label_instructions = critique_label.format(G=G, Q=Q)
    print("critique_label_instructions :", critique_label_instructions)
    # Use LLM to judge if Q correctly criticized it

    Q_identified_wrong = None
    max_retries = 10

    for attempt in range(max_retries):
        try:
            response = structured_llm.invoke(critique_label_instructions)
            print(f"LLM response (attempt {attempt + 1}):", response)

            if response is not None and hasattr(response, "score") and response.score is not None:
                Q_identified_wrong = response.score
                break
            else:
                print(f"⚠️ Attempt {attempt + 1}: score is None or invalid response.")

        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed with exception: {e}")

    print(" .score :", Q_identified_wrong)
    if Q_identified_wrong in ["True", "true", True]:
        Q_identified_wrong = 1
    if Q_identified_wrong in["Wrong", "wrong", False]:
        Q_identified_wrong = 0

    global TP, FP, FN
    TP = 0
    FP = 0
    FN = 0

    if Q_identified_wrong and is_G_wrong:
        TP += 1  # correct discrimination
    elif Q_identified_wrong and not is_G_wrong:
        FP += 1  # criticized a correct response
    elif not Q_identified_wrong and is_G_wrong:
        FN += 1  # missed a wrong response

    return [
        {"key": "is_G_wrongScore", "score": is_G_wrong},
        {"key": "Q_identified_wrongScore", "score": Q_identified_wrong},
        # {"key": "Expected", "score": expected},
        {"key": "TP", "score": TP},
        {"key": "FP", "score": FP},
        {"key": "FN", "score": FN}
    ]

client = Client()

experiment_results = client.evaluate(
    target,  # your traced MAS function
    # data="CriticBench",
    data=client.list_examples(dataset_name="CriticBench", limit=2),
    evaluators=[accuracy, f1score],
    experiment_prefix="CriticBench",
    max_concurrency=2,
)

# G accuracy Calculation
# G_finalscore = G_accuracy/3825
G_finalscore = G_accuracy/1
print(f"G Final Score: {G_finalscore}")

# C accuracy Calculation
# C_finalscore = C_accuracy/3825
C_finalscore = C_accuracy/1
print(f"C Final Score: {C_finalscore}")

# F1 Score Calculation
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print(f"Q F1 Score:  {f1_score:.3f}")

# import json
# from agentevals.trajectory.llm import (create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE)
# evaluator = create_trajectory_llm_as_judge(
#     prompt=TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
#     model="openai:o3-mini"
# )
