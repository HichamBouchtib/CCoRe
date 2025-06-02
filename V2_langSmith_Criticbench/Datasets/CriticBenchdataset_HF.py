from dotenv import load_dotenv
load_dotenv()
from langsmith import Client
from datasets import load_dataset
import os

os.environ["HF_TOKEN"] = "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"
datasetCriticBench = load_dataset("llm-agents/CriticBench")
# print(datasetCriticBench["test"][0])

client = Client()

# Create dataset in LangSmith
dataset = client.create_dataset(
    dataset_name="CriticBench",
    description="Questions from CriticBench (test set)"
)

# Prepare examples
examples = []
for item in datasetCriticBench["test"]:
    examples.append({
        "inputs": {
            "query": item["question"],
            "topic": item.get("question_type", "General")
        },
        "outputs": {"answer": item["response"]}
    })

# Upload examples
client.create_examples(dataset_id=dataset.id, examples=examples)
print(f"✅ Uploaded {len(examples)} examples to LangSmith.")