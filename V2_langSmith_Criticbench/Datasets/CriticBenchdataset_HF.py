from dotenv import load_dotenv
load_dotenv()
from langsmith import Client
from datasets import load_dataset
import os

os.environ["HF_TOKEN"] = "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"
datasetCriticBench = load_dataset("llm-agents/CriticBench")
client = Client()
dataset = client.create_dataset(
    dataset_name="CriticBench_dataset",
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
        "outputs": {"answer": item["response"]},
        "metadata": {
            "topic": item.get("question_type", "General")
        }
    })

# from langsmith.schemas import ExampleCreate
# from collections import defaultdict
# # Load your examples from LangSmith base dataset
# all_examples = client.list_examples(dataset_name="CriticBench_dataset")

# # Group by topic
# topic_groups = defaultdict(list)
# for ex in all_examples:
#     topic = ex.inputs.get("topic", "General")
#     topic_groups[topic].append(ex)

# # Create new datasets and convert examples to ExampleCreate
# for topic, examples in topic_groups.items():
#     dataset_name = f"CriticBench_{topic.replace(' ', '_')}"
#     client.create_dataset(dataset_name=dataset_name)

#     # Convert each example to ExampleCreate format
#     converted = [
#         ExampleCreate(inputs=ex.inputs, outputs=ex.outputs, metadata=ex.metadata)
#         for ex in examples
#     ]

#     # Upload
#     client.create_examples(dataset_name=dataset_name, examples=converted)
#     print(f"✅ Created dataset split: {dataset_name} with {len(converted)} examples")

# Upload examples
client.create_examples(dataset_id=dataset.id, examples=examples)
print(f"✅ Uploaded {len(examples)} examples to LangSmith.")