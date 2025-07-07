from dotenv import load_dotenv
load_dotenv()

from langsmith import Client

client = Client()
dataset = client.create_dataset(
    dataset_name="MAS Agent QA Benchmark",
    description="Testing MAS pipeline on complex multi-hop QA"
)

examples = [
    {
        "inputs": {"query": "whats phishing ?"},
        "outputs": {"answer": "Phishing is a type of cyber attack where attackers impersonate legitimate entities to trick individuals into revealing sensitive information, such as passwords or credit card numbers."},
    },
    {
        "inputs": {"query": "How can I design a multi-layered cybersecurity defense for my website using a team of specialized AI agents, each responsible for different types of threats such as phishing, malware, bot attacks, and insider threats, while ensuring low false positives and scalable real-time protection?"},
        "outputs": {"answer": "too complicated sorrry"},
    }
]

client.create_examples(dataset_id=dataset.id, examples=examples)