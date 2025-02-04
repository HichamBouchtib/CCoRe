import requests
from llama_index.llms import CustomLLM

class OllamaLLM(CustomLLM):
    def __init__(self, server_url: str):
        self.server_url = server_url

    def generate(self, prompt: str) -> str:
        data = {
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(self.server_url, json=data)
        response.raise_for_status()
        return response.json().get("response", "")

# Instantiate the LLM
ollama_llm = OllamaLLM(server_url="http://127.0.0.1:5000/generate")
