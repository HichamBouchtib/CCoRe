import os
from langchain_ollama import ChatOllama

class LLMManager:
    """
    Manages different ChatOllama models and allows dynamic switching
    (self-adaptive agent) to find the best model for each domain.
    """
    def __init__(self, model_name: str = "llama2-7b"):
        """
        :param model_name: The default Llama model to load.
        """
        self.model_name = model_name
        self._llm_instance = None
        self._load_model()

    def _load_model(self):
        """Load or reload the ChatOllama model."""
        self._llm_instance = ChatOllama(model=self.model_name, temperature=0.2)

    def switch_model(self, new_model_name: str):
        """
        WiserAgent can call this method to load a more domain-specific model.
        e.g. 'llama2-13b-finetuned-cybersecurity'
        """
        if new_model_name != self.model_name:
            self.model_name = new_model_name
            self._load_model()
    # A simple predict function for demonstration.
    def predict(self, prompt: str) -> str:
        if not self._llm_instance:
            self._load_model()
        response = self._llm_instance.invoke(prompt)
        return response.content if response else ""
