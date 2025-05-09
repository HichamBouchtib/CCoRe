from langchain_ollama import ChatOllama

local_llm = "qwen2.5:latest"
# local_llm = "qwen2.5:3b"
llm = ChatOllama(model=local_llm, temperature=0)