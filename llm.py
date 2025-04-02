from langchain_ollama import ChatOllama

local_llm = "llama3.2:3b-instruct-fp16"
# local_llm = "qwen2.5:3b"
llm = ChatOllama(model=local_llm, temperature=0)
