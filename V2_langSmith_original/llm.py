from langchain_ollama import ChatOllama
# from langsmith import traceable
local_llm = "qwen2.5:latest"
# local_llm = "llama3.3:latest "
# llm = ChatGroq(model="llama3-8b-8192")

llm = ChatOllama(model=local_llm, temperature=0)


# Streaming::
# for chunk in llm.stream(messages):
#     print(chunk.text(), end="")

# Asynchronous call:
# await llm.ainvoke(messages)

# Abatch to reduce inference time when sending multiple messages at the same time:
# messages = [
#     ("human", "Say hello world!"),
#     ("human","Say goodbye world!")
# ]
# await llm.abatch(messages)

# JSON mode
# json_llm = ChatOllama(format="json")

# ToolCalling LLM
# from langchain_ollama import ChatOllama
# from pydantic import BaseModel, Field

# class Multiply(BaseModel):
#     a: int = Field(..., description="First integer")
#     b: int = Field(..., description="Second integer")

# ans = await chat.invoke("What is 45*67")
# ans.tool_calls