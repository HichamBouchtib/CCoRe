import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain_community.document_loaders import WikipediaLoader
from state import State
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from llm import llm


wikipedia_summary_instructions = """You are a knowledgeable and organized summarizer AI.

Your task is to process Wikipedia-like documents retrieved from a search and output a clear and well-structured summary based on the following instructions:

1. Read and understand the full content.
2. Remove duplicated or redundant sections.
3. Clean up any malformed formatting (e.g., mathematical symbols, excessive newlines, headers like == Title ==).
4. Organize the summary in sections with clear titles (e.g., Introduction, Key Concepts, Applications, etc.).
5. Ensure the final output is readable, compact, and suitable for display in an AI assistant interface.

The content to summarize is based on the query: "{query}".
"""

def search_wikipedia(state: State) -> None:
    """Retrieve info from Wikipedia using the state's context and store the results."""
    
    context = state["context"]
    req = context.current_search
    if not req or req.source != "wikipedia":
        raise ValueError("No pending web search request found")
    query = req.content
    print("WiserAgent Query (Wikipedia):", query)

    if not query:
        raise ValueError("No search query found in the context. Please set `state.context.search_query` before calling search_wikipedia.")

    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    search_text = "\n\n".join(doc.page_content for doc in search_docs)
    system_message = wikipedia_summary_instructions.format(query=query)
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=search_text)
    ]

    summary = llm.invoke(messages)
    for doc in search_docs:
            print("Document source :", doc.metadata["source"])
    print("Summary of the WikipediaSearch Result :", summary.content)
    tool_call_id = f"search_wiki_{len(state['messages'])}"
    state["messages"].append(
        AIMessage(
            content=f"📚 Wikipedia search completed by `{req.from_agent}` for the query: `{req.content}`",
            tool_calls=[
                {
                    "name": "search_result_summary",
                    "args": {
                        "result_type": "wikipedia",
                        "source": "Wikipedia",
                        "content": summary.content,
                        "agent": req.from_agent,
                        "query": req.content
                    },
                    "id": tool_call_id
                }
            ]
        )
    )
    context.complete_search(summary.content)
    return {"context": context}

# # test
# state = get_current_state(topic="Reinforcement Learning")
# state["context"].search_query = "History and applications of reinforcement learning"
# search_wikipedia(state)
