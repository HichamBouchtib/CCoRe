import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage
from llm import llm
from state import State
from dotenv import load_dotenv
load_dotenv()
tavily_key = os.getenv("TAVILY_API_KEY")

websearch_instructions = """transform the long question below into a small and concise query that can be used to do a web search. The quesiton: {query}"""

tavily_search = TavilySearchResults(max_results=3)

def search_web(state: State) -> None:
    """Retrieve info from the web using the state's context and store the results."""
    context = state["context"]
    req = context.current_search
    if not req or req.source != "web":
        raise ValueError("No pending web search request found")
    query = req.content
    if not query:
        raise ValueError("No search query found in the context. Please set `state.context.search_query` before calling search_web.")
    system_message = websearch_instructions.format(query=query)
    message = [SystemMessage(content=system_message)]
    keywords = llm.invoke(message)
    search_docs = tavily_search.invoke(keywords.content)

    formatted_search_docs = "\n".join(
        [
            # f'Link browsed :{doc["url"]}\nSummary of the content \n:{doc["content"]}\n'
            f'Link browsed: {doc["url"]}\n'
            for doc in search_docs
        ]
    )
    print("WebSearch result :\n",formatted_search_docs)
    context.complete_search(formatted_search_docs)
    return {"context": context}