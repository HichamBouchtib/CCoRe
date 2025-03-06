from typing import List
import operator
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

search_instructions = SystemMessage(content="""You will be given a conversation between an agent and a user.

Your goal is to generate a well-structured query for use in retrieval or web-search related to the conversation.

1. Analyze the full conversation.
2. Pay particular attention to the final question posed by the user or agent.
3. Convert this final question into a well-structured web or Wikipedia search query.
""")

tavily_search = TavilySearchResults(max_results=3)

def search_web(state):
    """
    Node function for searching the web via TavilySearch.
    Expects 'state' to have:
      - 'messages': a list of messages (HumanMessage, AIMessage, etc.)
      - 'llm': an LLM with a .with_structured_output() method
    """
    llm = state["llm"] 
    messages = state["messages"]

    structured_llm = llm.with_structured_output(SearchQuery)
    
    search_query = structured_llm.invoke([search_instructions] + messages)
    
    results = tavily_search.invoke(search_query.search_query)
    
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in results
        ]
    )
    return {"context": [formatted_search_docs]}

def search_wikipedia(state):
    """
    Node function for searching Wikipedia.
    Expects 'state' to have:
      - 'messages': a list of messages
      - 'llm': an LLM with a .with_structured_output() method
    """
    llm = state["llm"]
    messages = state["messages"]

    structured_llm = llm.with_structured_output(SearchQuery)
    
    search_query = structured_llm.invoke([search_instructions] + messages)
    
    loader = WikipediaLoader(query=search_query.search_query, load_max_docs=2)
    wiki_docs = loader.load()

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in wiki_docs
        ]
    )
    return {"context": [formatted_search_docs]}
