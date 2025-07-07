# Web search tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage
from llm import llm
from state import State
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

tavily_search = TavilySearchResults(max_results=3)

# Search query writing
search_instructions = SystemMessage(content=f"""You will be given a conversation between a WiserAgent and an other WiserAgent that generated a Task Graph to solve and answer a user's query. 

Your goal is to generate a well-structured query for use in web-search related to the conversation.
        
First, analyze the full conversation.

Pay particular attention to the final question posed by the WiserAgent.

Convert this final question into a well-structured web search query""")

def search_web(state: State):
    
    """ Retrieve from web search """

    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions]+state['questions'])
    
    # Search
    search_docs = tavily_search.invoke(search_query.search_query)

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 
