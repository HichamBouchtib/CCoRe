# Wikipedia search tool
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import SystemMessage
from llm import llm
from state import State
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

# Search query writing
search_instructions = SystemMessage(content=f"""You will be given a interview conversation between WiserAgent generating a Task Graph to solve and answer a user's query and a WiserAgent that is questioning the Task Graph.

Your goal is to generate a well-structured query for use in retrieval related to the conversation.
        
First, analyze the full conversation.

Pay particular attention to the question.

Convert this final question into a well-structured wikipedia search query in order to help the TG owner answer the question and refine its TG if necessary""")

def search_wikipedia(state: State):
    
    """ Retrieve docs from wikipedia """

    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions]+state['questions'])
    
    # Search
    search_docs = WikipediaLoader(query=search_query.search_query, 
                                  load_max_docs=2).load()

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 