from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict

class SearchRequest(BaseModel):
    source: Literal["web", "wikipedia"]
    content: str
    from_agent: str
    entry_idx: int
    question_idx: int

class Context(BaseModel):
    """
    Central store for search requests and results, tracking per-agent per-question context.
    """
    pending_searches: List[SearchRequest] = Field(default_factory=list)
    completed_web: Dict[str, str] = Field(default_factory=dict)
    completed_wikipedia: Dict[str, str] = Field(default_factory=dict)
    current_search: Optional[SearchRequest] = Field(None)
    awaiting_search: bool = False

    def add_search_request(self, req: SearchRequest):
        """
        Add a new search request and mark as awaiting search.
        """
        self.pending_searches.append(req)
        self.current_search = req
        self.awaiting_search = True

    def complete_search(self, result: str):
        """
        Store the search result for the current request and clear awaiting flag.
        """
        if self.current_search:
            key = f"{self.current_search.from_agent}|{self.current_search.content}"
            if self.current_search.source == "web":
                self.completed_web[key] = result
            else:
                self.completed_wikipedia[key] = result
            # remove from pending
            self.pending_searches = [r for r in self.pending_searches if r != self.current_search]
        self.current_search = None
        self.awaiting_search = False

    def get_web_context(self, agent: str, content: str) -> Optional[str]:
        """Retrieve stored web result for a given agent and question."""
        key = f"{agent}|{content}"
        return self.completed_web.get(key)

    def get_wikipedia_context(self, agent: str, content: str) -> Optional[str]:
        """Retrieve stored wikipedia result for a given agent and question."""
        key = f"{agent}|{content}"
        return self.completed_wikipedia.get(key)

    def clear_all(self):
        """Reset all search-related state."""
        self.pending_searches.clear()
        self.completed_web.clear()
        self.completed_wikipedia.clear()
        self.current_search = None
        self.awaiting_search = False

    def print_search_recap(self):
        """
        Print a summary of all searches performed, grouped by source and agent.
        """
        print("----- Search Recap -----")
        if not self.completed_web and not self.completed_wikipedia:
            print("No searches performed.")
            return
        # Recap web searches
        print("[Web]")
        for key, result in self.completed_web.items():
            agent, question = key.split("|", 1)
            print(f"'{agent}' searched for: '{question}'\n")
            print(f"Result snippet:\n{result[:200]}...\n")
        # Recap wikipedia searches
        print("[Wikipedia]")
        for key, result in self.completed_wikipedia.items():
            agent, question = key.split("|", 1)
            print(f"'{agent}' searched for: '{question}'\n")
            print(f"Result snippet:\n{result[:200]}...\n")