from typing import List
from pydantic import BaseModel, Field, field_validator


class WiserAgent(BaseModel):
    name: str = Field(
        description="Name of the WiserAgent."
    )
    WS: int = Field(default=50,
        description="Tracks the agent's accumulated wisdom score based on its task performance."
    )
    domain_expertise: str = Field(
        description="Which domain or topic the agent covers."
    )
    description: str = Field(
        description="Description of the wiseragent focus, concerns, skills and motives.",
    )
    preferred_llm: str = Field(default="llama3.2",
        description="Specifies the preferred LLM depending on its domain expertise.",
    )
    # knowledge_graph: Dict[str, str] = Field(default_factory=dict, 
    #     description="Structured knowledge graph for fast retrieval and representation."
    # )
    
    @property
    def persona(self) -> str:
        return (
            f"Name: {self.name}\n"
            f"Domain Expertise: {self.domain_expertise}\n"
            f"Description: {self.description}\n"
            f"Wisdom Score: {self.WS}\n"
            f"Preferred LLM: {self.preferred_llm}\n"
        )
    @field_validator("WS")
    def validate_wisdom_score(cls, value):
        if value < 0:
            raise ValueError("Wisdom Score (WS) cannot be negative.")
        return value

class WiserAgentsList(BaseModel):
    wiseragents: List[WiserAgent] = Field(
        description="Comprehensive list of WiserAgents with their domain expertise.",
    )

# class WiserAgentsState(TypedDict):
#     topic: str # Query topic
#     human_wiseragent_feedback: str # Human feedback
#     wisdom_score: int # Tracks WS of each agent
#     wiseragents: List[WiserAgent] # WiserAgents asking questions
#     # task_graph_themes: List[str]  # Stores themes proposed for Task Graphs