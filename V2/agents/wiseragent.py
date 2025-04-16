from typing import List
from pydantic import BaseModel, Field, field_validator


class WiserAgent(BaseModel):
    name: str = Field(
        description="Name of the WiserAgent."
    )
    domain_expertise: str = Field(
        description="Which domain or topic the agent covers."
    )
    description: str = Field(
        description="Description of the wiseragent focus, concerns, skills and motives.",
    )
    WS: int = Field(default=50,
        description="Tracks the agent's accumulated wisdom score based on its task performance."
    )
    preferred_llm: str = Field(default="qwen2.5:latest",
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
    
    # This is the method that will allow the agent to be serialized into JSON
    def json(self, *args, **kwargs):
        # You can customize this if needed, but Pydantic already handles it well
        return super().json(*args, **kwargs)
    
class WiserAgentsList(BaseModel):
    wiseragents: List[WiserAgent] = Field(
        description="Comprehensive list of WiserAgents with their domain expertise.",
    )