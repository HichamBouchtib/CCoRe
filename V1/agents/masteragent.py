from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, field_validator
from llm import llm
from langchain_core.messages import SystemMessage
from typing import List

class MasterAgent(BaseModel):
    name: str = Field(
        description="Name of the MasterAgent."
    )
    task: str = Field(
        description="Description of the Masteragent focus, concerns, skills and motives.",
    )
    preferred_llm: str = Field(default="llama3.2",
        description="Specifies the preferred LLM depending on its sub-domain expertise.",
    )
    WS: int = Field(default=20,
        description="Tracks the agent's accumulated wisdom score based on its task performance."
    )
    # knowledge_graph: Dict[str, str] = Field(default_factory=dict, 
    #     description="Structured knowledge graph for fast retrieval and representation."
    # )
    
    @property
    def persona(self) -> str:
        return (
            f"Name: {self.name}\n"
            f"Description: {self.task}\n"
            f"Preferred LLM: {self.preferred_llm}\n"
            f"Wisdom Score: {self.WS}\n"
        )
    
    @field_validator("WS")
    def validate_wisdom_score(cls, value):
        if value < 0:
            raise ValueError("Wisdom Score (WS) cannot be negative.")
        return value
    
    def generate_response(self):
        """
            Generates a placeholder-styled response
            :return: A formatted placeholder-styled string.
        """
        placeholder_instructions = """You are a MasterAgent named {name} specializing in executing this specific task {task} within a larger operation.  

        ### **Your Responsibilities:**
        1. **Generate a structured response** that represents the expected task outcome.
        2. **Define the necessary WorkerAgents** required to complete the task.
        3. **Insert placeholders** in the response for missing information, formatted as:  
        **[PLACEHOLDER for <WorkerAgent Name> response]**, where you determine the WorkerAgent's role.

        ### **Response Guidelines:**
        - Maintain a clear and professional tone.
        - Ensure the response logically integrates the WorkerAgents’ results.

        ### **Example (Port Scanner MasterAgent):**
        **MasterAgent's Task:** "Scan network ports for vulnerabilities."  
        **Generated Response:**  
        "The network checking results for the IP **[PLACEHOLDER for IP Fetcher Agent response]** after being scanned are **[PLACEHOLDER for Port Scanner Agent response]**."

        Each MasterAgent tailors their response based on their expertise while delegating detailed computations to WorkerAgents.
        """

        system_message = placeholder_instructions.format(name=self.name, task=self.task)
        
        response = llm.invoke([SystemMessage(content=system_message)])
    
        return response

class MasterAgentsList(BaseModel):
    Masteragents: List[MasterAgent] = Field(
        description="Comprehensive list of MasterAgents with their sub-domain expertise.",
    )

# class MasterAgentState(MessagesState):         
#     tg_candidates: List[dict]  # Ensure this is a list of dictionaries (TaskGraph representations)
#     master_agents: Dict[MasterAgent]  # Dictionary to store created MasterAgents
#     query: str
#     tg_chosen: dict
#     tg_response: str # the aggregated placeholder-styled response