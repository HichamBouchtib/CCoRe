from typing import List, Callable, Any
from pydantic import BaseModel, Field, field_validator
from llm import llm
from langchain_core.messages import SystemMessage

class WorkerAgent(BaseModel):
    """
    WorkerAgent is responsible for executing a specialized tool to perform a well-defined sub-subtask 
    and return a structured response based on the tool's result.
    """
    name: str = Field(description="Name of the WorkerAgent.")
    subtask: str = Field(description="Specific description of the WorkerAgent's sub-task.")
    tool: Callable[..., Any] = Field(description="Function representing the WorkerAgent’s specific tool call.")
    llm: str = Field("llama3.2", description="Preferred LLM for generating responses.")
    response: str = None

    @field_validator("subtask")
    @classmethod
    def validate_subtask(cls, subtask: str) -> str:
        """Ensures the subtask description is meaningful and non-empty."""
        if not subtask or len(subtask.strip()) < 5:
            raise ValueError("subtask description must be a meaningful statement (at least 5 characters).")
        return subtask

    def execute_subtask(self, **kwargs) -> str:
        """
        Executes the WorkerAgent’s tool and generates an informed response following structured instructions.
        Args: **kwargs: Parameters required by the tool.
        Returns: str: The final structured response based on the tool's result.
        """
        try:
            tool_result = self.tool(**kwargs)
        except Exception as e:
            return f"Error executing tool for {self.name}: {str(e)}"

        # Worker execution instructions
        workerExecution_instructions = f"""
        You are **{self.name}**, a specialized WorkerAgent assigned to the following subtask:  
        **{self.subtask}**.  

        ### **Your Responsibilities:**
        - You have executed your designated tool and obtained the result:  
          **{tool_result}**
        - Generate a structured, professional response integrating the tool's output.
        - Ensure clarity, accuracy, and proper formatting.

        ### **Example Formatting:**  
        **Subtask:** "Retrieve the IP address of a device."  
        **Tool Result:** "192.168.1.10"  
        **Generated Response:** "The detected IP address is **192.168.1.10**."  

        **Now, strictly follow this format to generate your final response.**
        """

        # Bind the tool to the LLM
        llm_with_tools = llm.bind_tools([self.tool])

        # Generate response using LLM
        response = llm_with_tools.invoke([SystemMessage(content=workerExecution_instructions)])

        self.response = response.content if hasattr(response, 'content') else str(response)

        return self.response
    
class WorkerAgentsList(BaseModel):
    """
    A structured list of WorkerAgents, each responsible for a specific subtask.
    """
    worker_agents: List[WorkerAgent] = Field(
        description="Comprehensive list of WorkerAgents with their assigned subtasks and tools."
    )

# class WorkerAgentState(MessagesState):         
#     Worker_agents: Dict[str, WorkerAgent]  # Dictionary to store created WorkerAgents
#     query: str
#     tg_response: str  # Aggregated placeholder-styled response
