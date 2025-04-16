import textwrap
import importlib.util
import sys
from llm import llm
from langchain_core.messages import SystemMessage
from typing import Dict, Callable, Any

def generate_tool_function(tool_name: str, task_description: str) -> Callable:
    """
    Generate a Python function dynamically using the LLM based on the task description.
    The function is compiled and assigned to the WorkerAgent.

    :param tool_name: Name of the tool function.
    :param task_description: Description of the task the function will perform.
    :return: A callable function that executes the task.
    """

    tool_generation_prompt = f"""
    Generate a Python function to perform the following task:

    **Task Name:** {tool_name}
    **Task Description:** {task_description}

    **Requirements:**
    - The function should be well-structured and follow Pythonic conventions.
    - Include necessary imports.
    - Return relevant data.

    **Expected Output Format:**
    ```python
    def {tool_name.lower()}():
        \"\"\" {task_description} \"\"\"
        # Implementation here
        return "Result"
    ```
    """

    system_message = SystemMessage(content=tool_generation_prompt)
    response = llm.invoke([system_message])

    # Extract function code
    function_code = response.content if hasattr(response, 'content') else str(response)

    # Ensure correct indentation for execution
    function_code = textwrap.dedent(function_code)

    # Create a new module for the generated function
    module_name = f"dynamic_tools_{tool_name.lower()}"
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(function_code, module.__dict__)  # Execute function code in module scope
    sys.modules[module_name] = module  # Register module in sys

    # Retrieve function from module
    generated_function = getattr(module, tool_name.lower(), None)

    if not callable(generated_function):
        raise ValueError(f"Generated function {tool_name.lower()} is not callable.")

    return generated_function