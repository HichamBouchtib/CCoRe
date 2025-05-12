import pprint
import sys
import os
import uuid
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain_core.messages import HumanMessage, SystemMessage
from agents.wiseragent import WiserAgentsList
from llm import llm
from state import State
from langchain_core.messages import HumanMessage, AIMessage
# import logging
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# # Then use
# logger.info("WiserAgents generated")

wiseragents_instructions = """You are tasked with creating a set of specialized AI Wiser Agents. Follow these instructions carefully:

1. Review the input topic:  
   **{topic}**
        
2. Examine any editorial feedback that has been optionally provided to guide creation of the WiserAgents:  
   **{human_wiseragent_feedback}**
    
3. Identify key subtopics based on the provided topic and feedback.

4. Select the appropriate number of subtopics (determined by your reasoning), ensuring each represents a distinct domain of expertise. 

5. Assign one WiserAgent per subtopic:     
6. Return the structured output as a list of `WiserAgent` objects, following this defined schema.
[
    {{
        'name': 'Name of the agent',
        'domain_expertise': 'Area of knowledge',
        'description': 'Short explanation of the agent’s role or specialization',
        'WS': {WS}
    }},
]
"""

def create_wiseragents(state: dict) -> dict:
    """ Create WiserAgents """

    feedback = state.get("human_wiseragent_feedback", "").strip()
    state["messages"] = [
        m for m in state["messages"]
        if not (isinstance(m, HumanMessage) and m.additional_kwargs.get("tag") == "system-feedback")
    ]
    if feedback:
        state["messages"].append(HumanMessage(content=feedback))


    if not state.get("topic") or not state["topic"].strip():
        print("No valid topic detected.")
        return {
            **state,
            "__end__": True
        }

    # Show progress before actual generation
    state.setdefault("messages", []).append(
        AIMessage(content="WiserAgents generating successfully")
    )
    print("Generating WiserAgents...\n")
    
    topic=state['topic']
    human_wiseragent_feedback=state.get('human_wiseragent_feedback', '')
    WS = state.get('WS', 50)

    system_message = wiseragents_instructions.format(topic=topic,
                                                        human_wiseragent_feedback=human_wiseragent_feedback,
                                                        WS=WS)
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content="Generate the appropriate set of WiserAgents.")
    ]
    structured_llm = llm.with_structured_output(WiserAgentsList)
    wiseragents_output = structured_llm.invoke(messages)

    # handle it if the output is a tuple
    if isinstance(wiseragents_output, tuple):
        wiseragents_output = wiseragents_output[0]

    # Extract the list from WiserAgentsList
    wiseragents_list = wiseragents_output.wiseragents if hasattr(wiseragents_output, "wiseragents") else wiseragents_output

    summary = "\n".join(
    f"- 🤖 {agent.name} ({agent.domain_expertise}): {agent.description} [WS={agent.WS}]"
    for agent in wiseragents_list
    )
    print("WiserAgents created successfully!: ", summary)

    # Reponse without Tool call metadata
    # response_message = f"**WiserAgents created for your topic: \n\n{summary}"
    # state.setdefault("messages", []).append(AIMessage(content=response_message))

   
    # Reponse with Tool call metadata
    tool_call_id = str(uuid.uuid4())
    state["messages"].append(
        AIMessage(
            content="✅ WiserAgents Generation...",
            tool_calls=[
                {
                    "name": "wiseragent_summary",
                    "args": {
                        "agents": [
                            {
                                "name": agent.name,
                                "domain_expertise": agent.domain_expertise,
                                "description": agent.description,
                                "WS": agent.WS
                            }
                            for agent in wiseragents_list
                        ]
                    },
                    "id": tool_call_id
                }
            ]
        )
    )
    state["messages"].append(
        AIMessage(content="Do you have any feedback regarding your WiserAgent ?")
    )

    # After using the feedback, reset it
    state["human_wiseragent_feedback"] = ""
    state["feedback_handled"] = False
    return {
        **state,
        "wiseragents": wiseragents_list,
        "topic": topic
    }
