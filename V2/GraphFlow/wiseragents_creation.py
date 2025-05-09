import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain_core.messages import HumanMessage, SystemMessage
from agents.wiseragent import WiserAgentsList, display_wiseragents, visualize_agents_pyvis
from llm import llm
from state import State
from IPython.display import display, IFrame

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

def create_wiseragents(state: State):

    """ Create WiserAgents """

    if state["last_topic"] == state["topic"] and state["wiseragents"]:
        print("✅ Skipping WiserAgent generation.")
        return state
    
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

    # Add interactive widget display here
    display_wiseragents(wiseragents_list)
    # agent in chrome
    iframe = visualize_agents_pyvis(wiseragents_list)
    display(iframe)
    
    # # printing
    # for agent in wiseragents_output.wiseragents:
    #     print(agent.persona)
    #     print("-" * 50)
    
    print("WiserAgents generated:")
    return {
        **state,
        "wiseragents": wiseragents_list,
        "last_topic": topic
    }

# test_state = {
#     "topic": "AI in Cyberattacks",
#     "human_wiseragent_feedback": "Nothing for now",
#     "WS": 50
# }

# result = create_wiseragents(test_state)
# print("\n WiserAgents Summary:\n")
# for agent in result["wiseragents"]:
#     print(f"Name: {agent.name}")
#     print(f"Expertise: {agent.domain_expertise}")
#     print(f"Description: {agent.description}")
#     print(f"Wisdom Score: {agent.WS}")
#     print(f"Preferred LLM: {agent.preferred_llm}")
#     print("-" * 40)

# check whether objects are instances of WiserAgent
# print("\n Parsed WiserAgents Objects:\n")
# for agent in result["wiseragents"]:
#     print(f" This is a {type(agent)} instance")
#     print(f" {agent.name} | {agent.domain_expertise}")