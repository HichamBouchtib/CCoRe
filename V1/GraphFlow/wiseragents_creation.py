from langchain_core.messages import HumanMessage, SystemMessage
from agents.wiseragent import WiserAgentsList
from llm import llm
from state import State

wiseragents_instructions = """You are tasked with creating a set of specialized AI Wiser Agents. Follow these instructions carefully:

1. Review the input topic:  
   **{topic}**
        
2. Examine any editorial feedback that has been optionally provided to guide creation of the WiserAgents:  
   **{human_wiseragent_feedback}**
    
3. Identify key subtopics based on the provided topic and feedback.

4. Select the appropriate number of subtopics (determined by your reasoning), ensuring each represents a distinct domain of expertise. 

5. Assign one WiserAgent per subtopic, with:  
   - A **clear domain specialization**  
   - A **well-defined role and responsibilities**  
   - An **initial Wisdom Score of {WS}**     

6. Return the structured output as a list of `WiserAgent` objects, following the defined schema."""

def create_wiseragents(state: State):
    
    """ Create WiserAgents """
    
    topic=state['topic']
    human_wiseragent_feedback=state.get('human_wiseragent_feedback', '')
    WS=state['WS']
        
    # Enforce structured output
    structured_llm = llm.with_structured_output(WiserAgentsList)

    # System message
    system_message = wiseragents_instructions.format(topic=topic,
                                                        human_wiseragent_feedback=human_wiseragent_feedback,
                                                        WS=WS)

    # Generate WiserAgents 
    wiseragents = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Generate the appropriate set of WiserAgents.")])
    
    # Write the list of wiseragents to state
    return {"wiseragents": wiseragents.wiseragents}
