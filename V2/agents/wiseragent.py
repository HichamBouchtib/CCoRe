from typing import List
from pydantic import BaseModel, Field, field_validator
from ipywidgets import VBox, Accordion, HTML
from pyvis.network import Network
from IPython.display import display, IFrame

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
    
    # JSON serialization method
    def json(self, *args, **kwargs):
        return super().json(*args, **kwargs)
    
class WiserAgentsList(BaseModel):
    wiseragents: List[WiserAgent] = Field(
        description="Comprehensive list of WiserAgents with their domain expertise.",
    )

# Vizualize WiserAgents in a card format
def display_wiseragents(wiseragents_list):
    """ Display WiserAgents in a card format """
    widgets = []
    for agent in wiseragents_list:
        html = HTML(f"""
        <b>Name:</b> {agent.name}<br>
        <b>Expertise:</b> {agent.domain_expertise}<br>
        <b>Description:</b> {agent.description}<br>
        <b>Wisdom Score:</b> {agent.WS}<br>
        <b>Preferred LLM:</b> {agent.preferred_llm}
        """)
        widgets.append(html)

    accordion = Accordion(children=widgets)
    for i, agent in enumerate(wiseragents_list):
        accordion.set_title(i, agent.name)

    display(accordion)

def visualize_agents_pyvis(wiseragents):
    net = Network(height="600px", width="100%", notebook=True, cdn_resources='remote')
    for agent in wiseragents:
        net.add_node(agent.name, 
                     title=f"{agent.description}<br>WS: {agent.WS}<br>LLM: {agent.preferred_llm}",
                     label=f"{agent.name} (WS: {agent.WS})")
    net.show_buttons(filter_=['physics'])
    net.show("agents/wiseragents_pyvis.html")
    return IFrame(src="agents/wiseragents_pyvis.html", width="100%", height="650px")
