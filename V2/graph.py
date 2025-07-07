from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from GraphFlow.wiseragents_creation import create_wiseragents
from GraphFlow.TG_generation import generate_task_graphs
from state import State
from GraphFlow.DeepMonologue.generate_interv_qsts import generate_questions
from tools.searchweb_tool import search_web
from tools.searchwikipedia_tool import search_wikipedia
from GraphFlow.DeepMonologue.generate_Interv_anwrs import generate_answers
from GraphFlow.ConditionalEdges.should_search import should_search
from GraphFlow.TGvoting import vote_TG
from GraphFlow.human_feedback import human_feedback
from GraphFlow.ConditionalEdges.should_continue import should_continue
from GraphFlow.answer_user import answer_user
from GraphFlow.ConditionalEdges.choose_Agentmode import choose_AgentMode
from GraphFlow.agent_mode import agent_mode
from GraphFlow.tool_calling import tool_calling
from GraphFlow.TG_refining import refine_task_graphs
# from tools.mcp_tools import mcp_tool

# Nodes
builder = StateGraph(State)
builder.add_node("create_wiseragents", create_wiseragents)
builder.add_node("human_feedback", human_feedback)
builder.add_node("agent_mode", agent_mode)
builder.add_node("generate_TG", generate_task_graphs)
builder.add_node("generate_questions", generate_questions)
builder.add_node("tool_calling", tool_calling)
builder.add_node("generate_answers", generate_answers)
builder.add_node("search_web", search_web)
builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("TG_refining", refine_task_graphs)
builder.add_node("vote_TG", vote_TG)
builder.add_node("answer_user", answer_user)
# builder.add_node("mcp_node", mcp_node)

# Edges
builder.add_edge(START, "create_wiseragents")
builder.add_edge("create_wiseragents", "human_feedback")
builder.add_conditional_edges("human_feedback", should_continue, ["create_wiseragents", "agent_mode"])
builder.add_conditional_edges("agent_mode", choose_AgentMode, ["answer_user", "generate_TG"])
builder.add_edge("generate_TG", "generate_questions")
builder.add_edge("generate_questions", "tool_calling")
builder.add_conditional_edges("tool_calling", should_search, ["search_web", "search_wikipedia", "generate_answers"])
# builder.add_conditional_edges("tool_calling", should_tool, ["search_web", "search_wikipedia", "mcp_tool","generate_answers"])
# builder.add_edge("mcp_tool", "generate_answers")
builder.add_edge("search_web", "generate_answers")
builder.add_edge("search_wikipedia", "generate_answers")
builder.add_edge("generate_answers", "TG_refining")
builder.add_edge("TG_refining", "vote_TG")
builder.add_edge("vote_TG", "answer_user")
builder.add_edge("answer_user", END)

memory = MemorySaver()

graph = builder.compile(checkpointer=memory).with_config(run_name="DeepMonologue")

# Pausing graph and resume later
# graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory).with_config(run_name="monologue")
# graph.resume(memory)