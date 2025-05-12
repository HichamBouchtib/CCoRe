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
from GraphFlow.user_query import user_query
# from tools.mcp_tools import mcp_tool
from GraphFlow.Initialize_state import initialize_state

# Nodes
builder = StateGraph(State)

builder.add_node("initialize_state", initialize_state)
builder.add_node("create_wiseragents", create_wiseragents)
builder.add_node("human_feedback", human_feedback)
# builder.add_node("agent_mode", agent_modeLang)
builder.add_node("user_query", user_query)
builder.add_node("agent_mode_logic", agent_mode)
builder.add_node("agent_mode_router", lambda state: state)  # empty router node (no logic)
builder.add_node("generate_TG", generate_task_graphs)
builder.add_node("generate_questions", generate_questions)
builder.add_node("tool_calling", tool_calling)
builder.add_node("generate_answers", generate_answers)
builder.add_node("search_web", search_web)
builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("vote_TG", vote_TG)
builder.add_node("answer_user", answer_user)
# builder.add_node("mcp_node", mcp_node)

# Edges
builder.set_entry_point("initialize_state")
# builder.add_edge(START, "initialize_state")

builder.add_edge("initialize_state", "create_wiseragents")
builder.add_edge("create_wiseragents", "human_feedback")
# builder.add_conditional_edges("human_feedback", should_continue, ["create_wiseragents", "agent_mode"])
# builder.add_conditional_edges("agent_mode", choose_AgentMode, ["answer_user", "generate_TG"])
builder.add_conditional_edges("human_feedback", should_continue, ["create_wiseragents", "user_query"])
builder.add_edge("user_query", "agent_mode_logic")
builder.add_edge("agent_mode_logic", "agent_mode_router")
builder.add_conditional_edges("agent_mode_router", choose_AgentMode, ["answer_user", "generate_TG"])
builder.add_edge("generate_TG", "generate_questions")
builder.add_edge("generate_questions", "tool_calling")
builder.add_conditional_edges("tool_calling", should_search, ["search_web", "search_wikipedia", "generate_answers"])
# builder.add_edge("mcp_tool", "generate_answers")
builder.add_edge("search_web", "generate_answers")
builder.add_edge("search_wikipedia", "generate_answers")
builder.add_edge("generate_answers", "vote_TG")
builder.add_edge("vote_TG", "answer_user")
builder.add_edge("answer_user", END)

memory = MemorySaver()

# graph = builder.compile(checkpointer=memory).with_config(run_name="DeepMonologue")

# Pausing graph
graph = builder.compile(interrupt_before=["human_feedback", "user_query"], checkpointer=memory).with_config(run_name="DeepMonologue")

# graph.resume(memory)
