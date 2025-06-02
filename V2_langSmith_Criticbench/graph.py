from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from GraphFlow.wiseragents_creation import create_wiseragents
from GraphFlow.TG_generation import generate_task_graphs
from state import State
from GraphFlow.generate_interv_qsts import generate_questions
from tools.searchweb_tool import search_web
from tools.searchwikipedia_tool import search_wikipedia
from GraphFlow.generate_Interv_anwrs import generate_answers
from GraphFlow.ConditionalEdges.should_search import should_search
from GraphFlow.TGvoting import vote_TG
from GraphFlow.human_feedback import human_feedback
from GraphFlow.ConditionalEdges.should_continue import should_continue
from GraphFlow.correction_response import Correction_response
from GraphFlow.tool_calling import tool_calling
from GraphFlow.user_query import user_query
from GraphFlow.Initialize_state import initialize_state
from GraphFlow.TG_refining import refine_task_graphs
from GraphFlow.generation_response import Generation_response
from GraphFlow.critique_response import Critique_response

# Nodes
builder = StateGraph(State)

builder.add_node("initialize_state", initialize_state)
builder.add_node("create_wiseragents", create_wiseragents)
builder.add_node("human_feedback", human_feedback)
builder.add_node("user_query", user_query)
builder.add_node("G_response", Generation_response)
# builder.add_node("G_response", answer_user)
builder.add_node("generate_TG", generate_task_graphs)
builder.add_node("generate_questions", generate_questions)
builder.add_node("tool_calling", tool_calling)
builder.add_node("generate_answers", generate_answers)
builder.add_node("Q_response", Critique_response)
builder.add_node("search_web", search_web)
builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("TG_refining", refine_task_graphs)
builder.add_node("vote_TG", vote_TG)
builder.add_node("C_response", Correction_response)

# Edges
builder.set_entry_point("initialize_state")
builder.add_edge("initialize_state", "create_wiseragents")
# builder.add_edge("create_wiseragents", "human_feedback")
# builder.add_conditional_edges("human_feedback", should_continue, ["create_wiseragents", "user_query"])
builder.add_edge("create_wiseragents", "user_query")
builder.add_edge("user_query", "G_response")
builder.add_edge("G_response", "generate_TG")
builder.add_edge("generate_TG", "generate_questions")
builder.add_edge("generate_questions", "tool_calling")
builder.add_conditional_edges("tool_calling", should_search, ["search_web", "search_wikipedia", "generate_answers"])
builder.add_edge("search_web", "generate_answers")
builder.add_edge("search_wikipedia", "generate_answers")
builder.add_edge("generate_answers", "Q_response")
builder.add_edge("Q_response", "TG_refining")
builder.add_edge("TG_refining", "vote_TG")
builder.add_edge("vote_TG", "C_response")
memory = MemorySaver()

# LangSmith
# graph = builder.compile(checkpointer=memory).with_config(run_name="DeepMonologue")


graph = builder.compile().with_config(run_name="DeepMonologue")
# graph = builder.compile(interrupt_before=["user_query"], checkpointer=memory).with_config(run_name="DeepMonologue")


