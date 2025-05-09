# from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from GraphFlow.wiseragents_creation import create_wiseragents
from GraphFlow.TG_generation import generate_task_graphs
from state import State
from GraphFlow.TGInterview.generate_interv_qsts import generate_questions
from tools.searchweb_tool import search_web
from tools.searchwikipedia_tool import search_wikipedia
from GraphFlow.TGInterview.generate_Interv_anwrs import generate_answers
from GraphFlow.TGInterview.save_interview import save_interview
from GraphFlow.route_messages import route_messages
from GraphFlow.TGvoting import vote_TG
# from GraphFlow.filter_messages import filter_messages
from GraphFlow.human_feedback import human_feedback
from GraphFlow.should_continue import should_continue

# WiserAgent node stage
builder = StateGraph(State)
builder.add_node("create_wiseragents", create_wiseragents)
builder.add_node("human_feedback", human_feedback)
# TG node stage
builder.add_node("generate_TG", generate_task_graphs)
builder.add_node("generate_questions", generate_questions)
builder.add_node("search_web", search_web)
builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("generate_answers", generate_answers)
builder.add_node("save_interview", save_interview)
builder.add_node("vote_TG", vote_TG)

# WiserAgent edge stage
builder.add_edge(START, "create_wiseragents")
builder.add_edge("create_wiseragents", "human_feedback")
builder.add_conditional_edges("human_feedback", should_continue, ["create_wiseragents", "generate_TG"]) # connection
# TG edge stage
builder.add_edge("generate_TG", "generate_questions")
builder.add_edge("generate_questions", "search_wikipedia")
builder.add_edge("generate_questions", "search_web")
builder.add_edge("search_wikipedia", "generate_answers")
builder.add_edge("search_web", "generate_answers")
builder.add_conditional_edges("generate_answers", route_messages,['generate_questions','save_interview'])
builder.add_edge("save_interview", "vote_TG")
builder.add_edge("vote_TG", END)

memory = MemorySaver()
graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)

# Interview 
graph = builder.compile(checkpointer=memory).with_config(run_name="Conduct Interviews")

# View
# display(Image(graph.get_graph().draw_mermaid_png()))