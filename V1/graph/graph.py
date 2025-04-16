from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from GraphFlow.wiseragents_creation import create_wiseragents
from llm import llm
from GraphFlow.TG_generation import generate_task_graphs
from V1.state import State
from GraphFlow.TGInterview.Interview_qsts import generate_TGinterview_questions
from tools.searchweb_tool import search_web
from tools.searchwikipedia_tool import search_wikipedia
from GraphFlow.TGInterview.Interview_answers import generate_answer
from GraphFlow.TGInterview.save_interview import save_interview
from GraphFlow.route_messages import route_messages
from GraphFlow.masteragents_creation import instantiate_masteragents
from GraphFlow.masteragents_voting import evaluate_task_graphs
from GraphFlow.execute_TG import execute_task_graph
from GraphFlow.workeragents_creation import instantiate_workeragents
from GraphFlow.generate_finalResponse import generate_final_response
# from GraphFlow.filter_messages import filter_messages

def human_feedback(state: State):
    """ No-op node that should be interrupted on """
    pass

def should_continue(state: State):
    """ Return the next node to execute """

    # Check if human feedback
    human_wiseragent_feedback=state.get('human_wiseragent_feedback', None)
    if human_wiseragent_feedback:
        return "create_wiseragents"
    
    # Otherwise end
    return END

# WiserAgent node stage
builder = StateGraph(State)
builder.add_node("create_wiseragents", create_wiseragents)
builder.add_node("human_feedback", human_feedback)
# TG node stage
builder.add_node("generate_TG", generate_task_graphs)
builder.add_node("generate_question", generate_TGinterview_questions)
builder.add_node("search_web", search_web)
builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("generate_answer", generate_answer)
builder.add_node("save_interview", save_interview)
# masteragent nodes stage
builder = StateGraph(State)
builder.add_node("instantiate_masteragents", instantiate_masteragents)
builder.add_node("evaluate_task_graphs", evaluate_task_graphs)
 
# workeragent Node stage 
builder = StateGraph(State)
builder.add_node("instantiate_workeragents", instantiate_workeragents)
builder.add_node("generate_final_response", generate_final_response)


# WiserAgent edge stage
builder.add_edge(START, "create_wiseragents")
builder.add_edge("create_wiseragents", "human_feedback")
builder.add_conditional_edges("human_feedback", should_continue, ["create_wiseragents", "generate_TG"]) # connection
# TG edge stage
builder.add_edge("generate_TG", "generate_question")
builder.add_edge("generate_question", "search_wikipedia")
builder.add_edge("generate_question", "search_web")
builder.add_edge("search_wikipedia", "generate_answer")
builder.add_edge("search_web", "generate_answer")
builder.add_conditional_edges("generate_answer", route_messages,['generate_question','save_interview'])
# builder.add_conditional_edges("generate_answer", route_messages1,['filter_messages','save_interview'])
builder.add_edge("save_interview", "instantiate_masteragents") # connection
# masteragent edge stage
builder.add_edge("instantiate_masteragents", "evaluate_task_graphs")
builder.add_edge("evaluate_task_graphs", "execute_task_graph")
builder.add_edge("execute_task_graph", "instantiate_workeragents") # connection
# workeragent edge stage 
builder.add_edge("instantiate_workeragents", "generate_final_response")
builder.add_edge("generate_final_response", END)

memory = MemorySaver()
graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)

# Interview 
graph = builder.compile(checkpointer=memory).with_config(run_name="Conduct Interviews")

# View
display(Image(graph.get_graph().draw_mermaid_png()))