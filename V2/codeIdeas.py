from langgraph.graph import MessagesState
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, field_validator, ValidationError
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import NodeInterrupt
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults
from typing import List, Optional, Annotated
from operator import add
from typing_extensions import TypedDict
from langgraph.constants import Send
import operator

# API keys
os.environ["OPENAI_API_KEY"] = "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"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langchain-academy"

sys_msg = SystemMessage(content="You are a helpful assistant tasked with...") # System message

# llm = ChatOpenAI(model="gpt-4o")
model = ChatOpenAI(model="gpt-4o", temperature=0.5)

# # # State
# class State(TypedDict):
    # foo: Annotated[list[int], add]
class State(MessagesState):
    summary: str
# # State handling:
# class PydanticState(BaseModel):
#   name: str
#   mood: str # "happy" or "sad"
# @field_validator('mood')
# @classmethod
# def validate_mood(cls, value):
    # Ensure the mood is either "happy" or "sad"
    # if value not in ["happy", "sad"]:
    #     raise ValueError("Each mood must be either 'happy' or 'sad'")
    # return value
# try:
# 	state = PydanticState(name="John Doe", mood="mad")
# except ValidationError as e:
# 	print("Validation Error:", e)
# builder = StateGraph(PydanticState)
# State with reducers: Use MessagesState, which includes the messages key with add_messages reducer
# class ExtendedMessagesState(MessagesState):
    # Add any keys needed beyond messages, which is pre-built 
    # added_key_1: str
    # added_key_2: str

# # State with Private and Global Schemas
# class OverallState(TypedDict):
#     foo: int
# class PrivateState(TypedDict):
#     baz: int
# def node_1(state: OverallState) -> PrivateState:
#     print("---Node 1---")
#     return {"baz": state['foo'] + 1}
# def node2(state: PrivateState) -> OverallState:
    # print("---Node 2---")
    # return {"foo": state['baz'] + 1}_
# builder = StateGraph(OverallState)

# # State Input/Output in order to control the flow 
# class InputState(TypedDict):
#     question: str
# class OutputState(TypedDict):
#     answer: str
# class OverallState(TypedDict):
#     question: str
#     answer: str
#     notes: str
# def thinking_node(state: InputState):
    # return {"answer": "bye", "notes": "... his is name is Lance"}
# def answer_node(state: OverallState) -> OutputState:
    # return {"answer": "bye Lance"}


# # # TOOLS
# def multiply(a: int, b: int) -> int:
#     """Multiply a and b.

#     Args:
#         a: first int
#         b: second int
#     """
#     return a * b
# def add(a: int, b: int) -> int:
#     """Adds a and b.

#     Args:
#         a: first int
#         b: second int
#     """
#     return a + b
# def divide(a: int, b: int) -> float:
#     """Divide a and b.

#     Args:
#         a: first int
#         b: second int
#     """
#     return a / b
# tools = [add, multiply, divide]
# llm_with_tools = llm.bind_tools(tools)
# def assistant(state: MessagesState):
#    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
# def chat_model_node(state: MessagesState):
#   return {"messages": llm.invoke(state["messages"])}
# def filter_messages(state: MessagesState): # decreasing the token usage in the long running conversations
    # # Delete all but the 2 most recent messages
    # delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    # return {"messages": delete_messages}
# def chat_model_node(state: MessagesState):
    # messages = trim_messages(     # Message trimming
    #         state["messages"],
    #         max_tokens=100,
    #         strategy="last",
    #         token_counter=ChatOpenAI(model="gpt-4o"),
    #         allow_partial=False,
    #     )
    # return {"messages": [llm.invoke(messages)]}


# # # Summarization
def call_model(state: State): # Node to call the model to produce the summary of old convo
    summary = state.get("summary", "")
    if summary:
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"
        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": response}
def summarize_conversation(state: State):   # An other approach to summarize the conversation
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"
    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}
def should_continue(state: State): # Determine whether to end or summarize the conversation
    """Return the next node to execute."""
    messages = state["messages"]
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    return END
# # Graph without summarization
# builder = StateGraph(MessagesState)
# builder.add_node("assistant", assistant)
# builder.add_node("tools", ToolNode(tools))
# builder.add_edge(START, "assistant")
# builder.add_conditional_edges("assistant", tools_condition)   # If the latest message (result) from assistant is a tool call then tools_condition routes to tools otherwise to END
# builder.add_edge("tools", "assistant")
# memory = MemorySaver()    # use an in memory checkpointer to support long-running conversations with interruption
# react_graph_memory = builder.compile(checkpointer=memory)
# display(Image(react_graph_memory.get_graph(xray=True).draw_mermaid_png()))

# # Graph with summarization
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)


# # # MEMORY
# memory = MemorySaver() # MemorySaver option
# Indefinite persistence memory
# conn = sqlite3.connect(":memory:", check_same_thread = False) # In memory sqlite DB
db_path = "state_db/example.db"
conn = sqlite3.connect(db_path, check_same_thread=False) # DB  
memory = SqliteSaver(conn) # Checkpointer
graph = workflow.compile(checkpointer=memory)
display(Image(graph.get_graph().draw_mermaid_png()))
config = {"configurable": {"thread_id": "1"}} # Create a thread
input_message = HumanMessage(content="hi! I'm Lance")
output = graph.invoke({"messages": [input_message]}, config) 
input_message = HumanMessage(content="what's my name?")
output = graph.invoke({"messages": [input_message]}, config) 
input_message = HumanMessage(content="i like the 49ers!")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()
graph.get_state(config).values.get("summary","")    # To visualize the summary from the saved graph state

# # # Human-in-the-loop (Approval, debugging and editing)
# # Streaming 
# for chunk in graph.stream({"messages": [HumanMessage(content="hi! I'm Lance")]}, config, stream_mode="updates"):    # Print the state in a streaming way using stream_mode="updates", "values" mode printes everything
#     chunk['conversation']["messages"].pretty_print()
# async for event in graph.astream_events({"messages": [input_message]}, config, version="v2"):   # .astream_events method streams back events as they happen inside nodes
#     print(f"Node: {event['metadata'].get('langgraph_node','')}. Type: {event['event']}. Name: {event['name']}")
# Get chat model tokens from a particular node 
# if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node','') == 'node1':
#     print(event["data"])
    
# # Breakpoint
# graph = builder.compile(interrupt_before=["tools"], checkpointer=memory)   # To pause the execution, compile the graph with `interrupt_before=["node_name"]` and to continue the execution, compile with None
# for event in graph.stream(initial_input, thread, stream_mode="values"):    # Run the graph until the first interruption
#     event['messages'][-1].pretty_print()
# Either ask for user approval to continue
# user_approval = input("Do you want to call the tool? (yes/no): ")
# if user_approval.lower() == "yes":
    # If approved, continue the graph execution
#     for event in graph.stream(None, thread, stream_mode="values"):
#         event['messages'][-1].pretty_print()
# else:
#     print("Operation cancelled by user.")
# Or add a Human-in-the-loop node and update the state from there

# # Dynamic Breakpoint
# def step_2(state: State) -> State:  # # Sometimes it is helpful to allow the graph dynamically interrupt itself (Internal Breakpoint) via the NodeInterrupt (e.g bcuz of input length)
#     # Let's optionally raise a NodeInterrupt if the length of the input is longer than 5 characters
#     if len(state['input']) > 5:
#         raise NodeInterrupt(f"Received input that is longer than 5 characters: {state['input']}")
# graph.update_state(thread,{"input": "hi"},)    # State updating, to over-write the existing message, supply the id; and to append pass a message without an id
# graph.get_state_history(thread)     # .get_state is for current/recent state and .get_state_history is for all (agent history)
# .values: the state value, .next: the next node, .config: the checkpoint and thread ID

# # Forking
# if we want to run from the same step but with a different input.
# graph.update_state({thread_id}, {state: "..."}) # forks the current checkpoint
# graph.update_state({checkpoint_id: thread_id}, {state: "..."}) # forks the specified checkpoint
# to_fork = all_states[-2]
# to_fork.values["messages"] # to print the message
# fork_config = graph.update_state(
#     to_fork.config,
#     {"messages": [HumanMessage(content='Multiply 5 and 3', id=to_fork.values["messages"][0].id)]},
# )
# Reinvoke The graph from the new checkpoint, the graphs knows this checkpoint has never beeen executed so it runs instead of replaying 
# for event in graph.stream(None, fork_config, stream_mode="values"):
#     event['messages'][-1].pretty_print()
# config = {"configurable": {"thread_id": "1"}}     # Specify a thread
# messages = [HumanMessage(content="Add 3 and 4.")]
# # Invoking the graph with Thread IDs
# messages = react_graph_memory.invoke({"messages": messages}, config)
# graph.invoke(PydanticState(name="Lance",mood="sad"))
# for m in messages['messages']:
#     m.pretty_print()

# # # Controllability
# # Parallelization
# class State(TypedDict):   # For 2 parallel nodes, the reducer may be
    # The operator.add reducer fn makes this append-only
    # state: Annotated[list, operator.add]
# def sorting_reducer(left, right):     # Setting the order of state update
    # """ Combines and sorts the values in a list"""
    # if not isinstance(left, list):
    #     left = [left]
    # if not isinstance(right, list):
    #     right = [right]
    # return sorted(left + right, reverse=False)

# E.g Working with LLM
# class State(TypedDict):
#     question: str
#     answer: str
#     context: Annotated[list, operator.add]
# def search_web(state):
#     """ Retrieve docs from web search """
#     # Search
#     tavily_search = TavilySearchResults(max_results=3)
#     search_docs = tavily_search.invoke(state['question'])
#      # Format
#     formatted_search_docs = "\n\n---\n\n".join(
#         [
#             f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>'
#             for doc in search_docs
#         ]
#     )
#     return {"context": [formatted_search_docs]} 
# def search_wikipedia(state):
#     """ Retrieve docs from wikipedia """
#     # Search
#     search_docs = WikipediaLoader(query=state['question'], 
#                                   load_max_docs=2).load()
#      # Format
#     formatted_search_docs = "\n\n---\n\n".join(
#         [
#             f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}">\n{doc.page_content}\n</Document>'
#             for doc in search_docs
#         ]
#     )
#     return {"context": [formatted_search_docs]} 
# def generate_answer(state):
#     """ Node to answer a question """
#     # Get state
#     context = state["context"]
#     question = state["question"]
#     # Template
#     answer_template = """Answer the question {question} using this context: {context}"""
#     answer_instructions = answer_template.format(question=question, 
#                                                        context=context)    
#     # Answer
#     answer = llm.invoke([SystemMessage(content=answer_instructions)]+[HumanMessage(content=f"Answer the question.")])
#     # Append it to state
#     return {"answer": answer}
# builder = StateGraph(State)
# builder.add_node("search_web",search_web)
# builder.add_node("search_wikipedia", search_wikipedia)
# builder.add_node("generate_answer", generate_answer)
# builder.add_edge(START, "search_wikipedia")
# builder.add_edge(START, "search_web")
# builder.add_edge("search_wikipedia", "generate_answer")
# builder.add_edge("search_web", "generate_answer")
# builder.add_edge("generate_answer", END)

# # Subgraphs: Perform 2 operations in 2 different sub-graphs/teams of agents.
class Log(TypedDict):   # The structure of the logs
    id: str
    question: str
    docs: Optional[List]
    answer: str
    grade: Optional[int]
    grader: Optional[str]
    feedback: Optional[str]
# Failure Analysis Sub-graph
class FailureAnalysisState(TypedDict):
    cleaned_logs: List[Log]
    failures: List[Log]
    fa_summary: str
    processed_logs: List[str]
class FailureAnalysisOutputState(TypedDict):
    fa_summary: str
    processed_logs: List[str]
def get_failures(state):
    """ Get logs that contain a failure """
    cleaned_logs = state["cleaned_logs"]
    failures = [log for log in cleaned_logs if "grade" in log]
    return {"failures": failures}
def generate_summary(state):
    """ Generate summary of failures """
    failures = state["failures"]
    # Add fxn: fa_summary = summarize(failures)
    fa_summary = "Poor quality retrieval of Chroma documentation."
    return {"fa_summary": fa_summary, "processed_logs": [f"failure-analysis-on-log-{failure['id']}" for failure in failures]}
fa_builder = StateGraph(FailureAnalysisState,output=FailureAnalysisOutputState)
fa_builder.add_node("get_failures", get_failures)
fa_builder.add_node("generate_summary", generate_summary)
fa_builder.add_edge(START, "get_failures")
fa_builder.add_edge("get_failures", "generate_summary")
fa_builder.add_edge("generate_summary", END)
graph = fa_builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))
# Summarization subgraph
class QuestionSummarizationState(TypedDict):
    cleaned_logs: List[Log]
    qs_summary: str
    report: str
    processed_logs: List[str]
class QuestionSummarizationOutputState(TypedDict):
    report: str
    processed_logs: List[str]
def generate_summary(state):
    cleaned_logs = state["cleaned_logs"]
    # Add fxn: summary = summarize(generate_summary)
    summary = "Questions focused on usage of ChatOllama and Chroma vector store."
    return {"qs_summary": summary, "processed_logs": [f"summary-on-log-{log['id']}" for log in cleaned_logs]}
def send_to_slack(state):
    qs_summary = state["qs_summary"]
    # Add fxn: report = report_generation(qs_summary)
    report = "foo bar baz"
    return {"report": report}
qs_builder = StateGraph(QuestionSummarizationState,output=QuestionSummarizationOutputState)
qs_builder.add_node("generate_summary", generate_summary)
qs_builder.add_node("send_to_slack", send_to_slack)
qs_builder.add_edge(START, "generate_summary")
qs_builder.add_edge("generate_summary", "send_to_slack")
qs_builder.add_edge("send_to_slack", END)
graph = qs_builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))
# Adding 2 sub-graphs together
class EntryGraphState(TypedDict):  # Entry Graph
    raw_logs: List[Log]
    cleaned_logs: Annotated[List[Log], add] # This will be USED BY in BOTH sub-graphs
    fa_summary: str # This will only be generated in the FA sub-graph
    report: str # This will only be generated in the QS sub-graph
    processed_logs:  Annotated[List[int], add] # This will be generated in BOTH sub-graph
# we add our sub-graphs as nodes! 
def clean_logs(state):
    # Get logs
    raw_logs = state["raw_logs"]
    # Data cleaning raw_logs -> docs 
    cleaned_logs = raw_logs
    return {"cleaned_logs": cleaned_logs}
entry_builder = StateGraph(EntryGraphState)
entry_builder.add_node("clean_logs", clean_logs)
entry_builder.add_node("question_summarization", qs_builder.compile())
entry_builder.add_node("failure_analysis", fa_builder.compile())
entry_builder.add_edge(START, "clean_logs")
entry_builder.add_edge("clean_logs", "failure_analysis")
entry_builder.add_edge("clean_logs", "question_summarization")
entry_builder.add_edge("failure_analysis", END)
entry_builder.add_edge("question_summarization", END)
graph = entry_builder.compile()

# # Map-reduce
# Map Break a task into smaller sub-tasks, processing each in parallel and Reduce Aggregate the results across all of the completed, parallelized sub-tasks.
# subjects_prompt = """Generate a list of 3 sub-topics that are all related to this overall topic: {topic}."""
# joke_prompt = """Generate a joke about {subject}"""
# best_joke_prompt = """Below are a bunch of jokes about {topic}. Select the best one! Return the ID of the best one, starting 0 as the ID for the first joke. Jokes: \n\n  {jokes}"""
# class Subjects(BaseModel):
#     subjects: list[str]
# class BestJoke(BaseModel):
#     id: int
# class OverallState(TypedDict):
#     topic: str
#     subjects: list
#     jokes: Annotated[list, operator.add]
#     best_selected_joke: str
# def generate_topics(state: OverallState):
    # prompt = subjects_prompt.format(topic=state["topic"])
    # response = model.with_structured_output(Subjects).invoke(prompt)
    # return {"subjects": response.subjects}
# Send allows to pass any State to the generate_joke node without having to align with the OverallState as a result Send creates a joke for each subject
# def continue_to_jokes(state: OverallState):
    # return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]
# Map: Create a set of jokes about a topic.
# class JokeState(TypedDict):
    # subject: str
# class Joke(BaseModel):
    # joke: str
# def generate_joke(state: JokeState):
    # prompt = joke_prompt.format(subject=state["subject"])
    # response = model.with_structured_output(Joke).invoke(prompt)
    # return {"jokes": [response.joke]}
# Reduce: Pick the best joke from the list
# def best_joke(state: OverallState):
    # jokes = "\n\n".join(state["jokes"])
    # prompt = best_joke_prompt.format(topic=state["topic"], jokes=jokes)
    # response = model.with_structured_output(BestJoke).invoke(prompt)
    # return {"best_selected_joke": state["jokes"][response.id]}
# Graph
# graph = StateGraph(OverallState)
# graph.add_node("generate_topics", generate_topics)
# graph.add_node("generate_joke", generate_joke)
# graph.add_node("best_joke", best_joke)
# graph.add_edge(START, "generate_topics")
# graph.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
# graph.add_edge("generate_joke", "best_joke")
# graph.add_edge("best_joke", END)
# app = graph.compile()
# Image(app.get_graph().draw_mermaid_png())
for s in app.stream({"topic": "animals"}):      # Call the graph
    print(s)


# # # lightweight MAS around chat models that customizes the research process
