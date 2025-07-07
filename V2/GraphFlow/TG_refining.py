import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from TG.task_graph import TaskGraph
from state import State
from llm import llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

TG_REFINING_INSTRUCTIONS = """
You are the WiserAgent {agent_info}, and you previously generated this Task Graph to solve the following query:
User Query: {query}

Here is the original Task Graph you proposed:
{tg}

Based on the following interview with other agents, please refine your Task Graph to improve it.

### Interview:
{interview_QA}

Return the updated TaskGraph in the same structured format (TaskGraph JSON schema).
"""

def refine_task_graphs(state: State):
    print("🔧 Refining Task Graphs based on interview insights...")
    query = state['query']
    interviews = state['interview']
    initial_tgs = state['tg_candidates']
    agents = state["wiseragents"]
    # final_tgs = initial_tgs.copy() 
    final_tgs = []

    for tg in initial_tgs:
        refined = False
        for interview in interviews:
            for entry in interview.entries:
                if tg.owner_agent.name == entry.task_graph.owner_agent.name:
                    interview_QA = ""
                    for qst in entry.Questions:
                        if qst.content != "No question, Thanks":
                            question=qst.content
                            for a in entry.Answers:  
                                if a.to_.name == qst.from_.name:
                                    answer=a.content
                                # print(f"answer to {a.to_.name}: {answer}")
                                    interview_QA += f"Question: {question} \nAnswer: {answer}\n"
                    # print(f"InterviewQA by {tg.owner_agent.name}: {interview_QA}")
                    # print("tg.owner_agent.name: ", tg.owner_agent.name)
                    # linking the wiseragent name to the real wiseragent object

                    for agent in agents:
                        if tg.owner_agent.name == agent.name:
                            agent_info = agent

                            # fix the owner_agent (string -> Wiseragent object) issue
                            entry.task_graph.owner_agent = agent
                            # print("entry.task_graph.owner_agent", entry.task_graph.owner_agent)
                            break
                    
                    
                    
                    print("interview_QA: ", interview_QA)
                    # print("entry.task_graph :", entry.task_graph)
                    # print("entry.task_graph.to_json_string() :", entry.task_graph.to_json_string())
                    system_message = TG_REFINING_INSTRUCTIONS.format(
                        query=query, 
                        agent_info=agent_info,
                        interview_QA=interview_QA,
                        tg=entry.task_graph)

                    # print("system message : ",system_message)                            
                    structured_llm = llm.with_structured_output(TaskGraph)
                    refined_tg = structured_llm.invoke([
                        SystemMessage(content=system_message),
                        HumanMessage(content="Regenerate a better Task Graph using the new context")
                    ])

                    if refined_tg:
                        print(f"✅ TG refined for: {tg.owner_agent.name}")
                        final_tgs.append(refined_tg)
                        refined = True
                        break  # stop after refining once  
            if refined:
                break  # stop checking other interviews
    
        if not refined:
            final_tgs.append(tg)                 
    print("final_tgs :", final_tgs)
    print("Task Graph refinement completed.")
    # state["messages"].append(AIMessage(content="Task Graphs refined after interview..."))
    return State(**{
        **state,
        "tg_candidates": final_tgs
    })




# def refine_task_graphs(state: State) -> State:
#     print("🔧 Refining Task Graphs based on interview insights...")

#     query = state["query"]
#     interviews = state["interview"]
#     tg_candidates = state["tg_candidates"]

#     refined_tgs = []
#     preserved_tgs = []

#     for tg in tg_candidates:
#         refined = False
#         agent_name = tg.owner_agent.name

#         for interview in interviews:
#             for entry in interview.entries:
#                 if agent_name != entry.task_graph.owner_agent:
#                     continue

#                 # collect Q/A pairs that are not "No question"
#                 qa_pairs = []
#                 for qst in entry.Questions:
#                     if qst.content.strip().lower() != "no question, thanks":
#                         answer = next(
#                             (a.content for a in entry.Answers if a.to_.name == qst.from_.name),
#                             None
#                         )
#                         if answer:
#                             qa_pairs.append((qst.content.strip(), answer.strip()))

#                 if qa_pairs:
#                     # Prepare interview context
#                     interview_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in qa_pairs])

#                     # Prompt LLM to regenerate TaskGraph
#                     prompt = TG_REFINING_INSTRUCTIONS.format(
#                         agent_name=agent_name,
#                         query=query,
#                         tg_json=tg.to_json_string(),
#                         interview_text=interview_text
#                     )

#                     try:
#                         structured_llm = llm.with_structured_output(TaskGraph)
#                         refined_tg = structured_llm.invoke([
#                             SystemMessage(content=prompt),
#                             HumanMessage(content="Please regenerate the Task Graph using the feedback above.")
#                         ])
#                         refined_tg.pretty_print()
#                         refined_tg.save_to_file()
#                         refined_tgs.append(refined_tg)
#                         refined = True
#                         break  # stop processing more entries for this TG once refined
#                     except Exception as e:
#                         print(f"❌ Error refining TG for {agent_name}: {e}")

#         if not refined:
#             preserved_tgs.append(tg)
#     print("Refined TGs: ", refined_tgs)
#     print("Preserved TGs: ", preserved_tgs)
#     all_tgs_updated = preserved_tgs + refined_tgs

#     print("✅ Task Graph refinement completed.")
#     return State(**{
#         **state,
#         "tg_candidates": all_tgs_updated
#     })
