import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from TG.task_graph import TaskGraph
from state import State
from llm import llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import ValidationError

TG_refining_instructions = """
You are the WiserAgent {agent_info}, and you previously generated this Task Graph to solve the following query:
User Query: {query}

Here is the original Task Graph you proposed:
{tg}

Your goal is to refine your Task Graph based on this feedback. This may include:
- Adding missing tasks or dependencies
- Clarifying vague task names or descriptions
- Incorporating overlooked constraints or logic
- Improving the TG to better reflect the user query

You have now received peer-review feedback through the following interview questions and answers:
{interview_QA}

DO NOT return the same TG unless you are absolutely sure no meaningful improvement is possible.
Return the full updated TG in strict JSON using the TaskGraph schema.
"""

def refine_task_graphs(state: State):
    
    query = state['query']
    interviews = state['interview']
    initial_tgs = state['tg_candidates']
    agents = state["wiseragents"] 
    final_tgs = []

    print("Refining Task Graphs based on interview insights...")
    subfolder = state.get("subfolder") + "/refined"
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
                    for agent in agents:
                        if tg.owner_agent.name == agent.name:
                            agent_info = agent

                            # fix the owner_agent (string -> Wiseragent object) issue
                            entry.task_graph.owner_agent = agent
                            break
                    # print("interview_QA: ", interview_QA)
                    # print("entry.task_graph :", entry.task_graph)
                    # print("entry.task_graph.to_json_string() :", entry.task_graph.to_json_string())
                    system_message = TG_refining_instructions.format(
                        query=query, 
                        agent_info=agent_info,
                        interview_QA=interview_QA,
                        tg=entry.task_graph)

                    # print("Refining prompt : ", system_message)                            
                    structured_llm = llm.with_structured_output(TaskGraph)
                    max_retries = 3
                    for attempt in range(1, max_retries + 1):
                        try:
                            refined_tg = structured_llm.invoke([
                                SystemMessage(content=system_message),
                                HumanMessage(content="Regenerate a better Task Graph using the new context")
                            ])
                            break  # Success: exit retry loop
                        except ValidationError as ve:
                            print(f"⚠️ ValidationError on attempt {attempt}: {ve}")
                            if attempt == max_retries:
                                print("❌ Max retries reached, keeping original TG.")
                                refined_tg = None
                            else:
                                print("Retrying LLM generation...")
                    # refined_tg = structured_llm.invoke([
                    #     SystemMessage(content=system_message),
                    #     HumanMessage(content="Regenerate a better Task Graph using the new context")
                    # ])

                    # print("refined_tg:", refined_tg)
                    if refined_tg:
                        # print("Is refined_tg == tg? ", refined_tg.to_json() == tg.to_json())
                        if refined_tg.to_json() == tg.to_json():
                            print("No meaningful improvement found, keeping the original TG.")
                            continue
                        final_tgs.append(refined_tg)
                        refined = True
                        print(f"✅ TG refined for: {tg.owner_agent.name}")
                        break  # stop after refining once  
            if refined:
                refined_tg.save_to_file(subfolder=subfolder)
                # refined_tg.save_to_file()
                break  # stop checking other interviews
    
        if not refined:
            final_tgs.append(tg)                 
    # print("final_tgs :", final_tgs)
    print("Task Graph refinement completed.")
    state["messages"].append(AIMessage(content="Task Graphs refined..."))
    state["tg_candidates"] = final_tgs
    
    return state

