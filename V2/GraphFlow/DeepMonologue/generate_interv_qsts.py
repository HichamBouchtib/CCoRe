import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from llm import llm
from langchain_core.messages import SystemMessage
from state import State
from interview.Interview import InterviewEntry
from interview.Interview import Interview
from ipywidgets import VBox, Accordion, HTML
from IPython.display import display
from interview.Interview import display_interview_cards

tg_interview_instructions = """You are a technical reviewer WiserAgent with expertise in {expertise}. 
You are reviewing a Task Graph (TG) created to solve the following query:
"{query}"

Here is the TG:
{tg_details}

Here are the questions already asked about this TG:
{already_asked}

Your task is to:
- Suggest **a new and non-redundant** technical question, if something is **missing or unclear**.
- If all the main aspects are covered, simply respond: "No question, Thanks"

Focus technicaly only on:
- Logical dependencies
- Missing or unnecessary tasks
- Incomplete specifications

Do not add greetings or generic commentary. Return only the final question (or the sentence above).
"""


def generate_questions(state: State):
    print("- Deep Monologue: Started")

    tgs = state["tg_candidates"]
    agents = state["wiseragents"]
    query = state["query"]
    interview_obj = Interview()

    for tg in tgs:
        owner_agent = tg.owner_agent
        tg_text = json.dumps(tg.to_json(), indent=2)
        entry = InterviewEntry(TG_Owner=owner_agent, task_graph=tg)

        # print(owner_agent.name, ": Do you have any questions for me?")

        asked_questions = []

        for agent in agents:
            if agent.name == owner_agent.name:
                continue  # Skip self-questioning

            # Join previous questions to give context
            previous_qst_block = "\n".join([f"- {q}" for q in asked_questions]) or "None"

            system_message = tg_interview_instructions.format(
                query=query,
                expertise=agent.domain_expertise,
                tg_details=tg_text,
                already_asked=previous_qst_block
            )

            question_msg = llm.invoke([SystemMessage(content=system_message)])
            question_content = question_msg.content if hasattr(question_msg, 'content') else str(question_msg)

            # Add only if it's not a duplicate
            if question_content not in asked_questions:
                asked_questions.append(question_content)
                entry.add_question(from_agent=agent, question=question_content)
                # print(agent.name, ":", question_content)
            else:
                entry.add_question(from_agent=agent, question="No question, Thanks")
                # print(agent.name, ":", "No question, Thanks")
        interview_obj.add_entry(entry)
    interview_obj.save_qsts_to_file()
    display_interview_cards([interview_obj])
    # # Interactive printing
    # # display_interview_cards(interview_obj)
    # widgets = []

    # for entry in interview_obj.entries:
    #     owner = entry.TG_Owner.name
    #     qa_blocks = []

    #     if not entry.Questions:
    #         qa_blocks.append(HTML(f"<i>No questions asked to {owner}</i>"))
    #     else:
    #         for q in entry.Questions:
    #             asker = q.from_.name
    #             content = q.content
    #             block = HTML(
    #                 f"<b>{asker} :</b><br>"
    #                 f"<span style='margin-left:20px;'>{content}</span><br><br>"
    #             )
    #             qa_blocks.append(block)

    #     vbox = VBox(qa_blocks)
    #     widgets.append(vbox)

    # accordion = Accordion(children=widgets)
    # for i, entry in enumerate(interview_obj.entries):
    #     accordion.set_title(i, f"{entry.TG_Owner.name} Interview ")

    # print("Displaying Interview Questions")
    # display(accordion)

    return {"interview": [interview_obj]}


# from agents.wiseragent import WiserAgent
# mock_state = State(
#     topic='AI in cyber attacks in website',
#     query='',
#     human_wiseragent_feedback='',
#     feedback_handled=True,
#     WS=50,
#     wiseragents=[
#         WiserAgent(
#             name='AIAttackExpert',
#             domain_expertise='AI-driven cyber attack techniques',
#             description='Specializes in understanding and predicting AI-based cyber attacks targeting websites.',
#             WS=50,
#             preferred_llm='qwen2.5:latest'
#         ),
#         WiserAgent(
#             name='DefensiveAIWhiz',
#             domain_expertise='AI-based defense mechanisms for web security',
#             description='Expert in developing and implementing AI-driven cybersecurity measures to protect websites from various threats.',
#             WS=50,
#             preferred_llm='qwen2.5:latest'
#         )
#     ],
#     max_num_turns=5,
#     interview=Interview(entries=[]),
#     tg_candidates=[
#         TaskGraph(
#             owner_agent=WiserAgent(
#                 name='Cybersecurity Specialist',
#                 domain_expertise='AI-Driven Cybersecurity',
#                 description='Expert in using AI to protect against cyber threats, focusing on phishing attacks.',
#                 WS=50,
#                 preferred_llm='qwen2.5:latest'
#             ),
#             tasks={
#                 'Automated Response Setup': 'Setting up automated responses to detected phishing attempts to minimize damage.',
#                 'Phishing Detection Training': 'Training an AI model to detect phishing attempts based on historical data and current trends.',
#                 'Regular Updates': 'Updating the AI model regularly with new data to improve its accuracy.',
#                 'User Education': 'Educating users about common phishing tactics and how to avoid them.',
#                 'Website Monitoring': 'Continuously monitoring the website for any signs of phishing activities.'
#             },
#             orders=[
#                 {'condition': 'success', 'from': 'Phishing Detection Training', 'to': 'Website Monitoring'},
#                 {'condition': 'success', 'from': 'Phishing Detection Training', 'to': 'User Education'},
#                 {'condition': 'phishing_detected', 'from': 'Website Monitoring', 'to': 'Automated Response Setup'},
#                 {'condition': 'phishing_detected', 'from': 'Website Monitoring', 'to': 'Regular Updates'},
#                 {'condition': 'completed', 'from': 'User Education', 'to': 'End'},
#                 {'condition': 'setup_complete', 'from': 'Automated Response Setup', 'to': 'End'},
#                 {'condition': 'updated', 'from': 'Regular Updates', 'to': 'End'}
#             ]
#         )
#     ],
#     context=[],
#     tg_chosen={},
#     tg_response=''
# )
# generate_questions(mock_state)
