import sys
import os
import json
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, PrivateAttr
from TG.task_graph import TaskGraph
from agents.wiseragent import WiserAgent
from IPython.display import display
from ipywidgets import VBox, Accordion, HTML

class Question(BaseModel):
    from_: WiserAgent
    content: str = Field(..., description="The question content")

class Answer(BaseModel):
    to_: WiserAgent
    content: str = Field(..., description="The answer content")

class InterviewEntry(BaseModel):
    TG_Owner: WiserAgent
    task_graph: TaskGraph = Field(..., description="The task graph instance")
    Questions: List[Question] = Field(default_factory=list, description="List of questions posed by other agents")
    Answers: List[Answer] = Field(default_factory=list, description="List of answers provided by the TG_Owner")

    def add_question(self, from_agent: WiserAgent, question: str) -> None:
        # print(f"from_agent: {from_agent}, type: {type(from_agent)}")
        if not isinstance(from_agent, WiserAgent):
            raise ValueError("from_agent must be an instance of WiserAgent.")
        self.Questions.append(Question(from_=from_agent, content=question))

    def add_answer(self, to_agent: WiserAgent, answer: str) -> None:
        self.Answers.append(Answer(to_=to_agent, content=answer))

    def to_json(self) -> Dict:
        """Convert to JSON-compatible dict with agent names instead of full objects."""
        return {
            "TG_Owner": self.TG_Owner.name,
            "task_graph": self.task_graph.to_json(),
            "Questions": [
                {"from": q.from_.name, "content": q.content}
                for q in self.Questions
            ],
            "Answers": [
                {"to": a.to_.name, "content": a.content}
                for a in self.Answers
            ]
        }
    
    def to_json_string(self, indent=4) -> str:
        return json.dumps(self.to_json(), indent=indent)

class Interview(BaseModel):
    entries: List[InterviewEntry] = Field(default_factory=list, description="List of interview sessions")
    _filename: Optional[str] = PrivateAttr(default=None)  # Internal attribute, not serialized

    def add_entry(self, entry: InterviewEntry) -> None:
        self.entries.append(entry)

    def get_by_owner(self, owner: str) -> Optional[InterviewEntry]:
        for entry in self.entries:
            if entry.TG_Owner == owner:
                return entry
        return None

    def to_json(self) -> List[Dict]:
        return [entry.to_json() for entry in self.entries]

    def to_json_string(self, indent: int = 4):
        # Convert the Interview object to a dictionary
        interview_dict = self.model_dump()

        # Manually serialize WiserAgent objects in the Agents field (if any)
        if 'Agents' in interview_dict:
            interview_dict['Agents'] = [agent.json() for agent in interview_dict['Agents']]

        # Manually convert Questions list (assuming it may contain custom objects)
        if 'Questions' in interview_dict:
            interview_dict['Questions'] = [q.dict() for q in interview_dict['Questions']]

        if 'Answers' in interview_dict:
            interview_dict['Questions'] = [a.dict() for a in interview_dict['Questions']]
        return json.dumps(interview_dict, indent=indent)

    def save_qsts_to_file(self, folder="interview/saved_interviews") -> None:
        os.makedirs(folder, exist_ok=True)
        existing_files = glob.glob(os.path.join(folder, "interview_*.json"))
        next_index = len(existing_files) + 1
        filename = os.path.join(folder, f"interview_{next_index}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=4)
        self._filename = filename
        print(f"Interview questions saved to {filename}")

    def save_answers_to_file(self) -> None:
        if not self._filename:
            raise ValueError("No filename found. Make sure save_qsts_to_file was called first.")
        with open(self._filename, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=4)
        print(f"Interview answers updated in {self._filename}")
    
    @staticmethod
    def load_from_file(filename: str) -> Optional["Interview"]:
        if not os.path.exists(filename):
            print(f"File {filename} does not exist.")
            return None
        with open(filename, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        entries = [InterviewEntry(**entry) for entry in json_data]
        interview = Interview(entries=entries)
        interview._filename = filename
        return interview

    @staticmethod
    def list_available(folder="interview/saved_interviews") -> List[str]:
        os.makedirs(folder, exist_ok=True)
        files = glob.glob(os.path.join(folder, "interview_*.json"))
        return files if files else ["No interviews found."]

# # Function to display interview cards
def display_interview_cards(interviews):
    widgets = []

    for interview in interviews:
        for entry in interview.entries:
            owner = entry.TG_Owner.name
            qa_blocks = []

            if not entry.Questions:
                qa_blocks.append(HTML(f"<i>No questions asked to {owner}</i>"))
            else:
                for question in entry.Questions:
                    asker = question.from_.name
                    q_content = question.content

                    # Find matching answer (to this asker)
                    matching_answer = next(
                        (a.content for a in entry.Answers if a.to_.name == asker),
                        "<i>No answer yet</i>"
                    )

                    # Build HTML block
                    block = HTML(
                        f"<b>👤 {asker} :</b><br>"
                        f"<span style='margin-left:20px;'>{q_content}</span><br><br>"
                        f"<b>💬 {owner} :</b><br>"
                        f"<span style='margin-left:20px;'>{matching_answer}</span><br><hr>"
                    )
                    qa_blocks.append(block)

            vbox = VBox(qa_blocks)
            widgets.append(vbox)

    accordion = Accordion(children=widgets)
    for i, entry in enumerate(interview.entries):
        accordion.set_title(i, f"{entry.TG_Owner.name} Interview")

    display(accordion)

# # test_interview
# sample_graph_data = {
#     "tasks": {
#         "Port Scanning": "Scanning the network for open ports using predefined tools.",
#         "Logfile Analyzing": "Analyzing the logs for any suspicious activities and detect potential threats.",
#         "Vulnerability Protectoring": "Mitigating detected vulnerabilities by applying security patches or blocking malicious activities."
#     },
#     "orders": [
#         {"from": "Port Scanning", "to": "Logfile Analyzing", "condition": "success"},
#         {"from": "Port Scanning", "to": "Vulnerability Protectoring", "condition": "failure"},
#         {"from": "Logfile Analyzing", "to": "Vulnerability Protectoring", "condition": "threats_found"},
#         {"from": "Logfile Analyzing", "to": "End", "condition": "no_threats"},
#         {"from": "Vulnerability Protectoring", "to": "End", "condition": "resolved"}
#     ]
# }
# task_graph = TaskGraph(**sample_graph_data)

# # Step 2: Create WiserAgent instance
# owner_agent1 = WiserAgent(
#     name="Port Scanner WiserAgent",
#     domain_expertise="Cybersecurity",
#     description="Specializes in port scanning.",
#     WS=50,
#     preferred_llm="llama3"
# )
# agent2 = WiserAgent(name="WiserAgent2", domain_expertise="General", description="", WS=50, preferred_llm="llama3") 
# agent3 = WiserAgent(name="WiserAgent3", domain_expertise="General", description="", WS=50, preferred_llm="llama3")

# # Step 3: Create an InterviewEntry with questions and answers
# entry = InterviewEntry(TG_Owner=owner_agent1, task_graph=task_graph)

# entry.add_question(agent2, "What is the purpose of your task graph?")
# entry.add_question(agent3, "How do you handle errors in execution?")

# entry.add_answer(owner_agent1, "The purpose is to identify and handle cyber threats efficiently.")
# entry.add_answer(owner_agent1, "We handle errors using fallback and alert mechanisms.")

# # Step 4: Add entry to Interview
# interview = Interview()
# interview.add_entry(entry)

# # Step 5: Serialize to JSON string
# json_str = interview.to_json_string()
# print("Serialized Interview:\n", json_str)

# # Step 6: Save interview to file
# interview.save_to_file(folder="V2/interview/interviews_saved")

# # Step 7: Load it back
# latest_file = Interview.list_available(folder="V2/interview/interviews_saved")[-1]
# loaded_interview = Interview.load_from_file(latest_file)

# print("\nLoaded Interview:\n", loaded_interview.to_json_string())

# # Step 8: Retrieve by TG_Owner name
# retrieved_entry = loaded_interview.get_by_owner(owner_agent1)
# print("\nRetrieved Entry TG_Owner:\n", retrieved_entry.TG_Owner.persona)

