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


