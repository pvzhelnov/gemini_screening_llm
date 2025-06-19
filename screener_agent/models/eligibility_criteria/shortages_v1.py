from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional

class ResponseValue(str, Enum):
    YES = "Yes"
    NO = "No"
    MAYBE_UNCLEAR = "Maybe/Unclear"

class Question(BaseModel):
    unique_question_id: int = Field(..., description="Unique identifier for the question")
    question_formulation: str = Field(..., description="The question text")
    response: ResponseValue = Field(..., description="The LLM's response to this question")
    reasoning: str = Field(..., description="Explanation for the response")

class DrugShortagesCriteria:
    """
    Drug shortages eligibility criteria based on the YAML schema.
    Contains the 4 key questions for screening studies.
    """
    
    QUESTION_1 = "Is this an observational study?"
    QUESTION_2 = "Does this study examine pharmaceutical products?"
    QUESTION_3 = "Does this study examine drug supply chain disruptions?"
    QUESTION_4 = "Does this study examine the impact on drug utilization and access?"
    
    @classmethod
    def get_all_questions(cls) -> List[str]:
        return [
            cls.QUESTION_1,
            cls.QUESTION_2,
            cls.QUESTION_3,
            cls.QUESTION_4
        ]
    
    @classmethod
    def get_question_by_id(cls, question_id: int) -> str:
        questions = {
            1: cls.QUESTION_1,
            2: cls.QUESTION_2,
            3: cls.QUESTION_3,
            4: cls.QUESTION_4
        }
        return questions.get(question_id, "")

class FlowDecision(str, Enum):
    INCLUDE = "Include"
    EXCLUDE = "Exclude"
