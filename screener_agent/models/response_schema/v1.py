from typing import List, Generic, TypeVar
from pydantic import BaseModel, Field
from enum import Enum

# --- Criteria Lists ---
from screener_agent.models.eligibility_criteria import *

# --- Enums for Decision ---
class FinalDecision(str, Enum):
    include = "include"
    exclude = "exclude"

# --- Criterion-Level Reasoning ---
T = TypeVar("T", bound=EligibilityCriterion)  # making passing of criterion type possible via generics
class EvaluatedCriterion(BaseModel, Generic[T]):
    criterion: T = Field(..., description="A specific inclusion or exclusion criterion")
    met: bool = Field(..., description="Whether the criterion was satisfied")
    reasoning: str = Field(..., description="Explanation of why the criterion was or wasn't met")

# --- Main Analysis Response ---
class ScreeningResponseSchema(BaseModel):
    include_criteria_met: List[EvaluatedCriterion[InclusionCriterion]] = Field(
        description="Evaluation of each inclusion criterion with explanations"
    )
    exclude_criteria_met: List[EvaluatedCriterion[ExclusionCriterion]] = Field(
        description="Evaluation of each exclusion criterion with explanations"
    )
    decision: FinalDecision = Field(
        description="The final decision: include or exclude"
    )
    reasoning_summary: str = Field(
        description="Concise explanation of the overall decision, incorporating key reasoning"
    )
