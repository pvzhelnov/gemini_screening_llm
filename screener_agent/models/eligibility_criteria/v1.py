from pydantic import Field
from enum import Enum

class EligibilityCriterion(str, Enum):
    @classmethod
    def list_criteria(self) -> str:  # used to print these in system instruction
        return {"\n".join(f"- {criterion.value}" for criterion in list(self))}

class InclusionCriterion(EligibilityCriterion):
    ENGLISH = "English language"
    RBC_EFFECT = "Effects on RBC properties/function"
    INTERACTION = "Liposome-RBC interaction"
    STUDY_TYPE = "Relevant study type"
    PROPERTY_IMPLICATIONS = "Liposome/RBC properties with interaction implications"
    APPLICATIONS = "Applications of liposome-RBC interactions"
    THEORETICAL = "Theoretical/computational study"

class ExclusionCriterion(EligibilityCriterion):
    NO_INTERACTION = "No liposome-RBC interaction implications"
    OTHER_CELLS = "Other cell types without RBC component"
    PASSING_MENTION = "Passing mention only"
    NO_FULL_TEXT = "Full text unavailable"  # Note: LLM can only infer this if mentioned in abstract
    DUPLICATE = "Duplicate publication"  # Note: LLM cannot verify this from a single entry
    NON_PEER_REVIEWED = "Non-peer-reviewed or preprint"  # LLM can try to infer (e.g. from "proceedings", "preprint server")
