from typing import List
from pydantic import BaseModel, Field
from screener_agent.models.eligibility_criteria.shortages_v1 import Question, FlowDecision

class DrugShortagesScreeningResponse(BaseModel):
    """
    Response schema for drug shortages screening based on the flow diagram.
    The LLM answers 4 sequential questions and follows the flow logic.
    """
    
    question_1_response: Question = Field(
        ..., 
        description="Response to: Is this an observational study?"
    )
    question_2_response: Question = Field(
        ..., 
        description="Response to: Does this study examine pharmaceutical products?"
    )
    question_3_response: Question = Field(
        ..., 
        description="Response to: Does this study examine drug supply chain disruptions?"
    )
    question_4_response: Question = Field(
        ..., 
        description="Response to: Does this study examine the impact on drug utilization and access?"
    )
    
    final_decision: FlowDecision = Field(
        ..., 
        description="Final decision: Include or Exclude based on flow diagram logic"
    )
    
    decision_reasoning: str = Field(
        ..., 
        description="Explanation of how the final decision was reached following the flow diagram"
    )
    
    def evaluate_flow_decision(self) -> FlowDecision:
        """
        Evaluate the final decision based on the flow diagram logic:
        - Question 1 (No) -> Exclude
        - Question 2 (No) -> Exclude  
        - Question 3 (No) -> Exclude
        - Question 4 (No) -> Exclude
        - Question 4 (Yes/Maybe) -> Include
        """
        
        # Step 1: Check Question 1
        if self.question_1_response.response.value == "No":
            return FlowDecision.EXCLUDE
            
        # Step 2: Check Question 2  
        if self.question_2_response.response.value == "No":
            return FlowDecision.EXCLUDE
            
        # Step 3: Check Question 3
        if self.question_3_response.response.value == "No":
            return FlowDecision.EXCLUDE
            
        # Step 4: Check Question 4 (final step)
        if self.question_4_response.response.value in ["Yes", "Maybe/Unclear"]:
            return FlowDecision.INCLUDE
        else:
            return FlowDecision.EXCLUDE
