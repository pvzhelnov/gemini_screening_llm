import os
import sys
import json
import yaml
import time
from typing import Dict, List, Any
from dotenv import load_dotenv
import google.generativeai as genai

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from screener_agent.models.eligibility_criteria.shortages_v1 import (
    DrugShortagesCriteria, 
    Question, 
    ResponseValue, 
    FlowDecision
)
from screener_agent.models.response_schema.shortages_v1 import DrugShortagesScreeningResponse
from utils.loggers import init_logger
from screener_agent.metrics_analyzer import ScreeningMetricsAnalyzer

load_dotenv()


class YAMLBasedScreeningAgent:
    """
    YAML-based screening agent that loads criteria from a YAML file and uses Pydantic models.
    """
    
    def __init__(self, yaml_path: str, model_name: str = 'gemini-2.0-flash-exp'):
        self.logger = init_logger(__name__)
        self.yaml_path = yaml_path
        self.model_name = model_name
        self.criteria_data = self._load_yaml_schema()
        self.model = self._initialize_gemini_model()
    
    def _load_yaml_schema(self) -> Dict[str, Any]:
        """Load the eligibility criteria schema from YAML file."""
        try:
            with open(self.yaml_path, 'r') as file:
                data = yaml.safe_load(file)
                self.logger.info(f"✅ Successfully loaded YAML schema from {self.yaml_path}")
                return data
        except Exception as e:
            self.logger.error(f"❌ Failed to load YAML schema: {e}")
            raise
    
    def _initialize_gemini_model(self):
        """Initialize the Gemini model."""
        try:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            model = genai.GenerativeModel(self.model_name)
            self.logger.info(f"✅ Successfully initialized Gemini model: {self.model_name}")
            return model
        except KeyError:
            self.logger.error("❌ GEMINI_API_KEY environment variable not set.")
            raise
        except Exception as e:
            self.logger.error(f"❌ Error configuring Gemini API: {e}")
            raise
    
    def _create_dynamic_prompt(self, title: str, abstract: str) -> str:
        """
        Create a screening prompt dynamically from the YAML schema using the Pydantic models.
        """
        questions = self.criteria_data['response']['content']['questions']
        flow_diagram = self.criteria_data['response']['content']['flow_diagram']
        
        # Create questions dynamically using the DrugShortagesCriteria class and ResponseValue enum
        all_questions = []
        for q_data in questions:
            q_id = q_data['question_uid']['unique_question_id']
            q_text = DrugShortagesCriteria.get_question_by_id(q_id)
            all_questions.append(f"{q_id}. {q_text}")
        
        # Get response values from the ResponseValue enum
        response_values = [v.value for v in ResponseValue]
        
        prompt = f"""
You are an expert scientific paper screener for drug shortage studies. Your task is to analyze a given title and abstract and answer 4 sequential questions to determine if the study should be included or excluded.

QUESTIONS TO ANSWER:
{chr(10).join(all_questions)}

For each question, you must answer with exactly one of these values: {response_values}

RESPONSE CRITERIA FROM YAML:
"""
        
        # Add response criteria from YAML using the Pydantic models
        for q_data in questions:
            q_id = q_data['question_uid']['unique_question_id']
            q_text = DrugShortagesCriteria.get_question_by_id(q_id)
            prompt += f"\nQuestion {q_id}: {q_text}\n"
            
            for response in q_data['responses']:
                value = response['value']
                notes = response.get('notes', [])
                
                prompt += f"   - Answer \"{value}\" if:\n"
                for note in notes:
                    prompt += f"     * {note}\n"
        
        # Add flow logic from YAML using the FlowDecision enum
        prompt += f"\nFLOW LOGIC:\n"
        for step in flow_diagram:
            step_id = step['step_id']
            if step['is_last_step']:
                prompt += f"- Step {step_id} (Final): "
                if 'on_yes' in step and 'decision' in step['on_yes']:
                    prompt += f"If \"{ResponseValue.YES.value}\" → {step['on_yes']['decision']}, "
                if 'on_maybe' in step and 'decision' in step['on_maybe']:
                    prompt += f"If \"{ResponseValue.MAYBE_UNCLEAR.value}\" → {step['on_maybe']['decision']}, "
                prompt += f"If \"{ResponseValue.NO.value}\" → {step['on_no']['decision']}\n"
            else:
                prompt += f"- Step {step_id}: If \"{ResponseValue.NO.value}\" → {step['on_no']['decision']}, otherwise continue\n"
        
        prompt += f"""

Here is the paper to analyze:
Title: "{title}"
Abstract: "{abstract}"

Provide your response as a valid JSON object with exactly this structure (replace the example values):
{{
    "question_1_response": {{
        "unique_question_id": 1,
        "question_formulation": "{DrugShortagesCriteria.get_question_by_id(1)}",
        "response": "Yes",
        "reasoning": "Your detailed reasoning for this answer"
    }},
    "question_2_response": {{
        "unique_question_id": 2,
        "question_formulation": "{DrugShortagesCriteria.get_question_by_id(2)}",
        "response": "No", 
        "reasoning": "Your detailed reasoning for this answer"
    }},
    "question_3_response": {{
        "unique_question_id": 3,
        "question_formulation": "{DrugShortagesCriteria.get_question_by_id(3)}",
        "response": "Maybe/Unclear",
        "reasoning": "Your detailed reasoning for this answer"
    }},
    "question_4_response": {{
        "unique_question_id": 4,
        "question_formulation": "{DrugShortagesCriteria.get_question_by_id(4)}",
        "response": "No",
        "reasoning": "Your detailed reasoning for this answer"
    }},
    "final_decision": "Exclude",
    "decision_reasoning": "Based on the flow logic, this study should be excluded because..."
}}

Make sure:
1. All strings are properly quoted
2. No trailing commas
3. Valid JSON format
4. Response values are exactly: "Yes", "No", or "Maybe/Unclear"
5. Final decision is exactly: "Include" or "Exclude"
"""
        
        return prompt
    
    def create_question_objects(self, responses: Dict[str, Any]) -> List[Question]:
        """
        Create Question objects using the Pydantic model from LLM responses.
        """
        questions = []
        for i in range(1, 5):
            response_key = f"question_{i}_response"
            if response_key in responses:
                q_data = responses[response_key]
                question = Question(
                    unique_question_id=q_data['unique_question_id'],
                    question_formulation=q_data['question_formulation'],
                    response=ResponseValue(q_data['response']),
                    reasoning=q_data['reasoning']
                )
                questions.append(question)
        return questions
    
    def screen_study(self, title: str, abstract: str) -> Dict[str, Any]:
        """
        Screen a single study using the YAML-based criteria and Pydantic models.
        """
        prompt = self._create_dynamic_prompt(title, abstract)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=4096,
                        temperature=0.2,
                        response_mime_type='application/json'
                        # Removed response_schema to avoid conflicts
                    )
                )
                
                # Parse the JSON response
                result_dict = json.loads(response.text)
                
                # Manual validation - check if all required keys are present
                required_keys = ['question_1_response', 'question_2_response', 'question_3_response', 'question_4_response', 'final_decision', 'decision_reasoning']
                if not all(key in result_dict for key in required_keys):
                    raise ValueError(f"Missing required keys in response. Got: {list(result_dict.keys())}")
                
                # Try to validate using Pydantic model (but don't fail if it doesn't work)
                try:
                    validated_response = DrugShortagesScreeningResponse(**result_dict)
                    return validated_response.model_dump()
                except Exception as pydantic_error:
                    self.logger.warning(f"Pydantic validation failed, using raw response: {pydantic_error}")
                    return result_dict
                
            except json.JSONDecodeError as e:
                self.logger.error(f"⚠️ JSONDecodeError attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    return self._create_error_response(f"JSON parsing failed: {e}")
            except Exception as e:
                self.logger.error(f"🚨 Error attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    return self._create_error_response(f"Screening failed: {e}")
        
        return self._create_error_response("Failed after all retries")
    
    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create an error response in the expected format using Pydantic models."""
        return {
            "question_1_response": {
                "unique_question_id": 1,
                "question_formulation": DrugShortagesCriteria.get_question_by_id(1),
                "response": ResponseValue.NO.value,
                "reasoning": f"Error occurred: {error_msg}"
            },
            "question_2_response": {
                "unique_question_id": 2,
                "question_formulation": DrugShortagesCriteria.get_question_by_id(2),
                "response": ResponseValue.NO.value,
                "reasoning": f"Error occurred: {error_msg}"
            },
            "question_3_response": {
                "unique_question_id": 3,
                "question_formulation": DrugShortagesCriteria.get_question_by_id(3),
                "response": ResponseValue.NO.value,
                "reasoning": f"Error occurred: {error_msg}"
            },
            "question_4_response": {
                "unique_question_id": 4,
                "question_formulation": DrugShortagesCriteria.get_question_by_id(4),
                "response": ResponseValue.NO.value,
                "reasoning": f"Error occurred: {error_msg}"
            },
            "final_decision": FlowDecision.EXCLUDE.value,
            "decision_reasoning": f"Study excluded due to processing error: {error_msg}"
        }
    
    def load_ris_file(self, ris_path: str) -> List[Dict[str, str]]:
        """
        Load and parse a RIS file to extract title and abstract.
        """
        studies = []
        current_study = {}
        record_count = 0
        
        try:
            with open(ris_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    original_line = line
                    line = line.strip()
                    
                    if line.startswith('TI  - '):
                        title = line[6:].strip()
                        current_study['title'] = title
                    elif line.startswith('AB  - '):
                        abstract = line[6:].strip()
                        current_study['abstract'] = abstract
                    elif line.startswith('N2  - '):  # Alternative abstract field
                        abstract = line[6:].strip()
                        current_study['abstract'] = abstract
                    elif line.startswith('ER  - ') or line == 'ER  -':
                        # End of record
                        record_count += 1
                        if 'title' in current_study or 'abstract' in current_study:
                            studies.append({
                                'title': current_study.get('title', ''),
                                'abstract': current_study.get('abstract', '')
                            })
                        current_study = {}
            
            self.logger.info(f"✅ Loaded {len(studies)} studies from {ris_path} (processed {record_count} records)")
            return studies
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load RIS file {ris_path}: {e}")
            return []
    
    def screen_ris_file(self, ris_path: str, expected_label: str = None) -> Dict[str, Any]:
        """
        Screen all studies in a RIS file and return results.
        """
        studies = self.load_ris_file(ris_path)
        # Limit to first 3 studies for testing
        studies = studies[:3]
        
        results = {
            'file': os.path.basename(ris_path),  # trimming to filename.ris for privacy
            'expected_label': expected_label,
            'total_studies': len(studies),
            'results': []
        }
        
        for i, study in enumerate(studies):
            self.logger.info(f"Screening study {i+1}/{len(studies)} from {ris_path}")
            
            result = self.screen_study(study['title'], study['abstract'])
            result['study_index'] = i
            result['title'] = study['title']
            result['abstract'] = study['abstract']
            
            results['results'].append(result)
            
            # Add small delay to respect API limits
            time.sleep(0.5)
        
        return results


def main():
    """Test the YAML-based screening agent."""
    logger = init_logger(__name__)
    
    # Initialize the agent
    yaml_path = os.getenv("ELIGIBILITY_SCHEMA_YAML_PATH")
    agent = YAMLBasedScreeningAgent(yaml_path)
    
    # Test files
    test_files = [
        (os.getenv("IRRELEVANT_RIS_PATH"), "dummy_dataset/dummy_shortages_irrelevant.ris"),
        (os.getenv("RELEVANT_RIS_PATH"), "dummy_dataset/dummy_shortages_relevant.ris")
    ]
    
    all_results = []
    
    for ris_path, expected_label in test_files:
        logger.info(f"\n🔍 Screening {ris_path} (expected: {expected_label})")
        results = agent.screen_ris_file(ris_path, expected_label)
        all_results.append(results)
        
        # Print summary
        include_count = sum(1 for r in results['results'] if r['final_decision'] == 'Include')
        exclude_count = sum(1 for r in results['results'] if r['final_decision'] == 'Exclude')
        
        logger.info(f"📊 Results for {ris_path}:")
        logger.info(f"   - Total studies: {results['total_studies']}")
        logger.info(f"   - Include: {include_count}")
        logger.info(f"   - Exclude: {exclude_count}")
        logger.info(f"   - Expected: {expected_label}")

    def save_results():
        # Get the current Unix timestamp
        timestamp = int(time.time())

        # Create a subdirectory based on the timestamp
        output_dir = "yaml_screening_results"
        os.makedirs(output_dir, exist_ok=True)

        # Define the file path within the subdirectory
        filename = f'{output_dir}_{timestamp}.json'
        output_file = os.path.join(output_dir, filename)

        # Save results
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return output_file
    
    # Save results and get the filepath
    results_file = save_results()
    logger.info(f"✅ Results saved to {results_file}")
    
    # Generate comprehensive metrics analysis
    logger.info("\n🔬 Generating comprehensive metrics analysis...")
    try:
        analyzer = ScreeningMetricsAnalyzer()
        analyzer.load_results_from_json(results_file)
        
        # Generate complete analysis with all visualizations and reports
        outputs = analyzer.generate_complete_analysis()
        
        logger.info("\n🎉 Metrics analysis complete! Generated files:")
        for output_type, filepath in outputs.items():
            logger.info(f"   • {output_type.replace('_', ' ').title()}: {filepath}")
        
    except Exception as e:
        logger.error(f"❌ Failed to generate metrics analysis: {e}")
        logger.error("Continuing without metrics analysis...")


if __name__ == "__main__":
    main()