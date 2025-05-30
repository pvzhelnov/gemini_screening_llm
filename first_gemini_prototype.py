import pandas as pd
import google.generativeai as genai
import json
import os
import time
from dotenv  import load_dotenv
import sys

from screener_agent.models.eligibility_criteria import InclusionCriterion, ExclusionCriterion
from screener_agent.models.response_schema import ScreeningResponseSchema
from utils.loggers import init_logger

load_dotenv()
# --- Configuration ---
logger = init_logger(__file__)
screener_agent_model_name = 'gemini-2.5-pro-exp-03-25'
path_to_save_csv = "analyzed_studies.csv"

# IMPORTANT: Set your API key here or as an environment variable
# Option 1: Set directly in code (less secure for shared scripts)
# API_KEY = "YOUR_GEMINI_API_KEY"
# genai.configure(api_key=API_KEY)

# Option 2: Set as environment variable (more secure)
# and then run: genai.configure(api_key=os.environ["GEMINI_API_KEY"])
# For this example, let's assume you've set it as an environment variable
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    logger.error("🚨 GEMINI_API_KEY environment variable not set.")
    logger.error("Please set it before running the script.")
    exit()
except Exception as e:
    logger.error(f"🚨 Error configuring Gemini API: {e}")
    exit()


# --- Define Criteria (as strings for the prompt) ---
# Imported above

# --- LLM Model Setup ---
# For text-based tasks, gemini-pro is suitable
model = genai.GenerativeModel(screener_agent_model_name)

# --- Helper Function to Create Prompt ---
def create_prompt(title, abstract):
    prompt = f"""
    You are an expert scientific paper screener. Your task is to analyze a given title and abstract and determine if it should be included or excluded based on the provided criteria.

    Here are the criteria:

    Inclusion Criteria:
    {InclusionCriterion.list_criteria()}

    Exclusion Criteria:
    {ExclusionCriterion.list_criteria()}

    Decision Logic:
    1. First, assess if any Exclusion Criteria are met. If ANY exclusion criterion is met, the decision is "exclude".
    2. If no exclusion criteria are met, then assess Inclusion Criteria.
    3. The paper MUST be in "English language". If not, it's "exclude".
    4. The decision is "include" if ("English language" IS met AND "Effects on RBC properties/function" IS met).
    5. OR the decision is "include" if ("English language" IS met AND "Liposome/RBC properties with interaction implications" IS met).
    6. If none of the above "include" conditions (4 or 5) are met (and no exclusion criteria were met), the decision is "exclude".

    Provide your response as a JSON object with the following keys:
    - "include_criteria_met": A list of strings of ALL satisfied inclusion criteria from the provided list.
    - "exclude_criteria_met": A list of strings of ALL satisfied exclusion criteria from the provided list.
    - "decision": A string, either "include" or "exclude", based on the logic above.
    - "reasoning_summary": A list of strings, detailing your step-by-step thought process. Explain why each listed criterion (include or exclude) was met or not, and how you arrived at the final decision based on the decision logic. Be specific about what parts of the title or abstract led to your conclusions for each criterion.

    Here is the paper:
    Title: "{title}"
    Abstract: "{abstract}"

    Respond ONLY with the JSON object. Do not add any text before or after the JSON.
    """
    return prompt

# --- Function to Process a Single Study Identifier ---
def process_study(study_data):
    if not isinstance(study_data, dict) or 'title' not in study_data or 'abstract' not in study_data:
        return {
            "include_criteria_met": [],
            "exclude_criteria_met": ["Invalid input format"],
            "decision": "exclude",
            "reasoning_summary": ["Input data was not a valid dictionary with 'title' and 'abstract'."]
        }

    title = study_data.get('title', '')
    abstract = study_data.get('abstract', '')

    if not title and not abstract:
         return {
            "include_criteria_met": [],
            "exclude_criteria_met": ["Missing title and abstract"],
            "decision": "exclude",
            "reasoning_summary": ["Both title and abstract are missing."]
        }

    prompt = create_prompt(title, abstract)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                # Safety settings can be adjusted if needed
                # safety_settings=[
                #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                # ],
                generation_config=genai.types.GenerationConfig(
                    # candidate_count=1, # Default
                    # stop_sequences=['...'], # If needed
                    max_output_tokens=20480, # Adjust if JSON is truncated
                    temperature=0.2, # Lower temperature for more deterministic classification,
                    response_mime_type='application/json',
                    response_schema=ScreeningResponseSchema
                )
            )
            
            # Debug: print raw response text
            logger.info("--- RAW LLM Response ---")
            logger.info(response.text)
            logger.info("------------------------")

            # The response text should be a JSON string.
            # Sometimes Gemini might add ```json ... ```, so we try to strip that.
            json_text = response.text.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            
            json_text = json_text.strip() # Clean up any leading/trailing whitespace

            result_dict = json.loads(json_text)
            
            # Validate expected keys
            expected_keys = {"include_criteria_met", "exclude_criteria_met", "decision", "reasoning_summary"}
            if not expected_keys.issubset(result_dict.keys()):
                raise ValueError(f"LLM response missing one or more expected keys. Got: {result_dict.keys()}")

            return result_dict

        except json.JSONDecodeError as e:
            logger.error(f"⚠️ JSONDecodeError for title '{title[:50]}...': {e}. Raw response: '{response.text[:200]}...'")
            if attempt < max_retries - 1:
                logger.info(f"Retrying ({attempt+1}/{max_retries})...")
                time.sleep(2**(attempt + 1)) # Exponential backoff
                continue
            else:
                return {
                    "include_criteria_met": [],
                    "exclude_criteria_met": ["LLM output not valid JSON after retries"],
                    "decision": "exclude",
                    "reasoning_summary": [f"LLM did not return valid JSON. Error: {e}. Last raw response: '{response.text[:200]}'"]
                }
        except genai.types.generation_types.BlockedPromptException as e:
             logger.error(f"🚫 Prompt blocked for title '{title[:50]}...': {e}")
             return {
                "include_criteria_met": [],
                "exclude_criteria_met": ["Content blocked by API"],
                "decision": "exclude",
                "reasoning_summary": [f"The prompt or content was blocked by the API due to safety settings. Details: {e}"]
            }
        except Exception as e:
            logger.error(f"🚨 An error occurred while processing title '{title[:50]}...': {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying ({attempt+1}/{max_retries})...")
                time.sleep(2**(attempt + 1)) # Exponential backoff
                continue
            else:
                return {
                    "include_criteria_met": [],
                    "exclude_criteria_met": ["LLM API call failed after retries"],
                    "decision": "exclude",
                    "reasoning_summary": [f"An unexpected error occurred with the LLM API call: {e}"]
                }
    return { # Should not be reached if retry logic is correct, but as a fallback
        "include_criteria_met": [],
        "exclude_criteria_met": ["Processing failed after multiple retries"],
        "decision": "exclude",
        "reasoning_summary": ["Failed to get a valid response from the LLM after all retries."]
    }


# --- Main Processing Logic ---
def main():
    try:
        my_df = pd.read_pickle("my_df.pkl")
        my_df = my_df.head(10).copy()
    except Exception as e:
        logger.error(f"Error occurred when trying to read dataset:\n{e}")
        sys.exit(1)

    results_dict = {}
    for index, row in my_df.iterrows():
        logger.info(f"\nProcessing row {index}...")
        study_info = row.to_dict()
        
        llm_output = process_study(study_info)
        llm_output['model_name'] = screener_agent_model_name
        results_dict[index] = llm_output
        
        # Optional: Print intermediate results
        logger.info(f"Title: {study_info.get('title', 'N/A')[:60]}...")
        logger.info(f"Abstract: {study_info.get('abstract', 'N/A')[:60]}...")
        logger.info(f"Decision: {llm_output.get('decision')}")
        logger.info(f"Include Met: {llm_output.get('include_criteria_met')}")
        logger.info(f"Exclude Met: {llm_output.get('exclude_criteria_met')}")
        logger.info(f"Summary: {llm_output.get('reasoning_summary')}") # Can be verbose
        
        # Be respectful of API rate limits if you have many rows
        time.sleep(1) # Add a small delay if needed
        break  # for now - so as not to overload the API limits
    
    try:
        # Make sure new cols are present in df index
        ground_truth_cols = ["include_criteria_met", "exclude_criteria_met", "decision", "reasoning_summary", 'model_name']
        new_cols = ['llm_analysis']
        new_cols.extend([f'llm_{key}' for key in ground_truth_cols])
        my_df[new_cols] = None

        def update_row(row: pd.Series, results_dict) -> pd.Series:
            index = row.name
            llm_output = results_dict.get(index, None)
            if llm_output:
                row[new_cols[0]] = llm_output
                for key in ground_truth_cols:
                    row[f'llm_{key}'] = llm_output.get(key)
            return row

        # Add results to the DataFrame
        my_df = my_df.apply(lambda row: update_row(row, results_dict), axis=1)

        logger.info("\n\n--- Final DataFrame with LLM Analysis ---")
        logger.info(my_df.info())

        # You can save it to a CSV or Excel file
        my_df.to_csv(path_to_save_csv, index=True)
        # my_df.to_excel("analyzed_studies.xlsx", index=False)
    except Exception as e:
        logger.error(f"Error occurred when trying to save results:\n{e}")

if __name__ == '__main__':
    main()