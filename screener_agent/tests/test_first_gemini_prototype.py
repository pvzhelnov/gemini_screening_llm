import unittest
from unittest.mock import patch
import pandas as pd
import os

from utils.loggers import init_logger

class TestMainFunction(unittest.TestCase):
    @patch('first_gemini_prototype.process_study')
    @patch('first_gemini_prototype.pd.read_pickle')
    @patch('first_gemini_prototype.sys.exit')
    @patch('first_gemini_prototype.time.sleep', return_value=None)  # Avoid waiting during tests
    def test_main_with_fake_process_study(self, mock_sleep, mock_exit, mock_read_pickle, mock_process_study):
        # Create fake DataFrame
        data = [
            {
                'index': 'rayyan-75172666',
                'title': 'PROCEEDINGS OF THE AUSTRALASIAN SOCIETY OF CLINICAL AND EXPE...',
                'abstract': '14th Annual Meeting, December 1980, Canberra 1. Effect of de...'
            },
            {
                'index': 'id1',
                'title': 'Fake 2nd study title',
                'abstract': 'Fake 2nd study abstract'
            }
        ]
        fake_df = pd.DataFrame(data)
        fake_df.set_index('index', inplace=True)

        mock_read_pickle.return_value = fake_df

        # Set fake return value for process_study
        mock_process_study.return_value = {
            'llm_analysis': {
                'include_criteria_met': [{'criterion': 'mock_inclusion_criterion_0', 'met': True, 'reasoning': 'mock_reasoning_0'},{'criterion': 'mock_inclusion_criterion_1', 'met': False, 'reasoning': 'mock_inclusion_reasoning_1'}],
                'exclude_criteria_met': [{'criterion': 'mock_exclusion_criterion_0', 'met': None, 'reasoning': 'mock_exclusion_reasoning_0'},{'criterion': 'mock_exclusion_criterion_1', 'met': None, 'reasoning': 'mock_exclusion_reasoning_1'}],
                'decision': 'mock_decision',
                'reasoning_summary': 'mock_reasoning_summary'
            },
            'include_criteria_met': [{'criterion': 'English language', 'met': True, 'reasoning': 'The title of the proceedings and the list of individual presentation titles are in English.'}, {'criterion': 'Effects on RBC properties/function', 'met': False, 'reasoning': 'No entry within the provided list of 130 proceedings details the effects of liposomes on RBC properties or function. Entry #97 discusses ethanol effects on erythrocyte membranes, which is not related to liposomes.'}, {'criterion': 'Liposome-RBC interaction', 'met': False, 'reasoning': 'No entry in the provided list of 130 proceedings explicitly describes or implies an interaction between liposomes and RBCs.'}, {'criterion': 'Relevant study type', 'met': False, 'reasoning': 'The document is a list of conference proceedings titles, not a specific research study (e.g., experimental, clinical, review) focused on the topic of liposome-RBC interactions.'}, {'criterion': 'Liposome/RBC properties with interaction implications', 'met': False, 'reasoning': 'While entry #41 mentions liposomes and entry #97 mentions RBCs, no information is presented about their respective properties in a way that implies a potential interaction between them.'}, {'criterion': 'Applications of liposome-RBC interactions', 'met': False, 'reasoning': 'No applications of liposome-RBC interactions are mentioned or suggested in the list of proceedings.'}, {'criterion': 'Theoretical/computational study', 'met': False, 'reasoning': 'None of the 130 titles suggest a theoretical or computational study focused on liposome-RBC interactions.'}],
            'exclude_criteria_met': [{'criterion': 'No liposome-RBC interaction implications', 'met': True, 'reasoning': "The provided text is a list of 130 conference proceeding titles. A scan of these titles shows that while entry #41 mentions 'insulin-liposomes' and entry #97 mentions 'human erythrocyte membranes', no single entry describes or implies an interaction between liposomes and RBCs. Therefore, the overall document does not provide information on liposome-RBC interaction implications."}, {'criterion': 'Other cell types without RBC component', 'met': True, 'reasoning': "Entry #41 ('Evaluation of insulin-liposomes in diabetic rats') discusses liposomes but in a context (diabetic rats, insulin delivery) that does not involve or mention RBCs, implying interactions with other cell types or systems relevant to insulin action, not RBCs."}, {'criterion': 'Passing mention only', 'met': False, 'reasoning': "There is no mention of liposome-RBC interaction to be considered 'passing'. Liposomes and RBCs are mentioned in separate contexts within the list of proceedings, with no linkage."}, {'criterion': 'Full text unavailable', 'met': True, 'reasoning': "The provided text is a title for conference proceedings and an 'abstract' which is actually a list of 130 individual presentation titles and authors. This is not a full research article or a detailed abstract of a single study that could be meaningfully screened for the specific interactions."}, {'criterion': 'Duplicate publication', 'met': False, 'reasoning': 'The document is a list of conference proceedings, not a single publication that could be a duplicate.'}, {'criterion': 'Non-peer-reviewed or preprint', 'met': True, 'reasoning': "The document is titled 'PROCEEDINGS OF THE AUSTRALASIAN SOCIETY OF CLINICAL AND EXPERIMENTAL PHARMACOLOGISTS', suggesting it is a collection of abstracts or short presentations from a conference. Such proceedings are often not subjected to the same rigorous peer-review process as full journal articles."}],
            'decision': 'exclude',
            'reasoning_summary': "The decision is 'exclude'. Firstly, the provided document is a list of titles from conference proceedings, not a full research paper or a detailed abstract of a single study, thus meeting the 'Full text unavailable' exclusion criterion. Secondly, such proceedings are often not rigorously peer-reviewed, meeting the 'Non-peer-reviewed or preprint' criterion. Most importantly, a review of the 130 titles reveals no entry that discusses or implies liposome-RBC interactions, thereby meeting the 'No liposome-RBC interaction implications' criterion; for instance, entry #41 mentions liposomes but not RBCs (meeting 'Other cell types without RBC component' for that specific abstract's context), and entry #97 mentions erythrocytes but not liposomes, with no entry connecting the two. As per the decision logic, meeting any exclusion criterion leads to exclusion."
        }

        test_path_to_save_csv = 'screener_agent/tests/data/test_analyzed_studies.csv'
        os.makedirs(os.path.dirname(test_path_to_save_csv), exist_ok=True)

        # Run main function
        from first_gemini_prototype import main  # Make sure this import happens after patching
        with patch.multiple(
            'first_gemini_prototype',
            logger=init_logger(__file__),
            path_to_save_csv=test_path_to_save_csv):
            main()
        test_df = pd.read_csv(test_path_to_save_csv, index_col='index')

        # Assertions
        mock_process_study.assert_called_once()
        self.assertIn('llm_analysis', test_df.columns)
        self.assertEqual(test_df.loc['rayyan-75172666', 'llm_decision'], 'exclude')

if __name__ == '__main__':
    unittest.main()
