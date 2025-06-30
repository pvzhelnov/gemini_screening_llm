import os
import json
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from utils.loggers import init_logger


class ScreeningMetricsAnalyzer:
    """
    Comprehensive metrics analyzer for screening agent performance.
    Calculates F1 score, accuracy, balanced accuracy, precision, recall, confusion matrix.
    Enhanced with detailed per-file analysis capabilities.
    """
    
    def __init__(self):
        self.logger = init_logger(__name__)
        self.results = None
        self.metrics = {}
        self.detailed_results = []
        self.per_file_metrics = {}
        
    def load_results_from_json(self, json_path: str) -> None:
        """Load screening results from a JSON file."""
        try:
            with open(json_path, 'r') as f:
                self.results = json.load(f)
            self.logger.info(f"✅ Loaded results from {json_path}")
        except Exception as e:
            self.logger.error(f"❌ Failed to load results: {e}")
            raise
    
    def load_results_from_directory(self, results_dir: str) -> None:
        """Load all JSON results from a directory."""
        json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        if not json_files:
            raise ValueError(f"No JSON files found in {results_dir}")
        
        # Use the most recent file
        latest_file = max(json_files, key=lambda f: os.path.getctime(os.path.join(results_dir, f)))
        self.load_results_from_json(os.path.join(results_dir, latest_file))
    
    def _extract_predictions_and_labels(self) -> Tuple[List[str], List[str]]:
        """Extract predicted and true labels from results."""
        if not self.results:
            raise ValueError("No results loaded. Call load_results_* first.")
        
        y_true = []
        y_pred = []
        self.detailed_results = []
        
        for dataset in self.results:
            expected_label = dataset.get('expected_label', 'unknown')
            file_name = dataset.get('file', 'unknown')
            
            for result in dataset.get('results', []):
                # Extract true label (map expected_label to Include/Exclude)
                if expected_label == 'relevant':
                    true_label = 'Include'
                elif expected_label == 'irrelevant':
                    true_label = 'Exclude'
                else:
                    true_label = 'Unknown'
                
                # Extract predicted label
                pred_label = result.get('final_decision', 'Unknown')
                
                y_true.append(true_label)
                y_pred.append(pred_label)
                
                # Store detailed result for analysis
                self.detailed_results.append({
                    'file': file_name,
                    'study_index': result.get('study_index', -1),
                    'title': result.get('title', ''),
                    'abstract': result.get('abstract', ''),
                    'expected_label': expected_label,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'correct': true_label == pred_label,
                    'question_1_response': result.get('question_1_response', {}).get('response', ''),
                    'question_2_response': result.get('question_2_response', {}).get('response', ''),
                    'question_3_response': result.get('question_3_response', {}).get('response', ''),
                    'question_4_response': result.get('question_4_response', {}).get('response', ''),
                    'decision_reasoning': result.get('decision_reasoning', '')
                })
        
        self.logger.info(f"Extracted {len(y_true)} predictions for analysis")
        return y_true, y_pred
    
    def _calculate_additional_metrics(self, y_true_binary, y_pred_binary):
        """Calculate sensitivity, specificity, NPV, and return as dict."""
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # recall
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        else:
            sensitivity = specificity = npv = 0
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'npv': npv
        }

    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive classification metrics."""
        y_true, y_pred = self._extract_predictions_and_labels()
        
        # Convert to binary labels for sklearn (0=Exclude, 1=Include)
        y_true_binary = [1 if label == 'Include' else 0 for label in y_true]
        y_pred_binary = [1 if label == 'Include' else 0 for label in y_pred]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true_binary, y_pred_binary),
            'balanced_accuracy': balanced_accuracy_score(y_true_binary, y_pred_binary),
            'precision': precision_score(y_true_binary, y_pred_binary, average='binary', zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, average='binary', zero_division=0),
            'f1_score': f1_score(y_true_binary, y_pred_binary, average='binary', zero_division=0),
            'precision_macro': precision_score(y_true_binary, y_pred_binary, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true_binary, y_pred_binary, average='macro', zero_division=0),
            'f1_score_macro': f1_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
        }
        
        # Add additional metrics
        add_metrics = self._calculate_additional_metrics(y_true_binary, y_pred_binary)
        metrics.update(add_metrics)
        
        # Confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true_binary, y_pred_binary, 
            target_names=['Exclude', 'Include'], 
            output_dict=True,
            zero_division=0
        )
        
        # Additional statistics
        total_studies = len(y_true)
        correct_predictions = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        
        metrics.update({
            'total_studies': total_studies,
            'correct_predictions': correct_predictions,
            'incorrect_predictions': total_studies - correct_predictions,
            'true_positives': int(cm[1, 1]) if cm.shape == (2, 2) else 0,
            'true_negatives': int(cm[0, 0]) if cm.shape == (2, 2) else 0,
            'false_positives': int(cm[0, 1]) if cm.shape == (2, 2) else 0,
            'false_negatives': int(cm[1, 0]) if cm.shape == (2, 2) else 0
        })
        
        self.metrics = metrics
        return metrics

    def get_metrics_table(self) -> str:
        """Return a markdown table of key metrics."""
        m = self.metrics
        table = (
            "| Metric | Value |\n"
            "|--------|-------|\n"
            f"| Sensitivity (Recall) | {m.get('sensitivity', 0):.3f} |\n"
            f"| Specificity | {m.get('specificity', 0):.3f} |\n"
            f"| Balanced Accuracy | {m.get('balanced_accuracy', 0):.3f} |\n"
            f"| Accuracy | {m.get('accuracy', 0):.3f} |\n"
            f"| Precision | {m.get('precision', 0):.3f} |\n"
            f"| NPV | {m.get('npv', 0):.3f} |\n"
            f"| F1 Score | {m.get('f1_score', 0):.3f} |\n"
        )
        return table

    def get_model_parameters_table(self, params=None) -> str:
        """Return a markdown table of model parameters. Pass a dict or use defaults."""
        # If params not provided, use defaults or placeholders
        if params is None:
            params = {
                'Temperature': 0.2,
                'Maximum length': 4096,
                'Stop sequences': '-',
                'Top-p': '-',
                'Top-k': '-',
                'Frequency penalty': '-',
                'Presence penalty': '-',
            }
        table = (
            "| Parameter | Value |\n"
            "|-----------|-------|\n"
            + "\n".join(f"| {k} | {v} |" for k, v in params.items())
        )
        return table

    def save_metrics_table_csv(self, save_path: str = None) -> str:
        """Save the key metrics table as a CSV file."""
        import pandas as pd
        m = self.metrics
        data = [
            ['Sensitivity (Recall)', m.get('sensitivity', 0)],
            ['Specificity', m.get('specificity', 0)],
            ['Balanced Accuracy', m.get('balanced_accuracy', 0)],
            ['Accuracy', m.get('accuracy', 0)],
            ['Precision', m.get('precision', 0)],
            ['NPV', m.get('npv', 0)],
            ['F1 Score', m.get('f1_score', 0)],
        ]
        df = pd.DataFrame(data, columns=['Metric', 'Value'])
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"metrics_table_{timestamp}.csv"
        df.to_csv(save_path, index=False)
        self.logger.info(f"✅ Metrics table saved to {save_path}")
        return save_path

    def save_model_parameters_table_csv(self, params=None, save_path: str = None) -> str:
        """Save the model parameters table as a CSV file."""
        import pandas as pd
        if params is None:
            params = {
                'Temperature': 0.2,
                'Maximum length': 4096,
                'Stop sequences': '-',
                'Top-p': '-',
                'Top-k': '-',
                'Frequency penalty': '-',
                'Presence penalty': '-',
            }
        data = [[k, v] for k, v in params.items()]
        df = pd.DataFrame(data, columns=['Parameter', 'Value'])
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"model_parameters_table_{timestamp}.csv"
        df.to_csv(save_path, index=False)
        self.logger.info(f"✅ Model parameters table saved to {save_path}")
        return save_path

    def output_ris_files_by_llm_label(self, relevant_path='llm_relevant.ris', irrelevant_path='llm_irrelevant.ris'):
        """Output two RIS files: one for LLM-relevant (Include) and one for LLM-irrelevant (Exclude) studies."""
        if not self.detailed_results:
            raise ValueError("No detailed results available. Call calculate_metrics() first.")
        relevant = [r for r in self.detailed_results if r['predicted_label'] == 'Include']
        irrelevant = [r for r in self.detailed_results if r['predicted_label'] == 'Exclude']
        def to_ris(records):
            ris = []
            for i, r in enumerate(records):
                # Use .get and handle list/str for abstract
                abstract = r.get('abstract', '')
                if isinstance(abstract, list):
                    abstract = ' '.join([str(a) for a in abstract if a])
                ris.append(f"TY  - JOUR\nTI  - {r.get('title', '')}\nAB  - {abstract}\nER  -\n")
            return ''.join(ris)
        with open(relevant_path, 'w', encoding='utf-8') as f:
            f.write(to_ris(relevant))
        with open(irrelevant_path, 'w', encoding='utf-8') as f:
            f.write(to_ris(irrelevant))
        self.logger.info(f"✅ Wrote {len(relevant)} relevant and {len(irrelevant)} irrelevant studies to RIS files.")
        return relevant_path, irrelevant_path

    def calculate_per_file_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Calculate comprehensive metrics for each file individually."""
        if not self.detailed_results:
            raise ValueError("No detailed results available. Call calculate_metrics() first.")
        
        # Group results by file
        files_data = {}
        for result in self.detailed_results:
            file_name = result['file']
            if file_name not in files_data:
                files_data[file_name] = {
                    'y_true': [],
                    'y_pred': [],
                    'detailed_results': []
                }
            
            # Convert labels to binary for sklearn
            true_binary = 1 if result['true_label'] == 'Include' else 0
            pred_binary = 1 if result['predicted_label'] == 'Include' else 0
            
            files_data[file_name]['y_true'].append(true_binary)
            files_data[file_name]['y_pred'].append(pred_binary)
            files_data[file_name]['detailed_results'].append(result)
        
        # Calculate metrics for each file
        per_file_metrics = {}
        
        for file_name, data in files_data.items():
            y_true = data['y_true']
            y_pred = data['y_pred']
            
            # Calculate metrics
            file_metrics = {
                'file_name': file_name,
                'total_studies': len(y_true),
                'accuracy': accuracy_score(y_true, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
            }
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            file_metrics['confusion_matrix'] = cm.tolist()
            
            # Extract confusion matrix elements
            if cm.shape == (2, 2):
                file_metrics.update({
                    'true_negatives': int(cm[0, 0]),
                    'false_positives': int(cm[0, 1]),
                    'false_negatives': int(cm[1, 0]),
                    'true_positives': int(cm[1, 1])
                })
            else:
                # Handle cases where not all classes are present
                file_metrics.update({
                    'true_negatives': 0,
                    'false_positives': 0,
                    'false_negatives': 0,
                    'true_positives': 0
                })
                if len(np.unique(y_true)) == 1 and len(np.unique(y_pred)) == 1:
                    if y_true[0] == y_pred[0]:
                        if y_true[0] == 1:
                            file_metrics['true_positives'] = len(y_true)
                        else:
                            file_metrics['true_negatives'] = len(y_true)
                    else:
                        if y_true[0] == 1:
                            file_metrics['false_negatives'] = len(y_true)
                        else:
                            file_metrics['false_positives'] = len(y_true)
            
            # Count correct/incorrect predictions
            correct_predictions = sum(1 for t, p in zip(y_true, y_pred) if t == p)
            file_metrics.update({
                'correct_predictions': correct_predictions,
                'incorrect_predictions': len(y_true) - correct_predictions
            })
            
            # Store detailed results for this file
            file_metrics['detailed_results'] = data['detailed_results']
            
            # Calculate study-level breakdown
            include_studies = [r for r in data['detailed_results'] if r['true_label'] == 'Include']
            exclude_studies = [r for r in data['detailed_results'] if r['true_label'] == 'Exclude']
            
            file_metrics.update({
                'total_include_studies': len(include_studies),
                'total_exclude_studies': len(exclude_studies),
                'correct_include_predictions': sum(1 for s in include_studies if s['correct']),
                'correct_exclude_predictions': sum(1 for s in exclude_studies if s['correct']),
                'include_accuracy': sum(1 for s in include_studies if s['correct']) / len(include_studies) if include_studies else 0,
                'exclude_accuracy': sum(1 for s in exclude_studies if s['correct']) / len(exclude_studies) if exclude_studies else 0
            })
            
            per_file_metrics[file_name] = file_metrics
        
        self.per_file_metrics = per_file_metrics
        self.logger.info(f"✅ Calculated per-file metrics for {len(per_file_metrics)} files")
        return per_file_metrics
    
    def create_confusion_matrix_plot(self, save_path: str = None) -> str:
        """Create and save confusion matrix visualization."""
        if not self.metrics:
            raise ValueError("Metrics not calculated. Call calculate_metrics() first.")
        
        cm = np.array(self.metrics['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Exclude', 'Include'], 
                   yticklabels=['Exclude', 'Include'])
        plt.title('Confusion Matrix - Screening Agent Performance')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add performance metrics as text
        accuracy = self.metrics['accuracy']
        f1 = self.metrics['f1_score']
        plt.figtext(0.02, 0.02, f'Accuracy: {accuracy:.3f} | F1-Score: {f1:.3f}', 
                   fontsize=10, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"confusion_matrix_{timestamp}.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Confusion matrix saved to {save_path}")
        return save_path
    
    def create_metrics_summary_plot(self, save_path: str = None) -> str:
        """Create a summary plot of all key metrics."""
        if not self.metrics:
            raise ValueError("Metrics not calculated. Call calculate_metrics() first.")
        
        metrics_to_plot = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score']
        values = [self.metrics[metric] for metric in metrics_to_plot]
        labels = ['Accuracy', 'Balanced\nAccuracy', 'Precision', 'Recall', 'F1-Score']
        
        plt.figure(figsize=(12, 6))
        
        # Bar plot
        bars = plt.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        plt.title('Screening Agent Performance Metrics', fontsize=16, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add horizontal line at 0.5 for reference
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random Baseline')
        plt.legend()
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"metrics_summary_{timestamp}.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Metrics summary plot saved to {save_path}")
        return save_path
    
    def save_detailed_results_csv(self, save_path: str = None) -> str:
        """Save detailed results to CSV."""
        if not self.detailed_results:
            raise ValueError("No detailed results available. Call calculate_metrics() first.")
        
        df = pd.DataFrame(self.detailed_results)
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"detailed_screening_results_{timestamp}.csv"
        
        df.to_csv(save_path, index=False)
        self.logger.info(f"✅ Detailed results saved to {save_path}")
        return save_path
    
    def save_metrics_summary_csv(self, save_path: str = None) -> str:
        """Save metrics summary to CSV."""
        if not self.metrics:
            raise ValueError("Metrics not calculated. Call calculate_metrics() first.")
        
        # Create a summary DataFrame
        summary_data = []
        
        # Basic metrics
        basic_metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score']
        for metric in basic_metrics:
            summary_data.append({
                'metric': metric,
                'value': self.metrics[metric],
                'category': 'performance'
            })
        
        # Macro metrics
        macro_metrics = ['precision_macro', 'recall_macro', 'f1_score_macro']
        for metric in macro_metrics:
            summary_data.append({
                'metric': metric,
                'value': self.metrics[metric],
                'category': 'macro_average'
            })
        
        # Confusion matrix elements
        cm_metrics = ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']
        for metric in cm_metrics:
            summary_data.append({
                'metric': metric,
                'value': self.metrics[metric],
                'category': 'confusion_matrix'
            })
        
        # Additional metrics
        additional_metrics = ['sensitivity', 'specificity', 'npv']
        for metric in additional_metrics:
            summary_data.append({
                'metric': metric,
                'value': self.metrics[metric],
                'category': 'additional_metrics'
            })
        
        # Count metrics
        count_metrics = ['total_studies', 'correct_predictions', 'incorrect_predictions']
        for metric in count_metrics:
            summary_data.append({
                'metric': metric,
                'value': self.metrics[metric],
                'category': 'counts'
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"metrics_summary_{timestamp}.csv"
        
        df_summary.to_csv(save_path, index=False)
        self.logger.info(f"✅ Metrics summary saved to {save_path}")
        return save_path
    
    def save_per_file_metrics_csv(self, save_path: str = None) -> str:
        """Save detailed per-file metrics to CSV."""
        if not self.per_file_metrics:
            self.calculate_per_file_metrics()
        
        # Prepare data for CSV
        csv_data = []
        for file_name, metrics in self.per_file_metrics.items():
            row = {
                'file_name': file_name,
                'total_studies': metrics['total_studies'],
                'correct_predictions': metrics['correct_predictions'],
                'incorrect_predictions': metrics['incorrect_predictions'],
                'accuracy': metrics['accuracy'],
                'balanced_accuracy': metrics['balanced_accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'true_positives': metrics['true_positives'],
                'true_negatives': metrics['true_negatives'],
                'false_positives': metrics['false_positives'],
                'false_negatives': metrics['false_negatives'],
                'total_include_studies': metrics['total_include_studies'],
                'total_exclude_studies': metrics['total_exclude_studies'],
                'correct_include_predictions': metrics['correct_include_predictions'],
                'correct_exclude_predictions': metrics['correct_exclude_predictions'],
                'include_accuracy': metrics['include_accuracy'],
                'exclude_accuracy': metrics['exclude_accuracy']
            }
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"per_file_metrics_{timestamp}.csv"
        
        df.to_csv(save_path, index=False)
        self.logger.info(f"✅ Per-file metrics CSV saved to {save_path}")
        return save_path
    
    def generate_complete_analysis(self, output_dir: str = "analysis_results") -> Dict[str, str]:
        """Generate complete analysis with all outputs including enhanced per-file analysis."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not self.metrics:
            self.calculate_metrics()
        self.calculate_per_file_metrics()
        outputs = {}
        outputs['confusion_matrix'] = self.create_confusion_matrix_plot(
            os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
        )
        outputs['metrics_plot'] = self.create_metrics_summary_plot(
            os.path.join(output_dir, f"metrics_summary_{timestamp}.png")
        )
        outputs['detailed_csv'] = self.save_detailed_results_csv(
            os.path.join(output_dir, f"detailed_results_{timestamp}.csv")
        )
        outputs['metrics_csv'] = self.save_metrics_summary_csv(
            os.path.join(output_dir, f"metrics_summary_{timestamp}.csv")
        )
        outputs['report'] = self.save_friendly_report(
            os.path.join(output_dir, f"performance_report_{timestamp}.txt")
        )
        # Save metrics table CSV
        outputs['metrics_table_csv'] = self.save_metrics_table_csv(
            os.path.join(output_dir, f"metrics_table_{timestamp}.csv")
        )
        # Save model parameters table CSV (use real params if available)
        outputs['model_parameters_table_csv'] = self.save_model_parameters_table_csv(
            {
                'Temperature': 0.2,
                'Maximum length': 4096,
                'Stop sequences': '-',
                'Top-p': '-',
                'Top-k': '-',
                'Frequency penalty': '-',
                'Presence penalty': '-',
            },
            os.path.join(output_dir, f"model_parameters_table_{timestamp}.csv")
        )
        # Output RIS files for relevant/irrelevant
        relevant_ris = os.path.join(output_dir, f"llm_relevant_{timestamp}.ris")
        irrelevant_ris = os.path.join(output_dir, f"llm_irrelevant_{timestamp}.ris")
        self.output_ris_files_by_llm_label(relevant_ris, irrelevant_ris)
        outputs['llm_relevant_ris'] = relevant_ris
        outputs['llm_irrelevant_ris'] = irrelevant_ris
        self.logger.info(f"✅ Complete analysis with per-file enhancements saved to {output_dir}")
        self.logger.info(f"✅ Generated {len(outputs)} output files")
        return outputs
    
    def generate_friendly_report(self) -> str:
        """Generate a friendly, human-readable performance report."""
        if not self.metrics:
            raise ValueError("Metrics not calculated. Call calculate_metrics() first.")
        report = []
        report.append("🔍 SCREENING AGENT PERFORMANCE REPORT")
        report.append("=" * 50)
        report.append("")
        # Overview
        total = self.metrics['total_studies']
        correct = self.metrics['correct_predictions']
        accuracy = self.metrics['accuracy']
        report.append(f"📊 OVERVIEW")
        report.append(f"Total studies: {total}")
        report.append(f"Correct predictions: {correct}")
        report.append(f"Accuracy: {accuracy:.3f}")
        report.append("")
        report.append("## Key Metrics Table\n")
        report.append(self.get_metrics_table())
        report.append("")
        report.append("## Model Parameters Table\n")
        report.append(self.get_model_parameters_table())
        return "\n".join(report)

    def save_friendly_report(self, save_path: str = None) -> str:
        """Save the friendly report to a text file."""
        report = self.generate_friendly_report()
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"performance_report_{timestamp}.txt"
        with open(save_path, 'w') as f:
            f.write(report)
        self.logger.info(f"✅ Friendly report saved to {save_path}")
        return save_path


def main():
    """Main function to run the metrics analysis."""
    logger = init_logger(__name__)
    
    # Initialize analyzer
    analyzer = ScreeningMetricsAnalyzer()
    
    # Load results from the most recent file
    results_dir = "yaml_screening_results"
    try:
        analyzer.load_results_from_directory(results_dir)
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return
    
    # Generate complete analysis
    try:
        outputs = analyzer.generate_complete_analysis()
        
        logger.info("🎉 Analysis complete! Generated files:")
        for output_type, filepath in outputs.items():
            logger.info(f"   • {output_type}: {filepath}")
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()