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
        
        # Calculate metrics if not done already
        if not self.metrics:
            self.calculate_metrics()
        
        # Calculate per-file metrics
        self.calculate_per_file_metrics()
        
        outputs = {}
        
        # Save all existing outputs
        outputs['confusion_matrix'] = self.create_confusion_matrix_plot(
            os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
        )
        outputs['metrics_plot'] = self.create_metrics_summary_plot(
            os.path.join(output_dir, f"metrics_summary_{timestamp}.png")
        )
        outputs['comprehensive_plot'] = self.create_comprehensive_analysis_plot(
            os.path.join(output_dir, f"comprehensive_analysis_{timestamp}.png")
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
        
        # Add new per-file analysis outputs
        outputs['per_file_detailed_plot'] = self.create_per_file_detailed_plot(
            os.path.join(output_dir, f"per_file_detailed_analysis_{timestamp}.png")
        )
        outputs['per_file_comparison_plot'] = self.create_per_file_comparison_plot(
            os.path.join(output_dir, f"per_file_comparison_{timestamp}.png")
        )
        outputs['per_file_metrics_csv'] = self.save_per_file_metrics_csv(
            os.path.join(output_dir, f"per_file_metrics_{timestamp}.csv")
        )
        outputs['per_file_report'] = self.save_per_file_report(
            save_path=os.path.join(output_dir, f"per_file_report_all_files_{timestamp}.txt")
        )
        
        # Generate individual file reports
        for file_name in self.per_file_metrics.keys():
            # Extract just the base filename and sanitize it
            base_name = os.path.basename(file_name)
            safe_name = base_name.replace('.ris', '').replace('dummy_shortages_', '')
            individual_report_path = os.path.join(output_dir, f"per_file_report_{safe_name}_{timestamp}.txt")
            outputs[f'individual_report_{safe_name}'] = self.save_per_file_report(
                file_name=file_name, 
                save_path=individual_report_path
            )
        
        # Print friendly reports to console
        print("\n" + self.generate_friendly_report())
        print("\n" + "="*80)
        print(self.generate_per_file_report())
        
        self.logger.info(f"✅ Complete analysis with per-file enhancements saved to {output_dir}")
        self.logger.info(f"✅ Generated {len(outputs)} output files")
        return outputs
    
    def create_per_file_detailed_plot(self, save_path: str = None) -> str:
        """Create detailed per-file analysis visualization."""
        if not self.per_file_metrics:
            self.calculate_per_file_metrics()
        
        n_files = len(self.per_file_metrics)
        fig, axes = plt.subplots(2, n_files, figsize=(5 * n_files, 10))
        
        if n_files == 1:
            axes = axes.reshape(2, 1)
        
        file_names = list(self.per_file_metrics.keys())
        
        for i, file_name in enumerate(file_names):
            metrics = self.per_file_metrics[file_name]
            
            # Top row: Confusion Matrix
            ax_cm = axes[0, i]
            cm = np.array(metrics['confusion_matrix'])
            
            # Handle different confusion matrix shapes
            if cm.size == 0:
                cm = np.array([[0, 0], [0, 0]])
            elif cm.shape == (1, 1):
                # Expand to 2x2 if only one class present
                cm = np.array([[cm[0, 0], 0], [0, 0]]) if metrics['total_exclude_studies'] > 0 else np.array([[0, 0], [0, cm[0, 0]]])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Exclude', 'Include'],
                       yticklabels=['Exclude', 'Include'],
                       ax=ax_cm)
            
            short_name = file_name.replace('.ris', '').replace('dummy_shortages_', '')
            ax_cm.set_title(f'{short_name}\nConfusion Matrix', fontsize=12, fontweight='bold')
            ax_cm.set_ylabel('True Label')
            ax_cm.set_xlabel('Predicted Label')
            
            # Bottom row: Performance Metrics
            ax_metrics = axes[1, i]
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            metric_values = [
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1_score']
            ]
            
            bars = ax_metrics.bar(metric_names, metric_values, 
                                 color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax_metrics.set_title(f'{short_name}\nPerformance Metrics', fontsize=12, fontweight='bold')
            ax_metrics.set_ylabel('Score')
            ax_metrics.set_ylim(0, 1)
            ax_metrics.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                ax_metrics.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                               f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            # Add study count info
            total_studies = metrics['total_studies']
            correct = metrics['correct_predictions']
            ax_metrics.text(0.5, 0.85, f'Studies: {total_studies}\nCorrect: {correct}',
                           transform=ax_metrics.transAxes, ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7),
                           fontsize=9)
        
        plt.suptitle('Per-File Detailed Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"per_file_detailed_analysis_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Per-file detailed plot saved to {save_path}")
        return save_path

    def create_per_file_comparison_plot(self, save_path: str = None) -> str:
        """Create a comparison plot showing metrics across all files."""
        if not self.per_file_metrics:
            self.calculate_per_file_metrics()
        
        file_names = list(self.per_file_metrics.keys())
        short_names = [f.replace('.ris', '').replace('dummy_shortages_', '') for f in file_names]
        
        # Prepare data for plotting
        metrics_data = {
            'Accuracy': [self.per_file_metrics[f]['accuracy'] for f in file_names],
            'Precision': [self.per_file_metrics[f]['precision'] for f in file_names],
            'Recall': [self.per_file_metrics[f]['recall'] for f in file_names],
            'F1-Score': [self.per_file_metrics[f]['f1_score'] for f in file_names]
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            ax = axes[i]
            bars = ax.bar(short_names, values, color=colors[i], alpha=0.7)
            ax.set_title(f'{metric_name} by File', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric_name)
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            # Add horizontal line at 0.5 for reference
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
            
            # Add average line
            avg_value = np.mean(values)
            ax.axhline(y=avg_value, color=colors[i], linestyle='-', alpha=0.8, linewidth=2,
                      label=f'Average: {avg_value:.3f}')
            ax.legend()
        
        plt.suptitle('Per-File Metrics Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"per_file_comparison_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Per-file comparison plot saved to {save_path}")
        return save_path

    def generate_per_file_report(self, file_name: str = None) -> str:
        """Generate detailed report for a specific file or all files."""
        if not self.per_file_metrics:
            self.calculate_per_file_metrics()
        
        if file_name and file_name not in self.per_file_metrics:
            raise ValueError(f"File '{file_name}' not found in metrics")
        
        files_to_report = [file_name] if file_name else list(self.per_file_metrics.keys())
        
        report = []
        report.append("📊 PER-FILE SCREENING ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        for i, fname in enumerate(files_to_report):
            if i > 0:
                report.append("\n" + "─" * 60 + "\n")
            
            metrics = self.per_file_metrics[fname]
            short_name = fname.replace('.ris', '').replace('dummy_shortages_', '')
            
            report.append(f"📁 FILE: {short_name}")
            report.append(f"   Full name: {fname}")
            report.append("")
            
            # Basic Statistics
            report.append("📈 BASIC STATISTICS")
            report.append(f"   • Total studies analyzed: {metrics['total_studies']}")
            report.append(f"   • Studies to include: {metrics['total_include_studies']}")
            report.append(f"   • Studies to exclude: {metrics['total_exclude_studies']}")
            report.append(f"   • Correct predictions: {metrics['correct_predictions']} ({metrics['accuracy']:.1%})")
            report.append(f"   • Incorrect predictions: {metrics['incorrect_predictions']} ({(1-metrics['accuracy']):.1%})")
            report.append("")
            
            # Performance Metrics
            report.append("🎯 PERFORMANCE METRICS")
            report.append(f"   • Overall Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
            report.append(f"   • Balanced Accuracy: {metrics['balanced_accuracy']:.3f} ({metrics['balanced_accuracy']*100:.1f}%)")
            report.append(f"   • Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
            report.append(f"   • Recall: {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
            report.append(f"   • F1-Score: {metrics['f1_score']:.3f} ({metrics['f1_score']*100:.1f}%)")
            report.append("")
            
            # Label-specific accuracy
            report.append("🏷️  LABEL-SPECIFIC ACCURACY")
            if metrics['total_include_studies'] > 0:
                report.append(f"   • Include studies accuracy: {metrics['include_accuracy']:.3f} ({metrics['include_accuracy']*100:.1f}%)")
                report.append(f"     - Correctly identified: {metrics['correct_include_predictions']}/{metrics['total_include_studies']}")
            else:
                report.append("   • Include studies accuracy: N/A (no include studies in this file)")
            
            if metrics['total_exclude_studies'] > 0:
                report.append(f"   • Exclude studies accuracy: {metrics['exclude_accuracy']:.3f} ({metrics['exclude_accuracy']*100:.1f}%)")
                report.append(f"     - Correctly identified: {metrics['correct_exclude_predictions']}/{metrics['total_exclude_studies']}")
            else:
                report.append("   • Exclude studies accuracy: N/A (no exclude studies in this file)")
            report.append("")
            
            # Confusion Matrix Details
            tp = metrics['true_positives']
            tn = metrics['true_negatives']
            fp = metrics['false_positives']
            fn = metrics['false_negatives']
            
            report.append("🔍 DETAILED BREAKDOWN")
            report.append(f"   • True Positives (Correctly included): {tp}")
            report.append(f"   • True Negatives (Correctly excluded): {tn}")
            report.append(f"   • False Positives (Incorrectly included): {fp}")
            report.append(f"   • False Negatives (Incorrectly excluded): {fn}")
            report.append("")
            
            # Performance Assessment
            report.append("💡 PERFORMANCE ASSESSMENT")
            
            if metrics['accuracy'] >= 0.9:
                assessment = "Excellent performance"
            elif metrics['accuracy'] >= 0.8:
                assessment = "Good performance"
            elif metrics['accuracy'] >= 0.7:
                assessment = "Fair performance"
            else:
                assessment = "Performance needs improvement"
            
            report.append(f"   • Overall: {assessment} (Accuracy: {metrics['accuracy']:.3f})")
            
            # Specific issues and recommendations
            issues = []
            recommendations = []
            
            if fp > 0:
                issues.append(f"{fp} false positive(s) - studies incorrectly included")
                recommendations.append("Review inclusion criteria to reduce false positives")
            
            if fn > 0:
                issues.append(f"{fn} false negative(s) - relevant studies missed")
                recommendations.append("Review exclusion criteria to improve sensitivity")
            
            if metrics['precision'] < 0.8:
                issues.append(f"Low precision ({metrics['precision']:.3f}) - many irrelevant studies included")
            
            if metrics['recall'] < 0.8:
                issues.append(f"Low recall ({metrics['recall']:.3f}) - missing relevant studies")
            
            if issues:
                report.append("")
                report.append("⚠️  IDENTIFIED ISSUES")
                for issue in issues:
                    report.append(f"   • {issue}")
            
            if recommendations:
                report.append("")
                report.append("🔧 RECOMMENDATIONS")
                for rec in recommendations:
                    report.append(f"   • {rec}")
            
            # Study details for incorrect predictions
            incorrect_studies = [s for s in metrics['detailed_results'] if not s['correct']]
            if incorrect_studies:
                report.append("")
                report.append("❌ INCORRECT PREDICTIONS")
                for study in incorrect_studies[:5]:  # Show first 5 incorrect predictions
                    report.append(f"   • Study {study['study_index']}: {study['title'][:60]}...")
                    report.append(f"     Expected: {study['true_label']}, Predicted: {study['predicted_label']}")
                
                if len(incorrect_studies) > 5:
                    report.append(f"   ... and {len(incorrect_studies) - 5} more incorrect predictions")
        
        report.append("")
        report.append(f"📅 Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(report)

    def save_per_file_report(self, file_name: str = None, save_path: str = None) -> str:
        """Save per-file report to a text file."""
        report = self.generate_per_file_report(file_name)
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_suffix = f"_{file_name.replace('.ris', '')}" if file_name else "_all_files"
            output_dir = "analysis_results"
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"per_file_report{file_suffix}_{timestamp}.txt")
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"✅ Per-file report saved to {save_path}")
        return save_path

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
        report.append(f"   • Total studies analyzed: {total}")
        report.append(f"   • Correct predictions: {correct} ({correct/total*100:.1f}%)")
        report.append(f"   • Incorrect predictions: {total-correct} ({(total-correct)/total*100:.1f}%)")
        report.append("")
        
        # Performance Metrics
        report.append(f"📈 PERFORMANCE METRICS")
        report.append(f"   • Accuracy: {self.metrics['accuracy']:.3f} ({self.metrics['accuracy']*100:.1f}%)")
        report.append(f"   • Balanced Accuracy: {self.metrics['balanced_accuracy']:.3f} ({self.metrics['balanced_accuracy']*100:.1f}%)")
        report.append(f"   • Precision: {self.metrics['precision']:.3f} ({self.metrics['precision']*100:.1f}%)")
        report.append(f"   • Recall: {self.metrics['recall']:.3f} ({self.metrics['recall']*100:.1f}%)")
        report.append(f"   • F1-Score: {self.metrics['f1_score']:.3f} ({self.metrics['f1_score']*100:.1f}%)")
        report.append("")
        
        # Confusion Matrix
        tp = self.metrics['true_positives']
        tn = self.metrics['true_negatives']
        fp = self.metrics['false_positives']
        fn = self.metrics['false_negatives']
        
        report.append(f"🎯 CONFUSION MATRIX")
        report.append(f"   • True Positives (Correctly included): {tp}")
        report.append(f"   • True Negatives (Correctly excluded): {tn}")
        report.append(f"   • False Positives (Incorrectly included): {fp}")
        report.append(f"   • False Negatives (Incorrectly excluded): {fn}")
        report.append("")
        
        # Performance Interpretation
        report.append(f"💡 PERFORMANCE INTERPRETATION")
        
        # Accuracy interpretation
        if accuracy >= 0.9:
            acc_desc = "Excellent"
        elif accuracy >= 0.8:
            acc_desc = "Good"
        elif accuracy >= 0.7:
            acc_desc = "Fair"
        else:
            acc_desc = "Needs Improvement"
        
        report.append(f"   • Overall Performance: {acc_desc} (Accuracy: {accuracy:.3f})")
        
        # Precision interpretation
        precision = self.metrics['precision']
        if precision >= 0.9:
            prec_desc = "Very reliable - few false positives"
        elif precision >= 0.8:
            prec_desc = "Reliable - some false positives"
        elif precision >= 0.7:
            prec_desc = "Moderately reliable - notable false positives"
        else:
            prec_desc = "Many false positives - review inclusion criteria"
        
        report.append(f"   • Precision: {prec_desc} ({precision:.3f})")
        
        # Recall interpretation
        recall = self.metrics['recall']
        if recall >= 0.9:
            rec_desc = "Excellent sensitivity - catches most relevant studies"
        elif recall >= 0.8:
            rec_desc = "Good sensitivity - catches most relevant studies"
        elif recall >= 0.7:
            rec_desc = "Moderate sensitivity - misses some relevant studies"
        else:
            rec_desc = "Low sensitivity - misses many relevant studies"
        
        report.append(f"   • Recall: {rec_desc} ({recall:.3f})")
        report.append("")
        
        # Recommendations
        report.append(f"🔧 RECOMMENDATIONS")
        
        if precision < 0.8 and recall >= 0.8:
            report.append("   • Consider tightening inclusion criteria to reduce false positives")
        elif recall < 0.8 and precision >= 0.8:
            report.append("   • Consider broadening inclusion criteria to catch more relevant studies")
        elif precision < 0.8 and recall < 0.8:
            report.append("   • Review and refine both inclusion and exclusion criteria")
        else:
            report.append("   • Performance is good! Consider testing on larger datasets")
        
        if fp > 0:
            report.append(f"   • Investigate {fp} false positive(s) to understand misclassification patterns")
        if fn > 0:
            report.append(f"   • Investigate {fn} false negative(s) to improve sensitivity")
        
        report.append("")
        report.append(f"📅 Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(report)

    def save_friendly_report(self, save_path: str = None) -> str:
        """Save the friendly report to a text file."""
        report = self.generate_friendly_report()
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"performance_report_{timestamp}.txt"
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"✅ Performance report saved to {save_path}")
        return save_path

    def create_comprehensive_analysis_plot(self, save_path: str = None) -> str:
        """Create a comprehensive plot with confusion matrix, core accuracy, and per-file accuracy."""
        if not self.metrics:
            raise ValueError("Metrics not calculated. Call calculate_metrics() first.")
        
        # Calculate per-file accuracy
        file_accuracy = {}
        for result in self.detailed_results:
            file_name = result['file']
            if file_name not in file_accuracy:
                file_accuracy[file_name] = {'correct': 0, 'total': 0}
            file_accuracy[file_name]['total'] += 1
            if result['correct']:
                file_accuracy[file_name]['correct'] += 1
        
        # Calculate accuracy rates
        for file_name in file_accuracy:
            file_accuracy[file_name]['accuracy'] = (
                file_accuracy[file_name]['correct'] / file_accuracy[file_name]['total']
            )
        
        # Create subplot layout
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])
        
        # 1. Confusion Matrix (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        cm = np.array(self.metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Exclude', 'Include'], 
                   yticklabels=['Exclude', 'Include'],
                   ax=ax1)
        ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # 2. Core Accuracy Metrics (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        metrics_to_plot = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score']
        values = [self.metrics[metric] for metric in metrics_to_plot]
        labels = ['Accuracy', 'Balanced\nAccuracy', 'Precision', 'Recall', 'F1-Score']
        
        bars = ax2.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax2.set_title('Core Performance Metrics', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Add horizontal line at 0.5 for reference
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random Baseline')
        ax2.legend()
        
        # 3. Per-File Accuracy (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        file_names = list(file_accuracy.keys())
        file_acc_values = [file_accuracy[f]['accuracy'] for f in file_names]
        
        # Shorten file names for display
        short_names = [f.replace('.ris', '').replace('dummy_shortages_', '') for f in file_names]
        
        bars_files = ax3.bar(short_names, file_acc_values, color=['#ff9999', '#66b3ff'])
        ax3.set_title('Per-File Accuracy', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Accuracy')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels and count info
        for i, (bar, acc, file_name) in enumerate(zip(bars_files, file_acc_values, file_names)):
            correct = file_accuracy[file_name]['correct']
            total = file_accuracy[file_name]['total']
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{acc:.3f}\n({correct}/{total})', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9)
        
        # Add horizontal line at 0.5 for reference
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        
        # 4. Classification Report as Text (bottom spanning all columns)
        ax4 = fig.add_subplot(gs[1, :])
        ax4.axis('off')
        
        # Create classification report text
        tp = self.metrics['true_positives']
        tn = self.metrics['true_negatives']
        fp = self.metrics['false_positives']
        fn = self.metrics['false_negatives']
        
        report_text = f"""
DETAILED CLASSIFICATION ANALYSIS

Overall Performance:
• Total Studies: {self.metrics['total_studies']}
• Correct Predictions: {self.metrics['correct_predictions']} ({self.metrics['accuracy']:.1%})
• Incorrect Predictions: {self.metrics['incorrect_predictions']} ({(1-self.metrics['accuracy']):.1%})

Confusion Matrix Breakdown:
• True Positives (Correctly Included): {tp}
• True Negatives (Correctly Excluded): {tn}  
• False Positives (Incorrectly Included): {fp}
• False Negatives (Incorrectly Excluded): {fn}

Per-File Performance:
"""
        
        for file_name in file_names:
            acc = file_accuracy[file_name]['accuracy']
            correct = file_accuracy[file_name]['correct']
            total = file_accuracy[file_name]['total']
            report_text += f"• {file_name}: {acc:.1%} ({correct}/{total} correct)\n"
        
        # Performance insights
        if self.metrics['accuracy'] < 0.7:
            report_text += f"\n⚠️  Performance Alert: Overall accuracy ({self.metrics['accuracy']:.1%}) is below 70%"
        if fp > 0:
            report_text += f"\n🔍 Review: {fp} false positive(s) need investigation"
        if fn > 0:
            report_text += f"\n📋 Review: {fn} false negative(s) indicate missed relevant studies"
        
        ax4.text(0.05, 0.95, report_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        # Overall title
        fig.suptitle('Comprehensive Screening Agent Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"comprehensive_analysis_{timestamp}.png"
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Comprehensive analysis plot saved to {save_path}")
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