#!/usr/bin/env python3
"""
Enhanced Visualization System for ML Training
===========================================

This module provides comprehensive visualization capabilities for model training,
including performance plots, class distribution analysis, and cross-batch comparisons.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

class EnhancedVisualizationSystem:
    """Comprehensive visualization system for ML training results."""
    
    def __init__(self, plots_dir: str = 'plots'):
        """Initialize visualization system."""
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True, parents=True)
        
        # Configure matplotlib for better plots
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 150
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
    def create_class_distribution_plots(self, y_original: np.ndarray, y_balanced: np.ndarray = None, 
                                      batch_id: Optional[int] = None) -> str:
        """Create class distribution visualization."""
        try:
            fig, axes = plt.subplots(1, 2 if y_balanced is not None else 1, figsize=(12, 5))
            if y_balanced is None:
                axes = [axes]
            
            # Original distribution
            unique, counts = np.unique(y_original, return_counts=True)
            colors = ['lightcoral', 'lightblue']
            
            bars1 = axes[0].bar(['Negative (0)', 'Positive (1)'], counts, color=colors, alpha=0.7)
            axes[0].set_title(f'Original Class Distribution')
            axes[0].set_ylabel('Sample Count')
            
            # Add value labels on bars
            for i, bar in enumerate(bars1):
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{int(height)}\n({height/len(y_original)*100:.1f}%)',
                           ha='center', va='bottom')
            
            # Balanced distribution (if available)
            if y_balanced is not None:
                unique_bal, counts_bal = np.unique(y_balanced, return_counts=True)
                bars2 = axes[1].bar(['Negative (0)', 'Positive (1)'], counts_bal, color=colors, alpha=0.7)
                axes[1].set_title('After SMOTE Balancing')
                axes[1].set_ylabel('Sample Count')
                
                # Add value labels on bars
                for i, bar in enumerate(bars2):
                    height = bar.get_height()
                    axes[1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{int(height)}\n({height/len(y_balanced)*100:.1f}%)',
                               ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            suffix = f'_batch_{batch_id}' if batch_id else ''
            plot_path = self.plots_dir / f'class_distribution{suffix}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Class distribution plot saved: {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to create class distribution plot: {e}")
            plt.close()
            return ""
    
    def create_model_performance_plots(self, model_results: Dict[str, Any], 
                                     batch_id: Optional[int] = None) -> str:
        """Create comprehensive model performance visualization."""
        try:
            # Extract performance metrics
            model_names = []
            accuracies = []
            roc_aucs = []
            pr_aucs = []
            f1_scores = []
            precisions = []
            recalls = []
            
            for name, results in model_results.items():
                if 'metrics' in results:
                    metrics = results['metrics']
                    model_names.append(name.replace('_', ' ').title())
                    accuracies.append(metrics.get('accuracy', 0))
                    roc_aucs.append(metrics.get('roc_auc', 0))
                    pr_aucs.append(metrics.get('average_precision', 0))
                    f1_scores.append(metrics.get('f1', 0))
                    precisions.append(metrics.get('precision', 0))
                    recalls.append(metrics.get('recall', 0))
            
            if not model_names:
                logger.warning("No model results found for performance plots")
                return ""
            
            # Create subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.ravel()
            
            metrics_data = [
                (accuracies, 'Accuracy', 'Accuracy Score'),
                (roc_aucs, 'ROC-AUC', 'ROC-AUC Score'),
                (pr_aucs, 'PR-AUC', 'PR-AUC Score'),
                (f1_scores, 'F1 Score', 'F1 Score'),
                (precisions, 'Precision', 'Precision Score'),
                (recalls, 'Recall', 'Recall Score')
            ]
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
            
            for i, (values, title, ylabel) in enumerate(metrics_data):
                if i < len(axes):
                    bars = axes[i].bar(model_names, values, color=colors, alpha=0.8)
                    axes[i].set_title(f'{title} by Model')
                    axes[i].set_ylabel(ylabel)
                    axes[i].set_ylim(0, 1.0)
                    axes[i].tick_params(axis='x', rotation=45)
                    
                    # Add value labels on bars
                    for j, bar in enumerate(bars):
                        height = bar.get_height()
                        axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{height:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            suffix = f'_batch_{batch_id}' if batch_id else ''
            plot_path = self.plots_dir / f'model_performance{suffix}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Model performance plot saved: {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to create model performance plot: {e}")
            plt.close()
            return ""
    
    def create_confusion_matrices(self, model_results: Dict[str, Any], 
                                batch_id: Optional[int] = None) -> str:
        """Create confusion matrix plots for all models."""
        try:
            model_cms = []
            model_names = []
            
            for name, results in model_results.items():
                if 'confusion_matrix' in results:
                    model_cms.append(results['confusion_matrix'])
                    model_names.append(name.replace('_', ' ').title())
            
            if not model_cms:
                logger.warning("No confusion matrices found")
                return ""
            
            # Create subplots grid
            n_models = len(model_cms)
            cols = min(3, n_models)
            rows = (n_models + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            if n_models == 1:
                axes = [axes]
            elif rows == 1:
                axes = [axes] if cols == 1 else axes
            else:
                axes = axes.ravel()
            
            for i, (cm, name) in enumerate(zip(model_cms, model_names)):
                if i < len(axes):
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                              ax=axes[i], cbar=True)
                    axes[i].set_title(f'{name} Confusion Matrix')
                    axes[i].set_xlabel('Predicted')
                    axes[i].set_ylabel('Actual')
            
            # Hide empty subplots
            for i in range(n_models, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            suffix = f'_batch_{batch_id}' if batch_id else ''
            plot_path = self.plots_dir / f'confusion_matrices{suffix}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Confusion matrices plot saved: {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to create confusion matrices plot: {e}")
            plt.close()
            return ""
    
    def create_feature_importance_plot(self, feature_importance: Dict[str, float], 
                                     top_k: int = 20, batch_id: Optional[int] = None) -> str:
        """Create feature importance visualization."""
        try:
            if not feature_importance:
                logger.warning("No feature importance data available")
                return ""
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:top_k]
            
            features, importances = zip(*top_features)
            
            # Create horizontal bar plot
            fig, ax = plt.subplots(figsize=(10, max(8, len(features)*0.4)))
            
            y_pos = np.arange(len(features))
            bars = ax.barh(y_pos, importances, alpha=0.8, color='forestgreen')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'Top {top_k} Feature Importances')
            ax.invert_yaxis()  # Top features at the top
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2.,
                       f'{width:.3f}', ha='left', va='center')
            
            plt.tight_layout()
            
            # Save plot
            suffix = f'_batch_{batch_id}' if batch_id else ''
            plot_path = self.plots_dir / f'feature_importance{suffix}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Feature importance plot saved: {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to create feature importance plot: {e}")
            plt.close()
            return ""
    
    def create_ensemble_weights_plot(self, ensemble_weights: Dict[str, float], 
                                   batch_id: Optional[int] = None) -> str:
        """Create ensemble model weights visualization."""
        try:
            if not ensemble_weights:
                logger.warning("No ensemble weights data available")
                return ""
            
            models = list(ensemble_weights.keys())
            weights = list(ensemble_weights.values())
            
            # Create pie chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Pie chart
            colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
            wedges, texts, autotexts = ax1.pie(weights, labels=models, autopct='%1.1f%%',
                                             colors=colors, startangle=90)
            ax1.set_title('Ensemble Model Weights Distribution')
            
            # Bar chart
            bars = ax2.bar(range(len(models)), weights, color=colors, alpha=0.8)
            ax2.set_xticks(range(len(models)))
            ax2.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=45)
            ax2.set_ylabel('Weight')
            ax2.set_title('Ensemble Model Weights')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{height:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            suffix = f'_batch_{batch_id}' if batch_id else ''
            plot_path = self.plots_dir / f'ensemble_weights{suffix}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Ensemble weights plot saved: {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to create ensemble weights plot: {e}")
            plt.close()
            return ""
    
    def create_training_summary_plot(self, training_results: Dict[str, Any], 
                                   batch_id: Optional[int] = None) -> str:
        """Create comprehensive training summary visualization."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Training timeline
            if 'training_timeline' in training_results:
                timeline = training_results['training_timeline']
                ax1.plot(timeline.get('timestamps', []), timeline.get('progress', []))
                ax1.set_title('Training Progress Over Time')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Progress (%)')
            else:
                ax1.text(0.5, 0.5, 'No timeline data available', ha='center', va='center')
                ax1.set_title('Training Timeline (N/A)')
            
            # 2. Class imbalance metrics
            if 'class_metrics' in training_results:
                metrics = training_results['class_metrics']
                before = [metrics.get('negative_before', 0), metrics.get('positive_before', 0)]
                after = [metrics.get('negative_after', 0), metrics.get('positive_after', 0)]
                
                x = np.arange(2)
                width = 0.35
                
                ax2.bar(x - width/2, before, width, label='Before SMOTE', alpha=0.7)
                ax2.bar(x + width/2, after, width, label='After SMOTE', alpha=0.7)
                ax2.set_xticks(x)
                ax2.set_xticklabels(['Negative', 'Positive'])
                ax2.set_title('Class Balance: Before vs After SMOTE')
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, 'No class metrics available', ha='center', va='center')
                ax2.set_title('Class Balance Metrics (N/A)')
            
            # 3. Model comparison radar chart (simplified)
            if 'model_results' in training_results:
                models = list(training_results['model_results'].keys())[:5]  # Top 5 models
                metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                
                if models:
                    for i, model in enumerate(models):
                        model_metrics = training_results['model_results'][model].get('metrics', {})
                        values = [model_metrics.get(metric, 0) for metric in metrics]
                        ax3.plot(metrics, values, marker='o', label=model.replace('_', ' ').title())
                    
                    ax3.set_ylim(0, 1)
                    ax3.set_title('Model Performance Comparison')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                else:
                    ax3.text(0.5, 0.5, 'No model results available', ha='center', va='center')
                    ax3.set_title('Model Comparison (N/A)')
            else:
                ax3.text(0.5, 0.5, 'No model results available', ha='center', va='center')
                ax3.set_title('Model Comparison (N/A)')
            
            # 4. Final ensemble performance
            if 'ensemble_metrics' in training_results:
                ensemble = training_results['ensemble_metrics']
                metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC']
                values = [
                    ensemble.get('accuracy', 0),
                    ensemble.get('precision', 0),
                    ensemble.get('recall', 0),
                    ensemble.get('f1', 0),
                    ensemble.get('roc_auc', 0),
                    ensemble.get('pr_auc', 0)
                ]
                
                bars = ax4.bar(metrics_names, values, color='gold', alpha=0.8)
                ax4.set_ylim(0, 1)
                ax4.set_title('Final Ensemble Performance')
                ax4.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            else:
                ax4.text(0.5, 0.5, 'No ensemble metrics available', ha='center', va='center')
                ax4.set_title('Ensemble Performance (N/A)')
            
            plt.tight_layout()
            
            # Save plot
            suffix = f'_batch_{batch_id}' if batch_id else ''
            plot_path = self.plots_dir / f'training_summary{suffix}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Training summary plot saved: {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to create training summary plot: {e}")
            plt.close()
            return ""
    
    def generate_all_plots(self, training_results: Dict[str, Any], 
                          y_original: np.ndarray = None, y_balanced: np.ndarray = None,
                          batch_id: Optional[int] = None) -> List[str]:
        """Generate all visualization plots for training results."""
        plot_paths = []
        
        logger.info(f"🎨 Generating comprehensive visualization plots...")
        
        # Class distribution plots
        if y_original is not None:
            path = self.create_class_distribution_plots(y_original, y_balanced, batch_id)
            if path:
                plot_paths.append(path)
        
        # Model performance plots
        if 'model_results' in training_results:
            path = self.create_model_performance_plots(training_results['model_results'], batch_id)
            if path:
                plot_paths.append(path)
        
        # Confusion matrices
        if 'model_results' in training_results:
            path = self.create_confusion_matrices(training_results['model_results'], batch_id)
            if path:
                plot_paths.append(path)
        
        # Feature importance
        if 'feature_importance' in training_results:
            path = self.create_feature_importance_plot(training_results['feature_importance'], batch_id=batch_id)
            if path:
                plot_paths.append(path)
        
        # Ensemble weights
        if 'ensemble_weights' in training_results:
            path = self.create_ensemble_weights_plot(training_results['ensemble_weights'], batch_id)
            if path:
                plot_paths.append(path)
        
        # Training summary
        path = self.create_training_summary_plot(training_results, batch_id)
        if path:
            plot_paths.append(path)
        
        logger.info(f"✅ Generated {len(plot_paths)} visualization plots")
        return plot_paths


def create_cross_batch_analysis(reports_dir: str = 'reports') -> str:
    """Create cross-batch performance analysis visualization."""
    try:
        reports_path = Path(reports_dir)
        if not reports_path.exists():
            logger.error(f"Reports directory not found: {reports_dir}")
            return ""
        
        # Collect data from all batch telemetry files
        batch_data = []
        for batch_dir in reports_path.glob('batch_*'):
            if batch_dir.is_dir():
                telemetry_file = batch_dir / f'{batch_dir.name}_telemetry.json'
                if telemetry_file.exists():
                    try:
                        with open(telemetry_file, 'r') as f:
                            data = json.load(f)
                        
                        batch_num = int(batch_dir.name.replace('batch_', ''))
                        batch_data.append({
                            'batch': batch_num,
                            'pr_auc': data.get('pr_auc_meta', 0),
                            'gate': data.get('gate', 'UNKNOWN'),
                            'models_trained': data.get('models_trained', 0),
                            'training_time': data.get('training_time_seconds', 0),
                            'samples': data.get('samples_processed', 0)
                        })
                    except Exception as e:
                        logger.warning(f"Failed to load telemetry for {batch_dir.name}: {e}")
        
        if not batch_data:
            logger.warning("No batch data found for cross-batch analysis")
            return ""
        
        # Sort by batch number
        batch_data.sort(key=lambda x: x['batch'])
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract data for plotting
        batches = [d['batch'] for d in batch_data]
        pr_aucs = [d['pr_auc'] for d in batch_data]
        gates = [d['gate'] for d in batch_data]
        training_times = [d['training_time'] / 60 for d in batch_data]  # Convert to minutes
        samples = [d['samples'] for d in batch_data]
        
        # 1. PR-AUC performance across batches
        colors = ['green' if gate == 'PASSED' else 'red' for gate in gates]
        bars1 = ax1.bar(batches, pr_aucs, color=colors, alpha=0.7)
        ax1.axhline(0.05, color='orange', linestyle='--', label='Minimum Threshold')
        ax1.set_xlabel('Batch Number')
        ax1.set_ylabel('PR-AUC Score')
        ax1.set_title('PR-AUC Performance Across Batches')
        ax1.legend()
        
        # Add value labels
        for bar, value in zip(bars1, pr_aucs):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 2. Training time analysis
        ax2.plot(batches, training_times, marker='o', linewidth=2, markersize=6)
        ax2.set_xlabel('Batch Number')
        ax2.set_ylabel('Training Time (minutes)')
        ax2.set_title('Training Time Across Batches')
        ax2.grid(True, alpha=0.3)
        
        # 3. Sample count per batch
        ax3.bar(batches, samples, color='lightblue', alpha=0.7)
        ax3.set_xlabel('Batch Number')
        ax3.set_ylabel('Sample Count')
        ax3.set_title('Sample Count per Batch')
        
        # 4. Success rate summary
        passed_count = sum(1 for gate in gates if gate == 'PASSED')
        failed_count = len(gates) - passed_count
        
        ax4.pie([passed_count, failed_count], labels=['Passed', 'Failed'], 
               colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%', startangle=90)
        ax4.set_title(f'Overall Success Rate\n({passed_count}/{len(gates)} batches passed)')
        
        plt.suptitle('Cross-Batch Performance Analysis', fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(reports_dir) / 'cross_batch_analysis.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Cross-batch analysis plot saved: {plot_path}")
        return str(plot_path)
        
    except Exception as e:
        logger.error(f"❌ Failed to create cross-batch analysis: {e}")
        plt.close()
        return ""


# Global visualization system instance
enhanced_visualizer = EnhancedVisualizationSystem()
