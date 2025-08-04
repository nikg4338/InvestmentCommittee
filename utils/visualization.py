#!/usr/bin/env python3
"""
Visualization Utilities
======================

Centralized visualization functions for training results, confusion matrices,
and performance metrics. Eliminates code duplication across the training pipeline.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

from config.training_config import VisualizationConfig, get_default_config

logger = logging.getLogger(__name__)

def ensure_reports_dir() -> str:
    """Ensure reports directory exists and return path"""
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)
    return reports_dir

def get_timestamp() -> str:
    """Get timestamp string for file naming"""
    return datetime.now().strftime('%Y-%m-%d_%H-%M')

def plot_confusion_matrices(confusion_matrices: Dict[str, np.ndarray], 
                          model_names: List[str],
                          config: Optional[VisualizationConfig] = None) -> Optional[str]:
    """
    Plot confusion matrices for all models in a grid layout.
    
    Args:
        confusion_matrices: Dictionary mapping model names to confusion matrices
        model_names: List of model names for consistent ordering
        config: Visualization configuration
        
    Returns:
        Path to saved plot file or None if visualization unavailable
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Matplotlib not available, skipping confusion matrix plots")
        return None

    if config is None:
        config = get_default_config().visualization
    
    # Determine batch identifier for title
    batch_id = getattr(config, 'batch_id', None) if config else None
    title_prefix = f"Batch {batch_id} - " if batch_id else ""

    # Calculate grid dimensions
    n_models = len(model_names)
    n_cols = min(3, n_models)  # Max 3 columns
    n_rows = (n_models + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, 
                           figsize=(config.matrix_figure_width * n_cols, 
                                  config.matrix_figure_height * n_rows),
                           dpi=config.chart_dpi)
    
    # Handle single subplot case
    if n_models == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()
    
    # Plot each confusion matrix
    for i, model_name in enumerate(model_names):
        ax = axes[i]
        
        if model_name in confusion_matrices:
            cm = confusion_matrices[model_name]
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Predicted 0', 'Predicted 1'],
                       yticklabels=['Actual 0', 'Actual 1'],
                       ax=ax)
            ax.set_title(f'{model_name.replace("_", " ").title()}')
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{model_name.replace("_", " ").title()}')
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Hide unused subplots
    for i in range(len(model_names), len(axes)):
        axes[i].set_visible(False)
    
    # Add main title with batch information
    fig.suptitle(f'{title_prefix}Confusion Matrices', fontsize=14, y=0.95)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Adjust to make room for suptitle
    
    # Save plot
    if config.save_plots:
        reports_dir = ensure_reports_dir()
        timestamp = get_timestamp()
        filename = f'confusion_matrices_{timestamp}.{config.plot_format}'
        filepath = os.path.join(reports_dir, filename)
        plt.savefig(filepath, dpi=config.chart_dpi, bbox_inches='tight')
        logger.info(f"Confusion matrices saved to {filepath}")
        plt.close()
        return filepath
    else:
        plt.show()
        return None

def plot_model_performance(metrics_df: pd.DataFrame,
                         config: Optional[VisualizationConfig] = None) -> Optional[str]:
    """
    Plot model performance comparison as bar charts.
    
    Args:
        metrics_df: DataFrame with models as rows and metrics as columns
        config: Visualization configuration
        
    Returns:
        Path to saved plot file or None if visualization unavailable
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Matplotlib not available, skipping performance plots")
        return None

    if config is None:
        config = get_default_config().visualization
    
    # Determine batch identifier for title
    batch_id = getattr(config, 'batch_id', None) if config else None
    title_prefix = f"Batch {batch_id} - " if batch_id else ""

    # Define metrics to plot (exclude non-numeric columns)
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    metrics_to_plot = [col for col in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc'] 
                      if col in numeric_cols]
                      
    if not metrics_to_plot:
        logger.warning("No numeric metrics found for plotting")
        return None
    
    # Create subplots
    n_metrics = len(metrics_to_plot)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols,
                           figsize=(config.chart_figure_width * n_cols,
                                  config.chart_figure_height * n_rows),
                           dpi=config.chart_dpi)
    
    # Handle single subplot case
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()
    
    # Plot each metric
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        # Create bar plot
        bars = ax.bar(range(len(metrics_df)), metrics_df[metric], 
                     alpha=0.8, color=plt.cm.Set3(np.linspace(0, 1, len(metrics_df))))
        
        # Customize plot
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_xlabel('Models')
        ax.set_ylabel(metric.title())
        ax.set_xticks(range(len(metrics_df)))
        ax.set_xticklabels([name.replace('_', ' ').title() for name in metrics_df.index], 
                          rotation=45, ha='right')
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Set y-axis limits
        ax.set_ylim(0, max(1.05, metrics_df[metric].max() * 1.1))
    
    # Hide unused subplots
    for i in range(len(metrics_to_plot), len(axes)):
        axes[i].set_visible(False)
    
    # Add main title with batch information
    fig.suptitle(f'{title_prefix}Model Performance Comparison', fontsize=14, y=0.95)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Adjust to make room for suptitle
    
    # Save plot
    if config.save_plots:
        reports_dir = ensure_reports_dir()
        timestamp = get_timestamp()
        filename = f'performance_comparison_{timestamp}.{config.plot_format}'
        filepath = os.path.join(reports_dir, filename)
        plt.savefig(filepath, dpi=config.chart_dpi, bbox_inches='tight')
        logger.info(f"Performance comparison saved to {filepath}")
        plt.close()
        return filepath
    else:
        plt.show()
        return None

def plot_training_curves(training_history: Dict[str, Dict[str, List[float]]],
                        config: Optional[VisualizationConfig] = None) -> Optional[str]:
    """
    Plot training curves for models that support them (like neural networks).
    
    Args:
        training_history: Dictionary mapping model names to history dictionaries
        config: Visualization configuration
        
    Returns:
        Path to saved plot file or None if visualization unavailable
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Matplotlib not available, skipping training curves")
        return None
    
    if config is None:
        config = get_default_config().visualization
    
    if not training_history:
        logger.info("No training history available for plotting")
        return None
    
    # Filter models with actual history data
    models_with_history = {name: hist for name, hist in training_history.items() 
                          if hist and isinstance(hist, dict)}
    
    if not models_with_history:
        logger.info("No models with training history found")
        return None
    
    # Create figure
    n_models = len(models_with_history)
    fig, axes = plt.subplots(1, n_models if n_models <= 3 else 3,
                           figsize=(config.chart_figure_width * min(n_models, 3),
                                  config.chart_figure_height),
                           dpi=config.chart_dpi)
    
    if n_models == 1:
        axes = [axes]
    elif not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    
    # Plot training curves for each model
    for i, (model_name, history) in enumerate(models_with_history.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Plot loss curves
        if 'loss' in history:
            epochs = range(1, len(history['loss']) + 1)
            ax.plot(epochs, history['loss'], label='Training Loss', color='blue')
            
        if 'val_loss' in history:
            epochs = range(1, len(history['val_loss']) + 1)
            ax.plot(epochs, history['val_loss'], label='Validation Loss', color='red')
        
        # Plot accuracy curves if available
        ax2 = ax.twinx()
        if 'accuracy' in history:
            epochs = range(1, len(history['accuracy']) + 1)
            ax2.plot(epochs, history['accuracy'], label='Training Accuracy', 
                    color='green', linestyle='--')
            
        if 'val_accuracy' in history:
            epochs = range(1, len(history['val_accuracy']) + 1)
            ax2.plot(epochs, history['val_accuracy'], label='Validation Accuracy', 
                    color='orange', linestyle='--')
        
        # Customize plot
        ax.set_title(f'{model_name.replace("_", " ").title()} Training Curves')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax2.set_ylabel('Accuracy')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    
    # Save plot
    if config.save_plots:
        reports_dir = ensure_reports_dir()
        timestamp = get_timestamp()
        filename = f'training_curves_{timestamp}.{config.plot_format}'
        filepath = os.path.join(reports_dir, filename)
        plt.savefig(filepath, dpi=config.chart_dpi, bbox_inches='tight')
        logger.info(f"Training curves saved to {filepath}")
        plt.close()
        return filepath
    else:
        plt.show()
        return None

def plot_threshold_analysis(threshold_results: Dict[str, Dict[str, float]],
                          config: Optional[VisualizationConfig] = None) -> Optional[str]:
    """
    Plot threshold optimization results.
    
    Args:
        threshold_results: Dictionary with threshold analysis results
        config: Visualization configuration
        
    Returns:
        Path to saved plot file or None if visualization unavailable
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Matplotlib not available, skipping threshold analysis plots")
        return None
    
    if config is None:
        config = get_default_config().visualization
    
    if not threshold_results:
        return None
    
    # Extract data for plotting
    models = list(threshold_results.keys())
    thresholds = [threshold_results[model].get('threshold', 0.5) for model in models]
    f1_scores = [threshold_results[model].get('f1_score', 0.0) for model in models]
    predicted_positives = [threshold_results[model].get('predicted_positives', 0) for model in models]
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, 
                                       figsize=(config.chart_figure_width * 3,
                                              config.chart_figure_height),
                                       dpi=config.chart_dpi)
    
    # Plot 1: Optimal thresholds
    bars1 = ax1.bar(range(len(models)), thresholds, alpha=0.8, color='skyblue')
    ax1.set_title('Optimal Thresholds')
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Threshold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=45, ha='right')
    
    # Add value labels
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: F1 scores at optimal thresholds
    bars2 = ax2.bar(range(len(models)), f1_scores, alpha=0.8, color='lightgreen')
    ax2.set_title('F1 Scores at Optimal Thresholds')
    ax2.set_xlabel('Models')
    ax2.set_ylabel('F1 Score')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=45, ha='right')
    
    # Add value labels
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Predicted positives
    bars3 = ax3.bar(range(len(models)), predicted_positives, alpha=0.8, color='coral')
    ax3.set_title('Predicted Positives')
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Count')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=45, ha='right')
    
    # Add value labels
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    if config.save_plots:
        reports_dir = ensure_reports_dir()
        timestamp = get_timestamp()
        filename = f'threshold_analysis_{timestamp}.{config.plot_format}'
        filepath = os.path.join(reports_dir, filename)
        plt.savefig(filepath, dpi=config.chart_dpi, bbox_inches='tight')
        logger.info(f"Threshold analysis saved to {filepath}")
        plt.close()
        return filepath
    else:
        plt.show()
        return None

def plot_class_distribution(y_train: pd.Series, y_test: pd.Series,
                          config: Optional[VisualizationConfig] = None) -> Optional[str]:
    """
    Plot class distribution in training and test sets.
    
    Args:
        y_train: Training labels
        y_test: Test labels
        config: Visualization configuration
        
    Returns:
        Path to saved plot file or None if visualization unavailable
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Matplotlib not available, skipping class distribution plots")
        return None

    if config is None:
        config = get_default_config().visualization
    
    # Determine batch identifier for title
    batch_id = getattr(config, 'batch_id', None) if config else None
    title_prefix = f"Batch {batch_id} - " if batch_id else ""

    # Calculate distributions
    train_counts = y_train.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, 
                                  figsize=(config.chart_figure_width,
                                         config.chart_figure_height),
                                  dpi=config.chart_dpi)
    
    # Training set distribution
    ax1.pie(train_counts.values, labels=[f'Class {i}' for i in train_counts.index],
           autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'Training Set Distribution\n(Total: {len(y_train)})')
    
    # Test set distribution
    ax2.pie(test_counts.values, labels=[f'Class {i}' for i in test_counts.index],
           autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'Test Set Distribution\n(Total: {len(y_test)})')
    
    # Add main title with batch information
    fig.suptitle(f'{title_prefix}Class Distribution', fontsize=14, y=0.95)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Adjust to make room for suptitle
    
    # Save plot
    if config.save_plots:
        reports_dir = ensure_reports_dir()
        timestamp = get_timestamp()
        filename = f'class_distribution_{timestamp}.{config.plot_format}'
        filepath = os.path.join(reports_dir, filename)
        plt.savefig(filepath, dpi=config.chart_dpi, bbox_inches='tight')
        logger.info(f"Class distribution saved to {filepath}")
        plt.close()
        return filepath
    else:
        plt.show()
        return None

def create_training_report(results: Dict[str, Any],
                         config: Optional[VisualizationConfig] = None) -> Dict[str, Optional[str]]:
    """
    Create comprehensive training report with all visualizations.
    
    Args:
        results: Dictionary containing all training results
        config: Visualization configuration
        
    Returns:
        Dictionary mapping plot types to file paths
    """
    if config is None:
        config = get_default_config().visualization
    
    plot_paths = {}
    
    # Plot confusion matrices
    if 'confusion_matrices' in results and 'model_names' in results:
        plot_paths['confusion_matrices'] = plot_confusion_matrices(
            results['confusion_matrices'], 
            results['model_names'], 
            config
        )
    
    # Plot model performance
    if 'metrics_df' in results:
        plot_paths['performance'] = plot_model_performance(
            results['metrics_df'], 
            config
        )
    
    # Plot training curves
    if 'training_history' in results:
        plot_paths['training_curves'] = plot_training_curves(
            results['training_history'], 
            config
        )
    
    # Plot threshold analysis
    if 'threshold_results' in results:
        plot_paths['threshold_analysis'] = plot_threshold_analysis(
            results['threshold_results'], 
            config
        )
    
    # Plot class distribution
    if 'y_train' in results and 'y_test' in results:
        plot_paths['class_distribution'] = plot_class_distribution(
            results['y_train'], 
            results['y_test'], 
            config
        )
    
    # Log summary
    saved_plots = {k: v for k, v in plot_paths.items() if v is not None}
    if saved_plots:
        logger.info(f"Training report generated with {len(saved_plots)} plots:")
        for plot_type, path in saved_plots.items():
            logger.info(f"  {plot_type}: {path}")
    else:
        logger.info("No plots generated (visualization disabled or unavailable)")
    
    return plot_paths
