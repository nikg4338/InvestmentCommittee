#!/usr/bin/env python3
"""
Evaluation Utilities
===================

Comprehensive model evaluation, metrics computation, and results export
for the investment committee training pipeline. Now includes ranking-based
evaluation metrics for improved portfolio construction insights.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve
)

from config.training_config import TrainingConfig, get_default_config

# Import ranking metrics for comprehensive evaluation
try:
    from utils.ranking_metrics import compute_ranking_metrics, log_ranking_performance
    RANKING_METRICS_AVAILABLE = True
except ImportError:
    RANKING_METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)

def compute_threshold_from_oof(y_oof: np.ndarray, p_oof: np.ndarray, metric: str = "f1") -> float:
    """
    Select a single threshold using ONLY OOF data.
    metric: "f1" or "youden" (PR-optimal fallback uses max F1 over grid).
    """
    y_oof = np.asarray(y_oof).ravel()
    p_oof = np.asarray(p_oof).ravel()
    grid = np.linspace(0.01, 0.99, 99)

    if metric.lower() == "f1":
        scores = [f1_score(y_oof, (p_oof >= t).astype(int)) for t in grid]
        return float(grid[int(np.argmax(scores))])

    # Youden-like on PR curve as a tie-breaker
    prec, rec, thr = precision_recall_curve(y_oof, p_oof)
    f1 = (2 * prec * rec) / np.clip(prec + rec, 1e-9, None)
    best = np.argmax(f1)
    # Map back to a probability threshold; if PR thr shorter than curve, clamp
    return float(np.clip(thr[best-1] if 0 < best < len(thr) else 0.5, 0.01, 0.99))

def apply_fixed_threshold(p: np.ndarray, t: float) -> np.ndarray:
    return (np.asarray(p).ravel() >= float(t)).astype(int)

logger = logging.getLogger(__name__)

if not RANKING_METRICS_AVAILABLE:
    logger.warning("⚠️  Ranking metrics not available - some evaluation features disabled")

def convert_regression_ensemble_to_binary(model_predictions: Dict[str, np.ndarray], 
                                        y_true: Optional[np.ndarray] = None,
                                        optimize_thresholds: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Convert regression predictions to binary decisions with optimized thresholds.
    
    Args:
        model_predictions: Dictionary mapping model names to continuous predictions
        y_true: True labels for threshold optimization (optional)
        optimize_thresholds: Whether to optimize thresholds using y_true
        
    Returns:
        Dictionary with binary predictions and threshold information
    """
    results = {}
    
    for model_name, predictions in model_predictions.items():
        # Check if this is a regression model
        is_regressor = 'regressor' in model_name.lower()
        
        if is_regressor and y_true is not None and optimize_thresholds:
            # Find optimal threshold for regression model
            try:
                optimal_threshold, best_f1 = find_optimal_threshold(
                    (y_true > 0).astype(int),  # Convert continuous targets to binary
                    predictions,
                    metric='f1'
                )
            except Exception as e:
                logger.warning(f"Threshold optimization failed for {model_name}: {e}")
                optimal_threshold = 0.0  # Default for regression (positive returns)
                best_f1 = 0.0
        elif is_regressor:
            # Default threshold for regression without optimization
            optimal_threshold = 0.0
            best_f1 = None
        else:
            # Classification model - default threshold
            optimal_threshold = 0.5
            best_f1 = None
        
        # Convert to binary predictions
        binary_predictions = (predictions > optimal_threshold).astype(int)
        
        results[model_name] = {
            'predictions': binary_predictions,
            'continuous_predictions': predictions,
            'threshold': optimal_threshold,
            'f1_score': best_f1,
            'is_regressor': is_regressor
        }
    
    return results

def optimize_ensemble_thresholds(y_true: np.ndarray, 
                                model_predictions: Dict[str, np.ndarray],
                                metric: str = 'f1') -> Dict[str, Tuple[float, float]]:
    """
    Optimize thresholds for all models in an ensemble.
    
    Args:
        y_true: True binary labels
        model_predictions: Dictionary mapping model names to probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall', 'balanced_accuracy')
        
    Returns:
        Dictionary mapping model names to (optimal_threshold, best_metric_value)
    """
    threshold_results = {}
    
    logger.info(f"🎯 Optimizing thresholds for {len(model_predictions)} models using {metric} metric")
    
    for model_name, y_proba in model_predictions.items():
        try:
            optimal_threshold, best_score = find_optimal_threshold(y_true, y_proba, metric)
            threshold_results[model_name] = (optimal_threshold, best_score)
            
            # Count predictions at optimal threshold
            optimal_predictions = (y_proba >= optimal_threshold).astype(int)
            predicted_positives = optimal_predictions.sum()
            
            logger.info(f"  {model_name}: threshold={optimal_threshold:.4f}, "
                       f"{metric}={best_score:.4f}, predictions={predicted_positives}")
            
        except Exception as e:
            logger.warning(f"  {model_name}: threshold optimization failed - {e}")
            threshold_results[model_name] = (0.5, 0.0)
    
    return threshold_results

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                 y_proba: Optional[np.ndarray] = None,
                                 model_name: str = "model") -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_proba: Predicted probabilities (optional)
        model_name: Name of the model for logging
        
    Returns:
        Dictionary of computed metrics
    """
    metrics = {}
    
    try:
        # Ensure inputs are binary for classification metrics
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        
        # Validate that inputs are binary (0 or 1)
        if not (np.all(np.isin(y_true, [0, 1])) and np.all(np.isin(y_pred, [0, 1]))):
            logger.warning(f"Non-binary values detected in {model_name}. Converting continuous to binary using threshold 0.5")
            if y_proba is not None:
                y_pred = (y_proba > 0.5).astype(int)
            else:
                # For continuous y_pred, apply threshold
                y_pred = (y_pred > 0.5).astype(int)
            # For continuous y_true, apply threshold (though this shouldn't happen in normal cases)
            if not np.all(np.isin(y_true, [0, 1])):
                y_true = (y_true > 0.0).astype(int)
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0.0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0.0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0.0)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Additional derived metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative predictive value
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive predictive value (same as precision)
        
        # Probability-based metrics (if probabilities available)
        if y_proba is not None:
            try:
                # Check if we have both classes in y_true for ROC-AUC
                if len(np.unique(y_true)) > 1:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                else:
                    metrics['roc_auc'] = 0.5  # Random performance for single class
                
                metrics['pr_auc'] = average_precision_score(y_true, y_proba, pos_label=1)
            except Exception as e:
                logger.warning(f"Could not compute AUC metrics for {model_name}: {e}")
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
        else:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
        
        # Prediction counts
        metrics['predicted_positives'] = int(np.sum(y_pred))
        metrics['predicted_negatives'] = int(len(y_pred) - np.sum(y_pred))
        metrics['actual_positives'] = int(np.sum(y_true))
        metrics['actual_negatives'] = int(len(y_true) - np.sum(y_true))
        
        # Rates and ratios
        total = len(y_true)
        metrics['positive_rate'] = metrics['predicted_positives'] / total if total > 0 else 0.0
        metrics['actual_positive_rate'] = metrics['actual_positives'] / total if total > 0 else 0.0
        
    except Exception as e:
        logger.error(f"Error computing metrics for {model_name}: {e}")
        # Return default metrics on error
        metrics = {
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'roc_auc': 0.0, 'pr_auc': 0.0, 'specificity': 0.0, 'sensitivity': 0.0,
            'npv': 0.0, 'ppv': 0.0, 'true_negatives': 0, 'false_positives': 0,
            'false_negatives': 0, 'true_positives': 0, 'predicted_positives': 0,
            'predicted_negatives': 0, 'actual_positives': 0, 'actual_negatives': 0,
            'positive_rate': 0.0, 'actual_positive_rate': 0.0
        }
    
    return metrics

def compute_enhanced_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
                           model_name: str = "Model") -> Dict[str, float]:
    """
    Compute enhanced metrics including ranking-based evaluation for F₁ optimization.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_proba: Predicted probabilities
        model_name: Name of the model for logging
        
    Returns:
        Dictionary with comprehensive metrics including ranking metrics
    """
    # Start with standard classification metrics
    metrics = compute_classification_metrics(y_true, y_pred, y_proba, model_name)
    
    # Add ranking-based metrics if available
    if RANKING_METRICS_AVAILABLE and y_proba is not None:
        try:
            ranking_metrics = compute_ranking_metrics(y_true, y_proba)
            metrics.update(ranking_metrics)
            
            # Log ranking performance for immediate feedback
            log_ranking_performance(ranking_metrics, model_name)
            
            logger.info(f"✅ Enhanced metrics computed for {model_name}: {len(metrics)} total metrics")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to compute ranking metrics for {model_name}: {e}")
    
    else:
        if not RANKING_METRICS_AVAILABLE:
            logger.info(f"📊 Standard metrics only for {model_name} (ranking metrics unavailable)")
        else:
            logger.warning(f"⚠️  No probabilities available for {model_name} - skipping ranking metrics")
    
    return metrics

def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                         metric: str = 'f1',
                         n_thresholds: int = 101) -> Tuple[float, float]:
    """
    Find optimal threshold for binary classification using comprehensive grid search.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall', 'balanced_accuracy')
        n_thresholds: Number of thresholds to test
        
    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    from sklearn.metrics import balanced_accuracy_score
    
    # Comprehensive threshold grid from 0.0 to 1.0
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    
    best_threshold = 0.5
    best_score = 0.0
    
    # Select metric function
    metric_functions = {
        'f1': lambda yt, yp: f1_score(yt, yp, zero_division=0),
        'precision': lambda yt, yp: precision_score(yt, yp, zero_division=0),
        'recall': lambda yt, yp: recall_score(yt, yp, zero_division=0),
        'balanced_accuracy': lambda yt, yp: balanced_accuracy_score(yt, yp)
    }
    
    metric_func = metric_functions.get(metric, metric_functions['f1'])
    
    threshold_scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Skip if all predictions are one class (unless it's a valid case)
        unique_preds = np.unique(y_pred)
        if len(unique_preds) < 2:
            # For extreme imbalance, allow all-negative predictions
            if len(unique_preds) == 1 and unique_preds[0] == 0:
                score = 0.0  # All negative predictions
            else:
                continue  # Skip all-positive (usually not desired)
        else:
            try:
                score = metric_func(y_true, y_pred)
            except (ValueError, ZeroDivisionError):
                score = 0.0
        
        threshold_scores.append((threshold, score))
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    # Fallback strategies if no good threshold found
    if best_score == 0.0:
        logger.warning(f"No threshold yielded positive {metric} score. Applying fallback strategies.")
        
        # Strategy 1: Find minimum non-zero probability
        nonzero_proba = y_proba[y_proba > 0]
        if len(nonzero_proba) > 0:
            min_nonzero = np.min(nonzero_proba)
            best_threshold = min_nonzero
            logger.info(f"Fallback 1: Using minimum non-zero probability {best_threshold:.6f}")
        else:
            # Strategy 2: Use very small threshold to guarantee some positives
            best_threshold = 1e-6
            logger.info(f"Fallback 2: Using emergency threshold {best_threshold:.6f}")
        
        # Recalculate score with fallback threshold
        y_pred_fallback = (y_proba >= best_threshold).astype(int)
        if len(np.unique(y_pred_fallback)) > 1:
            try:
                best_score = metric_func(y_true, y_pred_fallback)
            except:
                best_score = 0.0
    
    logger.info(f"Optimal threshold for {metric}: {best_threshold:.6f} (score: {best_score:.3f})")
    return best_threshold, best_score

def evaluate_model_performance(y_true: np.ndarray, y_proba: np.ndarray,
                              model_name: str = "model",
                              find_threshold: bool = True) -> Dict[str, Any]:
    """
    Comprehensive model performance evaluation.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        model_name: Name of the model
        find_threshold: Whether to find optimal threshold
        
    Returns:
        Dictionary with performance metrics and optimal threshold
    """
    results = {
        'model_name': model_name,
        'n_samples': len(y_true),
        'n_positive': int(np.sum(y_true)),
        'n_negative': int(len(y_true) - np.sum(y_true))
    }
    
    # Find optimal threshold if requested
    if find_threshold:
        optimal_threshold, optimal_f1 = find_optimal_threshold(y_true, y_proba, 'f1')
        results['optimal_threshold'] = optimal_threshold
        results['optimal_f1'] = optimal_f1
    else:
        optimal_threshold = 0.5
        results['optimal_threshold'] = optimal_threshold
        results['optimal_f1'] = 0.0
    
    # Get predictions using optimal threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)
    
    # Compute comprehensive metrics
    metrics = compute_classification_metrics(y_true, y_pred, y_proba, model_name)
    results.update(metrics)
    
    # Add threshold-specific information
    results['threshold_used'] = optimal_threshold
    results['predictions_at_threshold'] = y_pred.tolist()
    
    return results

def evaluate_ensemble_performance(y_true: np.ndarray,
                                 base_predictions: Dict[str, np.ndarray],
                                 meta_predictions: Optional[np.ndarray] = None,
                                 ensemble_predictions: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Evaluate performance of ensemble and individual models.
    
    Args:
        y_true: True binary labels
        base_predictions: Dictionary mapping model names to probabilities
        meta_predictions: Meta-model probabilities (optional)
        ensemble_predictions: Simple ensemble probabilities (optional)
        
    Returns:
        Dictionary with comprehensive ensemble evaluation
    """
    results = {
        'base_model_performance': {},
        'ensemble_performance': {},
        'comparison_metrics': {}
    }
    
    # Evaluate each base model
    for model_name, y_proba in base_predictions.items():
        try:
            model_results = evaluate_model_performance(y_true, y_proba, model_name)
            results['base_model_performance'][model_name] = model_results
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            continue
    
    # Evaluate meta-model if available
    if meta_predictions is not None:
        try:
            meta_results = evaluate_model_performance(y_true, meta_predictions, "meta_model")
            results['ensemble_performance']['meta_model'] = meta_results
        except Exception as e:
            logger.error(f"Error evaluating meta-model: {e}")
    
    # Evaluate simple ensemble if available
    if ensemble_predictions is not None:
        try:
            ensemble_results = evaluate_model_performance(y_true, ensemble_predictions, "simple_ensemble")
            results['ensemble_performance']['simple_ensemble'] = ensemble_results
        except Exception as e:
            logger.error(f"Error evaluating simple ensemble: {e}")
    
    # Compute comparison metrics
    if results['base_model_performance']:
        base_f1_scores = [
            model_data.get('f1', 0.0) 
            for model_data in results['base_model_performance'].values()
        ]
        
        results['comparison_metrics']['best_base_f1'] = max(base_f1_scores) if base_f1_scores else 0.0
        results['comparison_metrics']['mean_base_f1'] = np.mean(base_f1_scores) if base_f1_scores else 0.0
        results['comparison_metrics']['std_base_f1'] = np.std(base_f1_scores) if base_f1_scores else 0.0
        
        # Meta-model improvement
        if 'meta_model' in results['ensemble_performance']:
            meta_f1 = results['ensemble_performance']['meta_model'].get('f1', 0.0)
            results['comparison_metrics']['meta_improvement'] = meta_f1 - results['comparison_metrics']['best_base_f1']
        
        # Simple ensemble improvement
        if 'simple_ensemble' in results['ensemble_performance']:
            ensemble_f1 = results['ensemble_performance']['simple_ensemble'].get('f1', 0.0)
            results['comparison_metrics']['ensemble_improvement'] = ensemble_f1 - results['comparison_metrics']['best_base_f1']
    
    return results

def create_performance_summary(evaluation_results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a summary DataFrame of model performance metrics.
    
    Args:
        evaluation_results: Results from evaluate_ensemble_performance
        
    Returns:
        DataFrame with model performance summary
    """
    summary_data = []
    
    # Add base model results
    for model_name, results in evaluation_results.get('base_model_performance', {}).items():
        summary_data.append({
            'model': model_name,
            'type': 'base',
            'accuracy': results.get('accuracy', 0.0),
            'precision': results.get('precision', 0.0),
            'recall': results.get('recall', 0.0),
            'f1': results.get('f1', 0.0),
            'roc_auc': results.get('roc_auc', 0.0),
            'pr_auc': results.get('pr_auc', 0.0),
            'threshold': results.get('optimal_threshold', 0.5),
            'predicted_positives': results.get('predicted_positives', 0),
            'true_positives': results.get('true_positives', 0),
            'false_positives': results.get('false_positives', 0),
            'false_negatives': results.get('false_negatives', 0)
        })
    
    # Add ensemble results
    for model_name, results in evaluation_results.get('ensemble_performance', {}).items():
        summary_data.append({
            'model': model_name,
            'type': 'ensemble',
            'accuracy': results.get('accuracy', 0.0),
            'precision': results.get('precision', 0.0),
            'recall': results.get('recall', 0.0),
            'f1': results.get('f1', 0.0),
            'roc_auc': results.get('roc_auc', 0.0),
            'pr_auc': results.get('pr_auc', 0.0),
            'threshold': results.get('optimal_threshold', 0.5),
            'predicted_positives': results.get('predicted_positives', 0),
            'true_positives': results.get('true_positives', 0),
            'false_positives': results.get('false_positives', 0),
            'false_negatives': results.get('false_negatives', 0)
        })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        df = df.set_index('model')  # Set model names as index
        df = df.round(4)  # Round to 4 decimal places
        return df
    else:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'model', 'type', 'accuracy', 'precision', 'recall', 'f1', 
            'roc_auc', 'pr_auc', 'threshold', 'predicted_positives',
            'true_positives', 'false_positives', 'false_negatives'
        ])

def export_training_results(evaluation_results: Dict[str, Any],
                           config: Optional[TrainingConfig] = None,
                           save_detailed: bool = True) -> Dict[str, str]:
    """
    Export training results to CSV files and log summary.
    
    Args:
        evaluation_results: Results from evaluate_ensemble_performance
        config: Training configuration
        save_detailed: Whether to save detailed results
        
    Returns:
        Dictionary mapping export types to file paths
    """
    if config is None:
        config = get_default_config()
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    exported_files = {}
    
    # Create and export performance summary
    summary_df = create_performance_summary(evaluation_results)
    if not summary_df.empty:
        summary_path = f'logs/training_summary_{timestamp}.csv'
        # Preserve model names as a column for easier downstream parsing
        summary_df.reset_index().to_csv(summary_path, index=False)
        exported_files['summary'] = summary_path
        logger.info(f"Performance summary exported to {summary_path}")
        
        # Performance metrics are already logged in the CSV file above
        # Individual model results available in summary_df
    
    # Export detailed results if requested
    if save_detailed and evaluation_results:
        detailed_path = f'logs/detailed_results_{timestamp}.json'
        try:
            import json
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                else:
                    return obj
            
            serializable_results = convert_numpy(evaluation_results)
            
            with open(detailed_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            exported_files['detailed'] = detailed_path
            logger.info(f"Detailed results exported to {detailed_path}")
            
        except Exception as e:
            logger.error(f"Failed to export detailed results: {e}")
    
    # Log summary statistics
    if evaluation_results.get('comparison_metrics'):
        metrics = evaluation_results['comparison_metrics']
        logger.info("=== Training Summary ===")
        logger.info(f"Best base model F1: {metrics.get('best_base_f1', 0.0):.4f}")
        logger.info(f"Mean base model F1: {metrics.get('mean_base_f1', 0.0):.4f}")
        
        if 'meta_improvement' in metrics:
            logger.info(f"Meta-model improvement: {metrics['meta_improvement']:+.4f}")
        
        if 'ensemble_improvement' in metrics:
            logger.info(f"Ensemble improvement: {metrics['ensemble_improvement']:+.4f}")
    
    return exported_files

def create_confusion_matrices(evaluation_results: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Extract confusion matrices from evaluation results.
    
    Args:
        evaluation_results: Results from evaluate_ensemble_performance
        
    Returns:
        Dictionary mapping model names to confusion matrices
    """
    confusion_matrices = {}
    
    # Process base models
    for model_name, results in evaluation_results.get('base_model_performance', {}).items():
        if all(key in results for key in ['true_negatives', 'false_positives', 'false_negatives', 'true_positives']):
            cm = np.array([
                [results['true_negatives'], results['false_positives']],
                [results['false_negatives'], results['true_positives']]
            ])
            confusion_matrices[model_name] = cm
    
    # Process ensemble models
    for model_name, results in evaluation_results.get('ensemble_performance', {}).items():
        if all(key in results for key in ['true_negatives', 'false_positives', 'false_negatives', 'true_positives']):
            cm = np.array([
                [results['true_negatives'], results['false_positives']],
                [results['false_negatives'], results['true_positives']]
            ])
            confusion_matrices[model_name] = cm
    
    return confusion_matrices

def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                               model_name: str = "Model") -> None:
    """
    Print detailed classification report.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        model_name: Name of the model
    """
    logger.info(f"\n=== {model_name} Classification Report ===")
    
    try:
        report = classification_report(y_true, y_pred, output_dict=False, zero_division=0)
        logger.info(f"\n{report}")
    except Exception as e:
        logger.error(f"Failed to generate classification report for {model_name}: {e}")
        
        # Manual calculation as fallback
        metrics = compute_classification_metrics(y_true, y_pred, model_name=model_name)
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1']:.4f}")

def generate_training_report(evaluation_results: Dict[str, Any],
                           training_time: float,
                           config: Optional[TrainingConfig] = None) -> str:
    """
    Generate comprehensive training report as formatted string.
    
    Args:
        evaluation_results: Results from evaluate_ensemble_performance
        training_time: Total training time in seconds
        config: Training configuration
        
    Returns:
        Formatted training report string
    """
    if config is None:
        config = get_default_config()
    
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("INVESTMENT COMMITTEE TRAINING REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Training Time: {training_time:.1f} seconds")
    report_lines.append(f"Configuration: {config.__class__.__name__}")
    report_lines.append("")
    
    # Base model performance
    base_results = evaluation_results.get('base_model_performance', {})
    if base_results:
        report_lines.append("BASE MODEL PERFORMANCE:")
        report_lines.append("-" * 40)
        for model_name, results in base_results.items():
            report_lines.append(f"{model_name.upper()}:")
            report_lines.append(f"  Accuracy:  {results.get('accuracy', 0.0):.4f}")
            report_lines.append(f"  Precision: {results.get('precision', 0.0):.4f}")
            report_lines.append(f"  Recall:    {results.get('recall', 0.0):.4f}")
            report_lines.append(f"  F1-Score:  {results.get('f1', 0.0):.4f}")
            report_lines.append(f"  ROC-AUC:   {results.get('roc_auc', 0.0):.4f}")
            report_lines.append(f"  Threshold: {results.get('optimal_threshold', 0.5):.4f}")
            report_lines.append(f"  Predicted: {results.get('predicted_positives', 0)}")
            report_lines.append("")
    
    # Ensemble performance
    ensemble_results = evaluation_results.get('ensemble_performance', {})
    if ensemble_results:
        report_lines.append("ENSEMBLE PERFORMANCE:")
        report_lines.append("-" * 40)
        for model_name, results in ensemble_results.items():
            report_lines.append(f"{model_name.upper()}:")
            report_lines.append(f"  Accuracy:  {results.get('accuracy', 0.0):.4f}")
            report_lines.append(f"  Precision: {results.get('precision', 0.0):.4f}")
            report_lines.append(f"  Recall:    {results.get('recall', 0.0):.4f}")
            report_lines.append(f"  F1-Score:  {results.get('f1', 0.0):.4f}")
            report_lines.append(f"  ROC-AUC:   {results.get('roc_auc', 0.0):.4f}")
            report_lines.append(f"  Threshold: {results.get('optimal_threshold', 0.5):.4f}")
            report_lines.append(f"  Predicted: {results.get('predicted_positives', 0)}")
            report_lines.append("")
    
    # Comparison metrics
    comparison = evaluation_results.get('comparison_metrics', {})
    if comparison:
        report_lines.append("COMPARISON METRICS:")
        report_lines.append("-" * 40)
        report_lines.append(f"Best Base F1:        {comparison.get('best_base_f1', 0.0):.4f}")
        report_lines.append(f"Mean Base F1:        {comparison.get('mean_base_f1', 0.0):.4f}")
        
        if 'meta_improvement' in comparison:
            improvement = comparison['meta_improvement']
            report_lines.append(f"Meta Improvement:    {improvement:+.4f}")
        
        if 'ensemble_improvement' in comparison:
            improvement = comparison['ensemble_improvement']
            report_lines.append(f"Ensemble Improvement: {improvement:+.4f}")
    
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)
