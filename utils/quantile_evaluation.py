#!/usr/bin/env python3
"""
Enhanced Evaluation Utilities for Quantile Regression
====================================================

This module extends the evaluation utilities to handle quantile regression
predictions, uncertainty estimation, and risk-aware metrics for the
Investment Committee project.

Features:
- Quantile regression evaluation metrics (pinball loss, coverage, interval width)
- Uncertainty-based model comparison
- Risk-aware threshold optimization
- Integration with existing evaluation pipeline

Phase 3 of Advanced Signal Improvements: Quantile Loss Options
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any

from .evaluation import (
    evaluate_ensemble_performance, create_performance_summary,
    find_optimal_threshold, convert_regression_ensemble_to_binary
)
from .quantile_loss import (
    evaluate_quantile_predictions, calculate_prediction_intervals,
    create_quantile_ensemble_predictions, quantile_to_binary_predictions,
    risk_aware_threshold_selection
)

logger = logging.getLogger(__name__)

def is_quantile_model(model_name: str) -> bool:
    """
    Check if a model name corresponds to a quantile regression model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        True if it's a quantile model, False otherwise
    """
    return 'quantile' in model_name.lower()

def extract_quantile_predictions(model_predictions: Dict[str, Union[np.ndarray, Dict[float, np.ndarray]]]) -> Dict[str, Dict[float, np.ndarray]]:
    """
    Extract quantile predictions from model predictions dictionary.
    
    Args:
        model_predictions: Dictionary of model predictions (some may be quantile dictionaries)
        
    Returns:
        Dictionary mapping model names to their quantile predictions
    """
    quantile_preds = {}
    
    for model_name, predictions in model_predictions.items():
        if isinstance(predictions, dict) and all(isinstance(k, float) for k in predictions.keys()):
            # This is already a quantile prediction dictionary
            quantile_preds[model_name] = predictions
        elif is_quantile_model(model_name):
            # This should be quantile predictions but isn't formatted correctly
            logger.warning(f"Expected quantile predictions for {model_name} but got {type(predictions)}")
    
    return quantile_preds

def evaluate_quantile_ensemble_performance(y_true: np.ndarray,
                                         base_predictions: Dict[str, Union[np.ndarray, Dict[float, np.ndarray]]],
                                         meta_predictions: Optional[np.ndarray] = None,
                                         final_predictions: Optional[np.ndarray] = None,
                                         config: Optional[Any] = None) -> Dict[str, Any]:
    """
    Evaluate ensemble performance including quantile regression models.
    
    Args:
        y_true: True target values
        base_predictions: Dictionary of base model predictions (may include quantile predictions)
        meta_predictions: Meta-model predictions
        final_predictions: Final ensemble predictions
        config: Training configuration
        
    Returns:
        Dictionary of evaluation results including quantile metrics
    """
    # Start with standard evaluation
    standard_preds = {}
    quantile_preds = {}
    
    # Separate standard and quantile predictions
    for model_name, predictions in base_predictions.items():
        if isinstance(predictions, dict) and all(isinstance(k, float) for k in predictions.keys()):
            # Quantile predictions
            quantile_preds[model_name] = predictions
            # Use median quantile for standard evaluation if available
            if 0.5 in predictions:
                standard_preds[model_name] = predictions[0.5]
            elif predictions:
                # Use first available quantile
                first_quantile = min(predictions.keys())
                standard_preds[model_name] = predictions[first_quantile]
        else:
            # Standard predictions
            standard_preds[model_name] = predictions
    
    # Evaluate standard predictions
    evaluation_results = evaluate_ensemble_performance(
        y_true, standard_preds, meta_predictions, final_predictions
    )
    
    # Add quantile-specific evaluation
    if quantile_preds:
        logger.info(f"📊 Evaluating {len(quantile_preds)} quantile regression models...")
        
        quantile_evaluation = {}
        
        for model_name, model_quantile_preds in quantile_preds.items():
            logger.info(f"Evaluating quantile model: {model_name}")
            
            # Evaluate quantile predictions
            quantile_metrics = evaluate_quantile_predictions(y_true, model_quantile_preds)
            quantile_evaluation[model_name] = {
                'type': 'quantile_regression',
                **quantile_metrics
            }
            
            # Calculate prediction intervals
            try:
                intervals_80 = calculate_prediction_intervals(model_quantile_preds, 0.8)
                intervals_90 = calculate_prediction_intervals(model_quantile_preds, 0.9)
                
                quantile_evaluation[model_name].update({
                    'mean_interval_width_80': np.mean(intervals_80['interval_width']),
                    'mean_interval_width_90': np.mean(intervals_90['interval_width'])
                })
            except Exception as e:
                logger.warning(f"Failed to calculate intervals for {model_name}: {e}")
            
            # Risk-aware evaluation for different risk tolerances
            for risk_tolerance in ['conservative', 'moderate', 'aggressive']:
                try:
                    binary_preds, conversion_info = quantile_to_binary_predictions(
                        model_quantile_preds, y_true, 'risk_aware', risk_tolerance
                    )
                    
                    # Calculate binary metrics
                    from sklearn.metrics import f1_score, precision_score, recall_score
                    binary_true = (y_true > 0).astype(int)
                    
                    if len(np.unique(binary_preds)) > 1:
                        f1 = f1_score(binary_true, binary_preds, zero_division=0)
                        precision = precision_score(binary_true, binary_preds, zero_division=0)
                        recall = recall_score(binary_true, binary_preds, zero_division=0)
                        
                        quantile_evaluation[model_name][f'f1_{risk_tolerance}'] = f1
                        quantile_evaluation[model_name][f'precision_{risk_tolerance}'] = precision
                        quantile_evaluation[model_name][f'recall_{risk_tolerance}'] = recall
                        quantile_evaluation[model_name][f'threshold_{risk_tolerance}'] = conversion_info.get('threshold', 0.0)
                        
                except Exception as e:
                    logger.warning(f"Failed risk-aware evaluation for {model_name} ({risk_tolerance}): {e}")
        
        # Create quantile ensemble if we have multiple quantile models
        if len(quantile_preds) > 1:
            try:
                ensemble_method = getattr(config, 'quantile_ensemble_method', 'median') if config else 'median'
                ensemble_quantile_preds = create_quantile_ensemble_predictions(
                    quantile_preds, ensemble_method
                )
                
                if ensemble_quantile_preds:
                    ensemble_metrics = evaluate_quantile_predictions(y_true, ensemble_quantile_preds)
                    quantile_evaluation['quantile_ensemble'] = {
                        'type': 'quantile_ensemble',
                        'method': ensemble_method,
                        **ensemble_metrics
                    }
                    
                    # Risk-aware evaluation for ensemble
                    decision_strategy = getattr(config, 'quantile_decision_strategy', 'threshold_optimization') if config else 'threshold_optimization'
                    risk_tolerance = getattr(config, 'risk_tolerance', 'moderate') if config else 'moderate'
                    
                    binary_preds, conversion_info = quantile_to_binary_predictions(
                        ensemble_quantile_preds, y_true, decision_strategy, risk_tolerance
                    )
                    
                    binary_true = (y_true > 0).astype(int)
                    if len(np.unique(binary_preds)) > 1:
                        from sklearn.metrics import f1_score, precision_score, recall_score
                        f1 = f1_score(binary_true, binary_preds, zero_division=0)
                        precision = precision_score(binary_true, binary_preds, zero_division=0)
                        recall = recall_score(binary_true, binary_preds, zero_division=0)
                        
                        quantile_evaluation['quantile_ensemble'].update({
                            'f1_score': f1,
                            'precision': precision,
                            'recall': recall,
                            'decision_strategy': decision_strategy,
                            'risk_tolerance': risk_tolerance,
                            **conversion_info
                        })
                        
                        logger.info(f"Quantile ensemble F1: {f1:.4f} (strategy: {decision_strategy}, risk: {risk_tolerance})")
                    
            except Exception as e:
                logger.warning(f"Failed to create quantile ensemble: {e}")
        
        # Add quantile evaluation to main results
        evaluation_results['quantile_models'] = quantile_evaluation
    
    return evaluation_results

def create_quantile_performance_summary(evaluation_results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create performance summary including quantile regression metrics.
    
    Args:
        evaluation_results: Evaluation results from evaluate_quantile_ensemble_performance
        
    Returns:
        DataFrame with performance summary including quantile metrics
    """
    # Start with standard performance summary
    standard_df = create_performance_summary(evaluation_results)
    
    # Add quantile-specific metrics if available
    if 'quantile_models' in evaluation_results:
        quantile_rows = []
        
        for model_name, metrics in evaluation_results['quantile_models'].items():
            if metrics.get('type') == 'quantile_regression':
                # Extract key quantile metrics
                row = {
                    'Model': f"{model_name}_quantile",
                    'Type': 'Quantile Regression'
                }
                
                # Add pinball losses for available quantiles
                for key, value in metrics.items():
                    if 'pinball_loss' in key:
                        quantile = key.split('_')[1]
                        row[f'Pinball_Loss_Q{quantile}'] = value
                    elif 'quantile_score' in key:
                        quantile = key.split('_')[1]
                        row[f'QScore_Q{quantile}'] = value
                    elif key.startswith('coverage_'):
                        row[f'Coverage_{key.split("_")[1]}'] = value
                    elif key.startswith('f1_'):
                        risk = key.split('_')[1]
                        row[f'F1_{risk.title()}'] = value
                    elif key.startswith('mean_interval_width_'):
                        conf = key.split('_')[-1]
                        row[f'Interval_Width_{conf}'] = value
                
                quantile_rows.append(row)
        
        # Add quantile ensemble if available
        if 'quantile_ensemble' in evaluation_results['quantile_models']:
            ensemble_metrics = evaluation_results['quantile_models']['quantile_ensemble']
            row = {
                'Model': 'quantile_ensemble',
                'Type': 'Quantile Ensemble',
                'F1_Score': ensemble_metrics.get('f1_score', 0),
                'Precision': ensemble_metrics.get('precision', 0),
                'Recall': ensemble_metrics.get('recall', 0),
                'Decision_Strategy': ensemble_metrics.get('decision_strategy', ''),
                'Risk_Tolerance': ensemble_metrics.get('risk_tolerance', ''),
                'Positive_Rate': ensemble_metrics.get('positive_rate', 0)
            }
            quantile_rows.append(row)
        
        if quantile_rows:
            quantile_df = pd.DataFrame(quantile_rows)
            
            # Combine with standard results
            if not standard_df.empty:
                # Align columns
                all_columns = list(set(standard_df.columns) | set(quantile_df.columns))
                
                for col in all_columns:
                    if col not in standard_df.columns:
                        standard_df[col] = np.nan
                    if col not in quantile_df.columns:
                        quantile_df[col] = np.nan
                
                combined_df = pd.concat([standard_df, quantile_df], ignore_index=True)
                return combined_df
            else:
                return quantile_df
    
    return standard_df

def optimize_quantile_ensemble_thresholds(y_true: np.ndarray,
                                        quantile_predictions: Dict[str, Dict[float, np.ndarray]],
                                        metric: str = 'f1') -> Dict[str, Dict[str, float]]:
    """
    Optimize thresholds for quantile regression models.
    
    Args:
        y_true: True target values
        quantile_predictions: Dictionary of quantile predictions
        metric: Metric to optimize ('f1', 'precision', 'recall')
        
    Returns:
        Dictionary mapping model names to threshold optimization results
    """
    threshold_results = {}
    
    for model_name, model_quantile_preds in quantile_predictions.items():
        logger.info(f"Optimizing thresholds for quantile model: {model_name}")
        
        model_results = {}
        
        # Optimize threshold for each quantile
        for quantile, predictions in model_quantile_preds.items():
            try:
                optimal_threshold, optimal_score = find_optimal_threshold(
                    (y_true > 0).astype(int), predictions, metric=metric
                )
                
                model_results[f'quantile_{quantile:.2f}'] = {
                    'threshold': optimal_threshold,
                    f'{metric}_score': optimal_score
                }
                
            except Exception as e:
                logger.warning(f"Failed to optimize threshold for {model_name} quantile {quantile}: {e}")
        
        # Find best performing quantile
        if model_results:
            best_quantile = max(model_results.keys(), 
                              key=lambda q: model_results[q].get(f'{metric}_score', 0))
            model_results['best_quantile'] = {
                'quantile': best_quantile,
                **model_results[best_quantile]
            }
        
        threshold_results[model_name] = model_results
    
    return threshold_results

def convert_quantile_ensemble_to_binary(quantile_predictions: Dict[str, Dict[float, np.ndarray]],
                                       y_true: np.ndarray,
                                       decision_strategy: str = 'threshold_optimization',
                                       risk_tolerance: str = 'moderate',
                                       optimize_thresholds: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Convert quantile predictions to binary decisions with optimization.
    
    Args:
        quantile_predictions: Dictionary of quantile predictions
        y_true: True target values
        decision_strategy: Strategy for binary conversion
        risk_tolerance: Risk tolerance level
        optimize_thresholds: Whether to optimize thresholds
        
    Returns:
        Dictionary mapping model names to conversion results
    """
    conversion_results = {}
    
    for model_name, model_quantile_preds in quantile_predictions.items():
        logger.info(f"Converting quantile predictions to binary for {model_name}")
        
        try:
            binary_preds, conversion_info = quantile_to_binary_predictions(
                model_quantile_preds, y_true, decision_strategy, risk_tolerance
            )
            
            # Calculate performance metrics
            binary_true = (y_true > 0).astype(int)
            
            if len(np.unique(binary_preds)) > 1:
                from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
                
                f1 = f1_score(binary_true, binary_preds, zero_division=0)
                precision = precision_score(binary_true, binary_preds, zero_division=0)
                recall = recall_score(binary_true, binary_preds, zero_division=0)
                
                # For continuous predictions (use median quantile for ROC-AUC)
                if 0.5 in model_quantile_preds:
                    continuous_preds = model_quantile_preds[0.5]
                else:
                    # Use first available quantile
                    continuous_preds = list(model_quantile_preds.values())[0]
                
                try:
                    roc_auc = roc_auc_score(binary_true, continuous_preds)
                except Exception:
                    roc_auc = 0.5  # Neutral score if calculation fails
                
                conversion_results[model_name] = {
                    'is_quantile_model': True,
                    'predictions': binary_preds,
                    'continuous_predictions': continuous_preds,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'roc_auc': roc_auc,
                    'decision_strategy': decision_strategy,
                    'risk_tolerance': risk_tolerance,
                    **conversion_info
                }
                
                logger.info(f"  {model_name}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
                
            else:
                logger.warning(f"No variation in binary predictions for {model_name}")
                conversion_results[model_name] = {
                    'is_quantile_model': True,
                    'predictions': binary_preds,
                    'continuous_predictions': list(model_quantile_preds.values())[0],
                    'f1_score': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'roc_auc': 0.5,
                    'warning': 'No variation in predictions',
                    **conversion_info
                }
                
        except Exception as e:
            logger.error(f"Failed to convert quantile predictions for {model_name}: {e}")
            conversion_results[model_name] = {
                'is_quantile_model': True,
                'predictions': np.zeros(len(y_true), dtype=int),
                'continuous_predictions': np.zeros(len(y_true)),
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'roc_auc': 0.5,
                'error': str(e)
            }
    
    return conversion_results
