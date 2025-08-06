#!/usr/bin/env python3
"""
Quantile Loss Implementation for Regression Models
==================================================

This module implements quantile loss functions and utilities for regression models
in the Investment Committee project. Quantile regression provides uncertainty
estimation and risk-aware predictions for better investment decision making.

Features:
- Multiple quantile targets (0.1, 0.5, 0.9) for uncertainty estimation
- Pinball loss function for quantile regression
- Risk-aware thresholding based on quantile predictions
- Integration with existing regression models (LightGBM, XGBoost)
- Uncertainty-based ensemble weighting

Phase 3 of Advanced Signal Improvements: Quantile Loss Options
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Calculate the pinball loss (quantile loss) for quantile regression.
    
    Args:
        y_true: True target values
        y_pred: Predicted quantile values
        quantile: Quantile level (e.g., 0.1, 0.5, 0.9)
        
    Returns:
        Pinball loss value
    """
    error = y_true - y_pred
    loss = np.where(error >= 0, quantile * error, (quantile - 1) * error)
    return np.mean(loss)

def quantile_score(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Calculate the quantile score (normalized pinball loss).
    
    Args:
        y_true: True target values
        y_pred: Predicted quantile values
        quantile: Quantile level
        
    Returns:
        Quantile score (lower is better)
    """
    loss = pinball_loss(y_true, y_pred, quantile)
    # Normalize by naive baseline (predicting the empirical quantile)
    naive_pred = np.full_like(y_true, np.quantile(y_true, quantile))
    naive_loss = pinball_loss(y_true, naive_pred, quantile)
    
    if naive_loss == 0:
        return 0.0
    
    return loss / naive_loss

def calculate_prediction_intervals(quantile_predictions: Dict[float, np.ndarray],
                                 confidence_level: float = 0.8) -> Dict[str, np.ndarray]:
    """
    Calculate prediction intervals from quantile predictions.
    
    Args:
        quantile_predictions: Dictionary mapping quantile levels to predictions
        confidence_level: Confidence level for intervals (e.g., 0.8 for 80% interval)
        
    Returns:
        Dictionary with 'lower', 'median', 'upper' predictions and 'interval_width'
    """
    alpha = 1 - confidence_level
    lower_quantile = alpha / 2
    upper_quantile = 1 - alpha / 2
    
    # Find closest available quantiles
    available_quantiles = sorted(quantile_predictions.keys())
    
    # Get lower bound
    lower_q = min(available_quantiles, key=lambda x: abs(x - lower_quantile))
    # Get upper bound
    upper_q = min(available_quantiles, key=lambda x: abs(x - upper_quantile))
    # Get median
    median_q = min(available_quantiles, key=lambda x: abs(x - 0.5))
    
    lower_preds = quantile_predictions[lower_q]
    upper_preds = quantile_predictions[upper_q]
    median_preds = quantile_predictions[median_q]
    
    interval_width = upper_preds - lower_preds
    
    return {
        'lower': lower_preds,
        'median': median_preds,
        'upper': upper_preds,
        'interval_width': interval_width,
        'confidence_level': confidence_level
    }

def risk_aware_threshold_selection(quantile_predictions: Dict[float, np.ndarray],
                                 risk_tolerance: str = 'moderate') -> Tuple[np.ndarray, float]:
    """
    Select threshold for binary decisions based on risk tolerance and quantile predictions.
    
    Args:
        quantile_predictions: Dictionary mapping quantile levels to predictions
        risk_tolerance: Risk tolerance level ('conservative', 'moderate', 'aggressive')
        
    Returns:
        Tuple of (selected_predictions, risk_weight) for decision making
    """
    available_quantiles = sorted(quantile_predictions.keys())
    
    if risk_tolerance == 'conservative':
        # Use lower quantile for conservative decisions (q=0.1 or closest)
        target_quantile = 0.1
        risk_weight = 0.3  # Lower weight due to conservatism
    elif risk_tolerance == 'aggressive':
        # Use upper quantile for aggressive decisions (q=0.9 or closest)
        target_quantile = 0.9
        risk_weight = 1.2  # Higher weight for aggressive approach
    else:  # moderate
        # Use median for moderate decisions (q=0.5 or closest)
        target_quantile = 0.5
        risk_weight = 1.0  # Standard weight
    
    # Find closest available quantile
    selected_quantile = min(available_quantiles, key=lambda x: abs(x - target_quantile))
    selected_predictions = quantile_predictions[selected_quantile]
    
    logger.info(f"Risk-aware selection: {risk_tolerance} → quantile {selected_quantile:.1f}, weight {risk_weight:.1f}")
    
    return selected_predictions, risk_weight

def evaluate_quantile_predictions(y_true: np.ndarray,
                                quantile_predictions: Dict[float, np.ndarray],
                                metrics: List[str] = None) -> Dict[str, float]:
    """
    Evaluate quantile regression predictions using multiple metrics.
    
    Args:
        y_true: True target values
        quantile_predictions: Dictionary mapping quantile levels to predictions
        metrics: List of metrics to calculate
        
    Returns:
        Dictionary of evaluation metrics
    """
    if metrics is None:
        metrics = ['pinball_loss', 'quantile_score', 'coverage', 'interval_width']
    
    results = {}
    
    for quantile, y_pred in quantile_predictions.items():
        prefix = f'q_{quantile:.1f}'
        
        if 'pinball_loss' in metrics:
            results[f'{prefix}_pinball_loss'] = pinball_loss(y_true, y_pred, quantile)
        
        if 'quantile_score' in metrics:
            results[f'{prefix}_quantile_score'] = quantile_score(y_true, y_pred, quantile)
    
    # Calculate interval-based metrics if we have multiple quantiles
    if len(quantile_predictions) >= 2 and ('coverage' in metrics or 'interval_width' in metrics):
        quantile_levels = sorted(quantile_predictions.keys())
        
        # Calculate coverage for different confidence levels
        for conf_level in [0.6, 0.8, 0.9]:
            try:
                intervals = calculate_prediction_intervals(quantile_predictions, conf_level)
                
                if 'coverage' in metrics:
                    # Calculate empirical coverage
                    within_interval = (y_true >= intervals['lower']) & (y_true <= intervals['upper'])
                    empirical_coverage = np.mean(within_interval)
                    results[f'coverage_{int(conf_level*100)}'] = empirical_coverage
                
                if 'interval_width' in metrics:
                    # Calculate mean interval width
                    mean_width = np.mean(intervals['interval_width'])
                    results[f'interval_width_{int(conf_level*100)}'] = mean_width
                    
            except Exception as e:
                logger.warning(f"Failed to calculate interval metrics for {conf_level}: {e}")
    
    return results

def create_quantile_ensemble_predictions(base_quantile_predictions: Dict[str, Dict[float, np.ndarray]],
                                       ensemble_method: str = 'median',
                                       weights: Optional[Dict[str, float]] = None) -> Dict[float, np.ndarray]:
    """
    Create ensemble predictions from multiple models' quantile predictions.
    
    Args:
        base_quantile_predictions: Dict mapping model names to their quantile predictions
        ensemble_method: Method for combining predictions ('mean', 'median', 'weighted')
        weights: Model weights for weighted ensemble (only used if ensemble_method='weighted')
        
    Returns:
        Dictionary mapping quantile levels to ensemble predictions
    """
    if not base_quantile_predictions:
        return {}
    
    # Get all available quantile levels
    all_quantiles = set()
    for model_preds in base_quantile_predictions.values():
        all_quantiles.update(model_preds.keys())
    
    ensemble_predictions = {}
    
    for quantile in sorted(all_quantiles):
        # Collect predictions for this quantile from all models that have it
        quantile_preds = []
        model_weights = []
        
        for model_name, model_preds in base_quantile_predictions.items():
            if quantile in model_preds:
                quantile_preds.append(model_preds[quantile])
                if weights is not None and model_name in weights:
                    model_weights.append(weights[model_name])
                else:
                    model_weights.append(1.0)
        
        if not quantile_preds:
            continue
        
        # Combine predictions
        if ensemble_method == 'mean':
            ensemble_pred = np.mean(quantile_preds, axis=0)
        elif ensemble_method == 'median':
            ensemble_pred = np.median(quantile_preds, axis=0)
        elif ensemble_method == 'weighted' and weights is not None:
            # Weighted average
            quantile_array = np.array(quantile_preds)
            weight_array = np.array(model_weights)
            weight_array = weight_array / np.sum(weight_array)  # Normalize weights
            ensemble_pred = np.average(quantile_array, axis=0, weights=weight_array)
        else:
            # Default to mean
            ensemble_pred = np.mean(quantile_preds, axis=0)
        
        ensemble_predictions[quantile] = ensemble_pred
    
    logger.info(f"Created quantile ensemble with {len(ensemble_predictions)} quantile levels using {ensemble_method}")
    
    return ensemble_predictions

def quantile_to_binary_predictions(quantile_predictions: Dict[float, np.ndarray],
                                 y_true: Optional[np.ndarray] = None,
                                 decision_strategy: str = 'threshold_optimization',
                                 risk_tolerance: str = 'moderate') -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Convert quantile predictions to binary predictions for trading decisions.
    
    Args:
        quantile_predictions: Dictionary mapping quantile levels to predictions
        y_true: True values for threshold optimization (optional)
        decision_strategy: Strategy for binary conversion ('threshold_optimization', 'risk_aware', 'median_based')
        risk_tolerance: Risk tolerance for risk-aware strategy
        
    Returns:
        Tuple of (binary_predictions, conversion_info)
    """
    conversion_info = {'strategy': decision_strategy, 'risk_tolerance': risk_tolerance}
    
    if decision_strategy == 'risk_aware':
        # Use risk-aware quantile selection
        selected_preds, risk_weight = risk_aware_threshold_selection(quantile_predictions, risk_tolerance)
        
        # Find optimal threshold for the selected quantile
        if y_true is not None:
            from sklearn.metrics import f1_score
            thresholds = np.linspace(selected_preds.min(), selected_preds.max(), 100)
            best_f1 = 0
            best_threshold = 0
            
            for threshold in thresholds:
                binary_pred = (selected_preds > threshold).astype(int)
                # Handle both continuous and binary targets
                if np.all(np.isin(y_true, [0, 1])):
                    # y_true is already binary
                    binary_true = y_true.astype(int)
                else:
                    # y_true is continuous, convert to binary
                    binary_true = (y_true > 0).astype(int)
                
                if len(np.unique(binary_pred)) > 1:
                    f1 = f1_score(binary_true, binary_pred, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
            
            conversion_info['threshold'] = best_threshold
            conversion_info['f1_score'] = best_f1
            binary_predictions = (selected_preds > best_threshold).astype(int)
        else:
            # Use zero threshold as default
            conversion_info['threshold'] = 0.0
            binary_predictions = (selected_preds > 0).astype(int)
            
        conversion_info['risk_weight'] = risk_weight
        # Find the selected quantile by comparing array contents
        selected_quantile = None
        for q, preds in quantile_predictions.items():
            if np.array_equal(preds, selected_preds):
                selected_quantile = q
                break
        conversion_info['selected_quantile'] = selected_quantile or 0.5
        
    elif decision_strategy == 'median_based':
        # Use median quantile (0.5) for decisions
        median_quantile = min(quantile_predictions.keys(), key=lambda x: abs(x - 0.5))
        median_preds = quantile_predictions[median_quantile]
        
        if y_true is not None:
            # Optimize threshold for median predictions
            from sklearn.metrics import f1_score
            thresholds = np.linspace(median_preds.min(), median_preds.max(), 100)
            best_f1 = 0
            best_threshold = 0
            
            for threshold in thresholds:
                binary_pred = (median_preds > threshold).astype(int)
                # Handle both continuous and binary targets
                if np.all(np.isin(y_true, [0, 1])):
                    # y_true is already binary
                    binary_true = y_true.astype(int)
                else:
                    # y_true is continuous, convert to binary
                    binary_true = (y_true > 0).astype(int)
                
                if len(np.unique(binary_pred)) > 1:
                    f1 = f1_score(binary_true, binary_pred, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
            
            conversion_info['threshold'] = best_threshold
            conversion_info['f1_score'] = best_f1
            binary_predictions = (median_preds > best_threshold).astype(int)
        else:
            conversion_info['threshold'] = 0.0
            binary_predictions = (median_preds > 0).astype(int)
            
        conversion_info['selected_quantile'] = median_quantile
        
    else:  # threshold_optimization (default)
        # Use all quantiles to find best performing one
        best_f1 = 0
        best_binary_pred = None
        best_info = {}
        
        for quantile, preds in quantile_predictions.items():
            if y_true is not None:
                from sklearn.metrics import f1_score
                thresholds = np.linspace(preds.min(), preds.max(), 100)
                
                for threshold in thresholds:
                    binary_pred = (preds > threshold).astype(int)
                    # Handle both continuous and binary targets
                    if np.all(np.isin(y_true, [0, 1])):
                        # y_true is already binary
                        binary_true = y_true.astype(int)
                    else:
                        # y_true is continuous, convert to binary
                        binary_true = (y_true > 0).astype(int)
                    
                    if len(np.unique(binary_pred)) > 1:
                        f1 = f1_score(binary_true, binary_pred, zero_division=0)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_binary_pred = binary_pred
                            best_info = {
                                'threshold': threshold,
                                'f1_score': f1,
                                'selected_quantile': quantile
                            }
        
        if best_binary_pred is not None:
            binary_predictions = best_binary_pred
            conversion_info.update(best_info)
        else:
            # Fallback to median quantile with zero threshold
            median_quantile = min(quantile_predictions.keys(), key=lambda x: abs(x - 0.5))
            median_preds = quantile_predictions[median_quantile]
            binary_predictions = (median_preds > 0).astype(int)
            conversion_info.update({
                'threshold': 0.0,
                'selected_quantile': median_quantile,
                'fallback': True
            })
    
    positive_count = np.sum(binary_predictions)
    conversion_info['positive_predictions'] = positive_count
    conversion_info['positive_rate'] = positive_count / len(binary_predictions) if len(binary_predictions) > 0 else 0
    
    logger.info(f"Quantile to binary conversion: {positive_count} positive predictions ({conversion_info['positive_rate']:.3f})")
    
    return binary_predictions, conversion_info

def get_default_quantile_levels() -> List[float]:
    """
    Get default quantile levels for quantile regression.
    
    Returns:
        List of quantile levels for uncertainty estimation
    """
    return [0.1, 0.25, 0.5, 0.75, 0.9]

def validate_quantile_levels(quantile_levels: List[float]) -> List[float]:
    """
    Validate and sort quantile levels.
    
    Args:
        quantile_levels: List of quantile levels to validate
        
    Returns:
        Sorted and validated quantile levels
    """
    valid_quantiles = []
    
    for q in quantile_levels:
        if 0 < q < 1:
            valid_quantiles.append(q)
        else:
            logger.warning(f"Invalid quantile level {q}, must be between 0 and 1")
    
    if not valid_quantiles:
        logger.warning("No valid quantile levels provided, using defaults")
        valid_quantiles = get_default_quantile_levels()
    
    return sorted(set(valid_quantiles))  # Remove duplicates and sort
