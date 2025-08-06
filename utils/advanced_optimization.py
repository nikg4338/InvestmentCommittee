#!/usr/bin/env python3
"""
Advanced Optimization Utilities for Investment Committee
=======================================================

This module provides advanced optimization techniques for threshold tuning,
enhanced sampling strategies, and multi-class handling for extreme imbalance scenarios.

Key Features:
- Top-K% threshold optimization
- Enhanced SMOTE for regression targets
- Multi-class to binary conversion
- F1-optimized threshold selection
- Portfolio-aware optimization metrics
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import (
    precision_recall_curve, f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score
)

logger = logging.getLogger(__name__)

def find_optimal_threshold_advanced(y_true: np.ndarray, y_pred: np.ndarray, 
                                  strategy: str = 'f1', 
                                  top_k_percent: float = None,
                                  min_precision: float = 0.05,
                                  min_recall: float = 0.1) -> Tuple[float, float, Dict[str, Any]]:
    """
    Advanced optimal threshold finding with multiple strategies.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities or continuous predictions
        strategy: Optimization strategy ('f1', 'precision', 'recall', 'top_k_percent', 'pr_auc')
        top_k_percent: If using top_k_percent strategy, percentage of top predictions to mark as positive
        min_precision: Minimum precision constraint
        min_recall: Minimum recall constraint
        
    Returns:
        Tuple of (optimal_threshold, best_score, detailed_results)
    """
    logger.info(f"🎯 Finding optimal threshold using {strategy} strategy...")
    
    if strategy == 'top_k_percent' and top_k_percent is not None:
        # Top-K% strategy: select top K% of predictions as positive
        k_samples = int(len(y_pred) * top_k_percent / 100)
        k_samples = max(1, k_samples)  # Ensure at least 1 sample
        
        if k_samples >= len(y_pred):
            threshold = np.min(y_pred)
        else:
            threshold = np.partition(y_pred, -k_samples)[-k_samples]
        
        y_pred_binary = (y_pred >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        
        results = {
            'strategy': f'top_{top_k_percent}%',
            'selected_samples': k_samples,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'positive_rate': k_samples / len(y_pred) * 100
        }
        
        logger.info(f"🎯 Top {top_k_percent}% strategy: threshold={threshold:.4f}, F1={f1:.3f}, precision={precision:.3f}, recall={recall:.3f}")
        logger.info(f"   Selected {k_samples}/{len(y_pred)} samples ({100*k_samples/len(y_pred):.1f}%)")
        return threshold, f1, results
    
    elif strategy == 'pr_auc':
        # Optimize for precision-recall AUC (good for imbalanced data)
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        
        # Calculate PR-AUC for each threshold
        pr_auc_scores = []
        for i, thresh in enumerate(thresholds):
            y_pred_thresh = (y_pred >= thresh).astype(int)
            if np.sum(y_pred_thresh) > 0:  # Avoid empty predictions
                pr_auc = average_precision_score(y_true, y_pred_thresh)
                pr_auc_scores.append(pr_auc)
            else:
                pr_auc_scores.append(0.0)
        
        best_idx = np.argmax(pr_auc_scores)
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_score = pr_auc_scores[best_idx]
        
        results = {
            'strategy': 'pr_auc',
            'precision': precision[best_idx],
            'recall': recall[best_idx],
            'f1_score': 2 * precision[best_idx] * recall[best_idx] / (precision[best_idx] + recall[best_idx] + 1e-8),
            'pr_auc': best_score
        }
        
    else:
        # Traditional threshold optimization with constraints
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        
        if strategy == 'f1':
            # F1 score optimization with constraints
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            
            # Apply constraints
            valid_mask = (precision >= min_precision) & (recall >= min_recall)
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)[0]
                best_idx = valid_indices[np.argmax(f1_scores[valid_indices])]
                best_score = f1_scores[best_idx]
                logger.info(f"   Applied constraints: min_precision={min_precision}, min_recall={min_recall}")
            else:
                logger.warning(f"   No thresholds meet constraints, using best F1 without constraints")
                best_idx = np.argmax(f1_scores)
                best_score = f1_scores[best_idx]
                
        elif strategy == 'precision':
            # Maximize precision while maintaining reasonable recall
            valid_indices = recall >= min_recall
            if np.any(valid_indices):
                best_idx = np.argmax(precision[valid_indices])
                best_idx = np.where(valid_indices)[0][best_idx]
                best_score = precision[best_idx]
            else:
                best_idx = np.argmax(precision)
                best_score = precision[best_idx]
                
        elif strategy == 'recall':
            # Maximize recall while maintaining reasonable precision
            valid_indices = precision >= min_precision
            if np.any(valid_indices):
                best_idx = np.argmax(recall[valid_indices])
                best_idx = np.where(valid_indices)[0][best_idx]
                best_score = recall[best_idx]
            else:
                best_idx = np.argmax(recall)
                best_score = recall[best_idx]
        
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        results = {
            'strategy': strategy,
            'precision': precision[best_idx],
            'recall': recall[best_idx],
            'f1_score': 2 * precision[best_idx] * recall[best_idx] / (precision[best_idx] + recall[best_idx] + 1e-8)
        }
    
    logger.info(f"🎯 {strategy} optimization: threshold={optimal_threshold:.4f}, score={best_score:.3f}")
    logger.info(f"   Final metrics: P={results['precision']:.3f}, R={results['recall']:.3f}, F1={results['f1_score']:.3f}")
    
    return optimal_threshold, best_score, results


def enhanced_smote_for_regression(X: np.ndarray, y: np.ndarray, 
                                 target_percentile: float = 90,
                                 sampling_strategy: float = 0.5,
                                 k_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhanced SMOTE for regression targets, upsampling high-return examples.
    
    Args:
        X: Feature matrix
        y: Continuous target values (returns)
        target_percentile: Percentile threshold for defining "high-return" examples
        sampling_strategy: Ratio of synthetic samples to create
        k_neighbors: Number of neighbors for SMOTE
        
    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    try:
        from imblearn.over_sampling import SMOTE
        from sklearn.neighbors import NearestNeighbors
        
        logger.info(f"🔄 Enhanced SMOTE for regression targets...")
        logger.info(f"   Target percentile: {target_percentile}%, sampling strategy: {sampling_strategy}")
        
        # Identify high-return examples
        high_return_threshold = np.percentile(y, target_percentile)
        high_return_mask = y >= high_return_threshold
        
        # Convert to binary for SMOTE
        y_binary = high_return_mask.astype(int)
        
        original_positive_count = np.sum(y_binary)
        logger.info(f"   Original high-return samples: {original_positive_count}/{len(y)} ({100*original_positive_count/len(y):.1f}%)")
        
        if original_positive_count < 2:
            logger.warning(f"   Too few high-return samples for SMOTE, returning original data")
            return X, y
        
        # Apply SMOTE to create synthetic high-return examples
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=min(k_neighbors, original_positive_count - 1),
            random_state=42
        )
        
        X_resampled, y_binary_resampled = smote.fit_resample(X, y_binary)
        
        # For synthetic samples, assign return values based on original high-return distribution
        synthetic_mask = len(X_resampled) > len(X)
        if synthetic_mask:
            n_synthetic = len(X_resampled) - len(X)
            # Sample from original high-return distribution
            high_return_values = y[high_return_mask]
            synthetic_returns = np.random.choice(high_return_values, size=n_synthetic, replace=True)
            
            # Combine original and synthetic targets
            y_resampled = np.concatenate([y, synthetic_returns])
        else:
            y_resampled = y
        
        logger.info(f"✅ Enhanced SMOTE complete:")
        logger.info(f"   Resampled shape: {X_resampled.shape} (was {X.shape})")
        logger.info(f"   High-return samples: {np.sum(y_resampled >= high_return_threshold)}/{len(y_resampled)} ({100*np.sum(y_resampled >= high_return_threshold)/len(y_resampled):.1f}%)")
        
        return X_resampled, y_resampled
        
    except ImportError:
        logger.warning("imblearn not available, returning original data")
        return X, y
    except Exception as e:
        logger.error(f"Enhanced SMOTE failed: {e}, returning original data")
        return X, y


def convert_multiclass_to_binary(y_multiclass: np.ndarray, 
                                strategy: str = 'top_class',
                                custom_mapping: Dict[int, int] = None) -> np.ndarray:
    """
    Convert multi-class targets to binary for final predictions.
    
    Args:
        y_multiclass: Multi-class predictions or labels
        strategy: Conversion strategy ('top_class', 'positive_classes', 'custom')
        custom_mapping: Custom mapping for 'custom' strategy
        
    Returns:
        Binary predictions (0 or 1)
    """
    if strategy == 'top_class':
        # Highest class becomes positive
        max_class = np.max(y_multiclass)
        return (y_multiclass == max_class).astype(int)
    
    elif strategy == 'positive_classes':
        # Upper half of classes become positive
        unique_classes = np.unique(y_multiclass)
        median_class = np.median(unique_classes)
        return (y_multiclass > median_class).astype(int)
    
    elif strategy == 'custom' and custom_mapping is not None:
        # Use custom mapping
        binary_result = np.zeros_like(y_multiclass)
        for original_class, binary_class in custom_mapping.items():
            binary_result[y_multiclass == original_class] = binary_class
        return binary_result
    
    else:
        logger.warning(f"Unknown strategy {strategy}, using top_class")
        return convert_multiclass_to_binary(y_multiclass, 'top_class')


def portfolio_aware_threshold_optimization(y_true: np.ndarray, y_pred: np.ndarray,
                                          portfolio_size: int = 20,
                                          risk_tolerance: str = 'moderate') -> Tuple[float, Dict[str, Any]]:
    """
    Optimize threshold for portfolio construction with risk management.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        portfolio_size: Target portfolio size
        risk_tolerance: Risk tolerance ('conservative', 'moderate', 'aggressive')
        
    Returns:
        Tuple of (optimal_threshold, portfolio_metrics)
    """
    logger.info(f"📊 Portfolio-aware threshold optimization for {portfolio_size} positions...")
    
    # Define risk tolerance parameters
    risk_params = {
        'conservative': {'min_precision': 0.15, 'max_positions': portfolio_size * 0.8},
        'moderate': {'min_precision': 0.10, 'max_positions': portfolio_size * 1.0},
        'aggressive': {'min_precision': 0.05, 'max_positions': portfolio_size * 1.2}
    }
    
    params = risk_params.get(risk_tolerance, risk_params['moderate'])
    min_precision = params['min_precision']
    max_positions = int(params['max_positions'])
    
    # Try different threshold strategies
    strategies = [
        ('top_k_percent', portfolio_size / len(y_pred) * 100),
        ('f1', None),
        ('precision', None)
    ]
    
    best_threshold = 0.5
    best_score = 0.0
    best_metrics = {}
    
    for strategy, param in strategies:
        try:
            if strategy == 'top_k_percent':
                threshold, score, metrics = find_optimal_threshold_advanced(
                    y_true, y_pred, strategy, top_k_percent=param
                )
            else:
                threshold, score, metrics = find_optimal_threshold_advanced(
                    y_true, y_pred, strategy, min_precision=min_precision
                )
            
            # Check portfolio constraints
            y_pred_binary = (y_pred >= threshold).astype(int)
            n_positions = np.sum(y_pred_binary)
            
            if n_positions <= max_positions and metrics['precision'] >= min_precision:
                portfolio_score = score * (1 + 0.1 * (1 - abs(n_positions - portfolio_size) / portfolio_size))
                
                if portfolio_score > best_score:
                    best_threshold = threshold
                    best_score = portfolio_score
                    best_metrics = metrics.copy()
                    best_metrics['portfolio_score'] = portfolio_score
                    best_metrics['n_positions'] = n_positions
                    best_metrics['risk_tolerance'] = risk_tolerance
            
        except Exception as e:
            logger.warning(f"Strategy {strategy} failed: {e}")
            continue
    
    logger.info(f"🎯 Best portfolio threshold: {best_threshold:.4f}")
    logger.info(f"   Portfolio size: {best_metrics.get('n_positions', 0)}/{portfolio_size}")
    logger.info(f"   Risk tolerance: {risk_tolerance}")
    logger.info(f"   Precision: {best_metrics.get('precision', 0):.3f}")
    logger.info(f"   Portfolio score: {best_score:.3f}")
    
    return best_threshold, best_metrics


def evaluate_threshold_robustness(y_true: np.ndarray, y_pred: np.ndarray,
                                 threshold: float, perturbation_range: float = 0.1) -> Dict[str, float]:
    """
    Evaluate how robust a threshold is to small perturbations.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        threshold: Threshold to evaluate
        perturbation_range: Range of perturbations to test (±)
        
    Returns:
        Dictionary with robustness metrics
    """
    logger.info(f"🔍 Evaluating threshold robustness around {threshold:.4f}...")
    
    # Test thresholds around the optimal one
    test_thresholds = np.linspace(
        threshold - perturbation_range,
        threshold + perturbation_range,
        21  # 21 points for ±10% range
    )
    
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for test_thresh in test_thresholds:
        y_pred_binary = (y_pred >= test_thresh).astype(int)
        
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    # Calculate robustness metrics
    f1_std = np.std(f1_scores)
    f1_range = np.max(f1_scores) - np.min(f1_scores)
    precision_std = np.std(precision_scores)
    recall_std = np.std(recall_scores)
    
    # Baseline metrics at optimal threshold
    y_pred_binary_opt = (y_pred >= threshold).astype(int)
    baseline_f1 = f1_score(y_true, y_pred_binary_opt, zero_division=0)
    
    robustness_metrics = {
        'threshold': threshold,
        'baseline_f1': baseline_f1,
        'f1_std': f1_std,
        'f1_range': f1_range,
        'precision_std': precision_std,
        'recall_std': recall_std,
        'robustness_score': baseline_f1 / (1 + f1_std),  # Higher is better
        'stability_ratio': 1 - (f1_range / (baseline_f1 + 1e-8))  # Closer to 1 is more stable
    }
    
    logger.info(f"🔍 Robustness analysis:")
    logger.info(f"   F1 stability: std={f1_std:.3f}, range={f1_range:.3f}")
    logger.info(f"   Robustness score: {robustness_metrics['robustness_score']:.3f}")
    logger.info(f"   Stability ratio: {robustness_metrics['stability_ratio']:.3f}")
    
    return robustness_metrics


def adaptive_threshold_selection(y_true_list: List[np.ndarray], 
                                y_pred_list: List[np.ndarray],
                                validation_strategy: str = 'cv_average') -> Tuple[float, Dict[str, Any]]:
    """
    Select threshold using cross-validation or multi-fold validation.
    
    Args:
        y_true_list: List of true labels from different folds/periods
        y_pred_list: List of predictions from different folds/periods
        validation_strategy: Strategy ('cv_average', 'robust_median', 'best_worst_case')
        
    Returns:
        Tuple of (adaptive_threshold, validation_metrics)
    """
    logger.info(f"🔄 Adaptive threshold selection using {validation_strategy}...")
    
    fold_thresholds = []
    fold_scores = []
    fold_metrics = []
    
    # Find optimal threshold for each fold
    for i, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list)):
        try:
            threshold, score, metrics = find_optimal_threshold_advanced(y_true, y_pred, 'f1')
            fold_thresholds.append(threshold)
            fold_scores.append(score)
            fold_metrics.append(metrics)
            
            logger.info(f"   Fold {i+1}: threshold={threshold:.4f}, F1={score:.3f}")
            
        except Exception as e:
            logger.warning(f"   Fold {i+1} failed: {e}")
            continue
    
    if not fold_thresholds:
        logger.error("No valid folds for threshold selection")
        return 0.5, {}
    
    # Select final threshold based on strategy
    if validation_strategy == 'cv_average':
        adaptive_threshold = np.mean(fold_thresholds)
        confidence = 1.0 - np.std(fold_thresholds) / (np.mean(fold_thresholds) + 1e-8)
        
    elif validation_strategy == 'robust_median':
        adaptive_threshold = np.median(fold_thresholds)
        confidence = 1.0 - np.std(fold_thresholds) / (np.median(fold_thresholds) + 1e-8)
        
    elif validation_strategy == 'best_worst_case':
        # Choose threshold that performs best in worst-case scenario
        min_scores = []
        for threshold in fold_thresholds:
            min_score = float('inf')
            for y_true, y_pred in zip(y_true_list, y_pred_list):
                y_pred_binary = (y_pred >= threshold).astype(int)
                score = f1_score(y_true, y_pred_binary, zero_division=0)
                min_score = min(min_score, score)
            min_scores.append(min_score)
        
        best_idx = np.argmax(min_scores)
        adaptive_threshold = fold_thresholds[best_idx]
        confidence = min_scores[best_idx] / np.mean(fold_scores)
    
    else:
        adaptive_threshold = np.mean(fold_thresholds)
        confidence = 0.5
    
    validation_metrics = {
        'adaptive_threshold': adaptive_threshold,
        'fold_thresholds': fold_thresholds,
        'fold_scores': fold_scores,
        'validation_strategy': validation_strategy,
        'confidence': confidence,
        'threshold_std': np.std(fold_thresholds),
        'score_mean': np.mean(fold_scores),
        'score_std': np.std(fold_scores)
    }
    
    logger.info(f"🎯 Adaptive threshold selected: {adaptive_threshold:.4f}")
    logger.info(f"   Strategy: {validation_strategy}")
    logger.info(f"   Confidence: {confidence:.3f}")
    logger.info(f"   Threshold stability: ±{np.std(fold_thresholds):.3f}")
    
    return adaptive_threshold, validation_metrics
