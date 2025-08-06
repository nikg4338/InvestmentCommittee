#!/usr/bin/env python3
"""
Ranking-Based Evaluation Metrics
================================

Advanced evaluation metrics for financial ML models that focus on ranking
quality rather than just threshold-based classification. These metrics are
particularly useful for portfolio construction and stock selection.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import precision_recall_curve, roc_curve

logger = logging.getLogger(__name__)

def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """
    Calculate precision at top-k predictions.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores/probabilities
        k: Number of top predictions to consider
        
    Returns:
        Precision at top-k
    """
    if len(y_true) == 0 or k <= 0:
        return 0.0
    
    # Sort predictions by score (descending)
    sorted_indices = np.argsort(y_scores)[::-1]
    
    # Take top-k predictions
    k = min(k, len(sorted_indices))
    top_k_indices = sorted_indices[:k]
    
    # Calculate precision
    return np.mean(y_true[top_k_indices])

def recall_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """
    Calculate recall at top-k predictions.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores/probabilities
        k: Number of top predictions to consider
        
    Returns:
        Recall at top-k
    """
    if len(y_true) == 0 or k <= 0:
        return 0.0
    
    total_positives = np.sum(y_true)
    if total_positives == 0:
        return 0.0
    
    # Sort predictions by score (descending)
    sorted_indices = np.argsort(y_scores)[::-1]
    
    # Take top-k predictions
    k = min(k, len(sorted_indices))
    top_k_indices = sorted_indices[:k]
    
    # Calculate recall
    true_positives_in_top_k = np.sum(y_true[top_k_indices])
    return true_positives_in_top_k / total_positives

def precision_at_percentile(y_true: np.ndarray, y_scores: np.ndarray, percentile: float) -> float:
    """
    Calculate precision at top percentile of predictions.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores/probabilities
        percentile: Percentile threshold (e.g., 0.1 for top 10%)
        
    Returns:
        Precision at top percentile
    """
    if len(y_true) == 0 or percentile <= 0 or percentile > 1:
        return 0.0
    
    k = max(1, int(len(y_scores) * percentile))
    return precision_at_k(y_true, y_scores, k)

def average_precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, max_k: int = None) -> float:
    """
    Calculate average precision across different k values (MAP@K).
    
    Args:
        y_true: True binary labels  
        y_scores: Predicted scores/probabilities
        max_k: Maximum k to consider (default: number of positives)
        
    Returns:
        Mean Average Precision at K
    """
    if len(y_true) == 0:
        return 0.0
    
    total_positives = np.sum(y_true)
    if total_positives == 0:
        return 0.0
    
    if max_k is None:
        max_k = int(total_positives)
    
    # Sort predictions by score (descending)
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_labels = y_true[sorted_indices]
    
    # Calculate precision at each relevant k
    precisions = []
    for k in range(1, min(max_k + 1, len(sorted_labels) + 1)):
        if sorted_labels[k-1] == 1:  # Only count when we hit a positive
            precision_k = np.mean(sorted_labels[:k])
            precisions.append(precision_k)
    
    return np.mean(precisions) if precisions else 0.0

def hit_rate_at_percentile(y_true: np.ndarray, y_scores: np.ndarray, percentile: float) -> float:
    """
    Calculate hit rate (fraction of top percentile that are positive).
    Same as precision_at_percentile but with clearer naming for financial context.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores/probabilities  
        percentile: Percentile threshold (e.g., 0.05 for top 5%)
        
    Returns:
        Hit rate in top percentile
    """
    return precision_at_percentile(y_true, y_scores, percentile)

def coverage_at_percentile(y_true: np.ndarray, y_scores: np.ndarray, percentile: float) -> float:
    """
    Calculate coverage (fraction of all positives captured in top percentile).
    Same as recall_at_percentile.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores/probabilities
        percentile: Percentile threshold (e.g., 0.1 for top 10%)
        
    Returns:
        Coverage in top percentile
    """
    k = max(1, int(len(y_scores) * percentile))
    return recall_at_k(y_true, y_scores, k)

def ranking_quality_score(y_true: np.ndarray, y_scores: np.ndarray, 
                         percentiles: list = [0.01, 0.02, 0.05, 0.1]) -> Dict[str, float]:
    """
    Comprehensive ranking quality assessment for multiple percentiles.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores/probabilities
        percentiles: List of percentiles to evaluate
        
    Returns:
        Dictionary with ranking metrics for each percentile
    """
    results = {}
    
    for p in percentiles:
        prefix = f"top_{int(p*100)}pct"
        
        results[f"{prefix}_precision"] = precision_at_percentile(y_true, y_scores, p)
        results[f"{prefix}_recall"] = coverage_at_percentile(y_true, y_scores, p)
        results[f"{prefix}_hit_rate"] = hit_rate_at_percentile(y_true, y_scores, p)
        
        # F1 score for this percentile
        prec = results[f"{prefix}_precision"]
        rec = results[f"{prefix}_recall"]
        if prec + rec > 0:
            results[f"{prefix}_f1"] = 2 * prec * rec / (prec + rec)
        else:
            results[f"{prefix}_f1"] = 0.0
    
    # Overall ranking metrics
    results["mean_average_precision"] = average_precision_at_k(y_true, y_scores)
    
    # Add baseline random performance for comparison
    random_baseline = np.mean(y_true)
    results["random_baseline"] = random_baseline
    
    # Calculate lift over random for top 5%
    top5_precision = results.get("top_5pct_precision", 0)
    if random_baseline > 0:
        results["top_5pct_lift"] = top5_precision / random_baseline
    else:
        results["top_5pct_lift"] = 0.0
    
    return results

def compute_ranking_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive ranking metrics for model evaluation.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores/probabilities
        
    Returns:
        Dictionary with all ranking metrics
    """
    try:
        # Core ranking metrics
        metrics = ranking_quality_score(y_true, y_scores)
        
        # Add specific metrics for extreme imbalance
        metrics.update({
            "precision_at_10": precision_at_k(y_true, y_scores, 10),
            "precision_at_50": precision_at_k(y_true, y_scores, 50),
            "precision_at_100": precision_at_k(y_true, y_scores, 100),
            "recall_at_100": recall_at_k(y_true, y_scores, 100),
        })
        
        # Add distribution statistics
        metrics.update({
            "score_mean": np.mean(y_scores),
            "score_std": np.std(y_scores),
            "score_max": np.max(y_scores),
            "score_min": np.min(y_scores),
            "positive_rate": np.mean(y_true),
        })
        
        logger.info(f"📊 Computed {len(metrics)} ranking metrics")
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Failed to compute ranking metrics: {e}")
        return {
            "precision_at_10": 0.0,
            "top_5pct_precision": 0.0,
            "mean_average_precision": 0.0,
            "random_baseline": np.mean(y_true) if len(y_true) > 0 else 0.0
        }

def log_ranking_performance(metrics: Dict[str, float], model_name: str = "Model"):
    """
    Log ranking performance metrics in a readable format.
    
    Args:
        metrics: Dictionary of ranking metrics
        model_name: Name of the model for logging
    """
    logger.info(f"\n📈 Ranking Performance for {model_name}:")
    logger.info(f"   Top 1% Precision: {metrics.get('top_1pct_precision', 0):.3f}")
    logger.info(f"   Top 2% Precision: {metrics.get('top_2pct_precision', 0):.3f}")
    logger.info(f"   Top 5% Precision: {metrics.get('top_5pct_precision', 0):.3f}")
    logger.info(f"   Top 10% Precision: {metrics.get('top_10pct_precision', 0):.3f}")
    logger.info(f"   MAP@K: {metrics.get('mean_average_precision', 0):.3f}")
    logger.info(f"   Top 5% Lift: {metrics.get('top_5pct_lift', 0):.2f}x")
    logger.info(f"   Random Baseline: {metrics.get('random_baseline', 0):.3f}")
