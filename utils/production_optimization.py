#!/usr/bin/env python3
"""
Enhanced Threshold Optimization for Production Trading
=====================================================

Optimized threshold selection for regressor models specifically designed for paper trading.
Focuses on precision-recall balance suitable for portfolio construction and risk management.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, precision_recall_curve
)

logger = logging.getLogger(__name__)

def find_portfolio_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                                   portfolio_size: int = 20,
                                   min_precision: float = 0.3,
                                   min_recall: float = 0.1,
                                   risk_tolerance: str = 'moderate') -> Tuple[float, Dict[str, float]]:
    """
    Find optimal threshold for portfolio trading with focus on precision-recall balance.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        portfolio_size: Target number of positions in portfolio
        min_precision: Minimum acceptable precision
        min_recall: Minimum acceptable recall
        risk_tolerance: 'conservative', 'moderate', or 'aggressive'
        
    Returns:
        Tuple of (optimal_threshold, metrics_dict)
    """
    
    # Risk tolerance parameters
    risk_params = {
        'conservative': {'precision_weight': 0.7, 'min_precision': 0.4, 'min_recall': 0.05},
        'moderate': {'precision_weight': 0.6, 'min_precision': 0.3, 'min_recall': 0.1},
        'aggressive': {'precision_weight': 0.5, 'min_precision': 0.25, 'min_recall': 0.15}
    }
    
    params = risk_params.get(risk_tolerance, risk_params['moderate'])
    min_precision = max(min_precision, params['min_precision'])
    min_recall = max(min_recall, params['min_recall'])
    precision_weight = params['precision_weight']
    
    # Portfolio-focused threshold grid
    # Start with percentile-based thresholds for portfolio sizing
    percentiles = np.linspace(80, 99.5, 50)  # Focus on top predictions
    portfolio_thresholds = [np.percentile(y_proba, p) for p in percentiles]
    
    # Add linear grid in high-probability range
    high_prob_range = np.linspace(0.3, 0.95, 50)
    
    # Combine and sort thresholds
    all_thresholds = np.unique(np.concatenate([portfolio_thresholds, high_prob_range]))
    
    best_threshold = 0.5
    best_score = 0.0
    best_metrics = {}
    
    valid_candidates = []
    
    logger.info(f"🎯 Portfolio threshold optimization: target {portfolio_size} positions, {risk_tolerance} risk")
    
    for threshold in all_thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Skip if no positive predictions
        if np.sum(y_pred) == 0:
            continue
            
        # Calculate metrics
        try:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Portfolio size constraint
            n_positions = np.sum(y_pred)
            size_penalty = abs(n_positions - portfolio_size) / portfolio_size
            
            # Apply minimum constraints
            if precision < min_precision or recall < min_recall:
                continue
                
            # Portfolio-focused scoring function
            portfolio_score = (
                precision_weight * precision + 
                (1 - precision_weight) * recall - 
                0.1 * size_penalty  # Penalty for deviating from target size
            )
            
            metrics = {
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'portfolio_size': n_positions,
                'portfolio_score': portfolio_score,
                'size_deviation': size_penalty
            }
            
            valid_candidates.append(metrics)
            
            if portfolio_score > best_score:
                best_score = portfolio_score
                best_threshold = threshold
                best_metrics = metrics
                
        except (ValueError, ZeroDivisionError):
            continue
    
    # If no valid candidates found, use a more conservative approach
    if not valid_candidates:
        logger.warning("No thresholds met minimum requirements. Using conservative fallback.")
        
        # Conservative fallback: aim for high precision, accept lower recall
        for threshold in np.linspace(0.5, 0.9, 20):
            y_pred = (y_proba >= threshold).astype(int)
            if np.sum(y_pred) == 0:
                continue
                
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            if precision >= 0.2:  # Very conservative minimum
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1_score(y_true, y_pred, zero_division=0),
                    'portfolio_size': np.sum(y_pred),
                    'portfolio_score': precision * 0.8 + recall * 0.2,
                    'fallback': True
                }
                break
    
    logger.info(f"✅ Optimal threshold: {best_threshold:.4f}")
    logger.info(f"   Portfolio size: {best_metrics.get('portfolio_size', 0)}")
    logger.info(f"   Precision: {best_metrics.get('precision', 0):.3f}")
    logger.info(f"   Recall: {best_metrics.get('recall', 0):.3f}")
    
    return best_threshold, best_metrics


def optimize_regressor_thresholds(models: Dict[str, Any], 
                                X_test: pd.DataFrame, 
                                y_test: pd.Series,
                                portfolio_size: int = 20) -> Dict[str, Dict[str, float]]:
    """
    Optimize thresholds for all regressor models using portfolio-focused approach.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        portfolio_size: Target portfolio size
        
    Returns:
        Dictionary mapping model names to optimized threshold info
    """
    
    optimized_thresholds = {}
    
    logger.info("🎯 Optimizing regressor thresholds for paper trading...")
    
    for model_name, model in models.items():
        if 'regressor' not in model_name.lower():
            continue
            
        logger.info(f"📊 Optimizing {model_name}...")
        
        try:
            # Get model predictions
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'predict'):
                # For regressors, we need to convert to probabilities
                y_pred_raw = model.predict(X_test)
                # Apply sigmoid to convert to [0,1] range
                y_proba = 1 / (1 + np.exp(-y_pred_raw))
            else:
                logger.warning(f"❌ {model_name} has no predict method")
                continue
                
            # Find optimal threshold
            threshold, metrics = find_portfolio_optimal_threshold(
                y_test.values, y_proba, 
                portfolio_size=portfolio_size,
                risk_tolerance='moderate'
            )
            
            optimized_thresholds[model_name] = {
                'threshold': threshold,
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1': metrics.get('f1', 0),
                'portfolio_size': metrics.get('portfolio_size', 0),
                'improvement': True
            }
            
        except Exception as e:
            logger.error(f"❌ Error optimizing {model_name}: {e}")
            optimized_thresholds[model_name] = {
                'threshold': 0.5,
                'error': str(e),
                'improvement': False
            }
    
    return optimized_thresholds


def create_production_threshold_config(optimized_thresholds: Dict[str, Dict[str, float]],
                                     output_path: str = "config/production_thresholds.json") -> None:
    """
    Create production configuration file with optimized thresholds.
    
    Args:
        optimized_thresholds: Results from threshold optimization
        output_path: Path to save configuration file
    """
    import json
    import os
    
    # Create config directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    production_config = {
        "metadata": {
            "generated_at": pd.Timestamp.now().isoformat(),
            "description": "Optimized thresholds for paper trading",
            "optimization_method": "portfolio_focused"
        },
        "model_thresholds": optimized_thresholds,
        "portfolio_settings": {
            "target_size": 20,
            "risk_tolerance": "moderate",
            "min_precision": 0.3,
            "min_recall": 0.1
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(production_config, f, indent=2)
    
    logger.info(f"✅ Production threshold config saved to {output_path}")


def validate_threshold_robustness(models: Dict[str, Any],
                                X_val: pd.DataFrame,
                                y_val: pd.Series,
                                thresholds: Dict[str, float],
                                n_bootstrap: int = 50) -> Dict[str, Dict[str, float]]:
    """
    Validate threshold robustness using bootstrap sampling.
    
    Args:
        models: Dictionary of trained models
        X_val: Validation features  
        y_val: Validation labels
        thresholds: Optimized thresholds to validate
        n_bootstrap: Number of bootstrap iterations
        
    Returns:
        Dictionary with robustness statistics for each model
    """
    
    logger.info("🔍 Validating threshold robustness with bootstrap sampling...")
    
    robustness_results = {}
    
    for model_name, threshold in thresholds.items():
        if model_name not in models:
            continue
            
        model = models[model_name]
        bootstrap_metrics = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            n_samples = len(X_val)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_val.iloc[indices]
            y_boot = y_val.iloc[indices]
            
            try:
                # Get predictions
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_boot)[:, 1]
                else:
                    y_pred_raw = model.predict(X_boot)
                    y_proba = 1 / (1 + np.exp(-y_pred_raw))
                
                # Apply threshold
                y_pred = (y_proba >= threshold).astype(int)
                
                # Calculate metrics
                precision = precision_score(y_boot, y_pred, zero_division=0)
                recall = recall_score(y_boot, y_pred, zero_division=0)
                f1 = f1_score(y_boot, y_pred, zero_division=0)
                
                bootstrap_metrics.append({
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
                
            except Exception as e:
                logger.warning(f"Bootstrap iteration {i} failed for {model_name}: {e}")
                continue
        
        if bootstrap_metrics:
            # Calculate robustness statistics
            precisions = [m['precision'] for m in bootstrap_metrics]
            recalls = [m['recall'] for m in bootstrap_metrics]
            f1s = [m['f1'] for m in bootstrap_metrics]
            
            robustness_results[model_name] = {
                'precision_mean': np.mean(precisions),
                'precision_std': np.std(precisions),
                'recall_mean': np.mean(recalls),
                'recall_std': np.std(recalls),
                'f1_mean': np.mean(f1s),
                'f1_std': np.std(f1s),
                'stability_score': 1.0 - (np.std(f1s) / max(np.mean(f1s), 0.001))  # Higher = more stable
            }
            
            logger.info(f"📊 {model_name}: F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}, "
                       f"Stability={robustness_results[model_name]['stability_score']:.3f}")
        else:
            robustness_results[model_name] = {'error': 'All bootstrap iterations failed'}
    
    return robustness_results
