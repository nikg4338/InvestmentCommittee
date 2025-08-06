#!/usr/bin/env python3
"""
Probability Analysis Utilities
==============================

Tools for analyzing and visualizing prediction probability distributions
to diagnose and improve model performance.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Import matplotlib if available
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

def analyze_probability_distribution(probabilities: np.ndarray, 
                                   model_name: str = "Model",
                                   y_true: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Analyze probability distribution statistics.
    
    Args:
        probabilities: Array of prediction probabilities
        model_name: Name of the model for logging
        y_true: True labels for additional analysis (optional)
        
    Returns:
        Dictionary with distribution statistics
    """
    stats = {
        'model_name': model_name,
        'count': len(probabilities),
        'min': float(probabilities.min()),
        'max': float(probabilities.max()),
        'mean': float(probabilities.mean()),
        'std': float(probabilities.std()),
        'median': float(np.median(probabilities)),
        'q25': float(np.percentile(probabilities, 25)),
        'q75': float(np.percentile(probabilities, 75)),
        'above_0.5': int((probabilities > 0.5).sum()),
        'above_0.7': int((probabilities > 0.7).sum()),
        'above_0.9': int((probabilities > 0.9).sum()),
        'below_0.1': int((probabilities < 0.1).sum()),
        'below_0.3': int((probabilities < 0.3).sum()),
    }
    
    # Add percentages
    total = len(probabilities)
    stats['pct_above_0.5'] = 100.0 * stats['above_0.5'] / total
    stats['pct_above_0.7'] = 100.0 * stats['above_0.7'] / total
    stats['pct_above_0.9'] = 100.0 * stats['above_0.9'] / total
    stats['pct_below_0.1'] = 100.0 * stats['below_0.1'] / total
    stats['pct_below_0.3'] = 100.0 * stats['below_0.3'] / total
    
    # Add class-specific statistics if labels provided
    if y_true is not None:
        pos_mask = (y_true == 1)
        neg_mask = (y_true == 0)
        
        if pos_mask.sum() > 0:
            stats['positive_class_mean'] = float(probabilities[pos_mask].mean())
            stats['positive_class_std'] = float(probabilities[pos_mask].std())
        
        if neg_mask.sum() > 0:
            stats['negative_class_mean'] = float(probabilities[neg_mask].mean())
            stats['negative_class_std'] = float(probabilities[neg_mask].std())
    
    return stats

def log_probability_analysis(probabilities: np.ndarray, 
                           model_name: str = "Model",
                           y_true: Optional[np.ndarray] = None) -> None:
    """
    Log detailed probability distribution analysis.
    
    Args:
        probabilities: Array of prediction probabilities
        model_name: Name of the model for logging
        y_true: True labels for additional analysis (optional)
    """
    stats = analyze_probability_distribution(probabilities, model_name, y_true)
    
    logger.info(f"\n📊 {model_name} Probability Distribution Analysis:")
    logger.info(f"  Count: {stats['count']:,}")
    logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    logger.info(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
    logger.info(f"  Median: {stats['median']:.4f}")
    logger.info(f"  Quartiles: Q25={stats['q25']:.4f}, Q75={stats['q75']:.4f}")
    
    logger.info(f"  Threshold Analysis:")
    logger.info(f"    > 0.5: {stats['above_0.5']:,} ({stats['pct_above_0.5']:.1f}%)")
    logger.info(f"    > 0.7: {stats['above_0.7']:,} ({stats['pct_above_0.7']:.1f}%)")
    logger.info(f"    > 0.9: {stats['above_0.9']:,} ({stats['pct_above_0.9']:.1f}%)")
    logger.info(f"    < 0.1: {stats['below_0.1']:,} ({stats['pct_below_0.1']:.1f}%)")
    logger.info(f"    < 0.3: {stats['below_0.3']:,} ({stats['pct_below_0.3']:.1f}%)")
    
    if y_true is not None and 'positive_class_mean' in stats:
        logger.info(f"  Class-specific Analysis:")
        logger.info(f"    Positive class (y=1): mean={stats['positive_class_mean']:.4f}, std={stats['positive_class_std']:.4f}")
        if 'negative_class_mean' in stats:
            logger.info(f"    Negative class (y=0): mean={stats['negative_class_mean']:.4f}, std={stats['negative_class_std']:.4f}")

def plot_probability_histogram(probabilities: np.ndarray, 
                             model_name: str = "Model",
                             y_true: Optional[np.ndarray] = None,
                             bins: int = 50,
                             save_path: Optional[str] = None) -> Optional[str]:
    """
    Plot probability distribution histogram.
    
    Args:
        probabilities: Array of prediction probabilities
        model_name: Name of the model for the title
        y_true: True labels for class-specific coloring (optional)
        bins: Number of histogram bins
        save_path: Path to save the plot (optional)
        
    Returns:
        Path to saved plot or None if not saved
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available, skipping probability histogram")
        return None
    
    plt.figure(figsize=(10, 6))
    
    if y_true is not None:
        # Plot separate histograms for each class
        pos_probs = probabilities[y_true == 1]
        neg_probs = probabilities[y_true == 0]
        
        plt.hist(neg_probs, bins=bins, alpha=0.7, label=f'Negative (n={len(neg_probs)})', 
                color='red', density=True)
        plt.hist(pos_probs, bins=bins, alpha=0.7, label=f'Positive (n={len(pos_probs)})', 
                color='green', density=True)
        plt.legend()
    else:
        # Single histogram
        plt.hist(probabilities, bins=bins, alpha=0.7, color='blue', density=True)
    
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold=0.5')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title(f'{model_name} - Probability Distribution')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    
    # Add statistics text
    stats = analyze_probability_distribution(probabilities, model_name, y_true)
    stats_text = f"Mean: {stats['mean']:.3f}\nStd: {stats['std']:.3f}\n>0.5: {stats['pct_above_0.5']:.1f}%"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Probability histogram saved to {save_path}")
        plt.close()
        return save_path
    else:
        plt.show()
        return None

def analyze_model_ensemble_probabilities(model_predictions: Dict[str, np.ndarray],
                                       y_true: Optional[np.ndarray] = None,
                                       save_plots: bool = False,
                                       plot_dir: str = "reports") -> Dict[str, Any]:
    """
    Comprehensive analysis of multiple model probability distributions.
    
    Args:
        model_predictions: Dictionary mapping model names to probability arrays
        y_true: True labels for additional analysis (optional)
        save_plots: Whether to save histogram plots
        plot_dir: Directory to save plots
        
    Returns:
        Dictionary with analysis results for all models
    """
    analysis_results = {}
    
    logger.info("\n🎯 Ensemble Probability Distribution Analysis")
    logger.info("=" * 60)
    
    for model_name, probabilities in model_predictions.items():
        # Analyze this model
        stats = analyze_probability_distribution(probabilities, model_name, y_true)
        analysis_results[model_name] = stats
        
        # Log analysis
        log_probability_analysis(probabilities, model_name, y_true)
        
        # Plot if requested
        if save_plots and MATPLOTLIB_AVAILABLE:
            plot_path = f"{plot_dir}/{model_name}_probability_distribution.png"
            plot_probability_histogram(probabilities, model_name, y_true, save_path=plot_path)
    
    # Summary comparison
    if len(model_predictions) > 1:
        logger.info(f"\n📈 Model Comparison Summary:")
        logger.info(f"{'Model':<15} {'Mean':<8} {'Std':<8} {'>0.5%':<8} {'Range':<12}")
        logger.info("-" * 55)
        
        for model_name, stats in analysis_results.items():
            range_str = f"[{stats['min']:.3f}, {stats['max']:.3f}]"
            logger.info(f"{model_name:<15} {stats['mean']:<8.4f} {stats['std']:<8.4f} "
                       f"{stats['pct_above_0.5']:<8.1f} {range_str:<12}")
    
    return analysis_results

def diagnose_probability_issues(probabilities: np.ndarray, 
                              model_name: str = "Model") -> List[str]:
    """
    Diagnose common issues with probability distributions.
    
    Args:
        probabilities: Array of prediction probabilities
        model_name: Name of the model
        
    Returns:
        List of diagnostic messages
    """
    issues = []
    stats = analyze_probability_distribution(probabilities, model_name)
    
    # Check for lack of spread
    if stats['std'] < 0.1:
        issues.append(f"{model_name}: Very low standard deviation ({stats['std']:.4f}) - model lacks confidence spread")
    
    # Check for extreme bias
    if stats['mean'] > 0.8:
        issues.append(f"{model_name}: High mean probability ({stats['mean']:.4f}) - model may be overconfident in positive class")
    elif stats['mean'] < 0.2:
        issues.append(f"{model_name}: Low mean probability ({stats['mean']:.4f}) - model may be overconfident in negative class")
    
    # Check for extreme skew
    if stats['pct_above_0.9'] > 50:
        issues.append(f"{model_name}: {stats['pct_above_0.9']:.1f}% predictions >0.9 - model may be overconfident")
    
    if stats['pct_below_0.1'] > 50:
        issues.append(f"{model_name}: {stats['pct_below_0.1']:.1f}% predictions <0.1 - model may be underconfident")
    
    # Check for no spread around 0.5
    mid_range = ((probabilities >= 0.3) & (probabilities <= 0.7)).sum()
    mid_pct = 100.0 * mid_range / len(probabilities)
    if mid_pct < 10:
        issues.append(f"{model_name}: Only {mid_pct:.1f}% predictions in [0.3, 0.7] range - model has poor calibration")
    
    return issues
