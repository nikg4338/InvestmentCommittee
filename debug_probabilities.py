#!/usr/bin/env python3
"""
Debug Probability Distributions
===============================

This script debugs the probability distributions output by our models
to understand why F1 scores are 0 even with class balancing.
"""

import numpy as np
import pandas as pd
from train_models import collect_alpaca_training_data, prepare_training_data, train_committee_models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_model_probabilities():
    """Debug the probability distributions of trained models."""
    
    # Collect data
    logger.info("Collecting training data...")
    df = collect_alpaca_training_data(batch_numbers=[1], max_symbols_per_batch=10, use_cached=True)
    
    if len(df) == 0:
        logger.error("No training data available")
        return
    
    # Prepare data
    feature_columns = [col for col in df.columns if col not in ['target', 'ticker']]
    X_train, X_test, y_train, y_test = prepare_training_data(df, feature_columns, target_column='target')
    
    logger.info(f"Test set class distribution: {pd.Series(y_test).value_counts().to_dict()}")
    
    # Train models
    logger.info("Training models...")
    models, metrics = train_committee_models(X_train, y_train, X_test, y_test)
    
    # Debug probabilities for each model
    for name, model in models.items():
        if name == 'stacked':
            continue  # Skip meta-model for now
            
        logger.info(f"\n=== {name.upper()} MODEL PROBABILITIES ===")
        
        try:
            proba = model.predict_proba(X_test)
            pos_proba = proba[:, -1] if proba.ndim > 1 else proba
            
            logger.info(f"Probability stats:")
            logger.info(f"  Min: {pos_proba.min():.6f}")
            logger.info(f"  Max: {pos_proba.max():.6f}")
            logger.info(f"  Mean: {pos_proba.mean():.6f}")
            logger.info(f"  Median: {np.median(pos_proba):.6f}")
            logger.info(f"  Std: {pos_proba.std():.6f}")
            
            # Count predictions above various thresholds
            thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
            for thresh in thresholds:
                count = (pos_proba >= thresh).sum()
                logger.info(f"  Predictions >= {thresh}: {count}/{len(pos_proba)} ({count/len(pos_proba)*100:.1f}%)")
            
            # Show actual test labels for comparison
            logger.info(f"Actual positive cases in test set: {y_test.sum()}/{len(y_test)}")
            
            # Show probabilities for actual positive cases
            if y_test.sum() > 0:
                pos_indices = np.where(y_test == 1)[0]
                logger.info(f"Probabilities for actual positive cases:")
                for i, idx in enumerate(pos_indices):
                    logger.info(f"  Positive case {i+1}: {pos_proba[idx]:.6f}")
                    
        except Exception as e:
            logger.error(f"Error analyzing {name}: {e}")

if __name__ == '__main__':
    debug_model_probabilities()
