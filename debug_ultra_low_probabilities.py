#!/usr/bin/env python3
"""
Debug script to analyze ultra-low probability distributions from the Committee of Five models
"""

import pandas as pd
import numpy as np
from train_models import collect_alpaca_training_data, prepare_training_data, Committee_of_Five_Training
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_ultra_low_probabilities():
    """Analyze ultra-low probability distributions to understand threshold issues"""
    
    logger.info("🔍 Starting ultra-low probability debugging...")
    
    # Load the same data used in training - using the same approach as train_models.py
    logger.info("Collecting training data...")
    df = collect_alpaca_training_data(batch_numbers=[1, 2], max_symbols_per_batch=10, use_cached=True)
    
    if len(df) == 0:
        logger.error("No training data available")
        return
    
    # Prepare data using the same approach
    feature_columns = [col for col in df.columns if col not in ['target', 'ticker']]
    X_train, X_test, y_train, y_test = prepare_training_data(df, feature_columns, target_column='target')
    
    logger.info(f"Data loaded: {len(X_train)} train, {len(X_test)} test samples")
    logger.info(f"Train class distribution: {pd.Series(y_train).value_counts().to_dict()}")
    logger.info(f"Test class distribution: {pd.Series(y_test).value_counts().to_dict()}")
    
    # Train models to get probability predictions
    logger.info("Training models...")
    trainer = Committee_of_Five_Training(
        n_folds=5,
        sampler='smoteenn',
        calibrate='isotonic',
        random_state=42
    )
    
    models = trainer.train(X_train, y_train, X_test)
    
    if models is None:
        logger.error("❌ Training failed!")
        return
    
    # Analyze each model's ultra-low probabilities
    logger.info("\n🔍 ULTRA-LOW PROBABILITY ANALYSIS:")
    logger.info("=" * 80)
    
    for model_name in ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'svm']:
        if model_name in models['test_predictions']:
            probas = models['test_predictions'][model_name]
            threshold = models['thresholds'][model_name]
            
            logger.info(f"\n📊 {model_name.upper()} Ultra-Low Analysis:")
            logger.info(f"   Original threshold: {threshold:.8f}")
            
            # Sort probabilities to understand distribution
            sorted_probas = np.sort(probas)
            logger.info(f"   Lowest 10 probabilities: {sorted_probas[:10]}")
            logger.info(f"   Highest 10 probabilities: {sorted_probas[-10:]}")
            
            # Ultra-detailed percentile analysis
            ultra_low_percentiles = [0.01, 0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.5, 99.9, 99.99]
            logger.info(f"   Ultra-detailed percentiles:")
            for p in ultra_low_percentiles:
                val = np.percentile(probas, p)
                logger.info(f"     {p:6.2f}th: {val:.10f}")
            
            # Count exact matches at various ultra-low levels
            exact_zeros = np.sum(probas == 0.0)
            logger.info(f"   Exact zeros: {exact_zeros}/{len(probas)}")
            
            # Test ultra-aggressive thresholds
            ultra_thresholds = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
            logger.info(f"   Ultra-aggressive threshold testing:")
            for thresh in ultra_thresholds:
                pos_preds = np.sum(probas >= thresh)
                logger.info(f"     {thresh:.0e}: {pos_preds:3d} positive predictions ({pos_preds/len(probas)*100:5.1f}%)")
            
            # Find the optimal ultra-low threshold that would give at least 1 positive prediction
            if probas.max() > 0:
                min_for_one_positive = sorted_probas[-1]  # Highest probability
                min_for_two_positive = sorted_probas[-2] if len(sorted_probas) > 1 else sorted_probas[-1]
                min_for_five_positive = sorted_probas[-5] if len(sorted_probas) > 4 else sorted_probas[-1]
                
                logger.info(f"   Thresholds for minimal positive predictions:")
                logger.info(f"     For 1+ positive:  {min_for_one_positive:.10f}")
                logger.info(f"     For 2+ positive:  {min_for_two_positive:.10f}")
                logger.info(f"     For 5+ positive:  {min_for_five_positive:.10f}")
                
                # Test F1 scores with these ultra-low thresholds
                from sklearn.metrics import f1_score
                
                for desc, thresh in [("1+ positive", min_for_one_positive), 
                                   ("2+ positive", min_for_two_positive),
                                   ("5+ positive", min_for_five_positive)]:
                    y_pred = (probas >= thresh).astype(int)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    pos_pred_count = np.sum(y_pred == 1)
                    logger.info(f"     {desc:12s}: F1={f1:.3f}, pos_preds={pos_pred_count}")
    
    logger.info("\n✅ Ultra-low probability analysis complete!")

if __name__ == "__main__":
    debug_ultra_low_probabilities()
