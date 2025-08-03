#!/usr/bin/env python3
"""
Analyze the specific positive case in test set to understand why models miss it
"""

import pandas as pd
import numpy as np
from train_models import collect_alpaca_training_data, prepare_training_data
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_positive_case():
    """Analyze the specific positive case in the test set"""
    
    logger.info("🔍 Analyzing the specific positive case in test set...")
    
    # Load and prepare the same data
    df = collect_alpaca_training_data(batch_numbers=[1, 2], max_symbols_per_batch=10, use_cached=True)
    feature_columns = [col for col in df.columns if col not in ['target', 'ticker']]
    X_train, X_test, y_train, y_test = prepare_training_data(df, feature_columns, target_column='target')
    
    # Find the positive case in test set
    positive_indices = np.where(y_test == 1)[0]
    logger.info(f"Positive case indices in test set: {positive_indices}")
    
    if len(positive_indices) > 0:
        pos_idx = positive_indices[0]
        logger.info(f"Analyzing positive case at index {pos_idx}")
        
        # Get the features for this positive case
        positive_case = X_test.iloc[pos_idx]
        logger.info(f"Positive case features:")
        for feature, value in positive_case.items():
            logger.info(f"  {feature}: {value}")
        
        # Compare with some negative cases
        negative_indices = np.where(y_test == 0)[0][:5]  # First 5 negative cases
        logger.info(f"\nComparing with first 5 negative cases:")
        
        for i, neg_idx in enumerate(negative_indices):
            negative_case = X_test.iloc[neg_idx]
            logger.info(f"\nNegative case {i+1} (index {neg_idx}):")
            for feature, value in negative_case.items():
                pos_val = positive_case[feature]
                diff = abs(pos_val - value) if pd.notna(pos_val) and pd.notna(value) else None
                logger.info(f"  {feature}: {value} (pos: {pos_val}, diff: {diff})")
        
        # Look at statistics
        logger.info(f"\nDataset statistics:")
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Total positive: {sum(df['target'] == 1)}")
        logger.info(f"Total negative: {sum(df['target'] == 0)}")
        logger.info(f"Positive rate: {sum(df['target'] == 1) / len(df) * 100:.3f}%")
        
        # Check if there's a ticker associated with this case
        if 'ticker' in df.columns:
            # Find original index
            test_tickers = X_test.index
            original_positive = df.loc[df.index.isin(test_tickers) & (df['target'] == 1)]
            if len(original_positive) > 0:
                logger.info(f"\nPositive case details:")
                logger.info(f"Ticker: {original_positive['ticker'].iloc[0]}")
                # Show all features for this specific case
                for col in df.columns:
                    if col != 'ticker':
                        val = original_positive[col].iloc[0]
                        logger.info(f"  {col}: {val}")
    
    else:
        logger.info("No positive cases found in test set!")
    
    logger.info("\n🎯 Key insights:")
    logger.info("1. With only 1 positive case in test set, F1=0 means models are missing this specific case")
    logger.info("2. Need to examine if this positive case is an outlier or has learnable patterns")
    logger.info("3. Consider if the current features provide enough signal to distinguish this case")

if __name__ == "__main__":
    analyze_positive_case()
