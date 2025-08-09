#!/usr/bin/env python3
"""
Investigate potential remaining data leakage in the leak-free dataset.
"""

import pandas as pd
import numpy as np

def investigate_leakage():
    # Load data
    df = pd.read_csv('data/leak_free_batch_1_data.csv')
    train_mask = df['temporal_split'] == 'train'
    
    feature_cols = [c for c in df.columns if c not in ['target', 'ticker', 'timestamp', 'temporal_split', 'data_collection_timestamp', 'data_source', 'leak_free_validated']]
    
    train_data = df[train_mask].copy()
    correlations = []
    
    for col in feature_cols:
        if train_data[col].nunique() > 1:
            corr = train_data[col].corr(train_data['target'])
            if not np.isnan(corr):
                correlations.append((col, abs(corr)))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    print('🔍 Top 10 feature-target correlations:')
    for col, corr in correlations[:10]:
        print(f'   {col}: {corr:.4f}')
    
    # Check for perfect predictors
    perfect_features = [col for col, corr in correlations if corr > 0.95]
    if perfect_features:
        print(f'🚨 POTENTIAL LEAKY FEATURES (>95% correlation): {perfect_features}')
    else:
        print('✅ No obvious perfect predictors found')
    
    # Check dataset structure
    print(f'\n📊 Dataset structure:')
    print(f'   Total samples: {len(train_data)}')
    print(f'   Positive: {train_data["target"].sum():.0f} ({train_data["target"].mean():.1%})')
    print(f'   Symbols: {train_data["ticker"].nunique()}')
    print(f'   Date range: {train_data["timestamp"].min()} to {train_data["timestamp"].max()}')
    
    # Check if we're only getting one symbol (potential issue)
    symbol_counts = train_data['ticker'].value_counts()
    print(f'\n📈 Symbols in training data:')
    print(symbol_counts)
    
    # If only one symbol, that might explain the high performance
    if len(symbol_counts) == 1:
        print('\n🚨 WARNING: Only one symbol in dataset!')
        print('This could cause overfitting and unrealistic performance.')
        
    return correlations, symbol_counts

if __name__ == "__main__":
    investigate_leakage()
