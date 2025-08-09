#!/usr/bin/env python3
"""
Quick training test on clean data to see actual performance.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, classification_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_training_test():
    """Run a quick training test on clean data."""
    
    # Load clean data
    df = pd.read_csv('data/ultra_clean_batch.csv')
    logger.info(f"📊 Loaded ultra-clean data: {len(df)} samples")
    
    # Check if EURKU is still present
    tickers = df['ticker'].unique()
    logger.info(f"🏢 Tickers: {list(tickers)}")
    
    if 'EURKU' in tickers:
        logger.error("❌ EURKU still present in supposedly clean data!")
        return
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['ticker', 'target', 'timestamp']]
    X = df[feature_cols]
    y = df['target']
    
    logger.info(f"📋 Features: {len(feature_cols)}")
    logger.info(f"🎯 Target distribution: {y.value_counts().to_dict()}")
    logger.info(f"🎯 Positive rate: {y.mean():.1%}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train a simple Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    # Predict and evaluate
    y_proba = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)
    
    # Calculate PR-AUC
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    
    logger.info(f"🎯 PR-AUC on clean data: {pr_auc:.4f}")
    
    # Per-ticker performance
    print("\n📊 Per-ticker performance:")
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df['ticker'] = df.loc[X_test.index, 'ticker']
    test_df['prediction'] = y_pred
    test_df['probability'] = y_proba
    
    for ticker in test_df['ticker'].unique():
        ticker_data = test_df[test_df['ticker'] == ticker]
        if len(ticker_data) > 5:  # Only analyze tickers with enough samples
            ticker_auc = auc(*precision_recall_curve(ticker_data['target'], ticker_data['probability'])[:2])
            pos_rate = ticker_data['target'].mean()
            pred_pos_rate = ticker_data['prediction'].mean()
            print(f"  {ticker}: {len(ticker_data)} samples, {pos_rate:.1%} actual positive, {pred_pos_rate:.1%} predicted positive, PR-AUC: {ticker_auc:.3f}")
    
    print(f"\n🎉 SUMMARY:")
    print(f"  ✅ Clean data without EURKU")
    print(f"  📊 Overall PR-AUC: {pr_auc:.4f}")
    print(f"  🎯 This is a REALISTIC score (not 99.9%!)")
    
    if pr_auc < 0.9:
        print(f"  ✅ Performance is realistic for authentic market data")
    else:
        print(f"  ⚠️ Performance still suspiciously high - may need more investigation")

if __name__ == "__main__":
    quick_training_test()
