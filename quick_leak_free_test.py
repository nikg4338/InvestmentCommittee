#!/usr/bin/env python3
"""
Quick test of leak-free data performance
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve, auc

def test_leak_free_performance():
    # Load leak-free data
    df = pd.read_csv('data/leak_free_batch_1_data.csv')
    print(f"📊 Dataset: {df.shape}")
    
    # Use only training data (as test data has no targets)
    train_mask = df['temporal_split'] == 'train'
    train_data = df[train_mask].copy()
    
    print(f"🎯 Training data: {len(train_data)} samples")
    
    # Prepare features (exclude metadata columns)
    feature_cols = [c for c in train_data.columns 
                   if c not in ['target', 'ticker', 'timestamp', 'temporal_split', 
                               'data_collection_timestamp', 'data_source', 'leak_free_validated']]
    
    X = train_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = train_data['target']
    
    print(f"📈 Features: {len(feature_cols)}")
    print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
    print(f"📊 Positive rate: {y.mean():.1%}")
    
    # Quick cross-validation test (proper temporal validation)
    from sklearn.model_selection import TimeSeriesSplit
    
    # Use time series split for realistic evaluation
    tscv = TimeSeriesSplit(n_splits=3)
    
    rf = RandomForestClassifier(n_estimators=50, random_state=42, 
                               class_weight='balanced', max_depth=8)
    
    # Evaluate with cross-validation
    cv_scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        
        rf.fit(X_train_cv, y_train_cv)
        y_pred_proba = rf.predict_proba(X_val_cv)[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_val_cv, y_pred_proba)
        pr_auc = auc(recall, precision)
        cv_scores.append(pr_auc)
    
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    
    print(f"\n🔄 TIME SERIES CROSS-VALIDATION RESULTS:")
    print(f"   Mean PR-AUC: {mean_cv_score:.4f} (+/- {std_cv_score:.4f})")
    print(f"   Individual scores: {[f'{s:.4f}' for s in cv_scores]}")
    
    if mean_cv_score < 0.6:
        print("   ✅ REALISTIC PERFORMANCE - Data leakage eliminated!")
    elif mean_cv_score < 0.8:
        print("   ⚠️  MODERATE PERFORMANCE - Could be legitimate")
    else:
        print("   🚨 HIGH PERFORMANCE - Possible remaining leakage")
    
    # Test on future data (test set)
    test_mask = df['temporal_split'] == 'test'
    if test_mask.sum() > 0:
        X_test = df.loc[test_mask, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Fit on all training data
        rf.fit(X, y)
        
        # Predict on test (future) data
        test_proba = rf.predict_proba(X_test)[:, 1]
        test_predictions = (test_proba > 0.5).astype(int)
        
        print(f"\n🔮 FUTURE PREDICTIONS (Test Set):")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Predicted positive rate: {test_predictions.mean():.1%}")
        print(f"   Average probability: {test_proba.mean():.3f}")
        print(f"   Prob range: {test_proba.min():.3f} - {test_proba.max():.3f}")
        
        # This should be realistic (not perfect predictions)
        if test_proba.std() < 0.1:
            print("   🚨 LOW VARIATION - Possible overfitting")
        else:
            print("   ✅ GOOD VARIATION - Model seems realistic")

if __name__ == "__main__":
    test_leak_free_performance()
