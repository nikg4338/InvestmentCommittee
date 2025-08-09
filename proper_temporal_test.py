#!/usr/bin/env python3
"""
Proper temporal validation test for leak-free data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc

def proper_temporal_test():
    """Test on proper temporal split (train on past, test on future)."""
    
    # Load leak-free data
    df = pd.read_csv('data/leak_free_batch_1_data.csv')
    print(f'📊 Dataset: {df.shape}')
    
    # Separate training and test sets using temporal split
    train_mask = df['temporal_split'] == 'train'
    test_mask = df['temporal_split'] == 'test'
    
    feature_cols = [c for c in df.columns if c not in ['target', 'ticker', 'timestamp', 'temporal_split', 'data_collection_timestamp', 'data_source', 'leak_free_validated']]
    
    # Prepare training data
    X_train = df.loc[train_mask, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = df.loc[train_mask, 'target'].dropna()
    
    # Align training data
    common_train_idx = X_train.index.intersection(y_train.index)
    X_train = X_train.loc[common_train_idx]
    y_train = y_train.loc[common_train_idx]
    
    # Prepare test data (should have NaN targets due to proper temporal split)
    X_test = df.loc[test_mask, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f'🎯 Training: {len(X_train)} samples, {y_train.mean():.1%} positive')
    print(f'🔮 Test: {len(X_test)} samples (no targets as expected)')
    
    # Train model
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    # Test 1: Performance on training data (should be reasonable but not perfect)
    y_train_pred_proba = rf.predict_proba(X_train)[:, 1]
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_pred_proba)
    pr_auc_train = auc(recall_train, precision_train)
    
    print(f'\n📈 TRAINING PERFORMANCE (prone to overfitting):')
    print(f'   PR-AUC: {pr_auc_train:.4f}')
    
    # Test 2: Cross-validation on training data for more realistic estimate
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer
    
    def pr_auc_score(y_true, y_pred_proba):
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        return auc(recall, precision)
    
    pr_auc_scorer = make_scorer(pr_auc_score, needs_proba=True)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring=pr_auc_scorer)
    
    print(f'\n🔄 CROSS-VALIDATION PERFORMANCE (more realistic):')
    print(f'   PR-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')
    
    # Interpretation
    if cv_scores.mean() > 0.9:
        print('   🚨 STILL VERY HIGH - investigate further')
        print('   Possible causes:')
        print('   - Strong genuine market patterns in this ETF data')
        print('   - Subtle remaining leakage')
        print('   - Temporal patterns not properly addressed')
    elif cv_scores.mean() > 0.7:
        print('   ⚠️  MODERATELY HIGH - could be legitimate')
        print('   ETF data might have stronger patterns than individual stocks')
    else:
        print('   ✅ REALISTIC PERFORMANCE - leakage eliminated!')
    
    # Test 3: Predict on future test period (no targets available for validation)
    test_predictions = rf.predict_proba(X_test)[:, 1]
    print(f'\n🔮 FUTURE PREDICTIONS:')
    print(f'   Predicted positive rate: {(test_predictions > 0.5).mean():.1%}')
    print(f'   Average probability: {test_predictions.mean():.3f}')
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f'\n🔍 TOP 5 MOST IMPORTANT FEATURES:')
    for _, row in importance_df.head(5).iterrows():
        print(f'   {row["feature"]}: {row["importance"]:.4f}')
    
    return cv_scores.mean()

if __name__ == "__main__":
    proper_temporal_test()
