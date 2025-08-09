#!/usr/bin/env python3
"""
Direct test of leak-free data training to verify realistic performance.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, classification_report
import warnings
warnings.filterwarnings('ignore')

def test_leak_free_training():
    """Test training directly on leak-free data."""
    
    print("🔍 TESTING LEAK-FREE BATCH 1 DATA")
    print("=" * 50)
    
    # Load the leak-free data
    df = pd.read_csv('data/leak_free_batch_1_data.csv')
    
    print(f"📊 Data shape: {df.shape}")
    print(f"✅ Leak-free validated: {df['leak_free_validated'].all()}")
    print(f"🎯 Target distribution: {df['target'].value_counts().to_dict()}")
    print(f"📈 Positive rate: {df['target'].mean():.3f}")
    
    # Prepare features (exclude metadata columns)
    exclude_cols = ['target', 'temporal_split', 'ticker', 'timestamp', 
                   'data_collection_timestamp', 'data_source', 'leak_free_validated']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].fillna(0)
    
    # Handle infinity values
    X = X.replace([np.inf, -np.inf], 0)
    
    y = df['target']
    
    print(f"🔧 Features: {len(feature_cols)}")
    print(f"   {feature_cols[:5]}... (showing first 5)")
    
    # Use only training data (test set has NaN targets for future predictions)
    train_mask = df['temporal_split'] == 'train'
    X_train_full = X[train_mask]
    y_train_full = y[train_mask]
    
    # Create our own train/test split from the training data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    print(f"📋 Train: {len(X_train)} samples ({y_train.mean():.3f} positive)")
    print(f"📋 Test:  {len(X_test)} samples ({y_test.mean():.3f} positive)")
    print(f"📋 Future: {len(df[df['temporal_split']=='test'])} samples (targets unknown)")
    
    # Train model
    print("\n🤖 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    # Predictions
    y_proba = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)
    
    # Calculate PR-AUC
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    
    print(f"\n📊 RESULTS:")
    print(f"   PR-AUC: {pr_auc:.4f}")
    print(f"   Test positive rate: {y_test.mean():.3f}")
    print(f"   Predicted positive rate: {y_pred.mean():.3f}")
    print(f"   Probability range: {y_proba.min():.3f} - {y_proba.max():.3f}")
    
    print(f"\n📈 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Verify this is leak-free by checking correlation
    feature_target_corr = X_test.corrwith(y_test).abs().max()
    print(f"\n🛡️  Maximum feature-target correlation: {feature_target_corr:.4f}")
    
    if pr_auc > 0.8:
        print("\n🚨 WARNING: PR-AUC > 0.8 suggests possible data leakage!")
    else:
        print(f"\n✅ SUCCESS: Realistic PR-AUC of {pr_auc:.4f} indicates leak-free training!")
    
    return pr_auc

if __name__ == "__main__":
    pr_auc = test_leak_free_training()
