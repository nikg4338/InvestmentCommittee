#!/usr/bin/env python3
"""
Data Leakage Fix and Clean Model Training
Remove leaked features and retrain models properly.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

def identify_and_remove_leakage():
    """Identify and remove data leakage from training data."""
    
    print("🔧 FIXING DATA LEAKAGE ISSUES")
    print("="*50)
    
    # Load original data
    df = pd.read_csv('alpaca_training_data.csv')
    print(f"Original data shape: {df.shape}")
    
    # Identify leaked features
    leaked_features = []
    feature_cols = [col for col in df.columns if col not in ['target', 'ticker', 'timestamp']]
    
    # Check correlations
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()['target'].abs()
    
    print(f"\n🔍 IDENTIFYING LEAKED FEATURES:")
    for feature, corr in correlations.sort_values(ascending=False).items():
        if feature != 'target' and corr > 0.95:
            leaked_features.append(feature)
            print(f"  🚨 LEAKED: {feature} (correlation: {corr:.4f})")
        elif feature != 'target' and corr > 0.8:
            print(f"  ⚠️ SUSPICIOUS: {feature} (correlation: {corr:.4f})")
    
    # Remove leaked features
    clean_features = [col for col in feature_cols if col not in leaked_features]
    
    # Create clean dataset
    clean_df = df[['ticker', 'timestamp', 'target'] + clean_features].copy()
    
    print(f"\n✅ CLEANING RESULTS:")
    print(f"  Original features: {len(feature_cols)}")
    print(f"  Leaked features removed: {len(leaked_features)}")
    print(f"  Clean features remaining: {len(clean_features)}")
    print(f"  Data integrity: {len(clean_df)} samples preserved")
    
    # Save clean data
    clean_filename = 'alpaca_training_data_clean.csv'
    clean_df.to_csv(clean_filename, index=False)
    
    print(f"\n💾 SAVED CLEAN DATA: {clean_filename}")
    
    # Show sample of remaining features
    print(f"\n📊 SAMPLE CLEAN FEATURES:")
    for i, feature in enumerate(clean_features[:10]):
        corr = correlations.get(feature, 0)
        print(f"  {feature}: {corr:.4f} correlation")
    
    return clean_filename, len(clean_features)

def create_clean_training_script():
    """Create script for training on clean data."""
    
    script_content = '''#!/usr/bin/env python3
"""
Clean Model Training
Train models on data without leakage for realistic performance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def train_clean_models():
    """Train models on clean data."""
    
    print("🧹 TRAINING CLEAN MODELS")
    print("="*40)
    
    # Load clean data
    df = pd.read_csv('alpaca_training_data_clean.csv')
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['target', 'ticker', 'timestamp']]
    X = df[feature_cols]
    y = df['target']
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Split data properly (time-aware)
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Features: {len(feature_cols)}")
    
    models = {}
    results = {}
    
    # Train CatBoost
    print("\\n🚀 Training CatBoost...")
    try:
        from catboost import CatBoostClassifier
        
        catboost = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            verbose=False,
            random_seed=42
        )
        
        catboost.fit(X_train, y_train)
        
        # Evaluate
        y_pred = catboost.predict(X_test)
        y_pred_proba = catboost.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        models['catboost'] = catboost
        results['catboost'] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # Train Random Forest
    print("\\n🌲 Training Random Forest...")
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        models['random_forest'] = rf
        results['random_forest'] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # Train XGBoost
    print("\\n⚡ Training XGBoost...")
    try:
        import xgboost as xgb
        
        xgboost = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        )
        
        xgboost.fit(X_train, y_train)
        
        y_pred = xgboost.predict(X_test)
        y_pred_proba = xgboost.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        models['xgboost'] = xgboost
        results['xgboost'] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # Save models
    os.makedirs('models/clean', exist_ok=True)
    
    for model_name, model in models.items():
        model_path = f'models/clean/{model_name}_clean.pkl'
        joblib.dump(model, model_path)
        print(f"💾 Saved {model_name} to {model_path}")
    
    # Save results
    results['training_date'] = datetime.now().isoformat()
    results['data_shape'] = [len(X_train) + len(X_test), len(feature_cols)]
    
    with open('models/clean/clean_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\n✅ CLEAN TRAINING COMPLETE")
    print(f"Expected accuracy range: 65-80% (realistic for financial data)")
    
    return results

if __name__ == "__main__":
    train_clean_models()
'''

    with open('train_clean_models.py', 'w') as f:
        f.write(script_content)
    
    print(f"💾 SAVED CLEAN TRAINING SCRIPT: train_clean_models.py")

def main():
    """Main function to fix data leakage."""
    
    print("🚨 DATA LEAKAGE REMEDIATION")
    print("="*50)
    
    # Step 1: Remove leaked features
    clean_file, feature_count = identify_and_remove_leakage()
    
    # Step 2: Create clean training script
    create_clean_training_script()
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"  1. Review clean features in {clean_file}")
    print(f"  2. Run: python train_clean_models.py")
    print(f"  3. Expect realistic accuracy: 65-80%")
    print(f"  4. Use clean models for paper trading")
    
    print(f"\n✅ REMEDIATION SETUP COMPLETE")

if __name__ == "__main__":
    main()
