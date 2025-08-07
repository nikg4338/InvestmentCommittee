#!/usr/bin/env python3
"""
Debug Class Imbalance Issue
===========================

This script diagnoses why you're getting 60/40 class distribution
instead of the expected extreme imbalance (~1-5% positive) for financial data.

It will:
1. Check your data source and target creation
2. Identify where the imbalance is being artificially balanced
3. Fix the target variable creation to ensure proper extreme imbalance
4. Test the corrected data with the enhanced meta-model training
"""

import pandas as pd
import numpy as np
import logging
import glob
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def diagnose_data_source():
    """Check all available data sources and their target distributions"""
    logger.info("🔍 PHASE 1: Diagnosing Data Sources")
    
    # Check batch data files
    data_files = glob.glob('data/batch_*_data.csv')
    if data_files:
        logger.info(f"Found {len(data_files)} batch data files")
        for file_path in sorted(data_files)[-3:]:  # Check last 3 files
            logger.info(f"\n📊 Analyzing {file_path}...")
            
            try:
                df = pd.read_csv(file_path)
                logger.info(f"  Shape: {df.shape}")
                
                # Check target column
                if 'target' in df.columns:
                    target_dist = df['target'].value_counts().sort_index()
                    positive_rate = df['target'].sum() / len(df) * 100
                    logger.info(f"  Target distribution: {dict(target_dist)}")
                    logger.info(f"  Positive rate: {positive_rate:.2f}%")
                    
                    # Check if this is the problem source
                    if positive_rate > 10:
                        logger.warning(f"  ❌ PROBLEM FOUND: {positive_rate:.1f}% positive rate is too high!")
                        logger.warning(f"      Expected: <5% for financial data")
                        
                        # Check target values
                        unique_vals = sorted(df['target'].unique())
                        logger.info(f"  Unique target values: {unique_vals}")
                        
                        # Sample the data to see what's happening
                        logger.info(f"  Sample rows with target=1:")
                        positive_samples = df[df['target'] == 1].head(3)
                        for idx, row in positive_samples.iterrows():
                            return_cols = [col for col in df.columns if 'return' in col.lower()]
                            if return_cols:
                                logger.info(f"    Row {idx}: {dict(row[return_cols])}")
                else:
                    logger.warning(f"  ❌ No 'target' column found in {file_path}")
                    target_cols = [col for col in df.columns if 'target' in col.lower() or 'return' in col.lower()]
                    logger.info(f"  Available target-like columns: {target_cols}")
                    
            except Exception as e:
                logger.error(f"  Error reading {file_path}: {e}")
    else:
        logger.warning("❌ No batch data files found in data/ directory")
    
    # Check filtered batch files
    logger.info(f"\n🔍 Checking filtered batch files...")
    if os.path.exists('filtered_iex_batches.json'):
        import json
        with open('filtered_iex_batches.json', 'r') as f:
            filtered_data = json.load(f)
        logger.info(f"  Filtered batches: {len(filtered_data)} batches")
        
        # Sample a batch to see the data structure
        if filtered_data:
            sample_batch = list(filtered_data.keys())[0]
            sample_data = filtered_data[sample_batch]
            logger.info(f"  Sample batch {sample_batch}: {len(sample_data)} symbols")
            if sample_data:
                if isinstance(sample_data, dict):
                    sample_symbol = list(sample_data.keys())[0]
                    sample_info = sample_data[sample_symbol]
                    if isinstance(sample_info, dict):
                        logger.info(f"  Sample symbol data keys: {list(sample_info.keys())}")
                    else:
                        logger.info(f"  Sample symbol data type: {type(sample_info)}")
                else:
                    logger.info(f"  Sample data type: {type(sample_data)}, length: {len(sample_data)}")
    
    return data_files

def check_target_creation_logic():
    """Check how targets are being created in data collection"""
    logger.info("\n🔍 PHASE 2: Checking Target Creation Logic")
    
    # Check data collection scripts
    scripts_to_check = [
        'data_collection_alpaca.py',
        'train_all_batches.py',
        'main.py'
    ]
    
    for script in scripts_to_check:
        if os.path.exists(script):
            logger.info(f"\n📝 Checking {script} for target creation...")
            try:
                with open(script, 'r') as f:
                    content = f.read()
                
                # Look for target creation patterns
                target_patterns = [
                    'target.*=',
                    'return.*>',
                    'percentile',
                    'top_.*percent',
                    'quantile',
                    'threshold.*return'
                ]
                
                for pattern in target_patterns:
                    import re
                    matches = re.findall(f'.*{pattern}.*', content, re.IGNORECASE)
                    if matches:
                        logger.info(f"  Found target creation pattern '{pattern}':")
                        for match in matches[:3]:  # Show first 3 matches
                            logger.info(f"    {match.strip()}")
                            
            except Exception as e:
                logger.warning(f"  Error reading {script}: {e}")

def create_proper_extreme_imbalance_target(df, return_column='1d_return', percentile_threshold=95):
    """
    Create a proper extreme imbalance target using top percentile approach
    """
    logger.info(f"\n🎯 PHASE 3: Creating Proper Extreme Imbalance Target")
    
    if return_column not in df.columns:
        # Find available return columns
        return_cols = [col for col in df.columns if 'return' in col.lower()]
        if return_cols:
            return_column = return_cols[0]
            logger.info(f"  Using available return column: {return_column}")
        else:
            logger.error(f"  ❌ No return columns found in data!")
            return df
    
    # Remove any NaN values from returns
    valid_mask = ~df[return_column].isnull()
    valid_returns = df.loc[valid_mask, return_column]
    
    logger.info(f"  Valid returns: {len(valid_returns)}/{len(df)} samples")
    logger.info(f"  Return stats: mean={valid_returns.mean():.4f}, std={valid_returns.std():.4f}")
    logger.info(f"  Return range: [{valid_returns.min():.4f}, {valid_returns.max():.4f}]")
    
    # Calculate percentile threshold for extreme imbalance
    threshold = np.percentile(valid_returns, percentile_threshold)
    logger.info(f"  {percentile_threshold}th percentile threshold: {threshold:.4f}")
    
    # Create binary target
    df['target'] = 0
    df.loc[valid_mask, 'target'] = (df.loc[valid_mask, return_column] > threshold).astype(int)
    
    # Check the resulting distribution
    target_dist = df['target'].value_counts().sort_index()
    positive_rate = df['target'].sum() / len(df) * 100
    
    logger.info(f"  ✅ NEW Target distribution: {dict(target_dist)}")
    logger.info(f"  ✅ NEW Positive rate: {positive_rate:.2f}%")
    
    if positive_rate > 10:
        logger.warning(f"  ⚠️ Still high positive rate! Trying higher percentile...")
        # Try 98th percentile
        threshold_98 = np.percentile(valid_returns, 98)
        df['target'] = 0
        df.loc[valid_mask, 'target'] = (df.loc[valid_mask, return_column] > threshold_98).astype(int)
        
        target_dist_98 = df['target'].value_counts().sort_index()
        positive_rate_98 = df['target'].sum() / len(df) * 100
        
        logger.info(f"  98th percentile threshold: {threshold_98:.4f}")
        logger.info(f"  ✅ FINAL Target distribution: {dict(target_dist_98)}")
        logger.info(f"  ✅ FINAL Positive rate: {positive_rate_98:.2f}%")
    
    return df

def test_enhanced_meta_model_with_extreme_imbalance():
    """Test the enhanced meta-model training with proper extreme imbalance"""
    logger.info(f"\n🧠 PHASE 4: Testing Enhanced Meta-Model with Extreme Imbalance")
    
    # Load or create test data
    data_files = glob.glob('data/batch_*_data.csv')
    if not data_files:
        logger.warning("  No batch data found, creating synthetic test data...")
        # Create synthetic financial data with extreme imbalance
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        feature_cols = [f'feature_{i}' for i in range(n_features)]
        
        # Generate realistic returns (mostly small, few large)
        base_returns = np.random.normal(0.001, 0.02, n_samples)  # 0.1% mean, 2% std
        # Add some extreme positive returns (top 2%)
        extreme_indices = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
        base_returns[extreme_indices] += np.random.exponential(0.05, len(extreme_indices))
        
        df = pd.DataFrame(X, columns=feature_cols)
        df['1d_return'] = base_returns
        
        logger.info(f"  Created synthetic data: {df.shape}")
    else:
        # Use real data
        latest_file = max(data_files)
        logger.info(f"  Loading real data from {latest_file}...")
        df = pd.read_csv(latest_file)
    
    # Fix the target variable
    df = create_proper_extreme_imbalance_target(df, percentile_threshold=98)  # Top 2%
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['target', '1d_return', '3d_return', '5d_return', '10d_return']]
    feature_cols = feature_cols[:20]  # Limit to 20 features for testing
    
    if len(feature_cols) == 0:
        logger.error("  ❌ No feature columns found!")
        return
    
    logger.info(f"  Using {len(feature_cols)} features")
    
    # Prepare data for training
    from train_models import prepare_training_data
    from config.training_config import get_extreme_imbalance_config
    
    config = get_extreme_imbalance_config()
    
    try:
        X_train, X_test, y_train, y_test = prepare_training_data(
            df, feature_cols, 'target', config
        )
        
        logger.info(f"  Train data: {X_train.shape}, positive rate: {y_train.sum()/len(y_train)*100:.2f}%")
        logger.info(f"  Test data: {X_test.shape}, positive rate: {y_test.sum()/len(y_test)*100:.2f}%")
        
        # Now test the enhanced meta-model training
        from utils.enhanced_meta_models import train_smote_enhanced_meta_model
        
        # Create simple base model predictions for meta-training
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_predict
        
        logger.info("  🎯 Creating base model predictions for meta-training...")
        
        # Train a simple base model to get meta-features
        base_model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
        meta_features_train = cross_val_predict(base_model, X_train, y_train, cv=3, method='predict_proba')[:, 1:]
        
        # Train base model on full train set for test predictions
        base_model.fit(X_train, y_train)
        meta_features_test = base_model.predict_proba(X_test)[:, 1:]
        
        logger.info(f"  Meta-features shape: train {meta_features_train.shape}, test {meta_features_test.shape}")
        
        # Test SMOTE-enhanced meta-model
        logger.info("  🧠 Testing SMOTE-enhanced meta-model...")
        positive_rate = y_train.sum() / len(y_train)
        logger.info(f"  Training positive rate: {positive_rate*100:.2f}%")
        
        if positive_rate < 0.02:
            logger.info("  ✅ Triggering SMOTE-enhanced meta-model (positive rate < 2%)")
            strategy = 'smote_enhanced'
        elif positive_rate < 0.05:
            logger.info("  ✅ Would trigger focal-loss meta-model (positive rate < 5%)")
            strategy = 'focal_loss'
        else:
            logger.info("  ✅ Would trigger optimal-threshold meta-model (positive rate >= 5%)")
            strategy = 'optimal_threshold'
        
        # Test the SMOTE-enhanced meta-model regardless
        meta_model, optimal_threshold = train_smote_enhanced_meta_model(
            meta_features_train, y_train, 
            meta_learner_type='logistic',  # LogisticRegression with class_weight='balanced'
            smote_ratio=0.5
        )
        
        # Get test predictions
        if hasattr(meta_model, 'predict_proba'):
            meta_test_proba = meta_model.predict_proba(meta_features_test)[:, 1]
        else:
            meta_test_proba = meta_model.predict(meta_features_test)
        
        # Evaluate meta-model
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
        
        meta_predictions = (meta_test_proba >= optimal_threshold).astype(int)
        
        f1 = f1_score(y_test, meta_predictions)
        precision = precision_score(y_test, meta_predictions, zero_division=0)
        recall = recall_score(y_test, meta_predictions, zero_division=0)
        roc_auc = roc_auc_score(y_test, meta_test_proba)
        
        logger.info(f"  ✅ Meta-model Results:")
        logger.info(f"    Strategy: {strategy}")
        logger.info(f"    Optimal threshold: {optimal_threshold:.4f}")
        logger.info(f"    F1 Score: {f1:.3f}")
        logger.info(f"    Precision: {precision:.3f}")
        logger.info(f"    Recall: {recall:.3f}")
        logger.info(f"    ROC-AUC: {roc_auc:.3f}")
        logger.info(f"    Predicted positives: {meta_predictions.sum()}/{len(y_test)} ({meta_predictions.sum()/len(y_test)*100:.1f}%)")
        
        # Check if meta-model is using class weights
        if hasattr(meta_model, 'class_weight'):
            logger.info(f"    Class weights: {meta_model.class_weight}")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Error in meta-model testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_full_diagnostic():
    """Run complete diagnostic and fix sequence"""
    logger.info("="*60)
    logger.info("🚀 STARTING CLASS IMBALANCE DIAGNOSTIC")
    logger.info("="*60)
    
    # Phase 1: Diagnose data sources
    data_files = diagnose_data_source()
    
    # Phase 2: Check target creation logic
    check_target_creation_logic()
    
    # Phase 3 & 4: Test with proper extreme imbalance
    success = test_enhanced_meta_model_with_extreme_imbalance()
    
    logger.info("\n" + "="*60)
    if success:
        logger.info("✅ DIAGNOSTIC COMPLETE - Enhanced meta-model working with extreme imbalance!")
        logger.info("💡 RECOMMENDATION: Update your data collection to use top 2% percentile targets")
        logger.info("🎯 Expected positive rate: 1-3% for proper extreme imbalance")
    else:
        logger.info("❌ DIAGNOSTIC FAILED - Need to investigate further")
    logger.info("="*60)

if __name__ == "__main__":
    run_full_diagnostic()
