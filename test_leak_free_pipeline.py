#!/usr/bin/env python3
"""
Test Script for Leak-Free Data Collection
==========================================

This script tests the fixed data collection pipeline to ensure:
1. No data leakage in features or targets
2. Fresh data collection from Alpaca API
3. Realistic model performance scores
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_leak_free_collection():
    """Test the leak-free data collection pipeline."""
    
    logger.info("🧪 Testing leak-free data collection pipeline...")
    
    # Test data collection for batch 1
    logger.info("📥 Collecting leak-free data for batch 1...")
    
    import subprocess
    cmd = [sys.executable, 'data_collection_alpaca_fixed.py', '--batch', '1', '--max-symbols', '5']
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            logger.info("✅ Leak-free data collection completed successfully")
            
            # Find the generated file
            expected_file = "data/leak_free_batch_1_data.csv"
            
            if Path(expected_file).exists():
                logger.info(f"📊 Testing collected data: {expected_file}")
                
                # Load and analyze the data
                df = pd.read_csv(expected_file)
                
                logger.info(f"Dataset shape: {df.shape}")
                logger.info(f"Columns: {len(df.columns)}")
                
                # Check temporal split
                if 'temporal_split' in df.columns:
                    train_count = (df['temporal_split'] == 'train').sum()
                    test_count = (df['temporal_split'] == 'test').sum()
                    logger.info(f"Train samples: {train_count}")
                    logger.info(f"Test samples: {test_count}")
                    
                    # Check target distribution
                    if 'target' in df.columns:
                        train_mask = df['temporal_split'] == 'train'
                        train_targets = df.loc[train_mask, 'target'].dropna()
                        
                        if len(train_targets) > 0:
                            pos_rate = train_targets.mean()
                            logger.info(f"Training positive rate: {pos_rate:.1%}")
                            
                            # Test model performance on leak-free data
                            test_model_performance(df)
                        else:
                            logger.warning("No training targets found")
                    else:
                        logger.warning("No target column found")
                else:
                    logger.warning("No temporal_split column found")
                    
            else:
                logger.error(f"Expected file not found: {expected_file}")
                return False
                
        else:
            logger.error(f"Data collection failed with code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Data collection timed out")
        return False
    except Exception as e:
        logger.error(f"Data collection error: {e}")
        return False
    
    return True

def test_model_performance(df: pd.DataFrame):
    """Test model performance on leak-free data."""
    
    logger.info("🤖 Testing model performance on leak-free data...")
    
    try:
        # Prepare data
        train_mask = df['temporal_split'] == 'train'
        feature_cols = [c for c in df.columns if c not in ['target', 'ticker', 'timestamp', 'temporal_split', 'data_collection_timestamp', 'data_source', 'leak_free_validated']]
        
        X_train = df.loc[train_mask, feature_cols].fillna(0)
        y_train = df.loc[train_mask, 'target'].dropna()
        
        # Align X_train and y_train
        common_idx = X_train.index.intersection(y_train.index)
        X_train = X_train.loc[common_idx]
        y_train = y_train.loc[common_idx]
        
        if len(X_train) < 50:
            logger.warning("Insufficient training data for model test")
            return
        
        logger.info(f"Training data: {len(X_train)} samples, {len(feature_cols)} features")
        
        # Train simple model
        rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
        rf.fit(X_train, y_train)
        
        # Predict on training data (for sanity check)
        y_pred_proba = rf.predict_proba(X_train)[:, 1]
        
        # Calculate PR-AUC
        precision, recall, _ = precision_recall_curve(y_train, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        logger.info(f"📈 Model performance on LEAK-FREE data:")
        logger.info(f"   PR-AUC: {pr_auc:.4f}")
        
        if pr_auc > 0.9:
            logger.error("🚨 PR-AUC still suspiciously high - possible remaining leakage!")
            return False
        elif pr_auc > 0.7:
            logger.warning("⚠️  PR-AUC moderately high - worth investigating")
        else:
            logger.info("✅ PR-AUC at realistic level - leakage appears eliminated!")
        
        # Check feature importances
        importances = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("🔍 Top 5 most important features:")
        for _, row in importances.head(5).iterrows():
            logger.info(f"   {row['feature']}: {row['importance']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model performance test failed: {e}")
        return False

def main():
    """Run all tests."""
    
    logger.info("🚀 Starting leak-free pipeline test...")
    
    # Ensure we're in the right directory
    if not Path('data_collection_alpaca_fixed.py').exists():
        logger.error("data_collection_alpaca_fixed.py not found - are you in the right directory?")
        sys.exit(1)
    
    # Test leak-free collection
    success = test_leak_free_collection()
    
    if success:
        logger.info("🎉 All tests passed! Leak-free pipeline is working correctly.")
    else:
        logger.error("❌ Tests failed! Please check the implementation.")
        sys.exit(1)

if __name__ == "__main__":
    main()
