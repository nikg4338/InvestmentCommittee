#!/usr/bin/env python3
"""
Test LightGBM Regressor Integration
==================================

This script tests the new LightGBM regressor with regression targets
and verifies proper integration with the training pipeline.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_lightgbm_regressor():
    """Test LightGBM regressor instantiation and basic functionality."""
    print("🧪 Testing LightGBM Regressor Integration...")
    
    try:
        # Test import
        from models.lightgbm_regressor import LightGBMRegressor
        print("✓ Successfully imported LightGBMRegressor")
        
        # Test instantiation
        model = LightGBMRegressor(name="TestRegressor")
        print("✓ Successfully instantiated LightGBMRegressor")
        
        # Test model registry
        from utils.stacking import MODEL_REGISTRY
        if 'lightgbm_regressor' in MODEL_REGISTRY:
            print("✓ LightGBMRegressor registered in MODEL_REGISTRY")
        else:
            print("❌ LightGBMRegressor NOT found in MODEL_REGISTRY")
            return False
            
        # Create dummy data for testing
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create regression targets (daily returns)
        y = pd.Series(np.random.normal(0, 0.02, n_samples))  # Daily returns with 2% std
        
        print(f"✓ Created test data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"✓ Target stats: mean={y.mean():.4f}, std={y.std():.4f}")
        
        # Test training
        try:
            model.train(X[:80], y[:80], X[80:], y[80:])
            print("✓ Successfully trained LightGBMRegressor")
            
            # Test prediction
            predictions = model.predict(X[80:])
            print(f"✓ Successfully made predictions: {len(predictions)} values")
            
            # Test binary prediction conversion
            reg_pred, binary_pred = model.predict(X[80:], return_probabilities=True)
            print(f"✓ Successfully converted to binary predictions")
            print(f"   Regression range: [{reg_pred.min():.4f}, {reg_pred.max():.4f}]")
            print(f"   Binary distribution: {binary_pred.sum()}/{len(binary_pred)} positive")
            print(f"   Optimal threshold: {model.optimal_threshold_:.4f}")
            
            # Test metrics
            metrics = model.get_metrics()
            print(f"✓ Model metrics: {list(metrics.keys())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Training/prediction failed: {str(e)}")
            return False
        
    except ImportError as e:
        print(f"❌ Import failed: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def test_data_collection_regression():
    """Test data collection with regression targets."""
    print("\n🧪 Testing Data Collection with Regression Targets...")
    
    try:
        from data_collection_alpaca import AlpacaDataCollector
        
        # Test instantiation (won't work without API keys but should import)
        collector = AlpacaDataCollector()
        print("✓ Successfully imported AlpacaDataCollector")
        
        # Create dummy price data for testing target creation
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': 100 + np.random.randn(50) * 2,
            'high': 102 + np.random.randn(50) * 2,
            'low': 98 + np.random.randn(50) * 2,
            'close': 100 + np.random.randn(50) * 2,
            'volume': 1000000 + np.random.randint(-100000, 100000, 50)
        })
        df = df.set_index('timestamp')
        
        print(f"✓ Created test price data: {len(df)} days")
        
        # Test regression target creation
        regression_target = collector._create_regression_target(df, 'TEST', target_horizon=3)
        print(f"✓ Created regression target: {len(regression_target)} values")
        
        valid_targets = regression_target.dropna()
        print(f"   Valid targets: {len(valid_targets)}/{len(regression_target)}")
        print(f"   Target range: [{valid_targets.min():.4f}, {valid_targets.max():.4f}]")
        print(f"   Target mean: {valid_targets.mean():.4f}, std: {valid_targets.std():.4f}")
        
        # Test create_target_variable with regression
        target_series = collector.create_target_variable(df, 'TEST', use_regression=True, target_horizon=3, create_all_horizons=False)
        print(f"✓ Created single regression target via create_target_variable")
        
        # Test create_target_variable with multiple horizons
        target_df = collector.create_target_variable(df, 'TEST', use_regression=True, create_all_horizons=True)
        print(f"✓ Created multi-horizon targets: {target_df.shape[1]} horizons")
        print(f"   Horizons: {list(target_df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data collection test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Testing LightGBM Regressor Integration\n")
    
    test1_success = test_lightgbm_regressor()
    test2_success = test_data_collection_regression()
    
    print(f"\n📊 Test Results:")
    print(f"   LightGBM Regressor: {'✓ PASS' if test1_success else '❌ FAIL'}")
    print(f"   Data Collection: {'✓ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print(f"\n🎉 All tests passed! LightGBM regressor integration is ready.")
        print(f"\n💡 Usage examples:")
        print(f"   # Train with regression targets")
        print(f"   python train_models.py --models lightgbm_regressor --target-column target")
        print(f"   ")
        print(f"   # Include regressor in ensemble")
        print(f"   python train_models.py --models xgboost lightgbm lightgbm_regressor catboost")
    else:
        print(f"\n❌ Some tests failed. Check the errors above.")
        sys.exit(1)
