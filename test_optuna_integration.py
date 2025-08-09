#!/usr/bin/env python3
"""
Test Optuna Integration After XGBoost Fixes
===========================================

This script specifically tests the Optuna optimization functions
to ensure they work correctly with XGBoost models.
"""

import logging
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise

def test_optuna_integration():
    """Test that Optuna optimization works with XGBoost models."""
    print("=== Testing Optuna Integration with XGBoost ===")
    
    from models.xgboost_model import XGBoostModel
    from utils.pipeline_improvements import tune_with_optuna
    from utils.enhanced_meta_models import optuna_optimize_base_model_for_f1
    from utils.stacking import get_optuna_param_space
    
    # Create test data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 8), columns=[f'feature_{i}' for i in range(8)])
    y = pd.Series(np.random.choice([0, 1], size=100, p=[0.7, 0.3]))
    
    print(f"Test data: {X.shape}, classes: {y.value_counts().to_dict()}")
    
    # Test 1: tune_with_optuna
    print("\n1. Testing tune_with_optuna...")
    try:
        param_space = get_optuna_param_space('xgboost')
        result1 = tune_with_optuna(XGBoostModel, X, y, param_space, n_trials=3)
        print(f"✅ tune_with_optuna succeeded: {len(result1)} parameters")
    except Exception as e:
        print(f"❌ tune_with_optuna failed: {str(e)}")
        return False
    
    # Test 2: optuna_optimize_base_model_for_f1
    print("\n2. Testing optuna_optimize_base_model_for_f1...")
    try:
        result2 = optuna_optimize_base_model_for_f1(XGBoostModel, X, y, n_trials=3)
        print(f"✅ optuna_optimize_base_model_for_f1 succeeded: {len(result2)} parameters")
    except Exception as e:
        print(f"❌ optuna_optimize_base_model_for_f1 failed: {str(e)}")
        return False
    
    # Test 3: Create models with optimized parameters
    print("\n3. Testing model creation with optimized parameters...")
    try:
        model1 = XGBoostModel(model_params=result1)
        model1.train(X, y)
        pred1 = model1.predict_proba(X)
        print(f"✅ Model 1 training successful: predictions {pred1.shape}")
        
        model2 = XGBoostModel(model_params=result2)
        model2.train(X, y)
        pred2 = model2.predict_proba(X)
        print(f"✅ Model 2 training successful: predictions {pred2.shape}")
    except Exception as e:
        print(f"❌ Model training failed: {str(e)}")
        return False
    
    print("\n🎉 All Optuna integration tests passed!")
    print("The XGBoost parameter compatibility issues have been resolved.")
    return True

if __name__ == "__main__":
    success = test_optuna_integration()
    if success:
        print("\n✅ Ready for production training with Optuna optimization!")
    else:
        print("\n❌ Issues remain - please check the errors above.")
