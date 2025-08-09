#!/usr/bin/env python3
"""
Test Script for Parameter Compatibility Fixes
=============================================

This script validates the fixes for XGBoost parameter compatibility
and single class sampling warnings.
"""

import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_xgboost_parameter_fix():
    """Test that XGBoost models can be created with Optuna parameters correctly."""
    print("\n=== Testing XGBoost Parameter Fix ===")
    
    from models.xgboost_model import XGBoostModel
    from utils.stacking import get_optuna_param_space
    
    # Get parameter space
    param_space = get_optuna_param_space('xgboost')
    print(f"Parameter space: {param_space}")
    
    # Test with actual parameter values
    test_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1
    }
    
    # Test old way (should fail)
    try:
        model_old = XGBoostModel(**test_params)
        print("❌ ERROR: Old way should have failed!")
        return False
    except Exception as e:
        print(f"✅ Expected error with old way: {str(e)}")
    
    # Test new way (should work)
    try:
        model_new = XGBoostModel(model_params=test_params)
        print("✅ SUCCESS: New way works!")
        print(f"Model parameters: {model_new.model_params}")
        return True
    except Exception as e:
        print(f"❌ ERROR with new way: {str(e)}")
        return False

def test_other_models_still_work():
    """Test that other models still work with direct parameter passing."""
    print("\n=== Testing Other Models Still Work ===")
    
    from models.lightgbm_model import LightGBMModel
    from models.catboost_model import CatBoostModel
    
    # Test LightGBM (should accept direct parameters)
    try:
        lgb_params = {
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        lgb_model = LightGBMModel(**lgb_params)
        print("✅ LightGBM model creation successful")
    except Exception as e:
        print(f"❌ LightGBM model creation failed: {str(e)}")
        return False
    
    # Test CatBoost (should accept direct parameters)
    try:
        cb_params = {
            'iterations': 100,
            'learning_rate': 0.05,
            'depth': 6
        }
        cb_model = CatBoostModel(**cb_params)
        print("✅ CatBoost model creation successful")
        return True
    except Exception as e:
        print(f"❌ CatBoost model creation failed: {str(e)}")
        return False

def test_single_class_handling():
    """Test handling of single class cases in sampling."""
    print("\n=== Testing Single Class Handling ===")
    
    from utils.sampling import prepare_balanced_data
    
    # Create single class data
    X_single = pd.DataFrame(np.random.randn(50, 5), columns=[f'feat_{i}' for i in range(5)])
    y_single = pd.Series([0] * 50)  # All same class
    
    print(f"Single class distribution: {y_single.value_counts().to_dict()}")
    
    try:
        X_balanced, y_balanced = prepare_balanced_data(X_single, y_single, method='smote')
        print("✅ Single class handling successful")
        print(f"Returned shape: {X_balanced.shape}")
        print(f"Returned distribution: {y_balanced.value_counts().to_dict()}")
        
        # Should return original data unchanged
        if X_balanced.shape == X_single.shape and y_balanced.equals(y_single):
            print("✅ Correctly returned original data for single class")
            return True
        else:
            print("❌ Did not return original data as expected")
            return False
    except Exception as e:
        print(f"❌ Single class handling failed: {str(e)}")
        return False

def test_stacking_parameter_logic():
    """Test the parameter passing logic in stacking module."""
    print("\n=== Testing Stacking Parameter Logic ===")
    
    from models.xgboost_model import XGBoostModel
    from models.lightgbm_model import LightGBMModel
    
    # Simulate the parameter passing logic from stacking module
    test_params = {
        'n_estimators': 50,
        'max_depth': 4,
        'learning_rate': 0.1
    }
    
    # Test XGBoost (should use model_params)
    try:
        model_class = XGBoostModel
        if model_class.__name__ in ['XGBoostModel']:
            fold_model = model_class(model_params=test_params)
        else:
            fold_model = model_class(**test_params)
        print("✅ XGBoost parameter logic works")
    except Exception as e:
        print(f"❌ XGBoost parameter logic failed: {str(e)}")
        return False
    
    # Test LightGBM (should use direct parameters)
    try:
        model_class = LightGBMModel
        if model_class.__name__ in ['XGBoostModel']:
            fold_model = model_class(model_params=test_params)
        else:
            fold_model = model_class(**test_params)
        print("✅ LightGBM parameter logic works")
        return True
    except Exception as e:
        print(f"❌ LightGBM parameter logic failed: {str(e)}")
        return False

def test_full_pipeline():
    """Test a mini version of the training pipeline."""
    print("\n=== Testing Full Pipeline ===")
    
    from models.xgboost_model import XGBoostModel
    from utils.sampling import prepare_balanced_data
    
    # Create test data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 8), columns=[f'feature_{i}' for i in range(8)])
    y = pd.Series(np.random.choice([0, 1], size=100, p=[0.7, 0.3]))
    
    print(f"Test data: {X.shape}, classes: {y.value_counts().to_dict()}")
    
    try:
        # Test balancing
        X_balanced, y_balanced = prepare_balanced_data(X, y, method='smote')
        print(f"Balanced: {X_balanced.shape}, classes: {y_balanced.value_counts().to_dict()}")
        
        # Test model training with Optuna-style parameters
        optuna_params = {
            'n_estimators': 50,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8
        }
        
        model = XGBoostModel(model_params=optuna_params)
        model.train(X_balanced, y_balanced)
        
        # Test predictions
        predictions = model.predict_proba(X)
        print(f"Predictions shape: {predictions.shape}")
        print("✅ Full pipeline test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Running Parameter Compatibility Tests")
    print("=" * 50)
    
    tests = [
        test_xgboost_parameter_fix,
        test_other_models_still_work,
        test_single_class_handling,
        test_stacking_parameter_logic,
        test_full_pipeline
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {str(e)}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    total_passed = sum(results)
    total_tests = len(results)
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 All tests passed! The parameter fixes are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
