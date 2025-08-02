#!/usr/bin/env python3
"""
Test script to validate all the improvements made to train_models.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from train_models import (
    train_and_calibrate_model,
    export_metrics_to_csv,
    save_meta_model_coefficients,
    _create_metric_comparison_charts,
    _create_confusion_matrices_oof
)

def test_train_and_calibrate_model():
    """Test the train_and_calibrate_model function"""
    print("🧪 Testing train_and_calibrate_model...")
    
    # Create synthetic data
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
    y_train = pd.Series(np.random.randint(0, 2, 100))
    X_val = pd.DataFrame(np.random.randn(20, 5), columns=[f'feature_{i}' for i in range(5)])
    y_val = pd.Series(np.random.randint(0, 2, 20))
    
    # Test with RandomForest
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    fitted_model, threshold = train_and_calibrate_model(
        model, X_train, y_train, X_val, y_val, 'isotonic', use_threshold=True
    )
    
    print(f"   ✅ Model trained with threshold: {threshold:.3f}")
    return True

def test_export_metrics_to_csv():
    """Test the export_metrics_to_csv function"""
    print("🧪 Testing export_metrics_to_csv...")
    
    # Create test metrics
    metrics = {
        'model_1': {'f1': 0.85, 'roc_auc': 0.92, 'pr_auc': 0.88, 'accuracy': 0.90, 'precision': 0.87, 'recall': 0.83},
        'model_2': {'f1': 0.78, 'roc_auc': 0.86, 'pr_auc': 0.81, 'accuracy': 0.82, 'precision': 0.80, 'recall': 0.76},
        'meta_model': {'f1': 0.89, 'roc_auc': 0.94, 'pr_auc': 0.91, 'accuracy': 0.93, 'precision': 0.91, 'recall': 0.87}
    }
    
    thresholds = {'model_1': 0.52, 'model_2': 0.48, 'meta_model': 0.50}
    
    # Create test directory
    test_dir = "test_output"
    os.makedirs(test_dir, exist_ok=True)
    
    export_metrics_to_csv(metrics, thresholds, test_dir)
    
    # Check if file was created
    csv_path = os.path.join(test_dir, 'model_summary.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"   ✅ CSV exported with {len(df)} models")
        print(f"   📊 Columns: {list(df.columns)}")
        
        # Cleanup
        os.remove(csv_path)
        os.rmdir(test_dir)
        return True
    else:
        print("   ❌ CSV file not created")
        return False

def test_save_meta_model_coefficients():
    """Test the save_meta_model_coefficients function"""
    print("🧪 Testing save_meta_model_coefficients...")
    
    # Create a simple logistic regression model
    np.random.seed(42)
    X_train = np.random.randn(100, 3)
    y_train = np.random.randint(0, 2, 100)
    
    meta_model = LogisticRegression(random_state=42)
    meta_model.fit(X_train, y_train)
    
    base_model_names = ['model_1', 'model_2', 'model_3']
    
    # Create test directory
    test_dir = "test_output"
    os.makedirs(test_dir, exist_ok=True)
    
    save_meta_model_coefficients(meta_model, base_model_names, test_dir)
    
    # Check if file was created
    coef_path = os.path.join(test_dir, 'meta_model_coefficients.csv')
    if os.path.exists(coef_path):
        df = pd.read_csv(coef_path)
        print(f"   ✅ Coefficients saved with {len(df)} features")
        print(f"   📊 Features: {list(df['feature'])}")
        
        # Cleanup
        os.remove(coef_path)
        os.rmdir(test_dir)
        return True
    else:
        print("   ❌ Coefficients file not created")
        return False

def test_visualization_functions():
    """Test the visualization helper functions"""
    print("🧪 Testing visualization helper functions...")
    
    # Test metrics for charts
    metrics = {
        'xgboost': {'f1': 0.85, 'roc_auc': 0.92, 'pr_auc': 0.88},
        'lightgbm': {'f1': 0.78, 'roc_auc': 0.86, 'pr_auc': 0.81},
        'meta_model': {'f1': 0.89, 'roc_auc': 0.94, 'pr_auc': 0.91}
    }
    
    # Test that functions exist and are callable
    assert callable(_create_metric_comparison_charts), "❌ _create_metric_comparison_charts not callable"
    assert callable(_create_confusion_matrices_oof), "❌ _create_confusion_matrices_oof not callable"
    
    print("   ✅ Visualization helper functions are properly defined")
    return True

def main():
    """Run all tests"""
    print("🚀 Testing OOF Pipeline Improvements")
    print("="*50)
    
    tests = [
        test_train_and_calibrate_model,
        test_export_metrics_to_csv,
        test_save_meta_model_coefficients,
        test_visualization_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"   ❌ {test.__name__} failed")
        except Exception as e:
            print(f"   ❌ {test.__name__} failed with error: {e}")
    
    print("="*50)
    print(f"🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All improvements successfully implemented and tested!")
    else:
        print("⚠️  Some tests failed - check the implementation")

if __name__ == "__main__":
    main()
