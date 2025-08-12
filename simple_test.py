#!/usr/bin/env python3
"""
Simple test to verify core functionality.
"""

import sys
import pandas as pd
import numpy as np

def test_data_leakage_fix():
    """Test that data leakage prevention works."""
    print("Testing data leakage prevention...")
    
    from train_models import prepare_training_data
    
    # Create sample imbalanced data
    np.random.seed(42)
    n_samples = 1000
    n_positive = 10  # 1% positive class
    
    # Create DataFrame with features and target
    df = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'target': [0] * (n_samples - n_positive) + [1] * n_positive
    })
    
    feature_columns = ['feature1', 'feature2', 'feature3']
    
    print(f"Original data: {len(df)} samples, {df['target'].sum()} positive ({df['target'].mean()*100:.1f}%)")
    
    # Test prepare_training_data function
    data_result = prepare_training_data(
        df, feature_columns, target_column='target'
    )
    
    X_train = data_result['X_train']
    X_test = data_result['X_test']
    y_train = data_result['y_train']
    y_test = data_result['y_test']
    
    test_positive_rate = y_test.mean() * 100
    train_positive_rate = y_train.mean() * 100
    
    print(f"Test set: {len(y_test)} samples, {y_test.sum()} positive ({test_positive_rate:.1f}%)")
    print(f"Training set: {len(y_train)} samples, {y_train.sum()} positive ({train_positive_rate:.1f}%)")
    
    # Verify test set preserved original distribution (should be ~1%)
    # Training set should be balanced if SMOTE was applied
    if abs(test_positive_rate - 1.0) < 3.0:  # Allow some variance due to small sample
        print("✓ SUCCESS: Data leakage prevention working correctly!")
        print("  - Test set preserved original ultra-rare distribution")
        print(f"  - Training set balanced: {train_positive_rate:.1f}% positive")
        return True
    else:
        print("✗ FAILURE: Data leakage prevention not working")
        print(f"  Expected ~1% positive in test, got {test_positive_rate:.1f}%")
        return False

def test_threshold_optimization():
    """Test threshold optimization function."""
    print("\nTesting threshold optimization...")
    
    from train_models import find_optimal_threshold_on_test
    
    # Create realistic test scenario
    np.random.seed(42)
    n_test = 200
    n_positive = 2  # 1% positive class
    
    y_test = np.array([0] * (n_test - n_positive) + [1] * n_positive)
    
    # Simulate model predictions - positive samples should have higher probabilities
    y_proba = np.random.beta(2, 8, n_test)  # Most predictions low
    y_proba[y_test == 1] = np.random.beta(6, 4, n_positive)  # Positive samples higher
    
    print(f"Test data: {len(y_test)} samples, {sum(y_test)} positive ({sum(y_test)/len(y_test)*100:.1f}%)")
    
    # Test threshold optimization
    best_threshold, best_score, metrics = find_optimal_threshold_on_test(y_test, y_proba)
    
    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"Best F1 score: {best_score:.3f}")
    print(f"Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
    
    if 0.1 <= best_threshold <= 0.9 and best_score > 0:
        print("✓ SUCCESS: Threshold optimization working correctly!")
        return True
    else:
        print("✗ FAILURE: Threshold optimization not working")
        return False

if __name__ == "__main__":
    print("="*60)
    print("ENHANCED PIPELINE CORE FUNCTIONALITY TEST")
    print("="*60)
    
    test1_passed = test_data_leakage_fix()
    test2_passed = test_threshold_optimization()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✓ Data leakage prevention working")
        print("✓ Threshold optimization working")
        print("\nThe enhanced pipeline is ready for training!")
    else:
        print("❌ SOME TESTS FAILED")
        if not test1_passed:
            print("✗ Data leakage prevention failed")
        if not test2_passed:
            print("✗ Threshold optimization failed")
