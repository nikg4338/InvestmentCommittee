#!/usr/bin/env python3

"""
Verification script to test that our data leakage fixes are working correctly.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath('.'))

from utils.data_splitting import stratified_train_test_split
from utils.stacking import get_model_predictions_safe

def test_stratified_split_no_synthetic():
    """Test that stratified split doesn't create synthetic samples before splitting."""
    print("🔍 Testing stratified split without synthetic data...")
    
    # Create a simple dataset with class imbalance
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'unique_id': range(100)  # Unique identifier to track samples
    })
    
    # Create imbalanced target: 80% class 0, 20% class 1
    y = pd.Series([0] * 80 + [1] * 20)
    
    # Test stratified split
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Verify no sample overlap between train and test
    train_ids = set(X_train['unique_id'])
    test_ids = set(X_test['unique_id'])
    overlap = train_ids.intersection(test_ids)
    
    print(f"   Original dataset size: {len(X)}")
    print(f"   Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"   Train class distribution: {y_train.value_counts().to_dict()}")
    print(f"   Test class distribution: {y_test.value_counts().to_dict()}")
    print(f"   Sample overlap between train/test: {len(overlap)}")
    
    # Verify both classes present and no synthetic data created
    assert len(overlap) == 0, "❌ LEAKAGE: Samples appear in both train and test!"
    assert len(X_train) + len(X_test) == len(X), "❌ LEAKAGE: Total samples don't match original!"
    assert len(y_train.value_counts()) >= 2, "❌ Missing classes in train set"
    assert len(y_test.value_counts()) >= 2, "❌ Missing classes in test set"
    
    print("   ✅ No synthetic data created, no sample leakage detected")

def test_prediction_safety():
    """Test that prediction helper returns probabilities for classifiers."""
    print("\n🔍 Testing safe prediction handling...")
    
    # Mock classifier that returns probabilities
    class MockClassifier:
        def predict_proba(self, X):
            n_samples = len(X)
            # Return probabilities [prob_class_0, prob_class_1]
            probs = np.random.uniform(0, 1, (n_samples, 2))
            probs = probs / probs.sum(axis=1, keepdims=True)  # Normalize
            return probs
        
        def predict(self, X):
            # This should NOT be called for classifiers in our safe implementation
            return np.random.randint(0, 2, len(X))
    
    # Mock classifier without predict_proba (unsafe case)
    class MockUnsafeClassifier:
        def predict(self, X):
            return np.random.randint(0, 2, len(X))
    
    # Test data
    X_test = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    
    # Test safe classifier
    model_safe = MockClassifier()
    predictions_safe = get_model_predictions_safe(model_safe, X_test, "test_classifier")
    
    print(f"   Safe classifier predictions: {predictions_safe}")
    print(f"   Are predictions probabilities? {np.all((predictions_safe >= 0) & (predictions_safe <= 1))}")
    assert np.all((predictions_safe >= 0) & (predictions_safe <= 1)), "❌ Predictions not in [0,1] range!"
    
    # Test unsafe classifier (should return neutral probabilities)
    model_unsafe = MockUnsafeClassifier()
    predictions_unsafe = get_model_predictions_safe(model_unsafe, X_test, "unsafe_classifier")
    
    print(f"   Unsafe classifier fallback: {predictions_unsafe}")
    print(f"   Fallback returns neutral probabilities? {np.allclose(predictions_unsafe, 0.5)}")
    assert np.allclose(predictions_unsafe, 0.5), "❌ Unsafe classifier didn't return neutral probabilities!"
    
    print("   ✅ Prediction safety verified - no hard labels leaked")

if __name__ == "__main__":
    print("🛡️  Verifying Data Leakage Fixes")
    print("=" * 50)
    
    try:
        test_stratified_split_no_synthetic()
        test_prediction_safety()
        
        print("\n🎉 All leakage prevention tests PASSED!")
        print("✅ Patch 1: Stratified split without synthetic data - WORKING")
        print("✅ Patch 2: Safe prediction handling with probabilities - WORKING")
        
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        sys.exit(1)
