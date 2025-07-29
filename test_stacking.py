#!/usr/bin/env python3
"""
Test script for model stacking implementation.
This script tests the predict_proba method and stacking functionality.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from models.neural_predictor import NeuralPredictor
from models.xgboost_model import XGBoostModel

def test_neural_predict_proba():
    """Test the predict_proba method of NeuralPredictor."""
    print("Testing NeuralPredictor.predict_proba()...")
    
    # Create a simple neural predictor
    neural_model = NeuralPredictor(model_type='mlp')
    
    # Create dummy data
    X_test = np.random.randn(100, 12)  # 100 samples, 12 features
    
    # Test predict_proba
    try:
        probs = neural_model.predict_proba(X_test)
        print(f"✅ predict_proba() works - returned {len(probs)} probabilities")
        print(f"   Probability range: {probs.min():.4f} to {probs.max():.4f}")
        return True
    except Exception as e:
        print(f"❌ predict_proba() failed: {e}")
        return False

def test_xgboost_predict_proba():
    """Test the predict_proba method of XGBoostModel."""
    print("\nTesting XGBoostModel.predict_proba()...")
    
    # Create a simple XGBoost model
    xgb_model = XGBoostModel()
    
    # Create dummy data
    X_test = np.random.randn(100, 12)  # 100 samples, 12 features
    y_test = np.random.randint(0, 2, 100)  # Binary labels
    
    # Train a simple model
    try:
        xgb_model.train(X_test, y_test)
        probs = xgb_model.predict_proba(X_test)
        print(f"✅ XGBoost predict_proba() works - returned shape {probs.shape}")
        print(f"   Probability range: {probs[:, 1].min():.4f} to {probs[:, 1].max():.4f}")
        return True
    except Exception as e:
        print(f"❌ XGBoost predict_proba() failed: {e}")
        return False

def test_stacking_logic():
    """Test the stacking logic with dummy data."""
    print("\nTesting stacking logic...")
    
    # Create dummy predictions
    n_samples = 100
    xgb_proba = np.random.random(n_samples)
    nn_proba = np.random.random(n_samples)
    y_true = np.random.randint(0, 2, n_samples)
    
    # Stack predictions
    stacked_features = np.column_stack((xgb_proba, nn_proba))
    
    # Train meta-model
    try:
        meta_model = LogisticRegression(random_state=42, max_iter=1000)
        meta_model.fit(stacked_features, y_true)
        
        # Evaluate
        final_preds = meta_model.predict(stacked_features)
        final_proba = meta_model.predict_proba(stacked_features)[:, 1]
        
        stacked_accuracy = accuracy_score(y_true, final_preds)
        stacked_roc_auc = roc_auc_score(y_true, final_proba)
        
        print(f"✅ Stacking logic works:")
        print(f"   Stacked accuracy: {stacked_accuracy:.4f}")
        print(f"   Stacked ROC AUC: {stacked_roc_auc:.4f}")
        return True
    except Exception as e:
        print(f"❌ Stacking logic failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING MODEL STACKING IMPLEMENTATION")
    print("=" * 60)
    
    tests = [
        test_neural_predict_proba,
        test_xgboost_predict_proba,
        test_stacking_logic
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("✅ All tests passed! Model stacking implementation is ready.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 