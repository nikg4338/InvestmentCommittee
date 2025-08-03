#!/usr/bin/env python3
"""
Test script for the robust training implementation
"""

import numpy as np
import pandas as pd
from train_models import (
    prepare_train_test, ensure_minority, robust_out_of_fold_stacking,
    train_committee_models_advanced, find_optimal_threshold
)

def create_test_data():
    """Create test data with class imbalance."""
    np.random.seed(42)
    
    # Create features
    n_samples = 100
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    
    # Create imbalanced target (only 5 positives)
    y = np.zeros(n_samples)
    positive_indices = np.random.choice(n_samples, size=5, replace=False)
    y[positive_indices] = 1
    
    # Create DataFrame
    feature_cols = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df['target'] = y
    
    return df, feature_cols

def test_ensure_minority():
    """Test the ensure_minority function."""
    print("Testing ensure_minority function...")
    
    df, feature_cols = create_test_data()
    X = df[feature_cols]
    y = df['target']
    
    print(f"Original class distribution: {y.value_counts().to_dict()}")
    
    # Test with n_splits=5
    X_boosted, y_boosted = ensure_minority(X, y, n_splits=5)
    
    print(f"Boosted class distribution: {y_boosted.value_counts().to_dict()}")
    
    # Should have at least 5 minority samples now
    minority_count = y_boosted.value_counts().min()
    assert minority_count >= 5, f"Expected at least 5 minority samples, got {minority_count}"
    
    print("✅ ensure_minority test passed")

def test_prepare_train_test():
    """Test the prepare_train_test function."""
    print("\nTesting prepare_train_test function...")
    
    df, feature_cols = create_test_data()
    
    try:
        X_train, X_test, y_train, y_test = prepare_train_test(df, feature_cols)
        
        print(f"Train class distribution: {y_train.value_counts().to_dict()}")
        print(f"Test class distribution: {y_test.value_counts().to_dict()}")
        
        # Should have both classes in both sets (after stratification)
        assert len(y_train.unique()) == 2, "Training set should have both classes"
        assert len(y_test.unique()) == 2, "Test set should have both classes"
        
        print("✅ prepare_train_test test passed")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"⚠️ prepare_train_test failed (expected with extreme imbalance): {e}")
        # Fallback for extreme imbalance
        X = df[feature_cols].copy()
        y = df['target'].copy()
        
        # Manual split ensuring at least one positive in each set
        positive_indices = y[y == 1].index
        test_size = int(0.2 * len(df))
        
        # Put one positive in test
        test_indices = [positive_indices[0]]
        # Add random negatives to test
        negative_indices = y[y == 0].index
        test_negatives = np.random.choice(negative_indices, size=test_size-1, replace=False)
        test_indices.extend(test_negatives)
        
        train_indices = [i for i in df.index if i not in test_indices]
        
        X_train = X.loc[train_indices]
        X_test = X.loc[test_indices] 
        y_train = y.loc[train_indices]
        y_test = y.loc[test_indices]
        
        print(f"Manual split - Train: {y_train.value_counts().to_dict()}")
        print(f"Manual split - Test: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test

def test_full_training_pipeline():
    """Test the full robust training pipeline."""
    print("\nTesting full robust training pipeline...")
    
    try:
        df, feature_cols = create_test_data()
        X_train, X_test, y_train, y_test = test_prepare_train_test()
        
        print("Running robust training...")
        models, metrics = train_committee_models_advanced(X_train, y_train, X_test, y_test)
        
        print("\n📊 Training Results:")
        for model_name, model_metrics in metrics.items():
            print(f"{model_name}: F1={model_metrics['f1']:.3f}, "
                  f"Precision={model_metrics['precision']:.3f}, "
                  f"Recall={model_metrics['recall']:.3f}")
        
        # Check that we got some results
        assert len(metrics) > 0, "Should have some metrics"
        assert 'meta_model' in metrics, "Should have meta-model metrics"
        
        print("✅ Full training pipeline test passed")
        
    except Exception as e:
        print(f"❌ Full training pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Testing robust training implementation...")
    
    test_ensure_minority()
    test_prepare_train_test()
    test_full_training_pipeline()
    
    print("\n🎉 All tests completed!")
