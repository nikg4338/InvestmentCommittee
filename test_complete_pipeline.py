#!/usr/bin/env python3
"""
Complete integration test for the adaptive OOF stacking pipeline.
Tests both singleton minority and normal scenarios end-to-end.
"""

import pandas as pd
import numpy as np
from train_models import prepare_training_data, train_committee_models_advanced
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data(n_majority=49, n_minority=1, n_features=10):
    """Create synthetic test data with specified class balance."""
    np.random.seed(42)
    
    # Create majority class samples
    X_majority = pd.DataFrame(
        np.random.randn(n_majority, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_majority = pd.Series([0.0] * n_majority)
    
    # Create minority class samples
    X_minority = pd.DataFrame(
        np.random.randn(n_minority, n_features) + 2,  # Shift to make classes separable
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_minority = pd.Series([1.0] * n_minority)
    
    # Combine
    X = pd.concat([X_majority, X_minority], ignore_index=True)
    y = pd.concat([y_majority, y_minority], ignore_index=True)
    
    return X, y

def test_singleton_minority_pipeline():
    """Test complete pipeline with singleton minority class."""
    print("\n🔬 Testing complete pipeline with singleton minority...")
    
    # Create extreme imbalance data
    X, y = create_test_data(n_majority=49, n_minority=1, n_features=8)
    print(f"📊 Original class distribution: {y.value_counts().to_dict()}")
    
    # Create combined dataframe for prepare_training_data function
    df = X.copy()
    df['target'] = y
    feature_columns = [col for col in df.columns if col != 'target']
    
    # Test data preparation
    X_train, X_test, y_train, y_test = prepare_training_data(
        df, feature_columns, 'target', test_size=0.2
    )
    print(f"📈 Train distribution: {y_train.value_counts().to_dict()}")
    print(f"📉 Test distribution: {y_test.value_counts().to_dict()}")
    
    # Verify singleton is in test set
    assert 1.0 in y_test.values, "Singleton minority should be in test set"
    assert 1.0 not in y_train.values, "Train set should not contain minority when singleton"
    print("✅ Data splitting correct: singleton minority in test set")
    
    # Test model training
    print("\n🚀 Training models with adaptive OOF...")
    models, metrics = train_committee_models_advanced(X_train, y_train, X_test, y_test)
    
    # Verify training completed successfully
    assert models is not None, "Models should be trained successfully"
    assert metrics is not None, "Metrics should be calculated"
    
    print("✅ Model training completed successfully!")
    print(f"📊 Number of models trained: {len(metrics)}")
    
    # Check that we got ensemble metrics
    if 'ensemble' in metrics:
        f1 = metrics['ensemble'].get('f1_score', metrics['ensemble'].get('f1', 0))
        roc_auc = metrics['ensemble'].get('roc_auc', 0)
        print(f"🏆 Ensemble F1: {f1:.3f}")
        print(f"🏆 Ensemble ROC-AUC: {roc_auc:.3f}")
    
    return True

def test_normal_case_pipeline():
    """Test complete pipeline with normal class distribution."""
    print("\n🔬 Testing complete pipeline with normal class distribution...")
    
    # Create balanced data
    X, y = create_test_data(n_majority=40, n_minority=10, n_features=8)
    print(f"📊 Original class distribution: {y.value_counts().to_dict()}")
    
    # Create combined dataframe for prepare_training_data function
    df = X.copy()
    df['target'] = y
    feature_columns = [col for col in df.columns if col != 'target']
    
    # Test data preparation
    X_train, X_test, y_train, y_test = prepare_training_data(
        df, feature_columns, 'target', test_size=0.2
    )
    print(f"📈 Train distribution: {y_train.value_counts().to_dict()}")
    print(f"📉 Test distribution: {y_test.value_counts().to_dict()}")
    
    # Verify both classes in both sets
    assert len(y_train.unique()) == 2, "Train set should contain both classes"
    assert len(y_test.unique()) == 2, "Test set should contain both classes"
    print("✅ Data splitting correct: both classes in train and test sets")
    
    # Test model training
    print("\n🚀 Training models with adaptive OOF...")
    models, metrics = train_committee_models_advanced(X_train, y_train, X_test, y_test)
    
    # Verify training completed successfully
    assert models is not None, "Models should be trained successfully"
    assert metrics is not None, "Metrics should be calculated"
    
    print("✅ Model training completed successfully!")
    print(f"📊 Number of models trained: {len(metrics)}")
    
    # Check stacking type
    if models.get('meta_model') is not None:
        print("📊 Using OOF stacking with meta-model")
    else:
        print("📊 Using simple stacking with ensemble")
    
    return True

def main():
    """Run complete pipeline tests."""
    print("🚀 Starting complete pipeline integration tests...")
    
    try:
        # Test singleton minority scenario
        test1_passed = test_singleton_minority_pipeline()
        
        # Test normal scenario
        test2_passed = test_normal_case_pipeline()
        
        if test1_passed and test2_passed:
            print("\n🎯 Final Results:")
            print("   Singleton minority pipeline: ✅ PASSED")
            print("   Normal case pipeline: ✅ PASSED")
            print("🏆 All pipeline integration tests passed!")
            return True
        else:
            print("❌ Some tests failed")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
