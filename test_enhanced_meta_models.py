#!/usr/bin/env python3
"""
Enhanced Meta-Model Validation Test
==================================

Test script to validate all enhanced meta-model training strategies for F₁ optimization.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_extreme_imbalance_dataset(n_samples=1000, n_features=10, positive_ratio=0.05):
    """Create synthetic dataset with extreme class imbalance."""
    logger.info(f"Creating dataset: {n_samples} samples, {positive_ratio:.1%} positive class")
    
    # Create imbalanced dataset with more signal
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,  # More informative features
        n_redundant=1,
        n_clusters_per_class=2,  # More clusters for better separation
        class_sep=1.2,  # Better class separation
        weights=[1-positive_ratio, positive_ratio],
        flip_y=0.01,  # Small amount of noise
        random_state=42
    )
    
    logger.info(f"Dataset created: {len(y)} samples, {y.sum()} positives ({100*y.mean():.1f}%)")
    return X, y

def test_enhanced_meta_models():
    """Test all enhanced meta-model strategies."""
    logger.info("🧪 Testing Enhanced Meta-Model Strategies")
    
    # Create test data with better signal
    X, y = create_extreme_imbalance_dataset(n_samples=800, positive_ratio=0.06)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Create more realistic meta-features (simulating 5 base model predictions)
    # These should have some correlation with the true labels
    np.random.seed(42)
    
    # Create meta-features that have some signal
    base_signal = y_train.reshape(-1, 1) * 0.3 + np.random.normal(0, 0.1, (len(y_train), 1))
    noise = np.random.beta(1, 20, (len(y_train), 4))  # Random noise for other models
    meta_X_train = np.hstack([base_signal, noise])
    
    base_signal_test = y_test.reshape(-1, 1) * 0.3 + np.random.normal(0, 0.1, (len(y_test), 1))
    noise_test = np.random.beta(1, 20, (len(y_test), 4))
    meta_X_test = np.hstack([base_signal_test, noise_test])
    
    # Create more realistic OOF predictions
    oof_predictions = {}
    test_predictions = {}
    
    for i, model_name in enumerate(['xgboost', 'lightgbm', 'catboost', 'random_forest', 'svm']):
        # Add some correlation with true labels
        signal_strength = [0.4, 0.35, 0.3, 0.25, 0.2][i]  # Decreasing signal strength
        base_prob = y_train * signal_strength + np.random.beta(1, 30, len(y_train))
        oof_predictions[model_name] = np.clip(base_prob, 0, 1)
        
        base_prob_test = y_test * signal_strength + np.random.beta(1, 30, len(y_test))
        test_predictions[model_name] = np.clip(base_prob_test, 0, 1)
    
    from utils.enhanced_meta_models import (
        train_meta_model_with_optimal_threshold,
        train_focal_loss_meta_model,
        train_dynamic_weighted_ensemble,
        train_feature_selected_meta_model,
        optuna_optimize_base_model_for_f1
    )
    
    results = {}
    
    # Test 1: Optimal threshold with class weighting
    logger.info("\n1️⃣ Testing Optimal Threshold Meta-Model...")
    try:
        meta_model, threshold = train_meta_model_with_optimal_threshold(
            meta_X_train, y_train, meta_learner_type='logistic', use_class_weights=True
        )
        test_proba = meta_model.predict_proba(meta_X_test)[:, 1]
        test_pred = (test_proba >= threshold).astype(int)
        
        from sklearn.metrics import f1_score, precision_score, recall_score
        f1 = f1_score(y_test, test_pred)
        results['optimal_threshold'] = {'f1': f1, 'threshold': threshold}
        logger.info(f"   ✅ F1: {f1:.4f}, Threshold: {threshold:.4f}")
        
    except Exception as e:
        logger.error(f"   ❌ Failed: {e}")
        results['optimal_threshold'] = {'f1': 0.0, 'error': str(e)}
    
    # Test 2: Focal loss meta-model
    logger.info("\n2️⃣ Testing Focal Loss Meta-Model...")
    try:
        meta_model, threshold = train_focal_loss_meta_model(meta_X_train, y_train)
        
        if hasattr(meta_model, 'predict'):
            test_proba = meta_model.predict(meta_X_test)
        else:
            test_proba = meta_model.predict_proba(meta_X_test)[:, 1]
            
        test_pred = (test_proba >= threshold).astype(int)
        f1 = f1_score(y_test, test_pred)
        results['focal_loss'] = {'f1': f1, 'threshold': threshold}
        logger.info(f"   ✅ F1: {f1:.4f}, Threshold: {threshold:.4f}")
        
    except Exception as e:
        logger.error(f"   ❌ Failed: {e}")
        results['focal_loss'] = {'f1': 0.0, 'error': str(e)}
    
    # Test 3: Dynamic weighted ensemble
    logger.info("\n3️⃣ Testing Dynamic Weighted Ensemble...")
    try:
        test_weighted, weights, threshold = train_dynamic_weighted_ensemble(
            oof_predictions, y_train, test_predictions, weight_metric='roc_auc'
        )
        test_pred = (test_weighted >= threshold).astype(int)
        f1 = f1_score(y_test, test_pred)
        results['dynamic_weights'] = {'f1': f1, 'threshold': threshold, 'weights': weights}
        logger.info(f"   ✅ F1: {f1:.4f}, Threshold: {threshold:.4f}")
        
    except Exception as e:
        logger.error(f"   ❌ Failed: {e}")
        results['dynamic_weights'] = {'f1': 0.0, 'error': str(e)}
    
    # Test 4: Feature-selected meta-model
    logger.info("\n4️⃣ Testing Feature-Selected Meta-Model...")
    try:
        meta_model, threshold, train_sel, test_sel = train_feature_selected_meta_model(
            meta_X_train, y_train, meta_X_test, k_best=3
        )
        test_proba = meta_model.predict_proba(test_sel)[:, 1]
        test_pred = (test_proba >= threshold).astype(int)
        f1 = f1_score(y_test, test_pred)
        results['feature_select'] = {'f1': f1, 'threshold': threshold}
        logger.info(f"   ✅ F1: {f1:.4f}, Threshold: {threshold:.4f}")
        
    except Exception as e:
        logger.error(f"   ❌ Failed: {e}")
        results['feature_select'] = {'f1': 0.0, 'error': str(e)}
    
    # Test 5: Optuna optimization (mock test)
    logger.info("\n5️⃣ Testing Optuna F₁ Optimization...")
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        # Use small subset for faster testing
        X_small = X_train[:100]
        y_small = y_train[:100]
        
        # Mock small optimization for speed
        best_params = optuna_optimize_base_model_for_f1(
            RandomForestClassifier, 
            X_small, y_small,  # Use numpy arrays directly
            n_trials=3,  # Reduced for speed
            optimize_metric='average_precision'
        )
        
        if best_params:
            results['optuna_f1'] = {'params': best_params, 'success': True}
            logger.info(f"   ✅ Optimization completed, {len(best_params)} parameters found")
        else:
            results['optuna_f1'] = {'success': False}
            logger.info("   ⚠️ No parameters found (may be normal)")
            
    except Exception as e:
        logger.error(f"   ❌ Failed: {e}")
        results['optuna_f1'] = {'success': False, 'error': str(e)}
    
    # Summary
    logger.info("\n📊 SUMMARY OF ENHANCED META-MODEL TESTS")
    logger.info("=" * 50)
    
    successful_tests = 0
    total_tests = 5
    
    for strategy, result in results.items():
        if 'error' not in result:
            if strategy == 'optuna_f1':
                status = "✅ PASS" if result.get('success', False) else "⚠️ PARTIAL"
                if result.get('success', False):
                    successful_tests += 1
            else:
                f1_score = result.get('f1', 0.0)
                status = "✅ PASS" if f1_score > 0 else "❌ FAIL"
                if f1_score > 0:
                    successful_tests += 1
                    
                logger.info(f"{status} {strategy.replace('_', ' ').title()}: F1={f1_score:.4f}")
        else:
            logger.info(f"❌ FAIL {strategy.replace('_', ' ').title()}: {result['error']}")
    
    logger.info(f"\n🎯 OVERALL: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests >= 4:
        logger.info("🎉 Enhanced meta-model strategies working properly!")
        return True
    else:
        logger.warning("⚠️ Some enhanced meta-model strategies failed")
        return False

if __name__ == "__main__":
    success = test_enhanced_meta_models()
    exit(0 if success else 1)
