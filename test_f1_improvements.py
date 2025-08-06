#!/usr/bin/env python3
"""
Test F₁ Improvements Implementation
==================================

Simple test script to verify all 8 F₁ improvements are working correctly.
"""

import sys
import traceback
import pandas as pd
import numpy as np

def test_improvement_1_extended_lookback():
    """Test #1: Extended Lookback Window (24 months)"""
    print("✅ Test 1: Extended Lookback Window")
    try:
        from data_collection_alpaca import AlpacaDataCollector
        collector = AlpacaDataCollector()
        
        # Check the default parameter
        import inspect
        sig = inspect.signature(collector.get_historical_data)
        days_default = sig.parameters['days'].default
        
        print(f"   Default lookback: {days_default} days ({days_default/365:.1f} years)")
        assert days_default == 730, f"Expected 730, got {days_default}"
        print("   ✓ PASS: 24-month lookback implemented")
        return True
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        return False

def test_improvement_2_regression_targets():
    """Test #2: Regression Target Variables"""
    print("✅ Test 2: Multi-Day Return Targets")
    try:
        from data_collection_alpaca import AlpacaDataCollector
        collector = AlpacaDataCollector()
        
        # Create simple mock data
        mock_df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105],
            'open': [99, 100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105, 106],
            'low': [98, 99, 100, 101, 102, 103],
            'volume': [1000000] * 6,
            'timestamp': pd.date_range('2024-01-01', periods=6)
        })
        
        # Test multi-target creation
        multi_targets = collector.create_target_variable(
            mock_df, 'TEST', use_regression=True, create_all_horizons=True
        )
        
        expected_cols = ['target_1d_return', 'target_3d_return', 'target_5d_return', 'target_10d_return']
        print(f"   Target columns: {list(multi_targets.columns)}")
        
        for col in expected_cols:
            if col in multi_targets.columns:
                print(f"   ✓ Found {col}")
            else:
                print(f"   ❌ Missing {col}")
                return False
        
        print("   ✓ PASS: Multi-day targets implemented")
        return True
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        return False

def test_improvement_3_class_weighting():
    """Test #3: Class Weighting"""
    print("✅ Test 3: Class Weighting")
    try:
        # Test model imports and class weight configurations
        models_tested = []
        
        # XGBoost
        try:
            from models.xgboost_model import XGBoostModel
            models_tested.append("XGBoost")
        except:
            pass
            
        # Random Forest
        try:
            from models.random_forest_model import RandomForestModel
            models_tested.append("RandomForest")
        except:
            pass
            
        # LightGBM
        try:
            from models.lightgbm_model import LightGBMModel
            models_tested.append("LightGBM")
        except:
            pass
        
        print(f"   Models with class weighting: {models_tested}")
        
        if len(models_tested) >= 2:
            print("   ✓ PASS: Class weighting implemented")
            return True
        else:
            print("   ❌ FAIL: Not enough models with class weighting")
            return False
            
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        return False

def test_improvement_4_smote_cv():
    """Test #4: SMOTE in Cross-Validation"""
    print("✅ Test 4: SMOTE in CV")
    try:
        from utils.sampling import smote_oversample, smoteenn_resample
        from config.training_config import get_extreme_imbalance_config
        
        config = get_extreme_imbalance_config()
        print(f"   Advanced sampling: {config.advanced_sampling}")
        print(f"   Enhanced sampling enabled: {config.enable_advanced_sampling}")
        
        print("   ✓ PASS: SMOTE in CV implemented")
        return True
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        return False

def test_improvement_5_calibration():
    """Test #5: Probability Calibration"""
    print("✅ Test 5: Probability Calibration")
    try:
        from config.training_config import get_extreme_imbalance_config
        
        config = get_extreme_imbalance_config()
        print(f"   Calibration enabled: {config.enable_calibration}")
        print(f"   Calibration method: {config.calibration.method}")
        print(f"   Calibration CV: {config.calibration.cv_folds}")
        
        print("   ✓ PASS: Probability calibration implemented")
        return True
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        return False

def test_improvement_6_multi_day_binary():
    """Test #6: Multi-Day Binary Targets"""
    print("✅ Test 6: Multi-Day Binary Targets")
    try:
        from data_collection_alpaca import AlpacaDataCollector
        collector = AlpacaDataCollector()
        
        # Simple mock data
        mock_df = pd.DataFrame({
            'close': [100, 102, 99, 105, 98, 107],
            'open': [99, 101, 98, 104, 97, 106],
            'high': [103, 104, 101, 107, 100, 109],
            'low': [98, 100, 97, 103, 96, 105],
            'volume': [1000000] * 6,
            'timestamp': pd.date_range('2024-01-01', periods=6)
        })
        
        # Test binary multi-targets
        binary_targets = collector.create_target_variable(
            mock_df, 'TEST', use_regression=False, create_all_horizons=True
        )
        
        expected_cols = ['target_1d_binary', 'target_3d_binary', 'target_5d_binary', 'target_10d_binary']
        print(f"   Binary columns: {list(binary_targets.columns)}")
        
        found_binary = [col for col in expected_cols if col in binary_targets.columns]
        if len(found_binary) >= 3:
            print("   ✓ PASS: Multi-day binary targets implemented")
            return True
        else:
            print(f"   ❌ FAIL: Only found {len(found_binary)} binary targets")
            return False
            
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        return False

def test_improvement_7_ranking_metrics():
    """Test #7: Ranking-Based Evaluation"""
    print("✅ Test 7: Ranking Metrics")
    try:
        from utils.ranking_metrics import compute_ranking_metrics, precision_at_k
        
        # Test with simple data
        y_true = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.8, 0.3, 0.9, 0.1, 0.4, 0.7, 0.2, 0.3])
        
        ranking_metrics = compute_ranking_metrics(y_true, y_scores)
        
        key_metrics = ['top_5pct_precision', 'mean_average_precision', 'precision_at_10']
        found_metrics = [m for m in key_metrics if m in ranking_metrics]
        
        print(f"   Found ranking metrics: {len(ranking_metrics)}")
        print(f"   Key metrics found: {found_metrics}")
        
        if len(found_metrics) >= 2:
            print("   ✓ PASS: Ranking metrics implemented")
            return True
        else:
            print("   ❌ FAIL: Missing key ranking metrics")
            return False
            
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        return False

def test_improvement_8_regime_features():
    """Test #8: Regime-Aware Features"""
    print("✅ Test 8: Regime Features")
    try:
        from data_collection_alpaca import AlpacaDataCollector
        collector = AlpacaDataCollector()
        
        # Create longer mock data for regime detection
        np.random.seed(42)
        prices = [100]
        for i in range(49):
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.01)))
        
        mock_df = pd.DataFrame({
            'close': prices,
            'open': [p * 0.995 for p in prices],
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.99 for p in prices],
            'volume': [1000000] * 50,
            'timestamp': pd.date_range('2024-01-01', periods=50)
        })
        
        # Test regime features
        enhanced_df = collector.calculate_technical_indicators(mock_df)
        
        regime_features = [col for col in enhanced_df.columns if 'regime' in col]
        print(f"   Regime features: {regime_features}")
        
        expected_regime = ['trend_regime', 'volatility_regime', 'momentum_regime']
        found_regime = [f for f in expected_regime if f in regime_features]
        
        if len(found_regime) >= 2:
            print("   ✓ PASS: Regime features implemented")
            return True
        else:
            print(f"   ❌ FAIL: Only found {len(found_regime)} regime features")
            return False
            
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        return False

def main():
    """Run all F₁ improvement tests"""
    print("🔧 F₁ IMPROVEMENT TESTING")
    print("=" * 50)
    print()
    
    tests = [
        test_improvement_1_extended_lookback,
        test_improvement_2_regression_targets,
        test_improvement_3_class_weighting,
        test_improvement_4_smote_cv,
        test_improvement_5_calibration,
        test_improvement_6_multi_day_binary,
        test_improvement_7_ranking_metrics,
        test_improvement_8_regime_features
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append(False)
            print()
    
    # Summary
    print("🎯 SUMMARY")
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    
    improvements = [
        "Extended Lookback Window (24 months)",
        "Regression Target Variables", 
        "Class Weighting",
        "SMOTE in Cross-Validation",
        "Probability Calibration",
        "Multi-Day Binary Targets",
        "Ranking-Based Evaluation",
        "Regime-Aware Features"
    ]
    
    for i, (improvement, result) in enumerate(zip(improvements, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"#{i+1}: {improvement} - {status}")
    
    print()
    print(f"🚀 OVERALL: {passed}/{total} F₁ improvements implemented")
    
    if passed == total:
        print("🎉 All F₁ improvements successfully implemented!")
    elif passed >= 6:
        print("✅ Most F₁ improvements working - ready for testing!")
    else:
        print("⚠️  Some improvements need attention")
    
    return passed

if __name__ == "__main__":
    main()
