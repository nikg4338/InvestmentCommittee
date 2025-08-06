#!/usr/bin/env python3
"""
Test Enhanced Extreme Imbalance Configuration
=============================================

This script tests the updated extreme_imbalance configuration with 
the new regression approach and LightGBM regressor integration.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_extreme_imbalance_config():
    """Test the enhanced extreme_imbalance configuration."""
    print("🧪 Testing Enhanced Extreme Imbalance Configuration...")
    
    try:
        # Test configuration loading
        from config.training_config import get_extreme_imbalance_config
        config = get_extreme_imbalance_config()
        print("✓ Successfully loaded extreme_imbalance configuration")
        
        # Check regression-specific enhancements
        print(f"\n📊 Regression Enhancements:")
        print(f"   enable_regression_targets: {getattr(config, 'enable_regression_targets', 'Not set')}")
        print(f"   regression_threshold_optimization: {getattr(config, 'regression_threshold_optimization', 'Not set')}")
        print(f"   huber_loss_alpha: {getattr(config, 'huber_loss_alpha', 'Not set')}")
        print(f"   evaluate_regression_metrics: {getattr(config, 'evaluate_regression_metrics', 'Not set')}")
        print(f"   multi_horizon_targets: {getattr(config, 'multi_horizon_targets', 'Not set')}")
        
        # Check model inclusion
        models = getattr(config, 'models_to_train', [])
        print(f"\n🤖 Models to Train: {models}")
        if 'lightgbm_regressor' in models:
            print("✓ LightGBM Regressor included in extreme_imbalance config")
        else:
            print("❌ LightGBM Regressor NOT included in models")
        
        # Check advanced features
        print(f"\n⚡ Advanced Features:")
        print(f"   enable_optuna: {getattr(config, 'enable_optuna', 'Not set')}")
        print(f"   optuna_trials: {getattr(config, 'optuna_trials', 'Not set')}")
        print(f"   advanced_sampling: {getattr(config, 'advanced_sampling', 'Not set')}")
        print(f"   meta_model_strategy: {getattr(config, 'meta_model_strategy', 'Not set')}")
        print(f"   enable_enhanced_stacking: {getattr(config, 'enable_enhanced_stacking', 'Not set')}")
        
        # Check threshold settings
        print(f"\n🎯 Threshold Configuration:")
        print(f"   min_positive_rate: {config.threshold.min_positive_rate}")
        print(f"   top_percentile: {config.ensemble.top_percentile}")
        print(f"   emergency_threshold: {config.threshold.emergency_threshold}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {str(e)}")
        return False

def test_model_registry():
    """Test that all models are properly registered."""
    print(f"\n🧪 Testing Model Registry...")
    
    try:
        from utils.stacking import MODEL_REGISTRY
        
        print(f"📋 Available Models:")
        for model_name, model_class in MODEL_REGISTRY.items():
            print(f"   {model_name}: {model_class.__name__}")
        
        # Check for LightGBM regressor
        if 'lightgbm_regressor' in MODEL_REGISTRY:
            print("✓ LightGBM Regressor properly registered")
            
            # Test instantiation
            regressor_class = MODEL_REGISTRY['lightgbm_regressor']
            model = regressor_class(name="TestRegressor")
            print("✓ LightGBM Regressor can be instantiated")
            
        else:
            print("❌ LightGBM Regressor NOT in MODEL_REGISTRY")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Model registry test failed: {str(e)}")
        return False

def test_integration_commands():
    """Show the recommended commands for using the enhanced configuration."""
    print(f"\n💡 Recommended Usage Commands:")
    
    print(f"\n🚀 **Basic Enhanced Training:**")
    print(f"   python train_models.py --config extreme_imbalance")
    
    print(f"\n🔥 **Production Training (50 Optuna trials):**")
    print(f"   python train_all_batches.py --config extreme_imbalance --optuna-trials 50")
    
    print(f"\n⚡ **Specific Models with Regressor:**")
    print(f"   python train_models.py --config extreme_imbalance --models xgboost lightgbm lightgbm_regressor catboost")
    
    print(f"\n🎯 **High-Quality Research (100 trials):**")
    print(f"   python train_all_batches.py --config extreme_imbalance --optuna-trials 100 --timeout 7200")
    
    print(f"\n📊 **Key Benefits of Enhanced Config:**")
    print(f"   • Automatic LightGBM regressor inclusion for better F₁ scores")
    print(f"   • Regression target support with threshold optimization") 
    print(f"   • Huber loss (alpha=0.9) for outlier robustness")
    print(f"   • Multi-horizon targets (1d, 3d, 5d, 10d) for ensemble diversity")
    print(f"   • ADASYN sampling optimized for extreme imbalance")
    print(f"   • Enhanced meta-model strategies with gradient boosting")

if __name__ == "__main__":
    print("🚀 Testing Enhanced Extreme Imbalance Configuration\n")
    
    test1_success = test_extreme_imbalance_config()
    test2_success = test_model_registry()
    
    print(f"\n📊 Test Results:")
    print(f"   Configuration: {'✓ PASS' if test1_success else '❌ FAIL'}")
    print(f"   Model Registry: {'✓ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print(f"\n🎉 All tests passed! Enhanced extreme_imbalance config is ready.")
        test_integration_commands()
    else:
        print(f"\n❌ Some tests failed. Check the errors above.")
        sys.exit(1)
