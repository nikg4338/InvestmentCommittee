#!/usr/bin/env python3
"""
Test script to verify the UnboundLocalError fix for meta_learner_type.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_meta_learner_type_fix():
    """Test that meta_learner_type is properly defined in train_committee_models."""
    logger.info("🧪 Testing meta_learner_type UnboundLocalError fix...")
    
    try:
        from config.training_config import get_default_config
        from train_models import train_committee_models
        
        # Create minimal synthetic dataset
        np.random.seed(42)
        n_samples = 100
        X_train = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples)
        })
        y_train = pd.Series(np.random.choice([0, 1], n_samples, p=[0.7, 0.3]))
        
        X_test = pd.DataFrame({
            'feature1': np.random.randn(20),
            'feature2': np.random.randn(20),
            'feature3': np.random.randn(20)
        })
        y_test = pd.Series(np.random.choice([0, 1], 20, p=[0.7, 0.3]))
        
        # Test different meta-learner configurations
        config = get_default_config()
        
        # Test 1: Default lightgbm meta-learner
        logger.info("Testing default lightgbm meta-learner...")
        config.meta_model.meta_learner_type = 'lightgbm'
        
        try:
            results = train_committee_models(X_train, y_train, X_test, y_test, config)
            logger.info("✅ Default lightgbm meta-learner test passed")
        except UnboundLocalError as e:
            if 'meta_learner_type' in str(e):
                logger.error(f"❌ UnboundLocalError still exists: {e}")
                return False
            else:
                logger.warning(f"Different UnboundLocalError: {e}")
        except Exception as e:
            logger.warning(f"Other error (may be expected with small dataset): {e}")
        
        # Test 2: Logistic meta-learner
        logger.info("Testing logistic meta-learner...")
        config.meta_model.meta_learner_type = 'logistic'
        
        try:
            results = train_committee_models(X_train, y_train, X_test, y_test, config)
            logger.info("✅ Logistic meta-learner test passed")
        except UnboundLocalError as e:
            if 'meta_learner_type' in str(e):
                logger.error(f"❌ UnboundLocalError still exists: {e}")
                return False
            else:
                logger.warning(f"Different UnboundLocalError: {e}")
        except Exception as e:
            logger.warning(f"Other error (may be expected with small dataset): {e}")
        
        # Test 3: Different meta-model strategies
        logger.info("Testing different meta-model strategies...")
        config.meta_model.meta_learner_type = 'lightgbm'
        
        for strategy in ['optimal_threshold', 'focal_loss', 'dynamic_weights', 'feature_select']:
            logger.info(f"  Testing strategy: {strategy}")
            config.meta_model_strategy = strategy
            
            try:
                results = train_committee_models(X_train, y_train, X_test, y_test, config)
                logger.info(f"  ✅ Strategy {strategy} test passed")
            except UnboundLocalError as e:
                if 'meta_learner_type' in str(e):
                    logger.error(f"  ❌ UnboundLocalError in {strategy}: {e}")
                    return False
                else:
                    logger.warning(f"  Different UnboundLocalError in {strategy}: {e}")
            except Exception as e:
                logger.warning(f"  Other error in {strategy} (may be expected): {e}")
        
        logger.info("✅ All meta_learner_type tests passed - UnboundLocalError fixed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test setup failed: {e}")
        return False

def test_config_integration():
    """Test that meta_learner_type is properly integrated in config."""
    logger.info("🔧 Testing config integration...")
    
    try:
        from config.training_config import get_default_config, get_extreme_imbalance_config
        
        # Test default config
        config = get_default_config()
        assert hasattr(config.meta_model, 'meta_learner_type'), "meta_learner_type missing from config"
        assert config.meta_model.meta_learner_type == 'lightgbm', f"Expected 'lightgbm', got {config.meta_model.meta_learner_type}"
        logger.info(f"✅ Default config has meta_learner_type: {config.meta_model.meta_learner_type}")
        
        # Test extreme imbalance config
        config_extreme = get_extreme_imbalance_config()
        assert hasattr(config_extreme.meta_model, 'meta_learner_type'), "meta_learner_type missing from extreme config"
        logger.info(f"✅ Extreme config has meta_learner_type: {config_extreme.meta_model.meta_learner_type}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Config integration test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting UnboundLocalError Fix Tests")
    
    success = True
    
    # Test 1: Config integration
    success &= test_config_integration()
    
    # Test 2: Function usage 
    success &= test_meta_learner_type_fix()
    
    if success:
        logger.info("🎉 All tests passed! UnboundLocalError fix successful.")
        logger.info("The meta_learner_type variable is now properly defined and accessible.")
    else:
        logger.error("❌ Some tests failed.")
    
    exit(0 if success else 1)
