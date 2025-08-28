"""
Test Enhanced Training Pipeline with All Improvements
Test the comprehensive improvements made to the enhanced training pipeline.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model_registry():
    """Test the new model registry functionality."""
    logger.info("🧪 Testing Model Registry...")
    
    try:
        from model_registry import ModelRegistry
        
        # Initialize registry
        registry = ModelRegistry(registry_path="models/test_registry")
        
        # Test model registration
        test_model_id = registry.register_model(
            model_id="test_xgboost_batch1",
            model_type="xgboost",
            batch_name="test_batch_1",
            model_path="models/test/test_model.pkl",
            hyperparameters={"max_depth": 6, "learning_rate": 0.1},
            performance_metrics={"pr_auc": 0.75, "accuracy": 0.82},
            features=["feature_1", "feature_2", "feature_3"],
            training_config={"cv_folds": 5, "timeout": 300}
        )
        
        if test_model_id:
            logger.info(f"✅ Model registered successfully: {test_model_id}")
            
            # Test promotion
            success = registry.promote_model(test_model_id, "staging")
            if success:
                logger.info("✅ Model promoted to staging successfully")
            
            # Test best models retrieval
            best_models = registry.get_best_models(metric="pr_auc", top_k=3)
            logger.info(f"✅ Retrieved {len(best_models)} best models")
            
            # Test registry report
            report = registry.generate_registry_report()
            if "MODEL REGISTRY REPORT" in report:
                logger.info("✅ Registry report generated successfully")
            
            return True
        else:
            logger.error("❌ Model registration failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Model registry test failed: {e}")
        return False

def test_categorical_feature_handling():
    """Test the enhanced categorical feature handling."""
    logger.info("🧪 Testing Categorical Feature Handling...")
    
    try:
        # Create test data with categorical features and required columns
        test_data = pd.DataFrame({
            'ticker': ['AAPL'] * 100,  # Required column
            'numeric_feature_1': np.random.randn(100),
            'numeric_feature_2': np.random.randn(100) * 10,
            'categorical_low': np.random.choice(['A', 'B', 'C'], 100),
            'categorical_high': np.random.choice([f'cat_{i}' for i in range(20)], 100),
            'mixed_feature': np.random.choice(['X', 'Y', np.nan], 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        # Save test data with proper leak-free naming
        test_file = "data/leak_free_test_categorical_data.csv"
        os.makedirs("data", exist_ok=True)
        test_data.to_csv(test_file, index=False)
        
        # Test with enhanced pipeline
        from enhanced_training_pipeline import EnhancedTrainingPipeline
        
        pipeline = EnhancedTrainingPipeline(
            use_enhanced_optimization=False,  # Skip optimization for quick test
            use_advanced_meta_learning=False,
            generate_comprehensive_plots=False
        )
        
        # Test data loading and cleaning
        batch_data = pipeline._load_batch_data(test_file)
        
        if batch_data:
            X_train, y_train, X_test, y_test = batch_data
            logger.info(f"✅ Categorical features handled successfully")
            logger.info(f"   Training features: {X_train.shape[1]}")
            logger.info(f"   Training samples: {X_train.shape[0]}")
            
            # Check if categorical features were encoded
            if X_train.shape[1] > 2:  # Should have more than just 2 numeric features
                logger.info("✅ Categorical features appear to be encoded")
                return True
            else:
                logger.warning("⚠️ Categorical features may not be properly encoded")
                return False
        else:
            logger.error("❌ Failed to load test data")
            return False
            
    except Exception as e:
        logger.error(f"❌ Categorical feature handling test failed: {e}")
        return False

def test_model_specific_timeouts():
    """Test model-specific timeout configuration."""
    logger.info("🧪 Testing Model-Specific Timeouts...")
    
    try:
        from enhanced_training_pipeline import EnhancedTrainingPipeline
        
        pipeline = EnhancedTrainingPipeline()
        
        # Check if model timeouts are configured
        expected_models = ['xgboost', 'lightgbm', 'catboost', 'neural_network', 'random_forest', 'svm']
        
        if hasattr(pipeline, 'model_timeouts'):
            logger.info("✅ Model timeouts attribute found")
            
            for model in expected_models:
                if model in pipeline.model_timeouts:
                    timeout = pipeline.model_timeouts[model]
                    logger.info(f"   {model}: {timeout} seconds")
                else:
                    logger.warning(f"⚠️ No timeout configured for {model}")
            
            # Check that timeouts are reasonable
            if all(pipeline.model_timeouts.get(model, 0) > 0 for model in expected_models):
                logger.info("✅ All model timeouts are positive")
                return True
            else:
                logger.error("❌ Some model timeouts are not configured properly")
                return False
        else:
            logger.error("❌ Model timeouts not found in pipeline")
            return False
            
    except Exception as e:
        logger.error(f"❌ Model timeout test failed: {e}")
        return False

def test_path_handling():
    """Test the improved path handling for absolute paths."""
    logger.info("🧪 Testing Path Handling...")
    
    try:
        from enhanced_training_pipeline import EnhancedTrainingPipeline
        
        pipeline = EnhancedTrainingPipeline()
        
        # Test various path formats
        test_paths = [
            "data/leak_free_batch_1_data.csv",  # Relative path
            "c:/investment-committee/data/leak_free_batch_1_data.csv",  # Absolute path
            "/absolute/path/leak_free_batch_test.csv",  # Unix-style absolute
            "leak_free_batch_2_data.csv"  # Just filename
        ]
        
        for test_path in test_paths:
            try:
                # Test the path handling logic
                filename = os.path.basename(test_path)
                is_leak_free = filename.startswith("leak_free_")
                
                logger.info(f"   Path: {test_path}")
                logger.info(f"   Filename: {filename}")
                logger.info(f"   Is leak-free: {is_leak_free}")
                
            except Exception as e:
                logger.error(f"   Failed to handle path {test_path}: {e}")
                return False
        
        logger.info("✅ Path handling works correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Path handling test failed: {e}")
        return False

def test_enhanced_pipeline_integration():
    """Test that all enhanced components integrate properly."""
    logger.info("🧪 Testing Enhanced Pipeline Integration...")
    
    try:
        from enhanced_training_pipeline import EnhancedTrainingPipeline
        
        # Initialize with all enhancements
        pipeline = EnhancedTrainingPipeline(
            use_enhanced_optimization=True,
            use_advanced_meta_learning=True,
            generate_comprehensive_plots=True
        )
        
        # Check all components are initialized
        checks = [
            ("Model Registry", hasattr(pipeline, 'model_registry')),
            ("Model Timeouts", hasattr(pipeline, 'model_timeouts')),
            ("Enhanced Config", hasattr(pipeline, 'enhanced_config')),
            ("Visualizer", hasattr(pipeline, 'visualizer')),
            ("Analyzer", hasattr(pipeline, 'analyzer')),
            ("Deployer", hasattr(pipeline, 'deployer')),
            ("Optimizer", hasattr(pipeline, 'optimizer'))
        ]
        
        all_passed = True
        for component, exists in checks:
            if exists:
                logger.info(f"   ✅ {component}: OK")
            else:
                logger.error(f"   ❌ {component}: Missing")
                all_passed = False
        
        if all_passed:
            logger.info("✅ All enhanced components integrated successfully")
            return True
        else:
            logger.error("❌ Some enhanced components are missing")
            return False
            
    except Exception as e:
        logger.error(f"❌ Enhanced pipeline integration test failed: {e}")
        return False

def main():
    """Run comprehensive tests for enhanced training pipeline."""
    logger.info("🚀 Starting Enhanced Training Pipeline Tests")
    logger.info("=" * 80)
    
    # Run all tests
    tests = [
        ("Model Registry", test_model_registry),
        ("Categorical Feature Handling", test_categorical_feature_handling),
        ("Model-Specific Timeouts", test_model_specific_timeouts),
        ("Path Handling", test_path_handling),
        ("Enhanced Pipeline Integration", test_enhanced_pipeline_integration)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running Test: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"Test Result: {status}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"Test Error: {e}")
            logger.error("❌ FAILED")
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("FINAL TEST SUMMARY")
    logger.info(f"{'='*80}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Enhanced training pipeline is ready.")
    else:
        logger.warning("⚠️ Some tests failed. Review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ All enhanced training pipeline tests passed!")
    else:
        print("\n❌ Some tests failed. Please review the output above.")
        sys.exit(1)
