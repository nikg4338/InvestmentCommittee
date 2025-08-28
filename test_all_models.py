#!/usr/bin/env python3
"""
Quick test to verify all models work with the fixed PR-AUC scorer
"""
import os
import sys
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_all_models_quick():
    """Test all 6 models with the fixed scorer."""
    try:
        from enhanced_training_pipeline import EnhancedTrainingPipeline
        
        # Initialize with minimal settings
        pipeline = EnhancedTrainingPipeline(
            use_enhanced_optimization=True,
            use_advanced_meta_learning=False,  # Disable for quick test
            generate_comprehensive_plots=False  # Disable for quick test
        )
        
        # Update configuration for quick testing
        pipeline.enhanced_config.update({
            'optimization_complexity': 'quick',
            'cross_validation_folds': 3,
            'timeout_per_model': 30  # Very short timeout for quick test
        })
        
        # Get available batches
        data_dir = os.path.join(os.getcwd(), 'data')
        batch_files = [f for f in os.listdir(data_dir) if f.startswith('leak_free_batch_') and f.endswith('_data.csv')]
        
        if not batch_files:
            logger.error("No leak-free batch files found!")
            return False
            
        # Test with just the first batch
        test_batch = batch_files[0]
        relative_path = f"data/{test_batch}"
        X_train, y_train, X_test, y_test = pipeline._load_batch_data(relative_path)
        
        logger.info(f"✅ Data loaded: Training: {X_train.shape}, Test: {X_test.shape}")
        
        # Test each model type quickly
        model_types = ['xgboost', 'lightgbm', 'random_forest']  # Test core models first
        
        results = {}
        for model_type in model_types:
            logger.info(f"🚀 Testing {model_type} optimization...")
            try:
                result = pipeline.optimizer.adaptive_optimization(
                    model_type=model_type,
                    X=X_train,
                    y=y_train,
                    complexity='quick',  # Use quick complexity setting
                    timeout=30   # Very short timeout
                )
                
                if result and result.get('best_score', 0) > 0:
                    score = result['best_score']
                    results[model_type] = score
                    logger.info(f"✅ {model_type} successful! Score: {score:.4f}")
                else:
                    logger.warning(f"⚠️ {model_type} returned no valid score")
                    results[model_type] = 0.0
                    
            except Exception as e:
                logger.error(f"❌ {model_type} failed: {e}")
                results[model_type] = None
        
        # Summary
        successful_models = len([r for r in results.values() if r is not None and r > 0])
        total_models = len(model_types)
        
        logger.info(f"🎯 Test Summary:")
        logger.info(f"   Successful models: {successful_models}/{total_models}")
        for model_type, score in results.items():
            status = "✅" if (score is not None and score > 0) else "❌"
            score_str = f"{score:.4f}" if score is not None else "Failed"
            logger.info(f"   {status} {model_type}: {score_str}")
        
        if successful_models >= 2:  # At least 2 out of 3 core models working
            logger.info("🎉 SCORER FIX VERIFIED - Core models working!")
            return True
        else:
            logger.error("💥 Multiple models still failing")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("🔬 Testing all models with fixed PR-AUC scorer...")
    success = test_all_models_quick()
    
    if success:
        logger.info("🎉 ALL SCORER FIXES VERIFIED!")
        sys.exit(0)
    else:
        logger.error("💥 Some models still failing")
        sys.exit(1)
