#!/usr/bin/env python3
"""
Quick Training Test - Single Batch
==================================
Test the enhanced training pipeline on a single batch with configurable CV folds.
"""

import logging
from enhanced_training_pipeline import EnhancedTrainingPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_test(cv_folds=3, optimization_complexity='quick', timeout_per_model=120):
    """Quick test with configurable parameters."""
    logger.info(f"🚀 Quick Training Test: {cv_folds} CV folds, {optimization_complexity} optimization")
    
    # Create pipeline with custom settings
    pipeline = EnhancedTrainingPipeline(
        use_enhanced_optimization=True,
        use_advanced_meta_learning=False,  # Disable for speed
        generate_comprehensive_plots=False  # Disable for speed
    )
    
    # Override configuration
    pipeline.enhanced_config.update({
        'cross_validation_folds': cv_folds,
        'optimization_complexity': optimization_complexity,
        'timeout_per_model': timeout_per_model,
        'enable_cross_batch_analysis': False,  # Disable for speed
        'enable_production_deployment': False  # Disable for speed
    })
    
    # Test on sample data if available
    import os
    sample_file = "data/leak_free_batch_sample_data.csv"
    if os.path.exists(sample_file):
        logger.info(f"📊 Testing on sample data: {sample_file}")
        batch_data = pipeline._load_batch_data(sample_file)
        
        if batch_data:
            X_train, y_train, X_test, y_test = batch_data
            results = pipeline.train_single_batch_enhanced(
                "quick_test", X_train, y_train, X_test, y_test
            )
            
            if results and results.get('models'):
                logger.info("✅ Quick test completed successfully!")
                logger.info(f"✅ Trained {len(results['models'])} models")
                logger.info(f"✅ Training time: {results.get('training_time', 0):.1f}s")
                return True
        else:
            logger.error("❌ Failed to load sample data")
    else:
        logger.error(f"❌ Sample data not found: {sample_file}")
        logger.info("💡 Run the integration test first to create sample data:")
        logger.info("   python test_enhanced_integration.py")
    
    return False

if __name__ == "__main__":
    import sys
    
    # Default settings
    cv_folds = 3
    optimization_complexity = 'quick'
    timeout_per_model = 120
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        cv_folds = int(sys.argv[1])
    if len(sys.argv) > 2:
        optimization_complexity = sys.argv[2]
    if len(sys.argv) > 3:
        timeout_per_model = int(sys.argv[3])
    
    print(f"""
🎯 Enhanced Training Quick Test
==============================
CV Folds: {cv_folds}
Optimization: {optimization_complexity}
Timeout per model: {timeout_per_model}s

Usage:
  python quick_training_test.py [cv_folds] [complexity] [timeout]
  
Examples:
  python quick_training_test.py 3 quick 120        # Fast test
  python quick_training_test.py 5 balanced 300     # Balanced test  
  python quick_training_test.py 7 intensive 600    # Thorough test
""")
    
    success = quick_test(cv_folds, optimization_complexity, timeout_per_model)
    
    if success:
        print("\n🎉 Ready for full training! Run: python enhanced_training_pipeline.py")
    else:
        print("\n❌ Quick test failed. Check the logs above.")
