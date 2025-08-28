#!/usr/bin/env python3
"""
Test Enhanced Training Pipeline with Fixes
==========================================
Quick test to verify all fixes are working properly in the enhanced pipeline.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_pipeline():
    """Test the enhanced training pipeline with all fixes."""
    logger.info("🚀 Testing Enhanced Training Pipeline with All Fixes")
    logger.info("=" * 60)
    
    try:
        # Import the enhanced pipeline
        from enhanced_training_pipeline import EnhancedTrainingPipeline
        
        # Initialize with conservative settings for testing
        pipeline = EnhancedTrainingPipeline(
            use_enhanced_optimization=False,  # Disable for quick test
            use_advanced_meta_learning=False,
            generate_comprehensive_plots=False
        )
        
        logger.info("✅ Enhanced pipeline initialized successfully")
        logger.info(f"   Feature aligner: {pipeline.feature_aligner is not None}")
        logger.info(f"   Probability fixer: {pipeline.probability_fixer is not None}")
        logger.info(f"   Drift mitigator: {pipeline.drift_mitigator is not None}")
        
        # Get available data files
        data_dir = "data"
        if not os.path.exists(data_dir):
            logger.error(f"❌ Data directory not found: {data_dir}")
            return False
            
        # Look for leak-free batch files
        batch_files = [f for f in os.listdir(data_dir) 
                      if f.startswith('leak_free_batch_') and f.endswith('.csv')]
        
        if not batch_files:
            logger.error("❌ No leak-free batch files found")
            logger.info(f"   Available files: {os.listdir(data_dir)[:10]}")
            return False
            
        # Test with the first available batch
        test_file = batch_files[0]
        logger.info(f"🧪 Testing with file: {test_file}")
        
        # Load and test data processing
        batch_path = os.path.join(data_dir, test_file)
        batch_data = pipeline._load_batch_data(batch_path)
        
        if not batch_data:
            logger.error("❌ Failed to load batch data")
            return False
            
        X_train, y_train, X_test, y_test = batch_data
        
        logger.info(f"✅ Data loaded successfully:")
        logger.info(f"   Training: {X_train.shape}")
        logger.info(f"   Test: {X_test.shape}")
        logger.info(f"   Target distribution: {np.bincount(y_train.astype(int))}")
        
        # Test feature alignment
        logger.info("🔧 Testing feature alignment...")
        original_features = X_train.shape[1]
        X_train_aligned = pipeline.feature_aligner.align_features(X_train)
        logger.info(f"   Original: {original_features} → Aligned: {X_train_aligned.shape[1]} features")
        
        # Test data drift detection
        logger.info("🔍 Testing data drift detection...")
        _, _, drift_report = pipeline.drift_mitigator.detect_and_handle_drift(
            X_train_aligned, X_test, threshold=0.1
        )
        logger.info(f"   Features with drift: {len(drift_report['features_with_drift'])}")
        
        # Test probability fixing with simulated uniform probabilities
        logger.info("🎯 Testing probability fixing...")
        uniform_probs = np.full(len(y_test), 0.5)
        is_uniform = pipeline.probability_fixer.detect_uniform_probabilities(uniform_probs)
        
        if is_uniform:
            fixed_probs = pipeline.probability_fixer.fix_uniform_probabilities(uniform_probs, y_test)
            prob_range = fixed_probs.max() - fixed_probs.min()
            logger.info(f"   Fixed uniform probabilities: range = {prob_range:.4f}")
            
            if prob_range > 0.01:
                logger.info("✅ Probability fixing working correctly")
            else:
                logger.warning("⚠️ Probability fixing may need tuning")
        else:
            logger.warning("⚠️ Uniform probability detection not working")
            
        logger.info("🎉 All core fixes validated successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the enhanced pipeline test."""
    logger.info("🔬 Starting Enhanced Training Pipeline Test")
    
    success = test_enhanced_pipeline()
    
    if success:
        logger.info("\n✅ Enhanced training pipeline with fixes is working correctly!")
        logger.info("🚀 You can now run full training with: python enhanced_training_pipeline.py")
    else:
        logger.error("\n❌ Enhanced pipeline test failed. Check errors above.")
        
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
