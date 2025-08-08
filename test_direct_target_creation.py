#!/usr/bin/env python3
"""
Direct Target Creation Test
===========================

Test the target creation logic directly without needing real market data.
"""

import logging
import pandas as pd
import numpy as np
from data_collection_alpaca import AlpacaDataCollector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_target_creation_directly():
    """Test target creation logic directly"""
    logger.info("🧪 Testing target creation logic directly...")
    
    # Create collector
    collector = AlpacaDataCollector()
    
    # Create realistic synthetic data
    np.random.seed(42)
    n_days = 500
    
    # Generate realistic price movements
    returns = np.random.normal(0.0005, 0.015, n_days)
    prices = [100.0]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Create DataFrame with realistic trading data
    df = pd.DataFrame({
        'close': prices[1:],
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices[1:]],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices[1:]],
        'open': [prices[i] * (1 + np.random.normal(0, 0.003)) for i in range(len(prices[1:]))],
        'volume': np.random.randint(500000, 5000000, n_days)
    }, index=pd.date_range('2023-01-01', periods=n_days, freq='D'))
    
    logger.info(f"📈 Generated {n_days} days of synthetic trading data")
    
    try:
        # Test 1: Enhanced target creation (what should happen now)
        logger.info("🔍 Test 1: Enhanced target creation with generous labeling")
        
        target_results = collector.create_target_variable(
            df, 
            symbol='TEST', 
            use_regression=False,  # Classification mode
            create_all_horizons=True,  # Multiple targets
            target_strategy='top_percentile'  # Generous labeling
        )
        
        if isinstance(target_results, pd.DataFrame):
            logger.info(f"✅ Created multiple targets: {list(target_results.columns)}")
            
            # Check the 3-day enhanced target specifically
            if 'target_3d_enhanced' in target_results.columns:
                target_values = target_results['target_3d_enhanced'].dropna()
                positive_count = (target_values == 1).sum()
                total_count = len(target_values)
                positive_rate = positive_count / total_count
                
                logger.info(f"📊 target_3d_enhanced: {positive_count}/{total_count} positive ({positive_rate:.1%})")
                
                if 0.20 <= positive_rate <= 0.30:
                    logger.info("✅ Test 1 PASSED: Generous labeling working in target_3d_enhanced")
                else:
                    logger.error(f"❌ Test 1 FAILED: Expected ~25%, got {positive_rate:.1%}")
                    return False
            else:
                logger.error("❌ Test 1 FAILED: target_3d_enhanced not found")
                return False
        else:
            logger.error("❌ Test 1 FAILED: Expected DataFrame, got Series")
            return False
        
        # Test 2: Single target creation
        logger.info("🔍 Test 2: Single target creation with generous labeling")
        
        single_target = collector.create_target_variable(
            df, 
            symbol='TEST', 
            use_regression=False,  # Classification mode
            create_all_horizons=False,  # Single target
            target_strategy='top_percentile'  # Generous labeling
        )
        
        if isinstance(single_target, pd.Series):
            target_values = single_target.dropna()
            positive_count = (target_values == 1).sum()
            total_count = len(target_values)
            positive_rate = positive_count / total_count
            
            logger.info(f"📊 Single target: {positive_count}/{total_count} positive ({positive_rate:.1%})")
            
            if 0.20 <= positive_rate <= 0.30:
                logger.info("✅ Test 2 PASSED: Generous labeling working in single target")
            else:
                logger.error(f"❌ Test 2 FAILED: Expected ~25%, got {positive_rate:.1%}")
                return False
        else:
            logger.error("❌ Test 2 FAILED: Expected Series, got DataFrame")
            return False
        
        # Test 3: Verify the enhanced target assignment logic
        logger.info("🔍 Test 3: Testing enhanced target assignment logic")
        
        # Simulate what happens in engineer_features_for_symbol
        if isinstance(target_results, pd.DataFrame):
            enhanced_target = 'target_3d_enhanced'
            
            if enhanced_target in target_results.columns:
                primary_target = target_results[enhanced_target]
                target_values = primary_target.dropna()
                positive_count = (target_values == 1).sum()
                total_count = len(target_values)
                positive_rate = positive_count / total_count
                
                logger.info(f"📊 Primary target assignment: {positive_count}/{total_count} positive ({positive_rate:.1%})")
                
                if 0.20 <= positive_rate <= 0.30:
                    logger.info("✅ Test 3 PASSED: Primary target assignment working")
                    return True
                else:
                    logger.error(f"❌ Test 3 FAILED: Expected ~25%, got {positive_rate:.1%}")
                    return False
            else:
                logger.error("❌ Test 3 FAILED: Enhanced target not found for assignment")
                return False
        
    except Exception as e:
        logger.error(f"❌ Error in target creation test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_old_vs_new_comparison():
    """Compare old extreme imbalance vs new generous labeling"""
    logger.info("🔍 Comparing old vs new labeling strategies...")
    
    collector = AlpacaDataCollector()
    
    # Create test data
    np.random.seed(42)
    n_days = 1000
    returns = np.random.normal(0.0005, 0.015, n_days)
    prices = [100.0]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    df = pd.DataFrame({
        'close': prices[1:],
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices[1:]],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices[1:]],
        'open': [prices[i] * (1 + np.random.normal(0, 0.003)) for i in range(len(prices[1:]))],
        'volume': np.random.randint(500000, 5000000, n_days)
    }, index=pd.date_range('2023-01-01', periods=n_days, freq='D'))
    
    # Test old extreme imbalance (95th percentile = 5% positive)
    forward_returns = (df['close'].shift(-3) / df['close']) - 1
    old_threshold = forward_returns.quantile(0.95)  # 95th percentile
    old_positives = (forward_returns >= old_threshold).sum()
    old_rate = old_positives / len(forward_returns.dropna())
    
    # Test new generous labeling (75th percentile = 25% positive)
    new_threshold = forward_returns.quantile(0.75)  # 75th percentile
    new_positives = (forward_returns >= new_threshold).sum()
    new_rate = new_positives / len(forward_returns.dropna())
    
    logger.info("📊 Comparison Results:")
    logger.info(f"   Old strategy (95th percentile): {old_positives} positives ({old_rate:.1%})")
    logger.info(f"   New strategy (75th percentile): {new_positives} positives ({new_rate:.1%})")
    logger.info(f"   Improvement: {new_rate/old_rate:.1f}x more positive samples")
    
    return True

def main():
    logger.info("🚀 Direct Target Creation Test")
    logger.info("=" * 50)
    
    # Test target creation logic
    success1 = test_target_creation_directly()
    
    logger.info("-" * 30)
    
    # Compare strategies
    success2 = test_old_vs_new_comparison()
    
    logger.info("=" * 50)
    if success1 and success2:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("✅ Generous labeling is working correctly")
        logger.info("✅ Target creation produces ~25% positive rate")
        logger.info("")
        logger.info("🚀 Ready to train batch 1 with generous labeling:")
        logger.info("   python train_all_batches.py --batch 1")
        logger.info("")
        logger.info("Expected improvements:")
        logger.info("   - ~25% positive rate instead of ~5%")
        logger.info("   - Better model learning with more positive examples")
        logger.info("   - More balanced confusion matrices")
        logger.info("   - Improved F1 scores and precision/recall balance")
    else:
        logger.error("❌ TESTS FAILED")

if __name__ == "__main__":
    main()
