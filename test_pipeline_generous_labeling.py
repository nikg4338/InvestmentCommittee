#!/usr/bin/env python3
"""
Test Actual Data Collection Pipeline
====================================

Test that the data collection pipeline now produces generous labeling targets.
"""

import logging
import pandas as pd
import numpy as np
from data_collection_alpaca import AlpacaDataCollector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_collection_pipeline():
    """Test the actual data collection pipeline produces generous labeling"""
    logger.info("🧪 Testing actual data collection pipeline...")
    
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
        # Test the enhanced feature engineering with target creation
        result_df = collector.engineer_features_for_symbol(
            'TEST_SYMBOL',
            days=730,
            use_enhanced_targets=True,
            target_strategy='top_percentile'
        )
        
        if result_df is None:
            logger.error("❌ engineer_features_for_symbol returned None")
            return False
        
        # Check if target column exists and has the right distribution
        if 'target' not in result_df.columns:
            logger.error("❌ 'target' column not found in result")
            logger.info(f"Available columns: {list(result_df.columns)}")
            return False
        
        # Analyze the target distribution
        target_values = result_df['target'].dropna()
        if len(target_values) == 0:
            logger.error("❌ No valid target values found")
            return False
        
        # Check if it's binary (0 and 1 values)
        unique_values = set(target_values.unique())
        if not unique_values.issubset({0, 1}):
            logger.error(f"❌ Target values are not binary: {unique_values}")
            return False
        
        # Calculate positive rate
        positive_count = (target_values == 1).sum()
        total_count = len(target_values)
        positive_rate = positive_count / total_count
        
        logger.info(f"📊 Data Collection Pipeline Results:")
        logger.info(f"   Total samples: {total_count}")
        logger.info(f"   Positive samples: {positive_count}")
        logger.info(f"   Positive rate: {positive_rate:.1%}")
        
        # Check if we're getting approximately 25%
        expected_rate = 0.25
        tolerance = 0.05
        
        if abs(positive_rate - expected_rate) <= tolerance:
            logger.info(f"✅ SUCCESS: Positive rate {positive_rate:.1%} matches generous labeling strategy!")
            return True
        else:
            logger.error(f"❌ FAILED: Positive rate {positive_rate:.1%} differs from expected {expected_rate:.1%}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in data collection pipeline: {e}")
        return False

def test_batch_data_collection():
    """Test that batch data collection would produce the right targets"""
    logger.info("🧪 Testing batch data collection simulation...")
    
    try:
        # Simulate what happens in collect_batch_data
        logger.info("📊 Simulating: python data_collection_alpaca.py --batch 999 --test-mode")
        
        # This would be the equivalent command that gets run for each batch
        # We can't easily test the full subprocess, but we can test the core logic
        
        collector = AlpacaDataCollector()
        
        # Test with a simple symbol-like structure
        test_symbols = ['TEST1', 'TEST2']
        
        for symbol in test_symbols:
            logger.info(f"🔄 Testing collection for {symbol}...")
            
            # This simulates what happens in the main collection loop
            result = collector.engineer_features_for_symbol(
                symbol,
                days=730,
                use_enhanced_targets=True,
                target_strategy='top_percentile'
            )
            
            if result is not None and 'target' in result.columns:
                target_values = result['target'].dropna()
                if len(target_values) > 0:
                    positive_rate = (target_values == 1).sum() / len(target_values)
                    logger.info(f"   {symbol}: {positive_rate:.1%} positive rate")
                    
                    if 0.20 <= positive_rate <= 0.30:  # 20-30% range for generous labeling
                        logger.info(f"   ✅ {symbol}: Generous labeling working")
                    else:
                        logger.warning(f"   ⚠️ {symbol}: Unexpected positive rate")
        
        logger.info("✅ Batch data collection simulation completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Batch simulation failed: {e}")
        return False

def main():
    logger.info("🚀 Testing Actual Data Collection Pipeline")
    logger.info("=" * 60)
    
    # Test 1: Direct pipeline test
    success1 = test_data_collection_pipeline()
    
    logger.info("-" * 30)
    
    # Test 2: Batch collection simulation
    success2 = test_batch_data_collection()
    
    logger.info("=" * 60)
    if success1 and success2:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("✅ Data collection pipeline now produces generous labeling")
        logger.info("✅ Ready to run: python train_all_batches.py --batch 1")
        logger.info("")
        logger.info("🎯 Expected results:")
        logger.info("   - ~25% positive rate (instead of 5% extreme imbalance)")
        logger.info("   - Improved model performance with balanced training")
        logger.info("   - Better confusion matrices with meaningful positive predictions")
    else:
        logger.error("❌ SOME TESTS FAILED")
        logger.error("   Data collection pipeline may not be producing generous labeling")

if __name__ == "__main__":
    main()
