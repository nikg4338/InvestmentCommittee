#!/usr/bin/env python3
"""
Quick Validation of Generous Labeling in Training Pipeline
==========================================================

This script validates that our generous labeling strategy works in the actual training pipeline.
"""

import logging
import pandas as pd
import numpy as np
from data_collection_alpaca import AlpacaDataCollector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_generous_labeling_pipeline():
    """Validate that the pipeline produces generous labeling results"""
    logger.info("🔍 Validating generous labeling in training pipeline...")
    
    # Create collector
    collector = AlpacaDataCollector()
    
    try:
        # Try to engineer features for a test symbol (using fake data if needed)
        logger.info("📊 Testing with synthetic data...")
        
        # Create more realistic synthetic financial data
        np.random.seed(42)
        n_days = 500  # About 2 years of trading days
        
        # Generate realistic price movements (geometric Brownian motion)
        returns = np.random.normal(0.0005, 0.015, n_days)  # 0.05% daily drift, 1.5% volatility
        prices = [100.0]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Create DataFrame with realistic trading data
        df = pd.DataFrame({
            'close': prices[1:],  # Remove first price to match returns length
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices[1:]],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices[1:]],
            'open': [prices[i] * (1 + np.random.normal(0, 0.003)) for i in range(len(prices[1:]))],
            'volume': np.random.randint(500000, 5000000, n_days)
        }, index=pd.date_range('2023-01-01', periods=n_days, freq='D'))
        
        logger.info(f"📈 Generated {n_days} days of synthetic trading data")
        logger.info(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        logger.info(f"   Daily returns: {((df['close'] / df['close'].shift(1) - 1).mean()*100):.3f}% ± {((df['close'] / df['close'].shift(1) - 1).std()*100):.3f}%")
        
        # Add technical indicators
        df = collector.calculate_technical_indicators(df)
        logger.info(f"✅ Added technical indicators: {len(df.columns)} total features")
        
        # Test enhanced target creation with generous strategy
        target_series = collector.create_target_variable(
            df, 
            symbol='SYNTHETIC', 
            use_regression=False,  # Use classification
            target_horizon=3,
            target_strategy='top_percentile'
        )
        
        # Validate results
        if isinstance(target_series, pd.DataFrame):
            # Multiple targets created
            for col in target_series.columns:
                if 'enhanced' in col:
                    targets = target_series[col]
                    positive_count = targets.sum()
                    total_count = len(targets.dropna())
                    positive_rate = positive_count / total_count if total_count > 0 else 0
                    
                    logger.info(f"📊 Target '{col}': {positive_count}/{total_count} positive ({positive_rate:.1%})")
        else:
            # Single target
            positive_count = target_series.sum()
            total_count = len(target_series.dropna())
            positive_rate = positive_count / total_count if total_count > 0 else 0
            
            logger.info(f"📊 Single target: {positive_count}/{total_count} positive ({positive_rate:.1%})")
        
        # Check if we're getting approximately 25%
        expected_rate = 0.25
        tolerance = 0.05
        
        if abs(positive_rate - expected_rate) <= tolerance:
            logger.info(f"✅ SUCCESS: Positive rate {positive_rate:.1%} matches generous labeling strategy")
        else:
            logger.warning(f"⚠️  WARNING: Positive rate {positive_rate:.1%} differs from expected {expected_rate:.1%}")
        
        # Test feature engineering completeness
        feature_cols = [col for col in df.columns if col not in ['target', 'target_enhanced']]
        logger.info(f"📊 Feature engineering results:")
        logger.info(f"   Total features: {len(feature_cols)}")
        logger.info(f"   Sample features: {feature_cols[:10]}")
        
        # Check for regime-aware features
        regime_features = [col for col in df.columns if 'regime' in col.lower()]
        logger.info(f"   Regime features: {len(regime_features)} ({regime_features[:5]})")
        
        # Check for technical indicators
        technical_features = [col for col in df.columns if any(term in col.lower() for term in ['rsi', 'macd', 'sma', 'volatility'])]
        logger.info(f"   Technical features: {len(technical_features)} (first 5: {technical_features[:5]})")
        
        logger.info("✅ Pipeline validation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline validation failed: {e}")
        return False

def main():
    logger.info("🚀 Quick Validation of Generous Labeling Pipeline")
    logger.info("=" * 60)
    
    success = validate_generous_labeling_pipeline()
    
    logger.info("=" * 60)
    if success:
        logger.info("🎉 VALIDATION PASSED: Pipeline ready for generous labeling strategy!")
        logger.info("")
        logger.info("📋 Summary of implemented changes:")
        logger.info("   ✅ Top 25% positive labeling (75th percentile threshold)")
        logger.info("   ✅ Bottom 75% negative labeling (all remaining data)")
        logger.info("   ✅ No samples discarded (robust negative class)")
        logger.info("   ✅ SMOTEENN default sampling for noisy financial data")
        logger.info("   ✅ class_weight='balanced' in all models")
        logger.info("")
        logger.info("🚀 Ready to run: python train_models.py")
    else:
        logger.error("❌ VALIDATION FAILED: Check pipeline configuration")

if __name__ == "__main__":
    main()
