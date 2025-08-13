#!/usr/bin/env python3
"""
Simple test to verify the trading engine components work individually.
"""

import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.production_config import ProductionConfig
from trading.execution.alpaca_client import AlpacaClient
from trading.production_trading_engine import ProductionModelEnsemble, ProductionModelConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test if models load correctly."""
    logger.info("Testing model loading...")
    
    try:
        config = ProductionModelConfig()
        ensemble = ProductionModelEnsemble(config)
        logger.info(f"Models loaded: {list(ensemble.models.keys())}")
        logger.info(f"Thresholds: {ensemble.thresholds}")
        return True
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return False

def test_alpaca_connection():
    """Test Alpaca API connection."""
    logger.info("Testing Alpaca connection...")
    
    try:
        client = AlpacaClient()
        account_info = client.get_account_info()
        logger.info(f"Account ID: {account_info.get('id', 'Unknown')}")
        return True
    except Exception as e:
        logger.error(f"Alpaca connection failed: {e}")
        return False

def test_basic_prediction():
    """Test basic model prediction."""
    logger.info("Testing basic prediction...")
    
    try:
        config = ProductionModelConfig()
        ensemble = ProductionModelEnsemble(config)
        
        # Create sample market data
        sample_data = {
            'price_change_1d': 0.01,
            'price_change_5d': 0.05,
            'price_change_10d': 0.03,
            'price_change_20d': 0.08,
            'price_vs_sma5': 0.02,
            'price_vs_sma10': 0.01,
            'price_vs_sma20': -0.01,
            'volatility_5d': 0.20,
            'volatility_10d': 0.25,
            'volatility_20d': 0.30,
            'volume_ratio': 1.2,
            'hl_ratio': 0.02,
            'hl_ratio_5d': 0.025,
            'rsi_14': 55.0,
            'macd': 0.1,
            'macd_signal': 0.08,
            'macd_histogram': 0.02,
            'bb_position': 0.6
        }
        
        predictions = ensemble.predict(sample_data)
        ensemble_pred, confidence = ensemble.ensemble_prediction(predictions)
        
        logger.info(f"Predictions: {predictions}")
        logger.info(f"Ensemble: {ensemble_pred:.3f}, Confidence: {confidence:.3f}")
        return True
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("TRADING SYSTEM COMPONENT TESTS")
    print("=" * 60)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Alpaca Connection", test_alpaca_connection),
        ("Basic Prediction", test_basic_prediction)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "PASS" if result else "FAIL"
            print(f"{test_name}: {status}")
        except Exception as e:
            results.append((test_name, False))
            print(f"{test_name}: FAIL - {e}")
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed

if __name__ == "__main__":
    main()
