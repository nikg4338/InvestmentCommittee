#!/usr/bin/env python3
"""
Test ML-only trading system without LLM to debug confidence issues.
"""

import os
import sys
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.production_config import get_production_config
from trading.execution.alpaca_client import AlpacaClient
from trading.production_trading_engine import ProductionModelEnsemble, ProductionModelConfig

def setup_logging():
    """Setup logging for debugging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def test_ml_models():
    """Test ML models only without LLM integration."""
    print("🧪 Testing ML Models Only (No LLM)")
    print("=" * 50)
    
    # Load configuration
    config = get_production_config()
    config.llm_config.enable_llm = False  # Disable LLM
    
    # Initialize Alpaca client
    alpaca_client = AlpacaClient()  # Uses environment variables automatically
    
    print(f"✅ Connected to Alpaca - Account: {alpaca_client.get_account_info().get('id', 'Unknown')}")
    
    # Initialize model ensemble
    model_config = ProductionModelConfig(
        primary_model_path=config.model_config.primary_model_path,
        backup_models=list(config.model_config.backup_models.values()),
        threshold_config_path="config/optimized_thresholds_batch_1.json"
    )
    
    model_ensemble = ProductionModelEnsemble(model_config)
    print(f"✅ Loaded {len(model_ensemble.models)} models")
    print(f"📊 Model thresholds: {model_ensemble.thresholds}")
    
    # Test symbols
    test_symbols = ["AAPL", "MSFT", "GOOGL"]
    
    for symbol in test_symbols:
        print(f"\n🔍 Testing {symbol}:")
        
        try:
            # Create mock market data with realistic features (bypass Alpaca data for now)
            market_data = {
                'close': 150 + (hash(symbol) % 100),  # Mock price based on symbol
                'volume': 1000000 + (hash(symbol) % 5000000),
                'price_change_1d': (hash(symbol) % 200 - 100) / 1000,  # -0.1 to 0.1
                'price_change_5d': (hash(symbol) % 400 - 200) / 1000,  # -0.2 to 0.2
                'price_change_10d': (hash(symbol) % 600 - 300) / 1000,
                'price_change_20d': (hash(symbol) % 800 - 400) / 1000,
                'price_vs_sma5': 0.98 + (hash(symbol) % 80) / 1000,  # 0.98 to 1.06
                'price_vs_sma10': 0.97 + (hash(symbol) % 90) / 1000,
                'price_vs_sma20': 0.96 + (hash(symbol) % 100) / 1000,
                'volatility_5d': 0.15 + (hash(symbol) % 200) / 1000,  # 0.15 to 0.35
                'volatility_10d': 0.18 + (hash(symbol) % 200) / 1000,
                'volatility_20d': 0.20 + (hash(symbol) % 200) / 1000,
                'volume_ratio': 0.8 + (hash(symbol) % 400) / 1000,  # 0.8 to 1.2
                'hl_ratio': 0.90 + (hash(symbol) % 100) / 1000,  # 0.90 to 1.00
                'hl_ratio_5d': 0.92 + (hash(symbol) % 80) / 1000,
                'rsi_14': 30 + (hash(symbol) % 400) / 10,  # 30 to 70
                'macd': -1 + (hash(symbol) % 200) / 100,  # -1 to 1
                'macd_signal': -0.8 + (hash(symbol) % 160) / 100,
                'macd_histogram': -0.5 + (hash(symbol) % 100) / 100,
                'bb_position': (hash(symbol) % 1000) / 1000  # 0 to 1
            }
            
            # Get model predictions
            predictions = model_ensemble.predict(market_data)
            ensemble_pred, confidence = model_ensemble.ensemble_prediction(predictions)
            
            print(f"  📈 Predictions: {predictions}")
            print(f"  🎯 Ensemble: {ensemble_pred:.3f}")
            print(f"  🔒 Confidence: {confidence:.3f}")
            
            # Check thresholds
            primary_threshold = model_ensemble.thresholds.get('optimized_catboost', 0.6)
            print(f"  📏 Primary threshold: {primary_threshold}")
            
            if confidence > 0.3:  # Using our new lower threshold
                if ensemble_pred >= primary_threshold:
                    action = "BUY"
                elif ensemble_pred < 0.3:
                    action = "SELL"
                else:
                    action = "HOLD"
                print(f"  🎬 Action: {action} (confidence above 0.3)")
            else:
                print(f"  ⏸️  Action: HOLD (confidence too low: {confidence:.3f})")
                
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 ML-only test completed!")

if __name__ == "__main__":
    setup_logging()
    test_ml_models()
