#!/usr/bin/env python3
"""
Simple ML Engine for Testing
============================

A minimal ML engine to test the graduated capital allocation system.
"""

import random
import logging

logger = logging.getLogger(__name__)


class SimpleMlEngine:
    """Simple ML engine that gives varied predictions for testing."""
    
    def __init__(self):
        self.models_loaded = True
        logger.info("Simple ML Engine initialized")
    
    def predict_confidence(self, symbol, features=None):
        """Generate varied confidence predictions instead of constant 46.9%."""
        # Generate realistic confidence values based on symbol characteristics
        base_confidence = 0.4 + (hash(symbol) % 100) / 250  # 0.4 to 0.8
        noise = random.uniform(-0.1, 0.1)
        confidence = max(0.1, min(0.95, base_confidence + noise))
        
        logger.info(f"ML prediction for {symbol}: {confidence:.3f} confidence")
        return confidence
    
    def get_quality_score(self, symbol, technical_data=None):
        """Generate quality scores for testing."""
        base_quality = 0.5 + (hash(symbol + "quality") % 100) / 200  # 0.5 to 1.0
        noise = random.uniform(-0.05, 0.05)
        quality = max(0.3, min(1.0, base_quality + noise))
        
        return quality


def test_ml_engine():
    """Test the simple ML engine."""
    engine = SimpleMlEngine()
    
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    print("Testing ML Engine:")
    for symbol in test_symbols:
        confidence = engine.predict_confidence(symbol)
        quality = engine.get_quality_score(symbol)
        print(f"{symbol}: Confidence={confidence:.3f}, Quality={quality:.3f}")


if __name__ == "__main__":
    test_ml_engine()
