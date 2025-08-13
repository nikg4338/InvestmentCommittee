#!/usr/bin/env python3
"""
Test Options Scanner - Verify Alpaca Options-Enabled Stocks
==========================================================

Test script to verify the options scanning functionality.
"""

import sys
import os

# Add trading module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'trading'))

from trading.execution.alpaca_client import AlpacaClient
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_options_scanner():
    """Test the options scanning functionality."""
    try:
        print("🔍 Testing Options Scanner...")
        print("=" * 50)
        
        # Initialize Alpaca client
        alpaca = AlpacaClient()
        print(f"✅ Alpaca client initialized successfully")
        
        # Test getting options-enabled stocks
        print("\n📊 Scanning for options-enabled stocks...")
        options_stocks = alpaca.get_options_enabled_stocks(limit=50)  # Limit to 50 for testing
        
        print(f"✅ Found {len(options_stocks)} options-enabled stocks")
        print(f"📝 First 10 symbols: {options_stocks[:10]}")
        
        # Test individual symbol checking
        print("\n🔍 Testing individual symbol checks...")
        test_symbols = ['AAPL', 'MSFT', 'TSLA', 'SPY', 'QQQ']
        
        for symbol in test_symbols:
            is_optionable = alpaca.is_options_enabled(symbol)
            status = "✅ Options Enabled" if is_optionable else "❌ No Options"
            print(f"  {symbol}: {status}")
        
        # Test filtering a list
        print(f"\n🎯 Testing symbol list filtering...")
        test_list = ['AAPL', 'MSFT', 'INVALID_SYMBOL', 'TSLA', 'GOOGL', 'META']
        optionable_from_list = alpaca.get_optionable_symbols_from_list(test_list)
        
        print(f"📝 Input symbols: {test_list}")
        print(f"✅ Optionable symbols: {optionable_from_list}")
        
        print("\n🎉 Options scanner test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing options scanner: {e}")
        logger.error(f"Options scanner test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_options_scanner()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
