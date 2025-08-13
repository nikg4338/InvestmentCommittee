#!/usr/bin/env python3
"""
Quick Test: Verify Trading System Components
===========================================

Test the key components to make sure they work before running the full system.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_alpaca_connection():
    """Test Alpaca API connection."""
    try:
        from trading.execution.alpaca_client import AlpacaClient
        alpaca = AlpacaClient()
        print("✓ Alpaca connection successful")
        return True
    except Exception as e:
        print(f"✗ Alpaca connection failed: {e}")
        return False

def test_trade_closer():
    """Test The Closer functionality."""
    try:
        from trading.trade_closer import TradeCloser
        from trading.execution.alpaca_client import AlpacaClient
        
        alpaca = AlpacaClient()
        closer = TradeCloser(alpaca)
        print("✓ Trade Closer initialization successful")
        return True
    except Exception as e:
        print(f"✗ Trade Closer failed: {e}")
        return False

def test_real_executor():
    """Test Real Alpaca Executor."""
    try:
        from trading.real_alpaca_executor import RealAlpacaExecutor
        from trading.execution.alpaca_client import AlpacaClient
        
        alpaca = AlpacaClient()
        executor = RealAlpacaExecutor(alpaca)
        print("✓ Real Alpaca Executor initialization successful")
        return True
    except Exception as e:
        print(f"✗ Real Alpaca Executor failed: {e}")
        return False

def test_models():
    """Test model loading."""
    try:
        import joblib
        models_dir = 'models/production'
        
        if not os.path.exists(models_dir):
            print(f"✗ Models directory not found: {models_dir}")
            return False
            
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        if not model_files:
            print(f"✗ No model files found in {models_dir}")
            return False
            
        print(f"✓ Found {len(model_files)} model files")
        return True
    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING TRADING SYSTEM COMPONENTS")
    print("=" * 60)
    
    tests = [
        ("Alpaca Connection", test_alpaca_connection),
        ("Trade Closer", test_trade_closer),
        ("Real Executor", test_real_executor),
        ("Models", test_models),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nTesting {name}...")
        success = test_func()
        results.append((name, success))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS:")
    print("=" * 60)
    
    all_passed = True
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{name}: {status}")
        if not success:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed! System should work correctly.")
        print("\nYou can now run:")
        print("  python autonomous_trading_launcher.py")
    else:
        print("✗ Some tests failed. Please fix issues before running the system.")
    
    return all_passed

if __name__ == "__main__":
    main()
