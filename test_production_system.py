#!/usr/bin/env python3
"""
Production Trading System Test
=============================

Test your production trading system with paper trading.
This script demonstrates all the capabilities working together.
"""

import os
import sys
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from run_production_trading import main, get_production_config
import argparse

def run_comprehensive_test():
    """Run a comprehensive test of the trading system."""
    
    print("=" * 80)
    print("🎯 PRODUCTION TRADING SYSTEM TEST")
    print("=" * 80)
    
    # Test with a small portfolio of symbols
    test_symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
    
    print(f"📊 Testing with {len(test_symbols)} symbols: {', '.join(test_symbols)}")
    print("🛡️  Running in PAPER TRADING mode (safe)")
    print("🧪 DRY RUN mode (no actual trades)")
    print()
    
    # Simulate command line arguments
    sys.argv = [
        "test_production_system.py",
        "--symbols"] + test_symbols + [
        "--dry-run",
        "--log-level", "INFO"
    ]
    
    try:
        # Run the main trading system
        main()
        
        print("\n✅ TEST COMPLETED SUCCESSFULLY!")
        print("Your trading system is ready for paper trading.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    
    return True

def show_next_steps():
    """Show next steps for the user."""
    print("\n" + "=" * 80)
    print("🚀 NEXT STEPS FOR PAPER TRADING")
    print("=" * 80)
    
    print("\n1. 🧪 Start with dry-run testing:")
    print("   python run_production_trading.py --dry-run")
    print("   (This shows what trades would be made without executing them)")
    
    print("\n2. 📊 Test with specific symbols:")
    print("   python run_production_trading.py --symbols AAPL MSFT --dry-run")
    
    print("\n3. 🎯 Run actual paper trading:")
    print("   python run_production_trading.py")
    print("   (This will execute trades with paper money)")
    
    print("\n4. 📈 Monitor results:")
    print("   Check results/trading_sessions/latest_session.json")
    print("   Check logs/trading_engine.log")
    
    print("\n5. ⚙️ Customize settings:")
    print("   Edit config/production_config.py")
    print("   Adjust portfolio size, confidence thresholds, etc.")
    
    print("\n⚠️  IMPORTANT SAFETY NOTES:")
    print("   • System defaults to paper trading (safe)")
    print("   • Always test thoroughly before live trading")
    print("   • Monitor performance and adjust as needed")
    print("   • Start with small position sizes")
    
    print("\n🎉 Your system is ready to trade!")

if __name__ == "__main__":
    print("Starting production trading system test...")
    
    success = run_comprehensive_test()
    
    if success:
        show_next_steps()
    else:
        print("\nPlease fix any issues and try again.")
        sys.exit(1)
