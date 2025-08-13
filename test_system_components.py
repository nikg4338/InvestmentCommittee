#!/usr/bin/env python3
"""
System Components Test - WORKING VERSION
=======================================

Test all system components to ensure they work correctly.
"""

import sys
import os
import asyncio
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test all imports."""
    print("=" * 60)
    print("TESTING SYSTEM IMPORTS")
    print("=" * 60)
    
    try:
        # Test Alpaca client
        from trading.execution.alpaca_client import AlpacaClient
        print("✓ AlpacaClient import successful")
        
        # Test Real Executor
        from trading.real_alpaca_executor import RealAlpacaExecutor
        print("✓ RealAlpacaExecutor import successful")
        
        # Test other components
        from trading.portfolio.position_manager import PositionManager
        print("✓ PositionManager import successful")
        
        from trading.strategy.risk_management import RiskManager
        print("✓ RiskManager import successful")
        
        from utils.trade_logger import TradeLogger
        print("✓ TradeLogger import successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_alpaca_connection():
    """Test Alpaca connection."""
    print("\n" + "=" * 60)
    print("TESTING ALPACA CONNECTION")
    print("=" * 60)
    
    try:
        from trading.execution.alpaca_client import AlpacaClient
        
        alpaca = AlpacaClient()
        print("✓ Alpaca client created successfully")
        print(f"✓ Paper trading enabled: {alpaca.paper_trading}")
        
        return True
        
    except Exception as e:
        print(f"✗ Alpaca connection error: {e}")
        return False

async def test_real_executor():
    """Test the real executor."""
    print("\n" + "=" * 60)
    print("TESTING REAL EXECUTOR")
    print("=" * 60)
    
    try:
        from trading.execution.alpaca_client import AlpacaClient
        from trading.real_alpaca_executor import RealAlpacaExecutor
        
        alpaca = AlpacaClient()
        executor = RealAlpacaExecutor(alpaca)
        print("✓ RealAlpacaExecutor created successfully")
        
        # Test a demo trade execution
        print("\nTesting demo trade execution...")
        success, details = await executor.execute_bull_put_spread(
            symbol="SPY",
            short_strike=145.0,
            long_strike=140.0,
            expiration_date="2025-09-19",
            contracts=1
        )
        
        if success:
            print("✓ Demo trade execution successful")
            print(f"  Order ID: {details.get('order_id')}")
            print(f"  Credit: ${details.get('filled_price', 0):.2f}")
            print(f"  Entry Time: {details.get('entry_time')}")
        else:
            print(f"✗ Demo trade execution failed: {details.get('error')}")
            
        return success
        
    except Exception as e:
        print(f"✗ Real executor error: {e}")
        return False

def test_working_trade_closer():
    """Test the working trade closer."""
    print("\n" + "=" * 60)
    print("TESTING WORKING TRADE CLOSER")
    print("=" * 60)
    
    try:
        # Import from the working file
        sys.path.insert(0, os.getcwd())
        from trade_closer_working import TradeCloser
        
        # Mock alpaca client
        class MockAlpaca:
            pass
        
        alpaca = MockAlpaca()
        closer = TradeCloser(alpaca)
        print("✓ TradeCloser created successfully")
        
        # Test registering a trade
        trade_id = f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        success = closer.register_trade(
            trade_id=trade_id,
            symbol="SPY",
            trade_type="bull_put_spread",
            entry_time=datetime.now(),
            profit_target=0.50,
            stop_loss=-200.0,
            time_decay_close=7,
            trade_params={'test': 'data'},
            entry_price=150.0
        )
        
        if success:
            print("✓ Trade registration successful")
            print(f"  Trade ID: {trade_id}")
        else:
            print("✗ Trade registration failed")
            
        return success
        
    except Exception as e:
        print(f"✗ Trade closer error: {e}")
        return False

def test_working_autonomous_committee():
    """Test the working autonomous committee."""
    print("\n" + "=" * 60)
    print("TESTING WORKING AUTONOMOUS COMMITTEE")
    print("=" * 60)
    
    try:
        # Import from the working file
        sys.path.insert(0, os.getcwd())
        from autonomous_committee_working import AutonomousCommittee
        
        committee = AutonomousCommittee()
        print("✓ AutonomousCommittee created successfully")
        print(f"  Models loaded: {len(committee.models)}")
        print(f"  Symbols loaded: {len(committee.symbols)}")
        print(f"  Market timezone: {committee.config.timezone}")
        
        return True
        
    except Exception as e:
        print(f"✗ Autonomous committee error: {e}")
        return False

async def run_all_tests():
    """Run all tests."""
    print("AUTONOMOUS TRADING SYSTEM - COMPONENT TESTS")
    print("=" * 80)
    
    results = []
    
    # Test imports
    results.append(test_imports())
    
    # Test Alpaca connection
    results.append(test_alpaca_connection())
    
    # Test real executor
    results.append(await test_real_executor())
    
    # Test working trade closer
    results.append(test_working_trade_closer())
    
    # Test working autonomous committee
    results.append(test_working_autonomous_committee())
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ ALL TESTS PASSED - System is ready for autonomous trading!")
        print("\nTo start autonomous trading tomorrow:")
        print("1. python autonomous_committee_working.py")
        print("2. python -m streamlit run trading_dashboard.py")
    else:
        print("✗ Some tests failed - please review errors above")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(run_all_tests())
