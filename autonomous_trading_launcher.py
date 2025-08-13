#!/usr/bin/env python3
"""
Autonomous Trading System Launcher
=================================

Launches the complete autonomous trading system with The Closer.
Fixed for Windows compatibility (no emojis to prevent Unicode errors).
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from autonomous_committee import AutonomousCommittee
from trading.trade_closer import TradeCloser
from trading.execution.alpaca_client import AlpacaClient

def setup_logging():
    """Set up comprehensive logging."""
    os.makedirs('logs', exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler('logs/autonomous_trading.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Console handler without emojis for Windows compatibility
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Set up root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def main():
    """Main launcher for autonomous trading system."""
    print("=" * 80)
    print("AUTONOMOUS INVESTMENT COMMITTEE")
    print("=" * 80)
    print("Fully automated trading system with intelligent trade management")
    print()
    print("Features:")
    print("• Autonomous trading during market hours (9:30 AM - 4:00 PM ET)")
    print("• ML model-driven bull put spread decisions")
    print("• 'The Closer' - intelligent trade management")
    print("• Automatic profit targets and stop losses")
    print("• Risk management and position sizing")
    print("• Comprehensive logging and performance tracking")
    print("=" * 80)
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Verify Alpaca connection
        logger.info("Verifying Alpaca connection...")
        alpaca = AlpacaClient()
        
        # Start the autonomous committee
        logger.info("Starting Autonomous Investment Committee...")
        committee = AutonomousCommittee()
        
        # Run the system
        asyncio.run(committee.start_autonomous_trading())
        
    except KeyboardInterrupt:
        print("\nWARNING: Shutdown requested by user")
        logger.info("User requested shutdown")
        
    except Exception as e:
        print(f"\nERROR: System error: {e}")
        logger.error(f"System error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
    finally:
        print("\nAutonomous trading system stopped")
        logger.info("Autonomous trading system stopped")

if __name__ == "__main__":
    main()
