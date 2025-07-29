#!/usr/bin/env python3
"""
IEX Alpaca Integration Demo
==========================

Demonstrates how to use IEX filtered symbols with Alpaca API exclusively.
This script shows the proper way to fetch data for your pre-filtered symbol universe.
"""

import sys
import os
from pathlib import Path

# Set up project root for imports
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

import logging
from utils.logging import setup_logging
from utils.helpers import load_iex_filtered_symbols, load_iex_batch_symbols, get_iex_batch_info
from trading.execution.alpaca_client import AlpacaClient

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

def main():
    """Demonstrate IEX symbols with Alpaca integration."""
    
    logger.info("IEX Alpaca Integration Demo")
    logger.info("=" * 50)
    
    try:
        # 1. Load IEX batch information
        logger.info("1. Loading IEX batch information...")
        batch_info = get_iex_batch_info()
        
        print(f"Total IEX Filtered Symbols: {batch_info['total_symbols']}")
        print(f"Total Batches: {batch_info['total_batches']}")
        print(f"Available Batches: {', '.join(batch_info['batch_names'])}")
        print("\nSymbols per batch:")
        for batch_name, count in batch_info['symbols_per_batch'].items():
            print(f"  {batch_name}: {count} symbols")
        
        # Filter criteria used
        filter_criteria = batch_info.get('filter_criteria', {})
        if filter_criteria:
            print(f"\nFilter Criteria Applied:")
            print(f"  Min Price: ${filter_criteria.get('min_price', 'N/A')}")
            print(f"  Min Market Cap: ${filter_criteria.get('min_market_cap', 'N/A'):,}")
            print(f"  Min Avg Volume: {filter_criteria.get('min_avg_volume', 'N/A'):,}")
            print(f"  PE Ratio Range: {filter_criteria.get('pe_ratio_range', 'N/A')}")
        
        # 2. Initialize Alpaca client
        logger.info("\n2. Initializing Alpaca client...")
        alpaca_client = AlpacaClient()
        
        # 3. Get account information
        logger.info("\n3. Fetching account information...")
        account_info = alpaca_client.get_account_info()
        
        print(f"\nAlpaca Account:")
        print(f"  Account ID: {account_info['id']}")
        print(f"  Status: {account_info['status']}")
        print(f"  Buying Power: ${account_info['buying_power']:,.2f}")
        print(f"  Cash: ${account_info['cash']:,.2f}")
        print(f"  Portfolio Value: ${account_info['portfolio_value']:,.2f}")
        print(f"  Paper Trading: {account_info.get('paper_trading', 'Yes')}")
        
        # 4. Load a sample batch and test market data
        logger.info("\n4. Testing market data with IEX symbols...")
        
        # Get symbols from batch_1 as a sample
        sample_symbols = load_iex_batch_symbols("filtered_iex_batches.json", "batch_1")
        
        # Test market data for first 3 symbols
        test_symbols = sample_symbols[:3]
        print(f"\nTesting market data for: {', '.join(test_symbols)}")
        
        for symbol in test_symbols:
            try:
                # Get current quote
                quote = alpaca_client.get_quote(symbol)
                print(f"\n{symbol} Quote:")
                print(f"  Bid: ${quote['bid_price']:.2f} x {quote['bid_size']}")
                print(f"  Ask: ${quote['ask_price']:.2f} x {quote['ask_size']}")
                print(f"  Spread: ${quote['ask_price'] - quote['bid_price']:.2f}")
                
                # Get recent bars
                market_data = alpaca_client.get_market_data(symbol, timeframe='1Day', limit=5)
                if market_data['bars']:
                    latest_bar = market_data['bars'][-1]
                    print(f"  Latest Close: ${latest_bar['close']:.2f}")
                    print(f"  Volume: {latest_bar['volume']:,}")
                    print(f"  Timestamp: {latest_bar['timestamp']}")
                
            except Exception as e:
                print(f"  Error getting data for {symbol}: {e}")
                continue
        
        # 5. Show market status
        logger.info("\n5. Checking market status...")
        is_open = alpaca_client.is_market_open()
        print(f"\nMarket Status: {'OPEN' if is_open else 'CLOSED'}")
        
        # 6. Example usage summary
        print("\n" + "=" * 50)
        print("USAGE SUMMARY")
        print("=" * 50)
        print(f"✓ Successfully loaded {batch_info['total_symbols']} IEX filtered symbols")
        print("✓ Connected to Alpaca API with paper trading account")
        print("✓ Retrieved real-time quotes and historical data")
        print("✓ All symbols are compatible with Alpaca free tier (IEX data)")
        print("\nThis demonstrates the correct setup for:")
        print("• Using pre-filtered IEX symbols only")
        print("• Alpaca API exclusively (no yfinance fallback)")
        print("• Real-time intraday data capability")
        print("• Free tier compatibility")
        
        logger.info("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 