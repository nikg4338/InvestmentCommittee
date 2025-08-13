#!/usr/bin/env python3
"""
Quick test of Alpaca Options API to see what's available
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trading'))

from execution.alpaca_client import AlpacaClient
from datetime import datetime, timedelta

def test_options_api():
    # Initialize client
    client = AlpacaClient()
    
    print("🔍 Testing Alpaca Options API...")
    print(f"Connected: {client.alpaca_connected}")
    
    # Test symbols
    test_symbols = ['SPY', 'AAPL', 'TSLA', 'QQQ', 'AMZN']
    
    for symbol in test_symbols:
        print(f"\n--- Testing {symbol} ---")
        
        # Test without expiration filter
        try:
            print(f"Testing {symbol} without expiration filter...")
            contracts = client.get_option_contracts(
                underlying_symbol=symbol,
                option_type='put',
                limit=50
            )
            print(f"Found {len(contracts)} put contracts for {symbol}")
            
            if contracts:
                # Show a few examples
                for i, contract in enumerate(contracts[:3]):
                    print(f"  Contract {i+1}: {contract.get('symbol', 'N/A')} - Strike: ${contract.get('strike_price', 'N/A')} - Exp: {contract.get('expiration_date', 'N/A')}")
            
        except Exception as e:
            print(f"Error testing {symbol}: {e}")
    
    # Test date ranges
    print(f"\n--- Testing date ranges ---")
    future_dates = []
    base_date = datetime.now()
    
    # Try next few Fridays (typical option expiration)
    for i in range(1, 8):
        future_date = base_date + timedelta(weeks=i)
        # Get next Friday
        days_until_friday = (4 - future_date.weekday()) % 7
        if days_until_friday == 0:
            days_until_friday = 7
        friday = future_date + timedelta(days=days_until_friday)
        future_dates.append(friday.strftime('%Y-%m-%d'))
    
    print(f"Testing upcoming Fridays: {future_dates[:3]}")
    
    for exp_date in future_dates[:3]:
        try:
            print(f"\nTesting SPY puts expiring {exp_date}...")
            contracts = client.get_option_contracts(
                underlying_symbol='SPY',
                expiration_date=exp_date,
                option_type='put',
                limit=10
            )
            print(f"Found {len(contracts)} SPY put contracts for {exp_date}")
            
            if contracts:
                for contract in contracts[:2]:
                    print(f"  {contract.get('symbol', 'N/A')} - Strike: ${contract.get('strike_price', 'N/A')}")
                    
        except Exception as e:
            print(f"Error testing {exp_date}: {e}")

if __name__ == "__main__":
    test_options_api()
