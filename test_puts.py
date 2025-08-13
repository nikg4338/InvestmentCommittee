#!/usr/bin/env python3
"""
Quick test to see what PUT options are available for trading
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trading'))

from execution.alpaca_client import AlpacaClient

client = AlpacaClient()

print("🔍 Checking PUT options availability...")

# Test PUT options for different expiration dates
symbols = ['SPY', 'QQQ', 'AAPL', 'AMZN']
exp_dates = ['2025-08-13', '2025-08-15', '2025-08-22', '2025-08-29']

for symbol in symbols:
    print(f"\n--- {symbol} PUT OPTIONS ---")
    
    for exp_date in exp_dates:
        try:
            contracts = client.get_option_contracts(
                underlying_symbol=symbol,
                expiration_date=exp_date,
                option_type='put',
                limit=5
            )
            print(f"  {exp_date}: {len(contracts)} put contracts")
            
            if contracts:
                for contract in contracts[:2]:
                    print(f"    {contract['symbol']} - Strike: ${contract['strike_price']}")
        except Exception as e:
            print(f"  {exp_date}: Error - {e}")

print("\n✅ Test completed!")
