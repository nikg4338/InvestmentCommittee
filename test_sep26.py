#!/usr/bin/env python3
"""
Test Sep 26, 2025 expiration to confirm contracts exist
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trading'))

from execution.alpaca_client import AlpacaClient

client = AlpacaClient()

print("🔍 Testing Sep 26, 2025 expiration date...")

symbols = ['SPY', 'QQQ', 'AMZN', 'WMT']

for symbol in symbols:
    try:
        contracts = client.get_option_contracts(
            underlying_symbol=symbol,
            expiration_date='2025-09-26',
            option_type='put',
            limit=3
        )
        print(f"\n{symbol}: {len(contracts)} PUT contracts for Sep 26, 2025")
        
        if contracts:
            for contract in contracts:
                print(f"  ✅ {contract['symbol']} - Strike: ${contract['strike_price']}")
        else:
            print(f"  ❌ No contracts found")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n🎯 Sep 26, 2025 contract test complete!")
