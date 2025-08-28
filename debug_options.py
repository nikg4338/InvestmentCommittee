#!/usr/bin/env python3
"""
Simple options contract test to see what's available from Alpaca
"""

import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add trading to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trading'))

try:
    from execution.alpaca_client import AlpacaClient
    print("✅ Successfully imported AlpacaClient")
    
    # Create client
    client = AlpacaClient()
    print("✅ Created AlpacaClient instance")
    
    # Test basic API access
    account = client.get_account_info()
    print(f"✅ Account ID: {account['id']}")
    
    # Test without any filters to see what's available
    print("\n🔍 Testing basic options contract call...")
    
    # Try a simple call without filters first
    try:
        # Build the simplest possible query using proper REST call
        query_params = {
            'underlying_symbols': 'SPY',
            'limit': 10
        }
        print(f"Testing with params: {query_params}")
        
        response = client.api._request('GET', '/options/contracts', data=query_params)
        print(f"Response type: {type(response)}")
        print(f"Response: {response}")
        
        if isinstance(response, dict) and 'option_contracts' in response:
            contracts = response['option_contracts']
            print(f"Found {len(contracts)} contracts via dict")
        else:
            print("No option_contracts found in response")
            print(f"Available keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
            
    except Exception as e:
        print(f"❌ Error with basic call: {e}")
        
    # Try different symbols
    for symbol in ['SPY', 'QQQ', 'AAPL']:
        print(f"\n🔍 Testing {symbol}...")
        try:
            contracts = client.get_option_contracts(symbol, limit=5)
            print(f"  Found {len(contracts)} contracts for {symbol}")
            if contracts:
                for i, contract in enumerate(contracts[:2]):
                    print(f"    {i+1}: {contract}")
        except Exception as e:
            print(f"  ❌ Error for {symbol}: {e}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
