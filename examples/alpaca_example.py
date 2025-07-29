# Example usage of Alpaca API client for Investment Committee
# This demonstrates how to use the AlpacaClient for paper trading

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from trading.execution.alpaca_client import AlpacaClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Example usage of the Alpaca API client.
    
    Note: You need to set your ALPACA_API_KEY and ALPACA_SECRET_KEY 
    environment variables in your .env file.
    """
    
    try:
        # Initialize Alpaca client
        print("Initializing Alpaca client...")
        alpaca = AlpacaClient()
        
        # Get account information
        print("\n=== Account Information ===")
        account = alpaca.get_account_info()
        print(f"Account ID: {account['id']}")
        print(f"Status: {account['status']}")
        print(f"Buying Power: ${account['buying_power']:,.2f}")
        print(f"Cash: ${account['cash']:,.2f}")
        print(f"Portfolio Value: ${account['portfolio_value']:,.2f}")
        print(f"Paper Trading: {account.get('paper_trading', 'Unknown')}")
        
        # Check if market is open
        print("\n=== Market Status ===")
        is_open = alpaca.is_market_open()
        print(f"Market is {'OPEN' if is_open else 'CLOSED'}")
        
        # Get current positions
        print("\n=== Current Positions ===")
        positions = alpaca.get_positions()
        if positions:
            for pos in positions:
                print(f"Symbol: {pos['symbol']}")
                print(f"  Quantity: {pos['qty']}")
                print(f"  Side: {pos['side']}")
                print(f"  Market Value: ${pos['market_value']:,.2f}")
                print(f"  Unrealized P&L: ${pos['unrealized_pl']:,.2f}")
                print(f"  Current Price: ${pos['current_price']:,.2f}")
                print("---")
        else:
            print("No open positions")
        
        # Get recent orders
        print("\n=== Recent Orders ===")
        orders = alpaca.get_orders(status='all', limit=10)
        if orders:
            for order in orders[-5:]:  # Show last 5 orders
                print(f"Order ID: {order['id']}")
                print(f"  Symbol: {order['symbol']}")
                print(f"  Side: {order['side']}")
                print(f"  Quantity: {order['qty']}")
                print(f"  Status: {order['status']}")
                print(f"  Created: {order['created_at']}")
                print("---")
        else:
            print("No recent orders")
        
        # Get market data example
        print("\n=== Market Data Example (SPY) ===")
        market_data = alpaca.get_market_data('SPY', timeframe='1Day', limit=5)
        if market_data['bars']:
            for bar in market_data['bars'][-3:]:  # Show last 3 bars
                print(f"Date: {bar['timestamp']}")
                print(f"  Open: ${bar['open']:.2f}")
                print(f"  High: ${bar['high']:.2f}")
                print(f"  Low: ${bar['low']:.2f}")
                print(f"  Close: ${bar['close']:.2f}")
                print(f"  Volume: {bar['volume']:,}")
                print("---")
        
        # Get quote example
        print("\n=== Current Quote Example (AAPL) ===")
        quote = alpaca.get_quote('AAPL')
        print(f"Symbol: {quote['symbol']}")
        print(f"Bid: ${quote['bid_price']:.2f} x {quote['bid_size']}")
        print(f"Ask: ${quote['ask_price']:.2f} x {quote['ask_size']}")
        print(f"Timestamp: {quote['timestamp']}")
        
        # Example of submitting a paper trade (commented out for safety)
        print("\n=== Paper Trading Example (COMMENTED OUT) ===")
        print("# Example of submitting a paper trade order:")
        print("# order = alpaca.submit_order(")
        print("#     symbol='AAPL',")
        print("#     qty=10,")
        print("#     side='buy',")
        print("#     order_type='market'")
        print("# )")
        print("# print(f'Order submitted: {order['id']}')")
        
    except Exception as e:
        logger.error(f"Error in Alpaca example: {e}")
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("1. Set your ALPACA_SECRET_KEY environment variable")
        print("2. Or add your API keys to the .env file")
        print("3. Installed the required packages: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 