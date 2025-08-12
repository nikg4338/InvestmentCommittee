#!/usr/bin/env python3
"""
Real Alpaca Portfolio Integration Test
Tests connection to Alpaca's paper trading account and displays current portfolio.
"""

import logging
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading.execution.alpaca_client import AlpacaClient
from trading.portfolio.position_manager import PositionManager
from trading.production_trading_engine import create_production_trading_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_alpaca_portfolio():
    """Test real Alpaca portfolio integration."""
    
    print("="*80)
    print("ALPACA PAPER TRADING PORTFOLIO TEST")
    print("="*80)
    
    try:
        # Initialize Alpaca client
        logger.info("Connecting to Alpaca...")
        alpaca_client = AlpacaClient()
        
        # Get account info
        logger.info("Fetching account information...")
        account_info = alpaca_client.get_account_info()
        
        print(f"\n📊 ACCOUNT OVERVIEW:")
        print(f"  Account ID: {account_info['id']}")
        print(f"  Status: {account_info['status']}")
        print(f"  Portfolio Value: ${account_info['portfolio_value']:,.2f}")
        print(f"  Cash: ${account_info['cash']:,.2f}")
        print(f"  Buying Power: ${account_info['buying_power']:,.2f}")
        print(f"  Equity: ${account_info['equity']:,.2f}")
        print(f"  Long Market Value: ${account_info.get('long_market_value', 0):,.2f}")
        
        # Get current positions
        logger.info("Fetching current positions...")
        positions = alpaca_client.get_positions()
        
        print(f"\n📈 CURRENT POSITIONS ({len(positions)} total):")
        if positions:
            for pos in positions:
                qty = float(pos['qty'])
                if qty != 0:  # Only show non-zero positions
                    market_value = pos['market_value']
                    cost_basis = pos['cost_basis']
                    unrealized_pl = pos['unrealized_pl']
                    unrealized_pct = pos['unrealized_plpc'] * 100
                    
                    print(f"  {pos['symbol']}: {qty:,.0f} shares")
                    print(f"    Market Value: ${market_value:,.2f}")
                    print(f"    Cost Basis: ${cost_basis:,.2f}")
                    print(f"    P&L: ${unrealized_pl:,.2f} ({unrealized_pct:+.2f}%)")
                    print(f"    Current Price: ${pos['current_price']:.2f}")
                    print()
        else:
            print("  No positions found")
        
        # Get recent orders
        logger.info("Fetching recent orders...")
        orders = alpaca_client.get_orders(status='all', limit=10)
        
        print(f"\n📋 RECENT ORDERS ({len(orders)} shown):")
        if orders:
            for order in orders[:5]:  # Show last 5 orders
                status_emoji = "✅" if order['status'] == 'filled' else "⏳" if order['status'] in ['new', 'accepted'] else "❌"
                print(f"  {status_emoji} {order['symbol']} - {order['side'].upper()} {order['qty']} @ {order['order_type']}")
                print(f"    Status: {order['status']} | Created: {order['created_at']}")
                if order['filled_qty'] > 0:
                    print(f"    Filled: {order['filled_qty']}/{order['qty']} @ ${order.get('avg_fill_price', 'N/A')}")
                print()
        else:
            print("  No orders found")
            
        # Test position manager integration
        print(f"\n🔧 TESTING POSITION MANAGER INTEGRATION:")
        position_manager = PositionManager(alpaca_client)
        
        # Test get_current_positions method
        current_positions = position_manager.get_current_positions()
        print(f"  Position Manager found {len(current_positions)} positions")
        
        # Test portfolio summary
        portfolio_summary = position_manager.get_portfolio_summary()
        print(f"  Portfolio Summary:")
        print(f"    Total Positions: {portfolio_summary.get('total_positions', 0)}")
        print(f"    Total Value: ${portfolio_summary.get('total_value', 0):,.2f}")
        print(f"    Realized P&L: ${portfolio_summary.get('realized_pnl', 0):,.2f}")
        print(f"    Unrealized P&L: ${portfolio_summary.get('unrealized_pnl', 0):,.2f}")
        
        print(f"\n✅ ALPACA INTEGRATION TEST COMPLETED SUCCESSFULLY")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing Alpaca portfolio: {e}")
        print(f"\n❌ ALPACA INTEGRATION TEST FAILED: {e}")
        return False

def test_buy_order():
    """Test placing a small buy order."""
    
    print(f"\n" + "="*80)
    print("TESTING LIVE BUY ORDER")
    print("="*80)
    
    try:
        # Create trading engine
        trading_engine = create_production_trading_engine(paper_trading=True)
        
        # Test symbol - using a cheap ETF for testing
        test_symbol = "SPY"
        position_size = 0.01  # 1% of portfolio
        
        print(f"\n🛒 Attempting to place BUY order for {test_symbol}")
        print(f"   Position size: {position_size:.1%} of portfolio")
        
        # Execute the trade
        success = trading_engine.position_manager.execute_trade(
            symbol=test_symbol,
            action="BUY",
            position_size=position_size,
            signal_metadata={'test': True, 'timestamp': datetime.now().isoformat()}
        )
        
        if success:
            print(f"✅ BUY order executed successfully for {test_symbol}")
            
            # Wait a moment then check positions
            import time
            time.sleep(2)
            
            positions = trading_engine.position_manager.get_current_positions()
            if test_symbol in positions:
                pos = positions[test_symbol]
                print(f"   Position confirmed: {pos.get('qty', 0)} shares")
            
        else:
            print(f"❌ BUY order failed for {test_symbol}")
            
        return success
        
    except Exception as e:
        logger.error(f"Error testing buy order: {e}")
        print(f"❌ BUY ORDER TEST FAILED: {e}")
        return False

if __name__ == "__main__":
    print(f"Starting Alpaca integration test at {datetime.now()}")
    
    # Test portfolio connection
    portfolio_test = test_alpaca_portfolio()
    
    if portfolio_test:
        # Ask user if they want to test a live order
        response = input(f"\n🤔 Test a live BUY order (1% of portfolio)? (y/N): ").strip().lower()
        if response == 'y':
            test_buy_order()
        else:
            print("Skipping live order test")
    
    print(f"\n" + "="*80)
    print("TEST COMPLETED")
    print("="*80)
