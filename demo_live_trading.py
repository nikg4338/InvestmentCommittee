#!/usr/bin/env python3
"""
Demo Live Trading System
Creates initial positions and then demonstrates the full trading cycle.
"""

import logging
import time
from datetime import datetime
from trading.production_trading_engine import create_production_trading_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_trading_cycle():
    """
    Demonstrate a complete trading cycle:
    1. Create some initial positions
    2. Run trading system to generate signals
    3. Execute trades based on signals
    """
    
    print("="*70)
    print("LIVE TRADING DEMONSTRATION")
    print("="*70)
    
    # Initialize trading engine
    logger.info("Initializing trading engine...")
    trading_engine = create_production_trading_engine(paper_trading=True)
    
    # Test symbols
    test_symbols = ["AAPL", "MSFT", "SPY"]
    
    print(f"\nTesting with symbols: {test_symbols}")
    
    # Step 1: Create some mock positions to demonstrate sell signals
    logger.info("Creating mock positions for demonstration...")
    for symbol in test_symbols:
        mock_position = {
            'id': f"{symbol}_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'symbol': symbol,
            'action': 'BUY',
            'shares': 10,
            'entry_price': 150.0,  # Mock price
            'entry_time': datetime.now(),
            'status': 'open'
        }
        trading_engine.position_manager.add_position(mock_position)
        print(f"  ✓ Added mock position: {symbol} (10 shares)")
    
    # Step 2: Update market context
    logger.info("Updating market context...")
    trading_engine.update_market_context()
    
    # Step 3: Generate trading signals
    logger.info("Generating trading signals...")
    signals = trading_engine.process_trading_universe(test_symbols)
    
    print(f"\nGenerated {len(signals)} signals:")
    for signal in signals:
        print(f"  {signal.symbol}: {signal.action} - Strength: {signal.signal_strength:.3f}, "
              f"Confidence: {signal.confidence:.3f}")
        if signal.llm_analysis:
            print(f"    LLM: {signal.llm_analysis.recommendation} ({signal.llm_analysis.confidence:.2f})")
    
    # Step 4: Execute trades
    print(f"\nExecuting trades...")
    high_confidence_signals = [s for s in signals if s.confidence > 0.3]
    
    if high_confidence_signals:
        execution_results = trading_engine.execute_trading_signals(high_confidence_signals)
        
        print(f"  ✓ Executed trades: {len(execution_results['executed_trades'])}")
        print(f"  ✗ Failed trades: {len(execution_results['failed_trades'])}")
        
        if execution_results['executed_trades']:
            print("  Executed symbols:", execution_results['executed_trades'])
        
        # Show portfolio summary
        portfolio = execution_results.get('portfolio_summary', {})
        if portfolio:
            print(f"\nPortfolio Summary:")
            print(f"  Total positions: {portfolio.get('total_positions', 0)}")
            print(f"  Realized P&L: ${portfolio.get('realized_pnl', 0):.2f}")
            if 'positions' in portfolio:
                print(f"  Current positions: {len(portfolio['positions'])}")
    else:
        print("  No high-confidence signals to execute")
    
    print("\n" + "="*70)
    print("DEMO COMPLETED")
    print("="*70)

if __name__ == "__main__":
    demo_trading_cycle()
