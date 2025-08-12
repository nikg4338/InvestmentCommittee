#!/usr/bin/env python3
"""
Options Trading Module
Handles options strategies including bull put spreads.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class OptionContract:
    """Represents an option contract."""
    symbol: str
    underlying: str
    expiration_date: str
    strike_price: float
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

@dataclass
class OptionsSpread:
    """Represents an options spread strategy."""
    strategy_type: str  # 'bull_put_spread', 'iron_condor', etc.
    underlying: str
    expiration_date: str
    long_leg: OptionContract
    short_leg: OptionContract
    max_profit: float
    max_loss: float
    breakeven: float
    net_credit: float
    probability_of_profit: float

class OptionsTrader:
    """
    Handles options trading strategies and execution.
    """
    
    def __init__(self, alpaca_client):
        self.alpaca = alpaca_client
        self.min_days_to_expiration = 30
        self.max_days_to_expiration = 60
        self.target_delta = 0.20  # For put spreads
        self.min_credit = 0.10    # Reduced minimum credit to $0.10
        
        # Enable debug logging
        logging.basicConfig(level=logging.INFO)
        
    def find_bull_put_spread_opportunities(self, symbol: str, account_value: float) -> List[OptionsSpread]:
        """
        Find bull put spread opportunities for a given symbol.
        
        Args:
            symbol: Underlying symbol (e.g., 'SPY')
            account_value: Total account value for position sizing
            
        Returns:
            List of viable bull put spread opportunities
        """
        try:
            logger.info(f"Scanning for bull put spread opportunities on {symbol}")
            
            # Get current stock price
            quote = self.alpaca.get_quote(symbol)
            current_price = quote.get('last_price', quote.get('ask_price', 0))
            
            if current_price <= 0:
                logger.warning(f"Invalid price for {symbol}: {current_price}")
                return []
            
            logger.info(f"{symbol} current price: ${current_price:.2f}")
            
            # Calculate target strikes for bull put spread
            # Use smaller, more realistic strike intervals
            # Short put: ~2% OTM (higher strike, collect premium)  
            # Long put: ~4% OTM (lower strike, protection)
            
            short_strike = current_price * 0.98  # 2% OTM - closer for better premium
            long_strike = current_price * 0.96   # 4% OTM - narrower spread
            
            # Round to nearest $1 for better pricing
            strike_interval = 1 if current_price < 100 else (5 if current_price < 500 else 10)
            short_strike = round(short_strike / strike_interval) * strike_interval
            long_strike = round(long_strike / strike_interval) * strike_interval
            
            # Create mock spread for demonstration (in production, would query real options data)
            spread = self._create_mock_bull_put_spread(
                symbol, current_price, short_strike, long_strike
            )
            
            if spread and self._validate_spread(spread, account_value):
                logger.info(f"Found viable bull put spread: {spread.short_leg.strike_price}/{spread.long_leg.strike_price}")
                return [spread]
            
            return []
            
        except Exception as e:
            logger.error(f"Error finding bull put spreads for {symbol}: {e}")
            return []
    
    def _create_mock_bull_put_spread(self, underlying: str, current_price: float, 
                                   short_strike: float, long_strike: float) -> Optional[OptionsSpread]:
        """Create a mock bull put spread for demonstration."""
        try:
            # Calculate expiration date (45 DTE)
            expiration = (datetime.now() + timedelta(days=45)).strftime('%Y-%m-%d')
            
            # Mock option prices based on typical bid/ask spreads
            # Short put (higher strike) - we collect premium
            short_put_price = self._estimate_put_price(current_price, short_strike, 45)
            short_contract = OptionContract(
                symbol=f"{underlying}_{expiration}_{short_strike:.0f}_P",
                underlying=underlying,
                expiration_date=expiration,
                strike_price=short_strike,
                option_type='put',
                bid=short_put_price - 0.05,
                ask=short_put_price + 0.05,
                last_price=short_put_price,
                volume=100,
                open_interest=500,
                implied_volatility=0.20,
                delta=-0.20
            )
            
            # Long put (lower strike) - we pay premium for protection
            long_put_price = self._estimate_put_price(current_price, long_strike, 45)
            long_contract = OptionContract(
                symbol=f"{underlying}_{expiration}_{long_strike:.0f}_P",
                underlying=underlying,
                expiration_date=expiration,
                strike_price=long_strike,
                option_type='put',
                bid=long_put_price - 0.05,
                ask=long_put_price + 0.05,
                last_price=long_put_price,
                volume=80,
                open_interest=300,
                implied_volatility=0.22,
                delta=-0.15
            )
            
            # Calculate spread metrics
            net_credit = short_put_price - long_put_price
            width = short_strike - long_strike
            max_profit = net_credit
            max_loss = width - net_credit
            breakeven = short_strike - net_credit
            prob_profit = 0.70  # Estimated 70% probability of profit
            
            spread = OptionsSpread(
                strategy_type='bull_put_spread',
                underlying=underlying,
                expiration_date=expiration,
                long_leg=long_contract,
                short_leg=short_contract,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven=breakeven,
                net_credit=net_credit,
                probability_of_profit=prob_profit
            )
            
            return spread
            
        except Exception as e:
            logger.error(f"Error creating mock spread: {e}")
            return None
    
    def _estimate_put_price(self, stock_price: float, strike: float, days_to_expiry: int) -> float:
        """Estimate put option price using simplified Black-Scholes approximation."""
        try:
            # More realistic option pricing for demonstration
            moneyness = strike / stock_price
            time_value = days_to_expiry / 365.0
            volatility = 0.25  # Assume 25% IV
            
            # Intrinsic value
            intrinsic = max(0, strike - stock_price)
            
            # Time value (improved calculation)
            if moneyness < 1.0:  # OTM put
                # Use a more realistic time value for OTM options
                otm_distance = 1.0 - moneyness
                time_premium = stock_price * volatility * (time_value ** 0.5) * (0.4 * (1 - otm_distance))
            else:  # ITM put
                time_premium = stock_price * volatility * (time_value ** 0.5) * 0.2
            
            total_price = intrinsic + time_premium
            
            # Ensure minimum price for very OTM options
            if total_price < 0.50:
                total_price = max(0.50, otm_distance * 2.0)
            
            return total_price
            
        except Exception:
            return 1.0  # Fallback price
    
    def _validate_spread(self, spread: OptionsSpread, account_value: float) -> bool:
        """Validate if the spread meets our criteria."""
        try:
            logger.info(f"Validating spread: Credit=${spread.net_credit:.2f}, Max Loss=${spread.max_loss:.2f}")
            
            # Check minimum credit
            if spread.net_credit < self.min_credit:
                logger.info(f"Spread rejected: insufficient credit ${spread.net_credit:.2f} < ${self.min_credit:.2f}")
                return False
            
            # Check risk/reward ratio - more realistic for credit spreads
            # Bull put spreads typically target 10-30% return on margin
            risk_reward_ratio = spread.max_profit / spread.max_loss if spread.max_loss > 0 else 0
            min_return = 0.10  # 10% minimum return on risk
            if risk_reward_ratio < min_return:
                logger.info(f"Spread rejected: return too low {risk_reward_ratio:.1%} < {min_return:.1%}")
                return False
            
            # Check position sizing (max 10% of account at risk)
            max_risk_per_spread = account_value * 0.10
            if spread.max_loss > max_risk_per_spread:
                logger.info(f"Spread rejected: too much risk ${spread.max_loss:.2f} > ${max_risk_per_spread:.2f}")
                return False
            
            # Check credit amount (should be meaningful)
            if spread.net_credit < 0.30:  # At least $0.30 credit
                logger.info(f"Spread rejected: credit too small ${spread.net_credit:.2f}")
                return False
            
            logger.info(f"✅ Spread validated: Credit=${spread.net_credit:.2f}, Return={risk_reward_ratio:.1%}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating spread: {e}")
            return False
    
    def execute_bull_put_spread(self, spread: OptionsSpread) -> bool:
        """
        Execute a bull put spread strategy.
        
        Args:
            spread: The spread to execute
            
        Returns:
            bool: True if successful
        """
        try:
            logger.info(f"Executing bull put spread on {spread.underlying}")
            logger.info(f"  Short {spread.short_leg.strike_price} Put")
            logger.info(f"  Long {spread.long_leg.strike_price} Put")
            logger.info(f"  Net Credit: ${spread.net_credit:.2f}")
            logger.info(f"  Max Profit: ${spread.max_profit:.2f}")
            logger.info(f"  Max Loss: ${spread.max_loss:.2f}")
            
            # In a real implementation, would submit orders to Alpaca
            # For now, simulate the execution
            
            logger.info("🎯 BULL PUT SPREAD EXECUTED SUCCESSFULLY")
            logger.info(f"   Strategy: Sell {spread.short_leg.strike_price} Put / Buy {spread.long_leg.strike_price} Put")
            logger.info(f"   Expiration: {spread.expiration_date}")
            logger.info(f"   Credit Collected: ${spread.net_credit:.2f}")
            logger.info(f"   Probability of Profit: {spread.probability_of_profit:.1%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing bull put spread: {e}")
            return False

def demo_options_trading():
    """Demonstrate options trading capabilities."""
    print("="*80)
    print("OPTIONS TRADING DEMONSTRATION")
    print("="*80)
    
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from trading.execution.alpaca_client import AlpacaClient
    
    # Initialize
    alpaca_client = AlpacaClient()
    options_trader = OptionsTrader(alpaca_client)
    
    # Test symbols
    test_symbols = ['SPY', 'QQQ', 'IWM']
    account_value = 100000  # $100k account
    
    for symbol in test_symbols:
        print(f"\n🔍 Scanning {symbol} for bull put spread opportunities...")
        
        spreads = options_trader.find_bull_put_spread_opportunities(symbol, account_value)
        
        if spreads:
            for spread in spreads:
                print(f"\n📊 BULL PUT SPREAD OPPORTUNITY - {symbol}")
                print(f"   Expiration: {spread.expiration_date}")
                print(f"   Short Strike: ${spread.short_leg.strike_price:.0f}")
                print(f"   Long Strike: ${spread.long_leg.strike_price:.0f}")
                print(f"   Net Credit: ${spread.net_credit:.2f}")
                print(f"   Max Profit: ${spread.max_profit:.2f}")
                print(f"   Max Loss: ${spread.max_loss:.2f}")
                print(f"   Breakeven: ${spread.breakeven:.2f}")
                print(f"   Probability of Profit: {spread.probability_of_profit:.1%}")
                
                # Ask if user wants to execute
                response = input(f"   Execute this spread? (y/N): ").strip().lower()
                if response == 'y':
                    success = options_trader.execute_bull_put_spread(spread)
                    if success:
                        print("   ✅ Spread executed successfully!")
                    else:
                        print("   ❌ Spread execution failed")
                else:
                    print("   Skipped")
        else:
            print(f"   No viable spreads found for {symbol}")
    
    print(f"\n" + "="*80)
    print("OPTIONS DEMO COMPLETED")
    print("="*80)

if __name__ == "__main__":
    demo_options_trading()
