#!/usr/bin/env python3
"""
Scalable Trading System
Processes batches of stocks from filtered_iex_batches.json for both stock trading and options strategies.
"""

import json
import logging
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import sys
import os
from dataclasses import dataclass
import random

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

@dataclass
class TradingOpportunity:
    """Represents a trading opportunity found during scanning."""
    symbol: str
    opportunity_type: str  # 'stock_buy', 'stock_sell', 'bull_put_spread', etc.
    confidence: float
    potential_profit: float
    risk_amount: float
    details: Dict
    priority_score: float  # Combined score for ranking

class ScalableTradingSystem:
    """
    Handles large-scale trading operations across hundreds of symbols.
    """
    
    def __init__(self):
        self.trading_engine = create_production_trading_engine(paper_trading=True)
        self.processed_symbols: Set[str] = set()
        self.opportunities: List[TradingOpportunity] = []
        self.rate_limit_delay = 0.5  # Seconds between API calls
        self.batch_size = 20  # Process this many symbols at once
        self.max_concurrent_positions = 10  # Maximum number of positions
        
    def load_filtered_symbols(self) -> Dict[str, List[str]]:
        """Load symbols from filtered_iex_batches.json"""
        try:
            with open('filtered_iex_batches.json', 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loaded {data['metadata']['total_symbols']} symbols across {data['metadata']['total_batches']} batches")
            return data['batches']
            
        except Exception as e:
            logger.error(f"Error loading filtered symbols: {e}")
            return {}
    
    def prioritize_symbols(self, symbols: List[str]) -> List[str]:
        """
        Prioritize symbols based on various factors.
        Focus on higher-volume, more liquid stocks for better options opportunities.
        """
        # High-priority symbols (ETFs and large-cap stocks likely to have options)
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for symbol in symbols:
            # ETFs and common large-cap symbols get priority
            if any(pattern in symbol for pattern in ['SPY', 'QQQ', 'IWM', 'EF', 'VT', 'GLD', 'SLV']):
                high_priority.append(symbol)
            elif len(symbol) <= 4 and symbol.isalpha():  # Traditional stock symbols
                medium_priority.append(symbol)
            else:
                low_priority.append(symbol)
        
        # Randomize within priority groups to avoid always hitting the same stocks
        random.shuffle(high_priority)
        random.shuffle(medium_priority)
        random.shuffle(low_priority)
        
        return high_priority + medium_priority + low_priority
    
    def scan_batch_for_opportunities(self, symbols: List[str]) -> List[TradingOpportunity]:
        """
        Scan a batch of symbols for trading opportunities.
        """
        opportunities = []
        
        logger.info(f"Scanning batch of {len(symbols)} symbols...")
        
        try:
            # Generate trading signals for the batch
            signals = self.trading_engine.process_trading_universe(symbols)
            
            for signal in signals:
                if signal.confidence > 0.7:  # High confidence only
                    # Create stock trading opportunity
                    opportunity = TradingOpportunity(
                        symbol=signal.symbol,
                        opportunity_type=f"stock_{signal.action.lower()}",
                        confidence=signal.confidence,
                        potential_profit=self._estimate_stock_profit(signal),
                        risk_amount=self._estimate_stock_risk(signal),
                        details={
                            'signal_strength': signal.signal_strength,
                            'action': signal.action,
                            'llm_analysis': signal.llm_analysis.recommendation if signal.llm_analysis else None
                        },
                        priority_score=signal.confidence * signal.signal_strength
                    )
                    opportunities.append(opportunity)
                    
                    # Also check for options opportunities on high-confidence signals
                    if signal.confidence > 0.85 and self._is_options_suitable(signal.symbol):
                        options_opp = self._analyze_options_opportunity(signal)
                        if options_opp:
                            opportunities.append(options_opp)
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
        except Exception as e:
            logger.error(f"Error scanning batch: {e}")
        
        return opportunities
    
    def _estimate_stock_profit(self, signal) -> float:
        """Estimate potential profit from stock trade."""
        # Simple estimation based on signal strength and typical price movements
        base_profit = 100.0  # Base $100 position
        return base_profit * signal.signal_strength * signal.confidence
    
    def _estimate_stock_risk(self, signal) -> float:
        """Estimate risk amount for stock trade."""
        # Risk is typically the position size for stocks
        return 1000.0  # $1000 typical position size
    
    def _is_options_suitable(self, symbol: str) -> bool:
        """Check if symbol is suitable for options trading."""
        # Focus on high-volume ETFs and large stocks that typically have liquid options
        suitable_patterns = [
            'SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VTI', 'GLD', 'SLV', 'TLT', 'HYG',
            'XLF', 'XLE', 'XLI', 'XLK', 'XLV', 'XLP', 'XLY', 'XLB', 'XLU', 'XLRE'
        ]
        return any(pattern in symbol for pattern in suitable_patterns)
    
    def _analyze_options_opportunity(self, signal) -> Optional[TradingOpportunity]:
        """Analyze options trading opportunity for a given signal."""
        try:
            # Get current price
            quote = self.trading_engine.alpaca_client.get_quote(signal.symbol)
            current_price = quote.get('last_price', quote.get('ask_price', 0))
            
            if current_price <= 0:
                return None
            
            # For SELL signals, consider bull put spreads (bullish/neutral strategy)
            if signal.action == 'SELL' and signal.confidence > 0.85:
                # Calculate theoretical bull put spread
                short_strike = round(current_price * 0.97)  # 3% OTM
                long_strike = round(current_price * 0.94)   # 6% OTM
                width = short_strike - long_strike
                
                # Estimate credit (simplified)
                estimated_credit = width * 0.30  # Assume 30% of width as credit
                max_profit = estimated_credit
                max_loss = width - estimated_credit
                
                # Only consider if risk/reward is acceptable
                if max_loss > 0 and (max_profit / max_loss) > 0.20:  # At least 20% return
                    return TradingOpportunity(
                        symbol=signal.symbol,
                        opportunity_type='bull_put_spread',
                        confidence=signal.confidence * 0.9,  # Slightly lower for options
                        potential_profit=max_profit * 100,  # Per contract
                        risk_amount=max_loss * 100,
                        details={
                            'strategy': 'bull_put_spread',
                            'short_strike': short_strike,
                            'long_strike': long_strike,
                            'width': width,
                            'estimated_credit': estimated_credit,
                            'current_price': current_price
                        },
                        priority_score=signal.confidence * (max_profit / max_loss)
                    )
            
            return None
            
        except Exception as e:
            logger.warning(f"Error analyzing options for {signal.symbol}: {e}")
            return None
    
    def run_scalable_scan(self, max_batches: int = 3, max_symbols_per_batch: int = 20) -> List[TradingOpportunity]:
        """
        Run a scalable scan across multiple batches of symbols.
        """
        print("="*80)
        print("🚀 SCALABLE TRADING SYSTEM - BATCH PROCESSING")
        print("="*80)
        
        # Load all symbols
        batches = self.load_filtered_symbols()
        if not batches:
            logger.error("No symbols loaded")
            return []
        
        # Get account info
        account_info = self.trading_engine.alpaca_client.get_account_info()
        print(f"\n💰 ACCOUNT STATUS:")
        print(f"   Portfolio Value: ${account_info['portfolio_value']:,.2f}")
        print(f"   Available Cash: ${account_info['cash']:,.2f}")
        print(f"   Buying Power: ${account_info['buying_power']:,.2f}")
        
        # Flatten and prioritize all symbols
        all_symbols = []
        for batch_name, symbols in batches.items():
            all_symbols.extend(symbols)
        
        prioritized_symbols = self.prioritize_symbols(all_symbols)
        
        print(f"\n🔍 SCANNING STRATEGY:")
        print(f"   Total symbols available: {len(all_symbols)}")
        print(f"   Max batches to process: {max_batches}")
        print(f"   Symbols per batch: {max_symbols_per_batch}")
        print(f"   Total symbols to scan: {max_batches * max_symbols_per_batch}")
        
        all_opportunities = []
        batch_count = 0
        
        # Process symbols in chunks
        for i in range(0, len(prioritized_symbols), max_symbols_per_batch):
            if batch_count >= max_batches:
                break
                
            batch_symbols = prioritized_symbols[i:i + max_symbols_per_batch]
            batch_count += 1
            
            print(f"\n📊 PROCESSING BATCH {batch_count}/{max_batches}")
            print(f"   Symbols: {batch_symbols[:5]}{'...' if len(batch_symbols) > 5 else ''}")
            
            start_time = time.time()
            batch_opportunities = self.scan_batch_for_opportunities(batch_symbols)
            scan_time = time.time() - start_time
            
            print(f"   ⏱️  Scan completed in {scan_time:.1f}s")
            print(f"   📈 Found {len(batch_opportunities)} opportunities")
            
            all_opportunities.extend(batch_opportunities)
            
            # Show top opportunities from this batch
            batch_opportunities.sort(key=lambda x: x.priority_score, reverse=True)
            for opp in batch_opportunities[:3]:  # Top 3
                print(f"      🎯 {opp.symbol}: {opp.opportunity_type} | Confidence: {opp.confidence:.1%} | Profit: ${opp.potential_profit:.0f}")
        
        # Sort all opportunities by priority
        all_opportunities.sort(key=lambda x: x.priority_score, reverse=True)
        
        print(f"\n🏆 SCAN COMPLETE - TOP OPPORTUNITIES:")
        print(f"   Total opportunities found: {len(all_opportunities)}")
        
        # Show top 10 opportunities
        for i, opp in enumerate(all_opportunities[:10], 1):
            type_emoji = "📈" if "buy" in opp.opportunity_type else "📉" if "sell" in opp.opportunity_type else "🎯"
            print(f"   {i:2d}. {type_emoji} {opp.symbol}: {opp.opportunity_type}")
            print(f"       Confidence: {opp.confidence:.1%} | Profit: ${opp.potential_profit:.0f} | Risk: ${opp.risk_amount:.0f}")
        
        return all_opportunities
    
    def execute_top_opportunities(self, opportunities: List[TradingOpportunity], max_executions: int = 5):
        """Execute the top trading opportunities."""
        
        if not opportunities:
            print("\n❌ No opportunities to execute")
            return
        
        print(f"\n🎯 EXECUTING TOP {min(len(opportunities), max_executions)} OPPORTUNITIES:")
        
        executed_count = 0
        for opp in opportunities[:max_executions]:
            try:
                if opp.opportunity_type.startswith('stock_'):
                    # Execute stock trade
                    action = opp.opportunity_type.split('_')[1].upper()
                    success = self.trading_engine.position_manager.execute_trade(
                        symbol=opp.symbol,
                        action=action,
                        position_size=0.02,  # 2% of portfolio per trade
                        signal_metadata={'batch_scan': True, 'priority_score': opp.priority_score}
                    )
                    
                    if success:
                        print(f"   ✅ {opp.symbol}: {action} executed successfully")
                        executed_count += 1
                    else:
                        print(f"   ❌ {opp.symbol}: {action} execution failed")
                        
                elif opp.opportunity_type == 'bull_put_spread':
                    # For now, just log the options opportunity
                    details = opp.details
                    print(f"   🎯 {opp.symbol}: Bull Put Spread identified")
                    print(f"      Sell ${details['short_strike']} Put / Buy ${details['long_strike']} Put")
                    print(f"      Estimated Credit: ${details['estimated_credit']:.2f}")
                    print(f"      ⚠️  Options execution not yet implemented")
                
                # Small delay between executions
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error executing opportunity for {opp.symbol}: {e}")
        
        print(f"\n📊 EXECUTION SUMMARY:")
        print(f"   Attempted: {min(len(opportunities), max_executions)}")
        print(f"   Successful: {executed_count}")
        print(f"   Failed: {min(len(opportunities), max_executions) - executed_count}")

def main():
    """Main function to run the scalable trading system."""
    
    print(f"🚀 Starting Scalable Trading System at {datetime.now()}")
    
    try:
        system = ScalableTradingSystem()
        
        # Run the scan with reasonable limits to avoid overwhelming the system
        opportunities = system.run_scalable_scan(max_batches=5, max_symbols_per_batch=15)  # 75 symbols total
        
        if opportunities:
            # Ask user about execution
            response = input(f"\n🤔 Execute top opportunities? (y/N): ").strip().lower()
            if response == 'y':
                system.execute_top_opportunities(opportunities, max_executions=3)
            else:
                print("Execution skipped by user")
        
        print(f"\n✅ Scalable trading system run completed!")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ System error: {e}")

if __name__ == "__main__":
    main()
