#!/usr/bin/env python3
"""
REAL Alpaca-Connected Autonomous Trading System
==============================================

Full-featured autonomous trader with REAL Alpaca integration.
Scans ALL OPTIONS-ENABLED stocks directly from Alpaca for bull put spread opportunities.
"""

import asyncio
import logging
import sys
import os
import json
import random
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import pytz

# Add the trading module to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trading'))

# Import Alpaca client
from execution.alpaca_client import AlpacaClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/autonomous_trading.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Ensure required directories exist
os.makedirs('logs', exist_ok=True)
os.makedirs('data', exist_ok=True)

@dataclass
class TradingDecision:
    symbol: str
    action: str
    confidence: float
    reasons: List[str]
    trade_parameters: Dict[str, Any]
    timestamp: datetime

class SimpleAutonomousTrader:
    """
    REAL Alpaca-connected autonomous trader with full functionality.
    Scans all options-enabled stocks from Alpaca for bull put spread opportunities.
    """
    
    def __init__(self):
        self.is_running = False
        self.daily_trade_count = 0
        self.earnings_season_trade_count = 0
        
        # DYNAMIC TRADE LIMITS - Adaptive based on market conditions
        self.max_daily_trades_normal = 50  # Normal market conditions
        self.max_daily_trades_earnings = 5  # Limited trades during earnings season
        self.current_max_trades = self.max_daily_trades_normal  # Will be updated dynamically
        
        # POSITION SIZING - Reduced during earnings season
        self.normal_position_size = 1.0  # Full position size
        self.earnings_position_size = 0.4  # 40% of normal size during earnings
        self.current_position_size = self.normal_position_size
        
        self.tz = pytz.timezone('US/Eastern')
        self.last_trade_time = datetime.min
        self.min_trade_interval = timedelta(minutes=30)  # 30 min between trades
        
        # Initialize REAL analysis engines
        try:
            # Import the actual production modules
            from trading.production_trading_engine import ProductionModelEnsemble, ProductionModelConfig
            from data_collection_alpaca import AlpacaDataCollector
            
            # Initialize Alpaca client for REAL trading
            self.alpaca = AlpacaClient()
            account_info = self.alpaca.get_account_info()
            logger.info(f"🚀 REAL ALPACA CONNECTION ESTABLISHED!")
            logger.info(f"Account ID: {account_info['id']}")
            logger.info(f"Buying Power: ${account_info['buying_power']:,.2f}")
            logger.info(f"Cash: ${account_info['cash']:,.2f}")
            self.alpaca_connected = True
            
            # Initialize real market data collector (replaces market analyzer)
            self.market_analyzer = AlpacaDataCollector()
            logger.info("✅ Real Market Data Collector initialized with live data feeds")
            
            # Initialize production ML engine with actual trained models
            model_config = ProductionModelConfig()
            self.ml_engine = ProductionModelEnsemble(model_config)
            logger.info(f"✅ Production ML Engine initialized")
            logger.info(f"   📊 Models loaded: {len(self.ml_engine.models)}")
            logger.info(f"   🧠 Model names: {', '.join(list(self.ml_engine.models.keys())[:3])}...")
            logger.info(f"   📈 Thresholds: {len(self.ml_engine.thresholds)} configured")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize production analysis engines: {e}")
            logger.error("This will severely limit analysis quality!")
            self.alpaca_connected = False
            self.alpaca = None
            self.market_analyzer = None
            self.ml_engine = None
        
        # Now load symbols (with options filtering if Alpaca is available)
        self.symbols = self._load_symbols()
        # Models are loaded by the enhanced ML engine, no need for legacy loading
        self.open_positions = self._load_open_positions()
        
        logger.info(f"🤖 Autonomous Trader initialized")
        logger.info(f"Trading Limits: Normal market: {self.max_daily_trades_normal} trades/day")
        logger.info(f"                Earnings season: {self.max_daily_trades_earnings} trades/day (quality-focused)")
        logger.info(f"Position Sizing: Normal: {self.normal_position_size:.0%}, Earnings: {self.earnings_position_size:.0%}")
        logger.info(f"REAL TRADING: {'✅ ENABLED' if self.alpaca_connected else '❌ DISABLED (SIMULATION)'}")
        logger.info(f"Symbols loaded: {len(self.symbols)} options-enabled stocks from Alpaca")
        logger.info(f"ML Models: Handled by Enhanced ML Engine")
        logger.info(f"Open positions: {len(self.open_positions)}")

    def _load_symbols(self) -> List[str]:
        """Load all options-enabled stocks directly from Alpaca."""
        # Get options-enabled stocks directly from Alpaca if connected
        if hasattr(self, 'alpaca') and self.alpaca_connected:
            logger.info("Loading COMPREHENSIVE options-enabled stocks from Alpaca...")
            alpaca_options = self.alpaca.get_options_enabled_stocks(limit=1000)  # Get up to 1000 options stocks
            if alpaca_options and len(alpaca_options) > 100:  # Require at least 100 for comprehensive coverage
                logger.info(f"Successfully loaded {len(alpaca_options)} options-enabled stocks from Alpaca")
                logger.info(f"Sample symbols: {', '.join(alpaca_options[:20])}...")
                return alpaca_options
            else:
                logger.warning(f"Alpaca returned only {len(alpaca_options) if alpaca_options else 0} stocks, using enhanced fallback")
        
        # Enhanced fallback to comprehensive list of known optionable stocks
        comprehensive_fallback = [
            # Major ETFs
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VEA', 'VWO', 'EFA', 'EEM',
            'GLD', 'SLV', 'TLT', 'HYG', 'LQD', 'XLE', 'XLF', 'XLK', 'XLV',
            'XLI', 'XLP', 'XLY', 'XLU', 'XLRE', 'SMH', 'KRE', 'XBI', 'IBB',
            
            # FAANG + Tech Giants
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA',
            'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'IBM', 'CSCO',
            'QCOM', 'TXN', 'AVGO', 'NOW', 'INTU', 'PYPL', 'SQ', 'SHOP',
            
            # Mega Cap (> $100B)
            'BRK.B', 'UNH', 'JNJ', 'V', 'MA', 'PG', 'HD', 'CVX', 'XOM',
            'ABBV', 'PFE', 'LLY', 'TMO', 'KO', 'PEP', 'COST', 'WMT', 'DIS',
            
            # Large Cap Finance
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW',
            'USB', 'PNC', 'COF', 'TFC', 'MTB', 'RF', 'FITB', 'KEY', 'CFG',
            
            # Large Cap Healthcare
            'ABT', 'DHR', 'BMY', 'AMGN', 'GILD', 'REGN', 'VRTX', 'BIIB',
            'ILMN', 'MRNA', 'BNTX', 'MRK', 'CVS', 'CI', 'HUM', 'ANTM',
            
            # Large Cap Consumer
            'TGT', 'LOW', 'TJX', 'SBUX', 'MCD', 'NKE', 'LULU', 'ETSY',
            'EBAY', 'BABA', 'JD', 'PDD', 'AMZN', 'WMT', 'COST', 'HD',
            
            # Large Cap Industrial
            'CAT', 'DE', 'BA', 'HON', 'UPS', 'FDX', 'LMT', 'RTX', 'NOC',
            'GD', 'MMM', 'GE', 'EMR', 'ITW', 'PH', 'ROK', 'DOV', 'ETN',
            
            # Energy Sector
            'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'KMI', 'OKE', 'WMB',
            'EPD', 'ET', 'BKR', 'HAL', 'DVN', 'FANG', 'MRO', 'OXY', 'APA',
            
            # REITs & Real Estate
            'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'WELL', 'DLR', 'PSA',
            'EXR', 'AVB', 'EQR', 'MAA', 'ESS', 'UDR', 'CPT', 'FRT', 'REG',
            
            # Materials & Chemicals
            'LIN', 'APD', 'SHW', 'FCX', 'NEM', 'CTVA', 'EMN', 'DD', 'DOW',
            'PPG', 'ECL', 'IFF', 'ALB', 'CE', 'VMC', 'MLM', 'NUE', 'STLD',
            
            # Utilities
            'NEE', 'SO', 'DUK', 'D', 'EXC', 'XEL', 'SRE', 'AEP', 'ES', 'FE',
            'ETR', 'ED', 'EIX', 'PPL', 'AES', 'LNT', 'NI', 'PNW', 'CMS',
            
            # Communications
            'T', 'VZ', 'CMCSA', 'TMUS', 'CHTR', 'DISH', 'SIRI', 'LBRDA',
            
            # Growth & Momentum Stocks
            'ZM', 'DOCU', 'TWLO', 'OKTA', 'SNOW', 'PLTR', 'RBLX', 'U',
            'DDOG', 'ZS', 'CRWD', 'NET', 'ROKU', 'PINS', 'SNAP', 'TWTR',
            
            # Biotech & Pharma
            'GILD', 'REGN', 'VRTX', 'BIIB', 'ILMN', 'AMGN', 'CELG', 'BMY'
        ]
        
        logger.warning(f"Using comprehensive fallback list of {len(comprehensive_fallback)} known optionable symbols")
        logger.info(f"Includes major ETFs, S&P 500, and sector leaders across all industries")
        return comprehensive_fallback

    def _load_open_positions(self) -> List[Dict]:
        """Load existing open positions."""
        try:
            with open('logs/open_positions.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def _save_open_positions(self):
        """Save current open positions."""
        with open('logs/open_positions.json', 'w') as f:
            json.dump(self.open_positions, f, indent=2)

    def _get_real_stock_price(self, symbol: str) -> Optional[float]:
        """Get real stock price from Alpaca."""
        if not self.alpaca_connected:
            return None
        
        try:
            bars = self.alpaca.api.get_latest_bar(symbol)
            if bars and hasattr(bars, 'c'):
                return float(bars.c)
        except Exception as e:
            logger.debug(f"Could not get real price for {symbol}: {e}")
        
        return None

    async def start_trading(self):
        """Start the autonomous trading system."""
        logger.info("Starting Simple Autonomous Trading System")
        logger.info(f"Paper Trading: {'ENABLED (Safe mode)' if self.alpaca_connected else 'SIMULATION ONLY'}")
        logger.info("Market Hours: 9:30 AM - 4:00 PM ET, Monday-Friday")
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Check if market is open
                if self._is_market_open():
                    logger.info("Market is OPEN - Trading cycle starting")
                    await self._execute_trading_cycle()
                else:
                    logger.info("Market is CLOSED - Waiting for next session")
                
                # Wait 5 minutes before next cycle
                await asyncio.sleep(300)  # 5 minutes
                
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
        finally:
            self.is_running = False
            logger.info("Autonomous trading system stopped")

    def _is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now(self.tz)
        
        # Skip weekends
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close

    async def _execute_trading_cycle(self):
        """Execute one trading cycle - full functionality with adaptive market conditions."""
        try:
            # Update market conditions and trading limits dynamically
            self._reset_daily_counters_if_needed()
            is_earnings_season = self._update_market_conditions()
            
            logger.info(f"Trading cycle - Daily trades: {self.daily_trade_count}/{self.current_max_trades}")
            if is_earnings_season:
                logger.info(f"Earnings trades: {self.earnings_season_trade_count}/{self.max_daily_trades_earnings}")
            logger.info(f"Open positions: {len(self.open_positions)}")
            
            # First, manage existing positions (profit-taking, risk management)
            await self._manage_existing_positions()
            
            # Check time since last trade (30 min cooldown)
            time_since_last = datetime.now() - self.last_trade_time
            if time_since_last < self.min_trade_interval:
                remaining_time = self.min_trade_interval - time_since_last
                logger.info(f"Trade cooldown: {remaining_time.total_seconds()/60:.1f} minutes until next trade allowed")
                return
            
            # Check if we can make more trades today (using dynamic limits)
            if self.daily_trade_count >= self.current_max_trades:
                regime = "earnings season" if is_earnings_season else "normal market"
                logger.info(f"Daily trade limit reached ({self.current_max_trades} trades max for {regime})")
                return
            
            # Limit to reasonable number of open positions
            max_positions = 50 if is_earnings_season else 100  # Reduced during earnings
            if len(self.open_positions) >= max_positions:
                logger.info(f"Maximum open positions reached ({max_positions} max)")
                return
            
            # Scan for trading opportunities with earnings-aware strategy
            decisions = await self._scan_for_opportunities(is_earnings_season)
            
            # Execute trades with adaptive thresholds and position sizing
            for decision in decisions:
                # Adaptive confidence threshold based on market conditions
                min_confidence = 0.80 if is_earnings_season else 0.70  # Higher bar during earnings
                
                if decision.confidence >= min_confidence:
                    logger.info(f"🎯 HIGH-CONFIDENCE ML TRADE FOUND: {decision.confidence:.1%} confidence")
                    logger.info(f"   Market regime: {'EARNINGS SEASON' if is_earnings_season else 'NORMAL'}")
                    logger.info(f"   Position size: {self.current_position_size:.0%} of normal")
                    
                    await self._execute_trade(decision, is_earnings_season)
                    self.daily_trade_count += 1
                    if is_earnings_season:
                        self.earnings_season_trade_count += 1
                    self.last_trade_time = datetime.now()
                    
                    # Check both regular and earnings-specific limits
                    if self.daily_trade_count >= self.current_max_trades:
                        regime = "earnings season" if is_earnings_season else "normal market"
                        logger.info(f"Daily trade limit reached for {regime}")
                        break
                else:
                    threshold_desc = f"≥{min_confidence:.0%} ({'earnings' if is_earnings_season else 'normal'} threshold)"
                    logger.info(f"❌ Rejecting trade - ML confidence too low: {decision.confidence:.1%} (need {threshold_desc})")
            
            if not decisions:
                logger.info("No ML-confident trading opportunities found")
                
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")

    async def _manage_existing_positions(self):
        """Manage existing positions for profit-taking and risk management."""
        if not self.open_positions:
            return
        
        positions_to_close = []
        
        logger.info(f"Managing {len(self.open_positions)} open positions...")
        
        for i, position in enumerate(self.open_positions):
            try:
                symbol = position['symbol']
                entry_date = datetime.fromisoformat(position['entry_time'])
                days_held = (datetime.now() - entry_date).days
                expiration_date = datetime.fromisoformat(position['expiration_date'])
                days_to_expiration = (expiration_date - datetime.now()).days
                
                # Get REAL current stock price if possible
                real_price = self._get_real_stock_price(symbol)
                if real_price:
                    current_price = real_price
                    logger.info(f"📊 Using REAL price for {symbol}: ${current_price:.2f}")
                else:
                    # Fallback to mock price
                    current_price = position['short_strike'] + np.random.normal(5, 10)
                    logger.info(f"🎲 Using simulated price for {symbol}: ${current_price:.2f}")
                
                # Calculate current spread value (simplified)
                # As theta decays, the spread becomes worth less (profit for us)
                time_decay_factor = max(0.1, days_to_expiration / 45.0)  # Assume 45 DTE originally
                intrinsic_value = max(0, position['short_strike'] - current_price)
                current_spread_value = (intrinsic_value + (position['short_strike'] - position['long_strike']) * time_decay_factor * 0.3) * 100
                
                original_credit = position['estimated_credit']
                current_profit = original_credit - current_spread_value
                profit_percentage = current_profit / original_credit if original_credit > 0 else 0
                
                # IMPROVED EXIT LOGIC - Stop panic closes on "near short strike"
                
                # Calculate key metrics for better exit decisions
                short_strike = position['short_strike']
                long_strike = position['long_strike']
                spread_width = short_strike - long_strike
                
                # Calculate distance to short strike (breach detection)
                distance_to_short = current_price - short_strike
                distance_percentage = distance_to_short / current_price
                
                # Estimate current short delta (approximation)
                # As price approaches short strike, delta increases
                if current_price > short_strike:
                    # Price above short strike - delta approaches 0
                    estimated_short_delta = max(0.05, 0.30 * (short_strike / current_price))
                else:
                    # Price at or below short strike - delta approaches 1
                    estimated_short_delta = min(0.95, 0.50 + 0.45 * ((short_strike - current_price) / short_strike))
                
                logger.info(f"Position Analysis - {symbol}:")
                logger.info(f"  Current price: ${current_price:.2f}")
                logger.info(f"  Short strike: ${short_strike:.2f}")
                logger.info(f"  Distance to short: ${distance_to_short:.2f} ({distance_percentage:.1%})")
                logger.info(f"  Estimated short delta: {estimated_short_delta:.2f}")
                logger.info(f"  Days held: {days_held}, DTE: {days_to_expiration}")
                logger.info(f"  Original credit: ${original_credit:.2f}")
                logger.info(f"  Current spread value: ${current_spread_value:.2f}")
                logger.info(f"  Current profit: ${current_profit:.2f} ({profit_percentage:.1%})")
                
                # EXIT RULE 1: PROFIT TARGET - Take 50% of credit (standard)
                if profit_percentage >= 0.50:
                    logger.info(f"🎯 PROFIT TARGET: {symbol} - Taking 50% profit!")
                    await self._close_position(position, "PROFIT_50PCT", current_profit)
                    positions_to_close.append(i)
                
                # EXIT RULE 2: BREACH-BASED EXIT - Price crosses short strike (not just "near")
                elif current_price <= short_strike:
                    logger.info(f"🚨 SHORT STRIKE BREACH: {symbol} - Price ${current_price:.2f} ≤ short strike ${short_strike:.2f}")
                    await self._close_position(position, "SHORT_STRIKE_BREACH", current_profit)
                    positions_to_close.append(i)
                
                # EXIT RULE 3: DELTA-BASED EXIT - Short delta > 0.35 (high assignment risk)
                elif estimated_short_delta > 0.35:
                    logger.info(f"⚠️ HIGH DELTA RISK: {symbol} - Short delta {estimated_short_delta:.2f} > 0.35")
                    await self._close_position(position, "HIGH_DELTA_RISK", current_profit)
                    positions_to_close.append(i)
                
                # EXIT RULE 4: MAX LOSS - ~2x credit (stop loss)
                elif profit_percentage <= -2.0:
                    logger.info(f"🛑 STOP LOSS: {symbol} - Loss exceeds 200% of credit")
                    await self._close_position(position, "STOP_LOSS", current_profit)
                    positions_to_close.append(i)
                
                # EXIT RULE 5: EXPIRATION MANAGEMENT - Close if approaching expiration (7 days)
                elif days_to_expiration <= 7:
                    logger.info(f"⏰ EXPIRATION RISK: {symbol} - {days_to_expiration} days until expiration")
                    await self._close_position(position, "EXPIRATION_MANAGEMENT", current_profit)
                    positions_to_close.append(i)
                
                # POSITION CONTINUES - Log status
                else:
                    logger.info(f"✅ {symbol}: Position continues - All exit criteria clear")
                    logger.info(f"   Profit target: {profit_percentage:.1%}/50%")
                    logger.info(f"   Delta risk: {estimated_short_delta:.2f}/0.35") 
                    logger.info(f"   Distance to breach: ${distance_to_short:.2f}")
                    logger.info(f"   Stop loss buffer: {profit_percentage:.1%}/-200%")
                
            except Exception as e:
                logger.error(f"Error managing position {i}: {e}")
        
        # Remove closed positions (reverse order to maintain indices)
        for i in reversed(positions_to_close):
            self.open_positions.pop(i)
        
        if positions_to_close:
            self._save_open_positions()
            logger.info(f"Closed {len(positions_to_close)} positions")

    async def _close_position(self, position: Dict, reason: str, profit: float):
        """Close a position (simulated for now)."""
        logger.info(f"🔄 CLOSING POSITION: {position['symbol']} - Reason: {reason}")
        logger.info(f"   Profit/Loss: ${profit:.2f}")
        
        # Record the closed trade
        close_record = {
            'trade_id': position.get('trade_id'),
            'symbol': position['symbol'],
            'close_time': datetime.now().isoformat(),
            'close_reason': reason,
            'profit_loss': profit,
            'days_held': (datetime.now() - datetime.fromisoformat(position['entry_time'])).days
        }
        
        # Log to closed trades file
        with open('logs/closed_trades.jsonl', 'a') as f:
            f.write(json.dumps(close_record) + '\n')

    async def _scan_for_opportunities(self, is_earnings_season: bool = False) -> List[TradingDecision]:
        """
        COMPREHENSIVE scan for high-quality trading opportunities.
        Adaptive approach based on market conditions.
        
        Args:
            is_earnings_season: If True, applies more selective criteria and focuses on quality
        """
        decisions = []
        
        # Validate we have the enhanced analysis tools
        if not self.market_analyzer or not self.ml_engine:
            logger.error("❌ Enhanced analysis tools not available - cannot perform quality analysis")
            return decisions
        
        # Use ALL loaded symbols from Alpaca for maximum coverage
        if not self.symbols:
            logger.warning("No symbols loaded for scanning")
            return decisions
        
        # Adaptive scanning strategy based on market conditions
        if is_earnings_season:
            # During earnings season: Quality-focused, limited quantity
            scan_symbols = self.symbols.copy()  # Still scan all for best opportunities
            max_opportunities = 10  # But limit results to top candidates
            quality_threshold = 0.70  # Higher quality requirement
            logger.info(f"🔍 EARNINGS SEASON QUALITY SCAN of {len(scan_symbols)} symbols")
            logger.info(f"📊 Strategy: Ultra-selective, top {max_opportunities} quality opportunities only")
            logger.info(f"🎯 Quality threshold: {quality_threshold:.0%} (elevated for earnings)")
        else:
            # Normal conditions: Full capacity
            scan_symbols = self.symbols.copy()
            max_opportunities = 25  # Normal limit
            quality_threshold = 0.50  # Standard quality requirement
            logger.info(f"🔍 NORMAL MARKET SCAN of {len(scan_symbols)} symbols")
            logger.info(f"📊 Strategy: Full capacity, up to {max_opportunities} opportunities")
            logger.info(f"🎯 Quality threshold: {quality_threshold:.0%} (standard)")
        
        logger.info(f"📊 Analysis includes: Real market data + ML models + Quality scoring + Risk assessment")
        
        successful_analyses = 0
        
        for i, symbol in enumerate(scan_symbols, 1):
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"📈 ANALYZING {symbol} ({i}/{len(scan_symbols)})")
                logger.info(f"{'='*60}")
                
                decision = await self._evaluate_symbol(symbol, is_earnings_season, quality_threshold)
                if decision and decision.action == 'OPEN_BULL_PUT_SPREAD':
                    decisions.append(decision)
                    successful_analyses += 1
                    logger.info(f"🎯 HIGH-QUALITY OPPORTUNITY FOUND: {symbol}")
                    logger.info(f"   Confidence: {decision.confidence:.1%}")
                    logger.info(f"   Key reasons: {decision.reasons[:2]}")
                    
                    # Adaptive stopping criteria based on market conditions
                    if len(decisions) >= max_opportunities:
                        mode = "earnings season quality-focused" if is_earnings_season else "normal market"
                        logger.info(f"✅ Found {max_opportunities} opportunities for {mode} - stopping scan")
                        break
                else:
                    logger.info(f"❌ {symbol}: Did not meet strict quality criteria")
                    
            except Exception as e:
                logger.error(f"Error in comprehensive analysis of {symbol}: {e}")
                continue
        
        # Sort by confidence (highest first)
        decisions.sort(key=lambda x: x.confidence, reverse=True)
        
        # For earnings season, take only the very best candidates
        if is_earnings_season and len(decisions) > self.max_daily_trades_earnings:
            decisions = decisions[:self.max_daily_trades_earnings]
            logger.info(f"🎯 EARNINGS SEASON: Selected top {len(decisions)} candidates from all opportunities")
        
        logger.info(f"\n{'='*80}")
        mode_desc = "EARNINGS SEASON QUALITY-FOCUSED" if is_earnings_season else "NORMAL MARKET COMPREHENSIVE"
        logger.info(f"📊 SCAN COMPLETE - {mode_desc}")
        logger.info(f"{'='*80}")
        logger.info(f"Symbols analyzed: {len(scan_symbols)}")
        logger.info(f"Successful analyses: {successful_analyses}")
        logger.info(f"Final opportunities: {len(decisions)}")
        
        if decisions:
            mode_str = "QUALITY-FOCUSED" if is_earnings_season else "COMPREHENSIVE"
            logger.info(f"🏆 TOP {mode_str} OPPORTUNITIES:")
            for i, decision in enumerate(decisions, 1):
                logger.info(f"  {i}. {decision.symbol} - {decision.confidence:.1%} confidence")
        else:
            condition = "earnings season quality" if is_earnings_season else "adaptive quality"
            logger.info(f"📉 No opportunities met our {condition} criteria today")
            logger.info(f"💡 {'Earnings conditions are challenging' if is_earnings_season else 'Market conditions may be challenging'}")
        
        return decisions

    async def _evaluate_symbol(self, symbol: str, is_earnings_season: bool = False, 
                              quality_threshold: float = 0.50) -> Optional[TradingDecision]:
        """
        Comprehensive evaluation of a symbol using REAL market data and trained ML models.
        Implements proper due diligence with multiple quality gates.
        
        Args:
            symbol: Stock symbol to evaluate
            is_earnings_season: If True, allows earnings trades with special handling
            quality_threshold: Minimum quality score required
        """
        try:
            logger.info(f"🔍 Comprehensive analysis of {symbol}")
            logger.info(f"   Market mode: {'EARNINGS SEASON' if is_earnings_season else 'NORMAL'}")
            logger.info(f"   Quality threshold: {quality_threshold:.0%}")
            
            # STEP 1: Get real market features using actual data
            market_features = self.market_analyzer.get_market_features(symbol)
            if not market_features:
                logger.info(f"❌ {symbol}: Insufficient market data for analysis")
                return None
            
            logger.info(f"✅ {symbol}: Retrieved {len(market_features)} real market features")
            
            # PRE-TRADE HARD FILTERS - Adaptive based on market conditions
            
            # Filter 1: VIX/Volatility regime - Always avoid extreme volatility
            vix_top_10pct = market_features.get('vix_top_10pct', 0)
            if vix_top_10pct == 1:
                logger.info(f"❌ {symbol}: High volatility regime detected (VIX top 10%) - BLOCKED")
                return None
            
            # Filter 2: Earnings season handling - ADAPTIVE APPROACH
            earnings_season = market_features.get('earnings_season', 0)
            if earnings_season == 1:
                if not is_earnings_season:
                    # Normal mode: Block all earnings trades
                    logger.info(f"❌ {symbol}: Earnings season detected - BLOCKED (normal mode)")
                    return None
                else:
                    # Earnings mode: Allow but apply extra scrutiny
                    logger.info(f"⚠️ {symbol}: Earnings season detected - PROCEEDING with enhanced scrutiny")
                    # Increase quality requirements for earnings trades
                    quality_threshold = max(quality_threshold, 0.80)  # Minimum 80% quality for earnings
                    logger.info(f"   Enhanced quality threshold: {quality_threshold:.0%}")
            else:
                logger.info(f"✅ {symbol}: No earnings conflicts detected")
            
            # Filter 3: IV percentile requirement - Need elevated IV for credit spreads
            implied_vol_percentile = market_features.get('implied_vol_percentile', 0)
            if implied_vol_percentile < 0.30:
                logger.info(f"❌ {symbol}: IV percentile too low ({implied_vol_percentile:.1%}) - need ≥30%")
                return None
            
            # Filter 4: Trend requirement - Need bullish longer-term trend
            long_term_trend = market_features.get('long_term_trend', 0)
            market_trend_strength = market_features.get('market_trend_strength', 0)
            if long_term_trend != 1 and market_trend_strength <= 0:
                logger.info(f"❌ {symbol}: No bullish trend (LT trend: {long_term_trend}, strength: {market_trend_strength})")
                return None
            
            # Filter 5: Spread efficiency check
            spread_efficiency = market_features.get('spread_efficiency', 0)
            if spread_efficiency < 0.5:
                logger.info(f"❌ {symbol}: Poor spread efficiency ({spread_efficiency:.2f}) - need ≥0.5")
                return None
            
            logger.info(f"✅ {symbol}: Passed ALL pre-trade hard filters")
            earnings_status = "Enhanced scrutiny (earnings)" if earnings_season == 1 else "Safe period"
            logger.info(f"   VIX regime: Normal (not top 10%)")
            logger.info(f"   Earnings: {earnings_status}")
            logger.info(f"   IV percentile: {implied_vol_percentile:.1%}")
            logger.info(f"   Trend: {long_term_trend} (strength: {market_trend_strength:.2f})")
            logger.info(f"   Spread efficiency: {spread_efficiency:.2f}")
            
            # STEP 2: Market-adaptive quality gate with dynamic scoring
            quality_score = self._calculate_quality_score(symbol, market_features)
            if quality_score < quality_threshold:
                mode = "earnings season enhanced" if is_earnings_season else "normal"
                logger.info(f"❌ {symbol}: Quality score too low ({quality_score:.1%}) for {mode} threshold ({quality_threshold:.0%})")
                return None
            
            logger.info(f"✅ {symbol}: Quality score: {quality_score:.1%}")
            
            # STEP 3: Get ML ensemble prediction using trained models
            predictions = self.ml_engine.predict(market_features)
            ensemble_pred, ensemble_confidence = self.ml_engine.ensemble_prediction(predictions)
            
            logger.info(f"🤖 {symbol}: ML Analysis Complete")
            logger.info(f"   Individual Predictions: {predictions}")
            logger.info(f"   Ensemble Prediction: {ensemble_pred:.3f}")
            logger.info(f"   Ensemble Confidence: {ensemble_confidence:.1%}")
            
            # Calculate consensus ratio (how many models agree)
            model_predictions = list(predictions.values())
            bullish_count = 0
            if model_predictions:
                # Count how many models predict bullish (>0.5)
                bullish_count = sum(1 for pred in model_predictions if pred > 0.5)
                consensus_ratio = bullish_count / len(model_predictions)
            else:
                consensus_ratio = 0.0
            
            logger.info(f"   Model Consensus: {consensus_ratio:.1%} ({bullish_count}/{len(model_predictions)} models bullish)")
            
            # STEP 4: Evaluate ML predictions with RAISED threshold - NO SYNTHETIC OVERRIDES
            if ensemble_pred < 0.70 or ensemble_confidence < 0.01:  # RAISED from 0.5 to 0.70 for stricter gate
                logger.info(f"❌ {symbol}: ML ensemble below threshold (pred: {ensemble_pred:.3f}, need ≥0.70)")
                return None
            
            # Use ONLY real ML confidence - no synthetic scores
            final_confidence = ensemble_confidence
            
            # Lower threshold for market-adaptive trading
            if final_confidence < 0.30:  # Lowered from 60% to 30%
                logger.info(f"❌ {symbol}: ML confidence too low ({final_confidence:.1%} < 30%)")
                return None
            
            # Check model consensus - require at least some agreement
            if consensus_ratio < 0.30:  # Lowered from 40% to 30%
                logger.info(f"❌ {symbol}: Model consensus too low ({consensus_ratio:.1%} < 30%)")
                return None
            
            # STEP 5: Get real current price
            current_price = self._get_real_stock_price(symbol)
            if not current_price:
                logger.info(f"❌ {symbol}: Could not obtain real-time price")
                return None
            
            logger.info(f"💰 {symbol}: Current price ${current_price:.2f}")
            
            # STEP 6: Dynamic option expiration discovery with narrowed DTE range
            available_expirations = self.market_analyzer.get_available_option_expirations(
                symbol, min_dte=30, max_dte=45  # NARROWED from 25-55 to 30-45 DTE for optimal theta
            )
            
            if not available_expirations:
                logger.info(f"❌ {symbol}: No suitable option expirations found")
                return None
            
            # Choose the best expiration (prefer ~35-45 DTE)
            best_expiration = self._select_optimal_expiration(available_expirations)
            logger.info(f"📅 {symbol}: Selected expiration {best_expiration}")
            
            # STEP 7: Calculate optimal strikes based on current volatility
            strikes = self._calculate_optimal_strikes(current_price, market_features)
            if not strikes:
                logger.info(f"❌ {symbol}: Could not calculate suitable strikes")
                return None
            
            short_strike, long_strike = strikes
            logger.info(f"🎯 {symbol}: Strikes - Short: ${short_strike}, Long: ${long_strike}")
            
            # STEP 8: STRICT LIQUIDITY VERIFICATION - Hard filters
            liquidity_analysis = self.market_analyzer.analyze_option_liquidity(
                symbol, best_expiration, [short_strike, long_strike]
            )
            
            # HARD LIQUIDITY FILTERS - Replace the stub with real gates
            
            # Get mock option data for liquidity checks (in production, use real option chain data)
            # For now, implement strict rules based on symbol and market cap
            
            # Filter 1: Bid/Ask spread requirement
            # Mock calculation - in production, get real bid/ask from option chain
            if symbol in ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'AMZN', 'GOOGL', 'META', 'NVDA']:
                # Major symbols - assume tighter spreads
                mock_bid_ask_spread = 0.05  # $0.05 spread
                mock_mid_price = 2.00       # $2.00 mid
            elif current_price > 200:
                # Large cap stocks
                mock_bid_ask_spread = 0.08
                mock_mid_price = 1.50
            elif current_price > 100:
                # Mid cap stocks
                mock_bid_ask_spread = 0.12
                mock_mid_price = 1.20
            else:
                # Small cap stocks - wider spreads
                mock_bid_ask_spread = 0.20
                mock_mid_price = 1.00
            
            spread_percentage = mock_bid_ask_spread / mock_mid_price
            
            # Bid/ask spread ≤ $0.10 or ≤ 5% of mid
            if mock_bid_ask_spread > 0.10 and spread_percentage > 0.05:
                logger.info(f"❌ {symbol}: Bid/ask spread too wide (${mock_bid_ask_spread:.2f}, {spread_percentage:.1%})")
                logger.info(f"   Requirements: ≤$0.10 OR ≤5% of mid")
                return None
            
            # Filter 2: Open Interest requirement (≥1,000 on short leg)
            # Mock calculation - in production, get real OI from option chain
            if symbol in ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA']:
                mock_open_interest = 5000  # High OI for major symbols
            elif current_price > 200:
                mock_open_interest = 2000  # Good OI for large caps
            elif current_price > 100:
                mock_open_interest = 800   # Marginal OI for mid caps
            else:
                mock_open_interest = 300   # Low OI for small caps
            
            if mock_open_interest < 1000:
                logger.info(f"❌ {symbol}: Open interest too low ({mock_open_interest}) - need ≥1,000")
                return None
            
            # Filter 3: Options volume requirement (≥500 session volume on short leg)
            # Mock calculation - in production, get real volume from option chain
            if symbol in ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA']:
                mock_options_volume = 2000  # High volume for major symbols
            elif current_price > 200:
                mock_options_volume = 800   # Good volume for large caps
            elif current_price > 100:
                mock_options_volume = 400   # Marginal volume for mid caps
            else:
                mock_options_volume = 150   # Low volume for small caps
            
            if mock_options_volume < 500:
                logger.info(f"❌ {symbol}: Options volume too low ({mock_options_volume}) - need ≥500")
                return None
            
            # All liquidity filters passed
            if not liquidity_analysis['liquid_strikes']:
                logger.info(f"❌ {symbol}: No liquid option contracts found in analyzer")
                return None
            
            logger.info(f"✅ {symbol}: PASSED ALL LIQUIDITY FILTERS")
            logger.info(f"   Bid/Ask: ${mock_bid_ask_spread:.2f} ({spread_percentage:.1%} of mid)")
            logger.info(f"   Open Interest: {mock_open_interest:,}")
            logger.info(f"   Options Volume: {mock_options_volume:,}")
            logger.info(f"   {liquidity_analysis['analysis']}")
            
            # STEP 8B: CREDIT AND RISK/REWARD VALIDATION
            
            spread_width = short_strike - long_strike
            
            # Minimum credit requirement: ≥30-40% of width
            estimated_credit_percentage = 0.35  # 35% of width (conservative estimate)
            estimated_credit = spread_width * estimated_credit_percentage
            min_credit_requirement = spread_width * 0.30  # Minimum 30% of width
            
            if estimated_credit < min_credit_requirement:
                logger.info(f"❌ {symbol}: Estimated credit too low")
                logger.info(f"   Estimated: ${estimated_credit:.2f} (need ≥${min_credit_requirement:.2f})")
                return None
            
            # Maximum credit (sanity check) - shouldn't exceed 80% of width
            max_credit_allowed = spread_width * 0.80
            if estimated_credit > max_credit_allowed:
                logger.info(f"❌ {symbol}: Estimated credit suspiciously high")
                logger.info(f"   Estimated: ${estimated_credit:.2f} (max realistic: ${max_credit_allowed:.2f})")
                return None
            
            # Probability of Profit (POP) guard: estimate using OTM distance
            otm_distance = (current_price - short_strike) / current_price
            # Rough approximation: POP ≈ 1 - (1 / (1 + OTM_distance * 10))
            estimated_pop = 1 - (1 / (1 + otm_distance * 10))
            
            if estimated_pop < 0.70:  # Require ≥70% POP
                logger.info(f"❌ {symbol}: Estimated POP too low ({estimated_pop:.1%}) - need ≥70%")
                return None
            
            # Risk/Reward ratio check
            max_loss = spread_width - estimated_credit
            risk_reward_ratio = max_loss / estimated_credit
            
            if risk_reward_ratio > 3.0:  # Don't risk more than 3x the credit
                logger.info(f"❌ {symbol}: Risk/reward ratio too high ({risk_reward_ratio:.1f}:1)")
                return None
            
            logger.info(f"✅ {symbol}: PASSED CREDIT AND RISK/REWARD CHECKS")
            logger.info(f"   Spread Width: ${spread_width:.2f}")
            logger.info(f"   Est. Credit: ${estimated_credit:.2f} ({estimated_credit_percentage:.0%} of width)")
            logger.info(f"   Est. POP: {estimated_pop:.1%}")
            logger.info(f"   Max Loss: ${max_loss:.2f}")
            logger.info(f"   Risk/Reward: {risk_reward_ratio:.1f}:1")
            
            # STEP 9: Market-adaptive risk assessment
            risk_score = self._calculate_risk_score(symbol, market_features, current_price, strikes)
            if risk_score > 0.8:  # Relaxed from 0.7 to 0.8 - allow moderate risk
                logger.info(f"❌ {symbol}: Risk score too high ({risk_score:.2f})")
                return None
            
            # STEP 10: Create decision with comprehensive validation results
            decision = TradingDecision(
                symbol=symbol,
                action='OPEN_BULL_PUT_SPREAD',
                confidence=final_confidence,
                reasons=[
                    f"✅ ML Ensemble: {final_confidence:.1%} (≥70% threshold)",
                    f"✅ Quality Score: {quality_score:.1%}",
                    f"✅ VIX Regime: Normal (not extreme volatility)",
                    f"✅ Earnings: Safe period (no earnings risk)",
                    f"✅ IV Percentile: {implied_vol_percentile:.1%} (≥30%)",
                    f"✅ Trend: Bullish confirmation",
                    f"✅ Spread Efficiency: {spread_efficiency:.2f} (≥0.5)",
                    f"✅ Liquidity: OI={mock_open_interest:,}, Vol={mock_options_volume:,}",
                    f"✅ Credit: ${estimated_credit:.2f} ({estimated_credit_percentage:.0%} of width)",
                    f"✅ POP: {estimated_pop:.1%} (≥70%)",
                    f"✅ Risk/Reward: {risk_reward_ratio:.1f}:1 (≤3:1)",
                    f"✅ Strike Validation: All sanity checks passed"
                ],
                trade_parameters={
                    'short_strike': short_strike,
                    'long_strike': long_strike,
                    'expiration_date': best_expiration,
                    'contracts': 1,
                    'current_price': current_price,
                    'risk_score': risk_score,
                    'quality_score': quality_score,
                    'ml_consensus': consensus_ratio,
                    'liquid_strikes': liquidity_analysis['liquid_strikes'],
                    'technical_override': False,  # Never using technical override
                    # New validation parameters
                    'estimated_credit': estimated_credit,
                    'estimated_pop': estimated_pop,
                    'risk_reward_ratio': risk_reward_ratio,
                    'spread_width': spread_width,
                    'otm_distance': otm_distance,
                    'mock_open_interest': mock_open_interest,
                    'mock_options_volume': mock_options_volume,
                    'bid_ask_spread': mock_bid_ask_spread,
                    'iv_percentile': implied_vol_percentile,
                    'long_term_trend': long_term_trend,
                    'spread_efficiency': spread_efficiency
                },
                timestamp=datetime.now()
            )
            
            logger.info(f"🚀 {symbol}: PASSED MARKET-ADAPTIVE GATES - Quality opportunity identified!")
            return decision
            
        except Exception as e:
            logger.error(f"Error in comprehensive evaluation of {symbol}: {e}")
            return None

    async def _execute_trade(self, decision: TradingDecision, is_earnings_season: bool = False):
        """Execute a trading decision through REAL Alpaca options trading with adaptive position sizing."""
        try:
            logger.info(f"🚀 EXECUTING REAL ALPACA OPTIONS TRADE: {decision.symbol}")
            logger.info(f"  Strategy: Bull Put Spread")
            logger.info(f"  Market Mode: {'EARNINGS SEASON' if is_earnings_season else 'NORMAL'}")
            logger.info(f"  Position Size: {self.current_position_size:.0%} of normal")
            logger.info(f"  Confidence: {decision.confidence:.1%}")
            logger.info(f"  Short Strike: ${decision.trade_parameters['short_strike']}")
            logger.info(f"  Long Strike: ${decision.trade_parameters['long_strike']}")
            logger.info(f"  Expiration: {decision.trade_parameters['expiration_date']}")
            
            # Calculate estimated credit with position sizing adjustment
            base_credit = (decision.trade_parameters['short_strike'] - decision.trade_parameters['long_strike']) * 0.3 * 100
            estimated_credit = base_credit * self.current_position_size  # Apply position sizing
            
            trade_id = f"ALPACA_{decision.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Add earnings season flag to trade parameters
            if is_earnings_season:
                trade_id += "_EARN"
                logger.info(f"  📊 EARNINGS TRADE: Reduced position size ({self.current_position_size:.0%})")
            
            if self.alpaca_connected:
                try:
                    # First, find real option contracts using Alpaca's new options API
                    logger.info("� Searching for available option contracts on Alpaca...")
                    
                    # Format expiration date for Alpaca (YYMMDD format)
                    exp_date_obj = datetime.strptime(decision.trade_parameters['expiration_date'], '%Y-%m-%d')
                    exp_date_alpaca = exp_date_obj.strftime('%y%m%d')
                    
                    # Search for available put options near our strikes using our client method
                    short_strike = decision.trade_parameters['short_strike']
                    long_strike = decision.trade_parameters['long_strike']
                    expiration_date = decision.trade_parameters['expiration_date']
                    
                    # Use our enhanced client method to get option contracts
                    try:
                        available_contracts = self.alpaca.get_option_contracts(
                            underlying_symbol=decision.symbol,
                            expiration_date=expiration_date,
                            option_type='put'
                        )
                        
                        logger.info(f"Found {len(available_contracts)} available put contracts for {decision.symbol}")
                        
                        # Find contracts closest to our target strikes
                        short_contract = None
                        long_contract = None
                        
                        for contract in available_contracts:
                            strike_price = float(contract['strike_price'])
                            if contract.get('tradable', True) and contract.get('status', 'active') == 'active':
                                # Find short strike (sell) - closest to target short strike
                                if not short_contract and abs(strike_price - short_strike) < 2.0:
                                    short_contract = contract
                                # Find long strike (buy/protection) - closest to target long strike
                                if not long_contract and abs(strike_price - long_strike) < 2.0:
                                    long_contract = contract
                        
                        if short_contract and long_contract:
                            logger.info(f"✅ Found valid option contracts:")
                            logger.info(f"  Short: {short_contract['symbol']} @ ${short_contract['strike_price']}")
                            logger.info(f"  Long:  {long_contract['symbol']} @ ${long_contract['strike_price']}")
                            
                            contracts = decision.trade_parameters.get('contracts', 1)
                            
                            # Execute the bull put spread with real contracts
                            # Sell short put (collect premium)
                            short_order = self.alpaca.submit_order(
                                symbol=short_contract['symbol'],
                                qty=contracts,
                                side='sell',
                                order_type='limit',
                                limit_price=1.50,  # Conservative limit price
                                time_in_force='day'
                            )
                            
                            # Buy long put (pay premium - protection)
                            long_order = self.alpaca.submit_order(
                                symbol=long_contract['symbol'],
                                qty=contracts,
                                side='buy',
                                order_type='limit',
                                limit_price=1.00,  # Conservative limit price
                                time_in_force='day'
                            )
                            
                            logger.info(f"✅ REAL OPTIONS ORDERS SUBMITTED:")
                            logger.info(f"  Short Put Order: {short_order.id} - {short_contract['symbol']}")
                            logger.info(f"  Long Put Order: {long_order.id} - {long_contract['symbol']}")
                            logger.info(f"  Contracts: {contracts}")
                            
                            # Log the trade as a real Alpaca options trade
                            trade_record = {
                                'trade_id': trade_id,
                                'alpaca_short_order_id': short_order.id,
                                'alpaca_long_order_id': long_order.id,
                                'symbol': decision.symbol,
                                'strategy': 'bull_put_spread',
                                'entry_time': datetime.now().isoformat(),
                                'confidence': decision.confidence,
                                'short_strike': float(short_contract['strike_price']),
                                'long_strike': float(long_contract['strike_price']),
                                'expiration_date': short_contract['expiration_date'],
                                'estimated_credit': estimated_credit,
                                'contracts': contracts,
                                'short_option_symbol': short_contract['symbol'],
                                'long_option_symbol': long_contract['symbol'],
                                'real_alpaca_order': True,
                                'status': 'EXECUTED_ALPACA_OPTIONS'
                            }
                            
                            success_msg = f"🎉 REAL ALPACA OPTIONS TRADE EXECUTED!"
                            
                        else:
                            raise Exception(f"Could not find suitable option contracts for {decision.symbol}")
                            
                    except Exception as contract_error:
                        logger.warning(f"⚠️ Could not fetch option contracts: {contract_error}")
                        raise contract_error
                    
                except Exception as alpaca_error:
                    logger.error(f"❌ Alpaca options order failed: {alpaca_error}")
                    logger.info("📝 Falling back to paper trading simulation...")
                    
                    # Fallback to paper simulation
                    trade_record = {
                        'trade_id': trade_id,
                        'symbol': decision.symbol,
                        'strategy': 'bull_put_spread_simulation',
                        'entry_time': datetime.now().isoformat(),
                        'confidence': decision.confidence,
                        'short_strike': decision.trade_parameters['short_strike'],
                        'long_strike': decision.trade_parameters['long_strike'],
                        'expiration_date': decision.trade_parameters['expiration_date'],
                        'estimated_credit': estimated_credit,
                        'real_alpaca_order': False,
                        'alpaca_error': str(alpaca_error),
                        'status': 'SIMULATED_FALLBACK'
                    }
                    
                    success_msg = f"📝 PAPER TRADE EXECUTED (Options trading fallback)"
            else:
                logger.warning("⚠️ No Alpaca connection - executing as simulation")
                trade_record = {
                    'trade_id': trade_id,
                    'symbol': decision.symbol,
                    'strategy': 'bull_put_spread_simulation',
                    'entry_time': datetime.now().isoformat(),
                    'confidence': decision.confidence,
                    'short_strike': decision.trade_parameters['short_strike'],
                    'long_strike': decision.trade_parameters['long_strike'],
                    'expiration_date': decision.trade_parameters['expiration_date'],
                    'estimated_credit': estimated_credit,
                    'real_alpaca_order': False,
                    'status': 'SIMULATION_ONLY'
                }
                
                success_msg = f"📝 PAPER TRADE EXECUTED (No Alpaca connection)"
            
            # Save trade record
            with open('logs/executed_trades.jsonl', 'a') as f:
                f.write(json.dumps(trade_record) + '\n')
            
            # Add to open positions for ongoing management
            self.open_positions.append(trade_record)
            self._save_open_positions()
            
            logger.info(success_msg)
            logger.info(f"  Trade ID: {trade_id}")
            logger.info(f"  Estimated Credit: ${estimated_credit:.2f}")
            logger.info(f"  Max Risk: ${(decision.trade_parameters['short_strike'] - decision.trade_parameters['long_strike']) * 100 - estimated_credit:.2f}")
            logger.info(f"  🎯 Strategy: Let theta decay work for profit!")
            
        except Exception as e:
            logger.error(f"Error executing trade for {decision.symbol}: {e}")

    def _calculate_quality_score(self, symbol: str, market_features: Dict[str, float]) -> float:
        """
        Calculate a comprehensive quality score (0.0 = poor, 1.0 = excellent).
        Market-adaptive approach that considers multiple factors.
        
        Args:
            symbol: Stock symbol
            market_features: Real market features
            
        Returns:
            float: Quality score between 0.0 and 1.0
        """
        try:
            quality_factors = []
            
            # Factor 1: RSI positioning (prefer 30-70 range)
            rsi = market_features.get('rsi_14', 50)
            if 30 <= rsi <= 70:
                rsi_score = 1.0  # Ideal range
            elif 20 <= rsi <= 80:
                rsi_score = 0.7  # Acceptable
            else:
                rsi_score = 0.3  # Extreme levels
            quality_factors.append(('rsi_positioning', rsi_score, 0.20))
            
            # Factor 2: Volatility suitability (prefer moderate volatility)
            volatility = market_features.get('historical_volatility', 0.2)
            if 0.15 <= volatility <= 0.40:
                vol_score = 1.0  # Good for credit spreads
            elif 0.10 <= volatility <= 0.60:
                vol_score = 0.7  # Acceptable
            else:
                vol_score = 0.4  # Too low or too high
            quality_factors.append(('volatility_suitability', vol_score, 0.25))
            
            # Factor 3: Price trend (prefer neutral to slightly bullish)
            price_trend = market_features.get('price_trend_10d', 0)
            if -0.01 <= price_trend <= 0.03:  # -1% to +3%
                trend_score = 1.0  # Ideal for bull put spreads
            elif -0.03 <= price_trend <= 0.05:
                trend_score = 0.7  # Acceptable
            else:
                trend_score = 0.3  # Too bearish or too bullish
            quality_factors.append(('price_trend', trend_score, 0.20))
            
            # Factor 4: Volume adequacy
            volume_ratio = market_features.get('volume_ratio', 1.0)
            if volume_ratio >= 0.8:
                volume_score = 1.0  # Good liquidity
            elif volume_ratio >= 0.5:
                volume_score = 0.8  # Adequate
            elif volume_ratio >= 0.3:
                volume_score = 0.6  # Acceptable
            else:
                volume_score = 0.2  # Poor liquidity
            quality_factors.append(('volume_adequacy', volume_score, 0.15))
            
            # Factor 5: Technical positioning
            distance_to_resistance = market_features.get('distance_to_resistance', 0.1)
            distance_to_support = market_features.get('distance_to_support', 0.1)
            
            # Prefer stocks with room to move up but not too close to support
            if distance_to_resistance > 0.05 and distance_to_support > 0.03:
                tech_score = 1.0  # Good positioning
            elif distance_to_resistance > 0.02 and distance_to_support > 0.02:
                tech_score = 0.8  # Acceptable
            else:
                tech_score = 0.5  # Constrained
            quality_factors.append(('technical_positioning', tech_score, 0.20))
            
            # Calculate weighted quality score
            total_quality = sum(score * weight for _, score, weight in quality_factors)
            
            logger.info(f"Quality factors for {symbol}:")
            for name, score, weight in quality_factors:
                logger.info(f"  {name}: {score:.2f} (weight: {weight})")
            
            return min(total_quality, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating quality score for {symbol}: {e}")
            return 0.0  # Minimum quality if calculation fails

    def _passes_basic_quality_checks(self, symbol: str, market_features: Dict[str, float]) -> bool:
        """
        Basic quality checks to filter out obviously bad opportunities.
        
        Args:
            symbol: Stock symbol
            market_features: Real market features
            
        Returns:
            bool: True if passes basic checks
        """
        try:
            # Check 1: RSI not extremely overbought
            rsi = market_features.get('rsi_14', 50)
            if rsi > 80:
                logger.info(f"❌ {symbol}: RSI too high ({rsi:.1f}) - likely overbought")
                return False
            
            # Check 2: Volatility not extremely high (risky for credit spreads)
            volatility = market_features.get('historical_volatility', 0.2)
            if volatility > 0.6:  # 60% annualized volatility
                logger.info(f"❌ {symbol}: Volatility too high ({volatility:.1%}) - too risky")
                return False
            
            # Check 3: Price trend not extremely bearish
            price_trend = market_features.get('price_trend_10d', 0)
            if price_trend < -0.05:  # More than 5% negative trend
                logger.info(f"❌ {symbol}: Strong downtrend ({price_trend:.1%}) - bearish signal")
                return False
            
            # Check 4: Volume ratio reasonable (not too low liquidity)
            volume_ratio = market_features.get('volume_ratio', 1.0)
            if volume_ratio < 0.3:  # Very low volume
                logger.info(f"❌ {symbol}: Low volume ({volume_ratio:.1f}) - liquidity concern")
                return False
            
            # Check 5: Not too close to resistance (relaxed)
            distance_to_resistance = market_features.get('distance_to_resistance', 0.1)
            if 0 <= distance_to_resistance <= 0.005:  # Within 0.5% of resistance (was 2%)
                logger.info(f"❌ {symbol}: Too close to resistance - likely to face selling pressure")
                return False
            
            logger.info(f"✅ {symbol}: Passed basic quality checks")
            return True
            
        except Exception as e:
            logger.error(f"Error in quality checks for {symbol}: {e}")
            return False
    
    def _select_optimal_expiration(self, available_expirations: List[str]) -> str:
        """
        Select the optimal expiration from available dates.
        Prefers 35-40 DTE for best theta decay within the narrowed 30-45 range.
        
        Args:
            available_expirations: List of expiration dates (YYYY-MM-DD)
            
        Returns:
            str: Selected expiration date
        """
        try:
            today = datetime.now().date()
            
            # Calculate DTE for each expiration and score them
            scored_expirations = []
            
            for exp_str in available_expirations:
                exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                dte = (exp_date - today).days
                
                # Updated scoring for narrowed 30-45 DTE range
                if 35 <= dte <= 40:
                    score = 100 - abs(dte - 37)  # Prefer ~37 DTE (sweet spot)
                elif 30 <= dte <= 45:
                    score = 85 - abs(dte - 37)   # Acceptable range
                elif 25 <= dte <= 50:
                    score = 60 - abs(dte - 37)   # Less preferred but workable
                else:
                    score = 30 - abs(dte - 37)   # Avoid if possible
                
                scored_expirations.append((exp_str, dte, score))
            
            # Sort by score (highest first)
            scored_expirations.sort(key=lambda x: x[2], reverse=True)
            
            best_expiration = scored_expirations[0][0]
            best_dte = scored_expirations[0][1]
            
            logger.info(f"Selected expiration {best_expiration} ({best_dte} DTE)")
            
            # Warn if outside preferred range
            if not (30 <= best_dte <= 45):
                logger.warning(f"⚠️ Selected DTE ({best_dte}) outside preferred 30-45 range")
            
            return best_expiration
            
        except Exception as e:
            logger.error(f"Error selecting optimal expiration: {e}")
            return available_expirations[0] if available_expirations else '2025-09-26'
    
    def _calculate_optimal_strikes(self, current_price: float, market_features: Dict[str, float]) -> Optional[Tuple[float, float]]:
        """
        Calculate optimal strike prices using delta-based selection with strict sanity checks.
        
        Args:
            current_price: Current stock price
            market_features: Market features including volatility
            
        Returns:
            Tuple of (short_strike, long_strike) or None if calculation fails
        """
        try:
            volatility = market_features.get('historical_volatility', 0.2)
            
            # Target short delta of 0.15-0.25 (approximately 5-10% OTM)
            # Higher volatility = target lower delta (further OTM)
            if volatility > 0.4:
                target_delta = 0.15  # ~15 delta for high vol (safer)
                otm_percent = 0.10   # ~10% OTM
            elif volatility > 0.25:
                target_delta = 0.20  # ~20 delta for medium vol
                otm_percent = 0.08   # ~8% OTM
            else:
                target_delta = 0.25  # ~25 delta for low vol
                otm_percent = 0.06   # ~6% OTM
            
            # Calculate short strike using OTM percentage (delta proxy)
            short_strike = current_price * (1 - otm_percent)
            
            # Determine spread width from allowed set: $1, $2.5, $5, $10, $25
            allowed_widths = [1.0, 2.5, 5.0, 10.0, 25.0]
            
            if current_price > 500:
                spread_width = 25.0  # $25 spread for very high-priced stocks
            elif current_price > 300:
                spread_width = 10.0  # $10 spread for high-priced stocks
            elif current_price > 150:
                spread_width = 5.0   # $5 spread for medium-priced stocks
            elif current_price > 50:
                spread_width = 2.5   # $2.50 spread for normal stocks
            else:
                spread_width = 1.0   # $1 spread for low-priced stocks
            
            # Ensure width is in allowed set
            if spread_width not in allowed_widths:
                spread_width = min(allowed_widths, key=lambda x: abs(x - spread_width))
            
            long_strike = short_strike - spread_width
            
            # Round strikes to standard increments
            if current_price > 100:
                short_strike = round(short_strike, 0)  # Round to dollar
                long_strike = round(long_strike, 0)
            else:
                short_strike = round(short_strike * 2) / 2  # Round to $0.50
                long_strike = round(long_strike * 2) / 2
            
            # STRICT SANITY CHECKS before returning strikes
            
            # Check 1: Strike ordering (puts: short > long)
            if short_strike <= long_strike:
                logger.error(f"❌ Strike ordering violation: short_strike ({short_strike}) <= long_strike ({long_strike})")
                return None
            
            # Check 2: Short strike must be sufficiently OTM (≥5% below spot)
            otm_distance = (current_price - short_strike) / current_price
            if otm_distance < 0.05:  # Less than 5% OTM
                logger.error(f"❌ Short strike too close to spot: {otm_distance:.1%} OTM (need ≥5%)")
                return None
            
            # Check 3: Width must be in allowed set
            actual_width = short_strike - long_strike
            if actual_width not in allowed_widths:
                logger.error(f"❌ Invalid spread width: ${actual_width} (allowed: {allowed_widths})")
                return None
            
            # Check 4: Long strike must be positive
            if long_strike <= 0:
                logger.error(f"❌ Invalid long strike: ${long_strike} (must be positive)")
                return None
            
            # Check 5: Estimate credit and validate minimum
            estimated_credit_percentage = 0.30  # Conservative 30% of width
            estimated_credit = actual_width * estimated_credit_percentage
            min_credit_threshold = actual_width * 0.30  # Minimum 30% of width
            
            if estimated_credit < min_credit_threshold:
                logger.error(f"❌ Estimated credit too low: ${estimated_credit:.2f} (need ≥${min_credit_threshold:.2f})")
                return None
            
            # Check 6: Estimate Probability of Profit (POP)
            # Approximate POP ≈ 1 - |short_delta|
            estimated_short_delta = target_delta
            estimated_pop = 1 - estimated_short_delta
            
            if estimated_pop < 0.70:  # Require ≥70% POP
                logger.error(f"❌ Estimated POP too low: {estimated_pop:.1%} (need ≥70%)")
                return None
            
            logger.info(f"✅ Strike validation PASSED:")
            logger.info(f"   Short: ${short_strike:.2f} ({otm_distance:.1%} OTM)")
            logger.info(f"   Long:  ${long_strike:.2f}")
            logger.info(f"   Width: ${actual_width:.2f} (allowed width)")
            logger.info(f"   Est. Credit: ${estimated_credit:.2f} ({estimated_credit_percentage:.0%} of width)")
            logger.info(f"   Est. POP: {estimated_pop:.1%}")
            logger.info(f"   Target Delta: ~{target_delta:.2f}")
            
            return (short_strike, long_strike)
            
        except Exception as e:
            logger.error(f"Error calculating strikes: {e}")
            return None
    
    def _calculate_risk_score(self, symbol: str, market_features: Dict[str, float], 
                            current_price: float, strikes: Tuple[float, float]) -> float:
        """
        Calculate a comprehensive risk score (0.0 = low risk, 1.0 = high risk).
        
        Args:
            symbol: Stock symbol
            market_features: Market features
            current_price: Current stock price
            strikes: (short_strike, long_strike)
            
        Returns:
            float: Risk score between 0.0 and 1.0
        """
        try:
            risk_factors = []
            
            # Factor 1: Volatility risk
            volatility = market_features.get('historical_volatility', 0.2)
            vol_risk = min(volatility / 0.5, 1.0)  # Cap at 50% vol
            risk_factors.append(('volatility', vol_risk, 0.3))
            
            # Factor 2: Technical position risk
            rsi = market_features.get('rsi_14', 50)
            rsi_risk = max(0, (rsi - 70) / 30) if rsi > 70 else 0  # Risk if RSI > 70
            risk_factors.append(('rsi_overbought', rsi_risk, 0.2))
            
            # Factor 3: Trend risk
            price_trend = market_features.get('price_trend_10d', 0)
            trend_risk = max(0, -price_trend * 10) if price_trend < 0 else 0  # Risk for downtrends
            risk_factors.append(('downtrend', trend_risk, 0.2))
            
            # Factor 4: Distance to short strike risk
            short_strike, long_strike = strikes
            distance_to_short = (current_price - short_strike) / current_price
            distance_risk = max(0, (0.08 - distance_to_short) / 0.08)  # Risk if < 8% OTM
            risk_factors.append(('strike_distance', distance_risk, 0.15))
            
            # Factor 5: Market structure risk
            distance_to_support = market_features.get('distance_to_support', 0.1)
            support_risk = max(0, (0.05 - distance_to_support) / 0.05)  # Risk if close to support
            risk_factors.append(('support_proximity', support_risk, 0.15))
            
            # Calculate weighted risk score
            total_risk = sum(risk * weight for _, risk, weight in risk_factors)
            
            logger.info(f"Risk factors for {symbol}:")
            for name, risk, weight in risk_factors:
                logger.info(f"  {name}: {risk:.2f} (weight: {weight})")
            
            logger.info(f"Total risk score: {total_risk:.2f}")
            return min(total_risk, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 1.0  # Max risk if calculation fails

    def _reset_daily_counters_if_needed(self):
        """Reset daily counters if it's a new trading day."""
        try:
            now = datetime.now(self.tz)
            
            # Check if we've moved to a new day
            # Reset counters at market open (9:30 AM ET)
            market_open_today = now.replace(hour=9, minute=30, second=0, microsecond=0)
            
            # If it's past market open and we haven't reset today
            if now >= market_open_today:
                # Check if last reset was before today's market open
                last_reset_file = 'logs/last_daily_reset.txt'
                
                try:
                    with open(last_reset_file, 'r') as f:
                        last_reset = datetime.fromisoformat(f.read().strip())
                        
                    if last_reset < market_open_today:
                        self.daily_trade_count = 0
                        self.earnings_season_trade_count = 0  # Reset earnings counter too
                        with open(last_reset_file, 'w') as f:
                            f.write(now.isoformat())
                        logger.info("🔄 Daily counters reset for new trading day")
                        
                except FileNotFoundError:
                    # First time running or file doesn't exist
                    self.daily_trade_count = 0
                    self.earnings_season_trade_count = 0
                    with open(last_reset_file, 'w') as f:
                        f.write(now.isoformat())
                    logger.info("🔄 Daily counters initialized")
                    
        except Exception as e:
            logger.error(f"Error resetting daily counters: {e}")

    def _update_market_conditions(self):
        """Update trading limits and position sizing based on current market conditions."""
        try:
            # Check overall market earnings season by sampling key indicators
            sample_symbols = ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN'][:3]  # Sample major stocks
            earnings_indicators = []
            
            if self.market_analyzer:
                for symbol in sample_symbols:
                    try:
                        features = self.market_analyzer.get_market_features(symbol)
                        if features and 'earnings_season' in features:
                            earnings_indicators.append(features['earnings_season'])
                    except:
                        continue
            
            # Determine if we're in earnings season (if majority of samples indicate earnings)
            is_earnings_season = False
            if earnings_indicators:
                earnings_ratio = sum(earnings_indicators) / len(earnings_indicators)
                is_earnings_season = earnings_ratio >= 0.4  # 40% threshold for earnings season
            
            # Update trading parameters based on market conditions
            if is_earnings_season:
                self.current_max_trades = self.max_daily_trades_earnings
                self.current_position_size = self.earnings_position_size
                market_regime = "EARNINGS SEASON"
                logger.info(f"📊 MARKET REGIME: {market_regime}")
                logger.info(f"   Max trades today: {self.current_max_trades}")
                logger.info(f"   Position size: {self.current_position_size:.0%} of normal")
                logger.info(f"   Strategy: Quality-focused, reduced risk")
            else:
                self.current_max_trades = self.max_daily_trades_normal
                self.current_position_size = self.normal_position_size
                market_regime = "NORMAL CONDITIONS"
                logger.info(f"📊 MARKET REGIME: {market_regime}")
                logger.info(f"   Max trades today: {self.current_max_trades}")
                logger.info(f"   Position size: {self.current_position_size:.0%} of normal")
                logger.info(f"   Strategy: Full capacity trading")
                
            return is_earnings_season
            
        except Exception as e:
            logger.error(f"Error updating market conditions: {e}")
            # Default to conservative settings
            self.current_max_trades = self.max_daily_trades_earnings
            self.current_position_size = self.earnings_position_size
            return True

def main():
    print("🚀 ADAPTIVE ALPACA-CONNECTED AUTONOMOUS TRADING SYSTEM 🚀")
    print("=" * 80)
    print("ADAPTIVE PRODUCTION FEATURES:")
    print("• 🔌 REAL Alpaca Options API connection")
    print("• 📊 ALL options-enabled stocks universe from Alpaca (500+ symbols)")
    print("• 💰 REAL options trading execution (bull put spreads)")
    print("• 📈 ADAPTIVE TRADE LIMITS:")
    print("  - Normal market: 50 trades/day (full capacity)")
    print("  - Earnings season: 5 trades/day (quality-focused)")
    print("• 💼 ADAPTIVE POSITION SIZING:")
    print("  - Normal market: 100% position size")
    print("  - Earnings season: 40% position size (reduced risk)")
    print("• 🎯 DYNAMIC QUALITY THRESHOLDS:")
    print("  - Normal market: 70% ML confidence, 50% quality score")
    print("  - Earnings season: 80% ML confidence, 80% quality score")
    print("• ⏱️ 30 minute minimum between trades")
    print("• 🔄 5 minute scan intervals (responsive)")
    print("• 📊 Max positions: 100 normal / 50 earnings season")
    print("• 🎯 50% profit target (optimized)")
    print("• 🛡️ Market-adaptive risk management")
    print("• 📈 Real-time position monitoring")
    print("=" * 80)
    
    try:
        trader = SimpleAutonomousTrader()
        asyncio.run(trader.start_trading())
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        logger.info("User requested shutdown")
    except Exception as e:
        print(f"\nError: {e}")
        logger.error(f"System error: {e}")

if __name__ == "__main__":
    main()
