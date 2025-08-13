#!/usr/bin/env python3
"""
The Closer - Intelligent Trade Management System - WORKING VERSION
================================================================

Monitors and closes bull put spread positions based on:
- Profit targets (50% of credit received)
- Stop losses ($200 max loss)
- Time decay (close 7 days before expiry)
"""

import logging
import asyncio
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class ManagedTrade:
    """Represents a trade being managed by The Closer."""
    trade_id: str
    symbol: str
    trade_type: str
    entry_time: datetime
    profit_target: float
    stop_loss: float
    time_decay_close: int  # Days before expiry to close
    trade_params: Dict[str, Any]
    entry_price: float
    status: str = "ACTIVE"
    
class TradeCloser:
    """
    The Closer - Intelligent trade management system.
    """
    
    def __init__(self, alpaca_client):
        self.alpaca = alpaca_client
        self.managed_trades: Dict[str, ManagedTrade] = {}
        self.data_file = 'data/managed_trades.json'
        self.completions_file = 'logs/trade_completions.jsonl'
        
        # Ensure directories exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Load existing trades
        self._load_managed_trades()
        
        logger.info(f"The Closer initialized with {len(self.managed_trades)} active trades")

    def register_trade(self,
                      trade_id: str,
                      symbol: str,
                      trade_type: str,
                      entry_time: datetime,
                      profit_target: float,
                      stop_loss: float,
                      time_decay_close: int,
                      trade_params: Dict[str, Any],
                      entry_price: float) -> bool:
        """
        Register a new trade for monitoring.
        
        Args:
            trade_id: Unique identifier for the trade
            symbol: Underlying symbol
            trade_type: Type of trade (e.g., 'bull_put_spread')
            entry_time: When the trade was entered
            profit_target: Target profit percentage (e.g., 0.50 for 50%)
            stop_loss: Maximum loss threshold (e.g., -200 for $200 loss)
            time_decay_close: Days before expiry to close
            trade_params: Full trade parameters from execution
            entry_price: Entry price/credit received
            
        Returns:
            True if registered successfully
        """
        try:
            trade = ManagedTrade(
                trade_id=trade_id,
                symbol=symbol,
                trade_type=trade_type,
                entry_time=entry_time,
                profit_target=profit_target,
                stop_loss=stop_loss,
                time_decay_close=time_decay_close,
                trade_params=trade_params,
                entry_price=entry_price,
                status="ACTIVE"
            )
            
            self.managed_trades[trade_id] = trade
            self._save_managed_trades()
            
            logger.info(f"Registered trade {trade_id} for monitoring")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Entry Price: ${entry_price:.2f}")
            logger.info(f"   Profit Target: {profit_target:.1%}")
            logger.info(f"   Stop Loss: ${stop_loss:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error registering trade {trade_id}: {e}")
            return False

    async def check_all_trades(self):
        """Check all managed trades for exit conditions."""
        if not self.managed_trades:
            return
            
        logger.debug(f"Checking {len(self.managed_trades)} managed trades")
        
        trades_to_close = []
        
        for trade_id, trade in self.managed_trades.items():
            if trade.status != "ACTIVE":
                continue
                
            try:
                should_close, reason, exit_price = await self._should_close_trade(trade)
                
                if should_close:
                    trades_to_close.append((trade, reason, exit_price))
                    
            except Exception as e:
                logger.error(f"Error checking trade {trade_id}: {e}")
                continue
        
        # Close trades that meet exit criteria
        for trade, reason, exit_price in trades_to_close:
            await self._close_trade(trade, reason, exit_price)

    async def _should_close_trade(self, trade: ManagedTrade) -> tuple[bool, str, float]:
        """
        Determine if a trade should be closed.
        
        Returns:
            (should_close, reason, estimated_exit_price)
        """
        try:
            # Get current position value
            current_value = await self._get_current_position_value(trade)
            if current_value is None:
                return False, "Cannot get position value", 0.0
            
            # Calculate P&L
            pnl = current_value - trade.entry_price
            pnl_percent = pnl / abs(trade.entry_price) if trade.entry_price != 0 else 0
            
            # Check profit target
            if pnl_percent >= trade.profit_target:
                return True, f"Profit target reached: {pnl_percent:.1%}", current_value
            
            # Check stop loss
            if pnl <= trade.stop_loss:
                return True, f"Stop loss triggered: ${pnl:.2f}", current_value
            
            # Check time decay
            if self._should_close_time_decay(trade):
                return True, f"Time decay close: {trade.time_decay_close} days to expiry", current_value
            
            # Check for unusual conditions
            days_held = self._get_days_held(trade)
            if days_held > 60:  # Maximum hold period
                return True, f"Maximum hold period reached: {days_held} days", current_value
            
            return False, "All conditions OK", current_value
            
        except Exception as e:
            logger.error(f"Error evaluating trade {trade.trade_id}: {e}")
            return False, f"Error: {e}", 0.0

    async def _get_current_position_value(self, trade: ManagedTrade) -> Optional[float]:
        """Get current value of the position."""
        try:
            # For demo purposes, simulate position value changes
            # In production, this would query Alpaca for real position values
            
            days_held = self._get_days_held(trade)
            
            # Simulate time decay and market movements
            base_decay = 0.02 * days_held  # 2% decay per day
            market_movement = 0.1 * (0.5 - hash(trade.trade_id) % 100 / 100)  # Random market movement
            
            # Current value = entry_price * (1 + market_movement - time_decay)
            current_value = trade.entry_price * (1 + market_movement - base_decay)
            
            # Ensure reasonable bounds
            current_value = max(0, min(current_value, trade.entry_price * 2))
            
            return current_value
            
        except Exception as e:
            logger.error(f"Error getting position value for {trade.trade_id}: {e}")
            return None

    def _should_close_time_decay(self, trade: ManagedTrade) -> bool:
        """Check if trade should be closed due to time decay."""
        try:
            expiration_str = trade.trade_params.get('expiration_date')
            if not expiration_str:
                return False
                
            expiration_date = datetime.strptime(expiration_str, '%Y-%m-%d')
            days_to_expiry = (expiration_date - datetime.now()).days
            
            return days_to_expiry <= trade.time_decay_close
            
        except Exception as e:
            logger.debug(f"Error checking time decay for {trade.trade_id}: {e}")
            return False

    def _get_days_held(self, trade: ManagedTrade) -> int:
        """Get number of days the trade has been held."""
        return (datetime.now() - trade.entry_time).days

    async def _close_trade(self, trade: ManagedTrade, reason: str, exit_price: float):
        """Close a trade and record the completion."""
        try:
            logger.info(f"CLOSING TRADE: {trade.trade_id}")
            logger.info(f"   Symbol: {trade.symbol}")
            logger.info(f"   Reason: {reason}")
            logger.info(f"   Entry Price: ${trade.entry_price:.2f}")
            logger.info(f"   Exit Price: ${exit_price:.2f}")
            
            # Calculate final P&L
            pnl = exit_price - trade.entry_price
            pnl_percent = pnl / abs(trade.entry_price) if trade.entry_price != 0 else 0
            days_held = self._get_days_held(trade)
            
            logger.info(f"   P&L: ${pnl:.2f} ({pnl_percent:.1%})")
            logger.info(f"   Days Held: {days_held}")
            
            # In production, execute the actual close order here
            # success = await self._execute_close_order(trade)
            
            # For now, simulate successful close
            success = True
            
            if success:
                # Update trade status
                trade.status = "CLOSED"
                
                # Record completion
                completion_record = {
                    'trade_id': trade.trade_id,
                    'symbol': trade.symbol,
                    'trade_type': trade.trade_type,
                    'entry_time': trade.entry_time.isoformat(),
                    'exit_time': datetime.now().isoformat(),
                    'entry_price': trade.entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_percent': pnl_percent,
                    'days_held': days_held,
                    'close_reason': reason,
                    'trade_params': trade.trade_params
                }
                
                self._record_completion(completion_record)
                
                # Remove from active trades
                del self.managed_trades[trade.trade_id]
                self._save_managed_trades()
                
                logger.info(f"Trade {trade.trade_id} closed successfully")
                
            else:
                logger.error(f"Failed to close trade {trade.trade_id}")
                
        except Exception as e:
            logger.error(f"Error closing trade {trade.trade_id}: {e}")

    def _save_managed_trades(self):
        """Save managed trades to disk."""
        try:
            trades_data = {}
            for trade_id, trade in self.managed_trades.items():
                trades_data[trade_id] = {
                    'trade_id': trade.trade_id,
                    'symbol': trade.symbol,
                    'trade_type': trade.trade_type,
                    'entry_time': trade.entry_time.isoformat(),
                    'profit_target': trade.profit_target,
                    'stop_loss': trade.stop_loss,
                    'time_decay_close': trade.time_decay_close,
                    'trade_params': trade.trade_params,
                    'entry_price': trade.entry_price,
                    'status': trade.status
                }
            
            with open(self.data_file, 'w') as f:
                json.dump(trades_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving managed trades: {e}")

    def _load_managed_trades(self):
        """Load managed trades from disk."""
        try:
            if not os.path.exists(self.data_file):
                return
                
            with open(self.data_file, 'r') as f:
                trades_data = json.load(f)
            
            for trade_id, data in trades_data.items():
                try:
                    trade = ManagedTrade(
                        trade_id=data['trade_id'],
                        symbol=data['symbol'],
                        trade_type=data['trade_type'],
                        entry_time=datetime.fromisoformat(data['entry_time']),
                        profit_target=data['profit_target'],
                        stop_loss=data['stop_loss'],
                        time_decay_close=data['time_decay_close'],
                        trade_params=data['trade_params'],
                        entry_price=data['entry_price'],
                        status=data.get('status', 'ACTIVE')
                    )
                    
                    if trade.status == "ACTIVE":
                        self.managed_trades[trade_id] = trade
                        
                except Exception as e:
                    logger.error(f"Error loading trade {trade_id}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error loading managed trades: {e}")

    def _record_completion(self, completion_record: Dict[str, Any]):
        """Record a completed trade."""
        try:
            with open(self.completions_file, 'a') as f:
                f.write(json.dumps(completion_record) + '\n')
        except Exception as e:
            logger.error(f"Error recording completion: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of The Closer."""
        active_trades = len([t for t in self.managed_trades.values() if t.status == "ACTIVE"])
        
        return {
            'active_trades': active_trades,
            'total_managed': len(self.managed_trades),
            'data_file': self.data_file,
            'completions_file': self.completions_file
        }

# Demo function
async def demo_trade_closer():
    """Demonstrate The Closer functionality."""
    print("=" * 60)
    print("THE CLOSER - Trade Management Demo")
    print("=" * 60)
    
    # Mock alpaca client
    class MockAlpaca:
        pass
    
    alpaca = MockAlpaca()
    closer = TradeCloser(alpaca)
    
    # Register a demo trade
    trade_id = f"DEMO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    closer.register_trade(
        trade_id=trade_id,
        symbol="SPY",
        trade_type="bull_put_spread",
        entry_time=datetime.now() - timedelta(days=5),
        profit_target=0.50,
        stop_loss=-200.0,
        time_decay_close=7,
        trade_params={
            'short_strike': 145.0,
            'long_strike': 140.0,
            'expiration_date': '2025-09-19',
            'contracts': 1
        },
        entry_price=150.0
    )
    
    print(f"\nRegistered demo trade: {trade_id}")
    print(f"Status: {closer.get_status()}")
    
    # Check the trade
    print(f"\nChecking trade conditions...")
    await closer.check_all_trades()
    
    print(f"\nFinal status: {closer.get_status()}")
    print("\nDemo completed!")

if __name__ == "__main__":
    # Set up logging for demo
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_trade_closer())
