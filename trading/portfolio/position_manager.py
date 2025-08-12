# Position manager module
# Tracks open positions, calculates P&L, and manages position lifecycle # position_manager.py
"""
Position Manager Module
Tracks open positions, calculates P&L, and manages position lifecycle.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

from trading.execution.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

class PositionManager:
    """
    Tracks open positions, updates status, calculates P&L, and manages lifecycle.
    """

    def __init__(self, alpaca_client: Optional[AlpacaClient] = None):
        self.alpaca = alpaca_client or AlpacaClient()
        self.open_positions: List[Dict[str, Any]] = []
        self.closed_positions: List[Dict[str, Any]] = []

    def load_positions(self, positions: List[Dict[str, Any]]):
        """
        Load open positions from a list (e.g., from broker, CSV, or DB).
        """
        self.open_positions = positions

    def add_position(self, position: Dict[str, Any]):
        """
        Add a new open position (dict with all relevant fields).
        """
        logger.info(f"Adding new position: {position.get('symbol', 'N/A')}")
        self.open_positions.append(position)

    def close_position(self, position_id: str, exit_price: float, exit_time: Optional[datetime] = None, status: str = 'closed'):
        """
        Close a position by ID, update P&L, and move to closed.
        """
        position = next((pos for pos in self.open_positions if pos.get('id') == position_id), None)
        if position:
            position['exit_price'] = exit_price
            position['exit_time'] = exit_time or datetime.now()
            position['status'] = status
            position['pnl'] = self.calculate_pnl(position)
            logger.info(f"Position closed: {position_id} | P&L: {position['pnl']:.2f}")
            self.closed_positions.append(position)
            self.open_positions = [pos for pos in self.open_positions if pos.get('id') != position_id]
        else:
            logger.warning(f"Tried to close unknown position: {position_id}")

    def update_positions_from_broker(self):
        """
        Sync open positions with live broker data (from Alpaca).
        """
        broker_positions = self.alpaca.get_positions()
        for broker_pos in broker_positions:
            local_pos = next((p for p in self.open_positions if p['symbol'] == broker_pos['symbol']), None)
            if local_pos:
                local_pos.update({
                    'current_price': broker_pos['current_price'],
                    'market_value': broker_pos['market_value'],
                    'unrealized_pl': broker_pos['unrealized_pl'],
                    'side': broker_pos['side'],
                })
            else:
                # Add unknown broker position to local open positions
                self.open_positions.append(broker_pos)

    def calculate_pnl(self, position: Dict[str, Any]) -> float:
        """
        Calculate P&L for a position (spread or single leg).
        """
        entry_price = position.get('entry_price', 0)
        exit_price = position.get('exit_price', 0)
        qty = position.get('qty', 1)
        direction = position.get('direction', 'credit')  # 'credit' (bull put) or 'debit'

        if direction == 'credit':
            pnl = (entry_price - exit_price) * qty * 100  # options multiplier
        else:
            pnl = (exit_price - entry_price) * qty * 100

        return pnl

    def get_open_positions_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.open_positions)

    def get_closed_positions_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.closed_positions)

    def report_positions(self):
        """
        Print a summary of open/closed positions and P&L.
        """
        open_df = self.get_open_positions_df()
        closed_df = self.get_closed_positions_df()

        print("\nOpen Positions:")
        print(open_df[["id", "symbol", "entry_price", "current_price", "qty", "direction", "status"]]
              if not open_df.empty else "None")
        print("\nClosed Positions:")
        print(closed_df[["id", "symbol", "entry_price", "exit_price", "pnl", "qty", "direction", "status"]]
              if not closed_df.empty else "None")
        print(f"\nTotal Realized P&L: {closed_df['pnl'].sum():.2f}" if not closed_df.empty else "")

    def get_current_positions(self) -> Dict[str, Dict]:
        """
        Get current positions as a dictionary keyed by symbol.
        Prioritizes real Alpaca portfolio data over internal tracking.
        
        Returns:
            Dict mapping symbol -> position details
        """
        try:
            position_dict = {}
            
            # First, get real positions from Alpaca broker
            if self.alpaca:
                try:
                    broker_positions = self.alpaca.get_positions()
                    for pos in broker_positions:
                        symbol = pos.get('symbol', '')
                        if symbol and abs(float(pos.get('qty', 0))) > 0:  # Only include positions with actual shares
                            position_dict[symbol] = pos
                    
                    if position_dict:
                        logger.info(f"Found {len(position_dict)} real positions from Alpaca: {list(position_dict.keys())}")
                except Exception as e:
                    logger.warning(f"Could not fetch broker positions: {e}")
            
            # Supplement with internal tracking for any missing positions
            for pos in self.open_positions:
                symbol = pos.get('symbol', '')
                if symbol and symbol not in position_dict:
                    # Only add if we don't already have real broker data for this symbol
                    position_dict[symbol] = pos
            
            return position_dict
            
        except Exception as e:
            logger.error(f"Error getting current positions: {e}")
            return {}

    def execute_trade(self, symbol: str, action: str, position_size: float, signal_metadata: Dict = None) -> bool:
        """
        Execute a trade through the Alpaca broker.
        
        Args:
            symbol: Trading symbol
            action: 'BUY' or 'SELL'
            position_size: Position size (as percentage of portfolio)
            signal_metadata: Additional signal information
            
        Returns:
            bool: True if trade executed successfully
        """
        try:
            if not self.alpaca:
                logger.error("No Alpaca client available for trade execution")
                return False
            
            logger.info(f"Executing {action} trade for {symbol} (size: {position_size:.2%})")
            
            # Get account info to calculate position sizing
            account_info = self.alpaca.get_account_info()
            account_value = account_info.get('portfolio_value', 100000)
            buying_power = account_info.get('buying_power', account_value)
            
            logger.info(f"Account value: ${account_value:,.2f}, Buying power: ${buying_power:,.2f}")
            
            if action == "BUY":
                # Calculate shares based on portfolio percentage and current buying power
                target_value = min(account_value * position_size, buying_power * 0.95)  # Use 95% of buying power max
                
                # Get current market price
                try:
                    quote = self.alpaca.get_quote(symbol)
                    current_price = quote.get('ask_price', quote.get('last_price', 100))
                except Exception as e:
                    logger.warning(f"Could not get quote for {symbol}: {e}")
                    current_price = 100  # Fallback price
                
                shares = int(target_value / current_price)
                
                if shares > 0:
                    # Submit market order to Alpaca
                    try:
                        order_response = self.alpaca.submit_order(
                            symbol=symbol,
                            qty=shares,
                            side='buy',
                            order_type='market',
                            time_in_force='day'
                        )
                        
                        logger.info(f"BUY order submitted to Alpaca: {shares} shares of {symbol} at market price")
                        logger.info(f"Order ID: {order_response['id']}, Status: {order_response['status']}")
                        
                        # Add to internal tracking
                        new_position = {
                            'id': f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            'symbol': symbol,
                            'action': action,
                            'shares': shares,
                            'entry_price': current_price,
                            'entry_time': datetime.now(),
                            'status': 'open',
                            'order_id': order_response['id'],
                            'signal_metadata': signal_metadata or {}
                        }
                        self.add_position(new_position)
                        
                        return True
                    except Exception as e:
                        logger.error(f"Failed to submit BUY order for {symbol}: {e}")
                        return False
                else:
                    logger.warning(f"Calculated 0 shares for {symbol} BUY order - insufficient funds or high price")
                    return False
                    
            elif action == "SELL":
                # Get current positions from Alpaca
                current_positions = self.get_current_positions()
                
                if symbol in current_positions:
                    position = current_positions[symbol]
                    shares_owned = abs(float(position.get('qty', 0)))
                    
                    if shares_owned > 0:
                        try:
                            # Submit market sell order to Alpaca
                            order_response = self.alpaca.submit_order(
                                symbol=symbol,
                                qty=shares_owned,
                                side='sell',
                                order_type='market',
                                time_in_force='day'
                            )
                            
                            logger.info(f"SELL order submitted to Alpaca: {shares_owned} shares of {symbol}")
                            logger.info(f"Order ID: {order_response['id']}, Status: {order_response['status']}")
                            
                            # Update internal tracking - mark position as closed
                            for pos in self.open_positions:
                                if pos.get('symbol') == symbol:
                                    pos['status'] = 'closing'
                                    pos['exit_order_id'] = order_response['id']
                            
                            return True
                        except Exception as e:
                            logger.error(f"Failed to submit SELL order for {symbol}: {e}")
                            return False
                    else:
                        logger.info(f"No shares to sell for {symbol} (qty: {shares_owned})")
                        return False
                else:
                    logger.info(f"No position found for {symbol} to sell")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Trade execution failed for {symbol}: {e}")
            return False

    def force_assign(self, position_id: str):
        """
        Mark a position as assigned (e.g., for short options assigned at expiry).
        """
        position = next((pos for pos in self.open_positions if pos.get('id') == position_id), None)
        if position:
            position['status'] = 'assigned'
            logger.info(f"Position assigned: {position_id}")
        else:
            logger.warning(f"Tried to assign unknown position: {position_id}")

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive portfolio summary including positions, P&L, and metrics.
        
        Returns:
            Dict with portfolio summary including:
            - total_positions: Number of open positions
            - total_value: Total portfolio value
            - realized_pnl: Total realized P&L from closed positions
            - unrealized_pnl: Total unrealized P&L from open positions
            - positions: List of current positions
            - performance_metrics: Basic performance statistics
        """
        try:
            # Get current positions from broker if available
            broker_positions = []
            if self.alpaca:
                try:
                    broker_positions = self.alpaca.get_positions()
                except Exception as e:
                    logger.warning(f"Could not fetch broker positions: {e}")
            
            # Calculate metrics
            open_df = self.get_open_positions_df()
            closed_df = self.get_closed_positions_df()
            
            total_positions = len(open_df) if not open_df.empty else 0
            realized_pnl = closed_df['pnl'].sum() if not closed_df.empty else 0.0
            
            # Calculate unrealized P&L (simplified)
            unrealized_pnl = 0.0
            if not open_df.empty and 'pnl' in open_df.columns:
                unrealized_pnl = open_df['pnl'].sum()
            
            # Get account info if available
            account_value = 0.0
            buying_power = 0.0
            if self.alpaca:
                try:
                    account_info = self.alpaca.get_account_info()
                    account_value = float(account_info.get('portfolio_value', 0))
                    buying_power = float(account_info.get('buying_power', 0))
                except Exception as e:
                    logger.warning(f"Could not fetch account info: {e}")
            
            # Create summary
            summary = {
                'total_positions': total_positions,
                'total_value': account_value,
                'buying_power': buying_power,
                'realized_pnl': realized_pnl,
                'unrealized_pnl': unrealized_pnl,
                'total_pnl': realized_pnl + unrealized_pnl,
                'positions': broker_positions if broker_positions else self.open_positions,
                'open_positions_count': len(broker_positions) if broker_positions else total_positions,
                'closed_positions_count': len(closed_df) if not closed_df.empty else 0,
                'performance_metrics': {
                    'win_rate': self._calculate_win_rate(closed_df),
                    'avg_win': self._calculate_avg_win(closed_df),
                    'avg_loss': self._calculate_avg_loss(closed_df),
                    'profit_factor': self._calculate_profit_factor(closed_df)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating portfolio summary: {e}")
            return {
                'total_positions': 0,
                'total_value': 0.0,
                'buying_power': 0.0,
                'realized_pnl': 0.0,
                'unrealized_pnl': 0.0,
                'total_pnl': 0.0,
                'positions': [],
                'open_positions_count': 0,
                'closed_positions_count': 0,
                'performance_metrics': {},
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_win_rate(self, closed_df: pd.DataFrame) -> float:
        """Calculate win rate from closed positions."""
        if closed_df.empty or 'pnl' not in closed_df.columns:
            return 0.0
        winning_trades = (closed_df['pnl'] > 0).sum()
        total_trades = len(closed_df)
        return (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    
    def _calculate_avg_win(self, closed_df: pd.DataFrame) -> float:
        """Calculate average winning trade."""
        if closed_df.empty or 'pnl' not in closed_df.columns:
            return 0.0
        winning_trades = closed_df[closed_df['pnl'] > 0]['pnl']
        return winning_trades.mean() if not winning_trades.empty else 0.0
    
    def _calculate_avg_loss(self, closed_df: pd.DataFrame) -> float:
        """Calculate average losing trade."""
        if closed_df.empty or 'pnl' not in closed_df.columns:
            return 0.0
        losing_trades = closed_df[closed_df['pnl'] < 0]['pnl']
        return losing_trades.mean() if not losing_trades.empty else 0.0
    
    def _calculate_profit_factor(self, closed_df: pd.DataFrame) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if closed_df.empty or 'pnl' not in closed_df.columns:
            return 0.0
        gross_profit = closed_df[closed_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(closed_df[closed_df['pnl'] < 0]['pnl'].sum())
        return (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

# Example usage
if __name__ == "__main__":
    # Example open position (dummy data)
    dummy_pos = {
        "id": "abc123",
        "symbol": "AAPL250726P00170000-AAPL250726P00165000",
        "entry_price": 1.50,
        "qty": 1,
        "direction": "credit",
        "status": "open"
    }
    manager = PositionManager()
    manager.add_position(dummy_pos)
    manager.close_position("abc123", exit_price=0.70)
    manager.report_positions()
