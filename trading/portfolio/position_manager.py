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
