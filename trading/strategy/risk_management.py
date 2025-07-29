# Risk management module
# Implements position sizing, stop-loss, and portfolio risk controls # risk_management.py
"""
Risk Management Module
Implements position sizing, stop-loss, and portfolio risk controls.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Provides risk controls: position sizing, stop-loss, max portfolio risk, etc.
    """

    def __init__(
        self,
        max_portfolio_risk_pct: float = 0.03,   # Max risk per trade as % of portfolio (e.g. 3%)
        max_total_risk_pct: float = 0.10,       # Max aggregate open risk as % of portfolio
        max_open_positions: int = 5,
        min_credit_rr: float = 0.30,            # Min risk/reward credit (ex: $0.30 per spread)
        stop_loss_pct: float = 0.75,            # Stop loss triggers at 75% of max loss
        min_cash_reserve_pct: float = 0.10,     # Always keep at least 10% in cash
    ):
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.max_total_risk_pct = max_total_risk_pct
        self.max_open_positions = max_open_positions
        self.min_credit_rr = min_credit_rr
        self.stop_loss_pct = stop_loss_pct
        self.min_cash_reserve_pct = min_cash_reserve_pct

    def position_size(
        self, account_equity: float, spread_max_loss: float
    ) -> int:
        """
        Calculate allowed contracts for this trade based on account size and max risk per trade.
        Args:
            account_equity: total portfolio equity
            spread_max_loss: dollar value of max loss for 1 spread
        Returns:
            Contracts allowed (int, >=0)
        """
        max_risk_per_trade = self.max_portfolio_risk_pct * account_equity
        contracts = int(max_risk_per_trade // spread_max_loss)
        logger.info(
            f"Position size calculation: equity={account_equity:.2f}, max_loss={spread_max_loss:.2f}, contracts={contracts}"
        )
        return max(contracts, 0)

    def can_open_new_position(
        self,
        open_positions: List[Dict[str, Any]],
        candidate_max_loss: float,
        account_equity: float,
        cash_available: float
    ) -> bool:
        """
        Check if new position can be opened given portfolio and risk rules.
        Args:
            open_positions: list of currently open positions (each must include 'max_loss')
            candidate_max_loss: max loss for this candidate position
            account_equity: portfolio equity
            cash_available: cash available for new trade
        Returns:
            True if allowed, False otherwise
        """
        if len(open_positions) >= self.max_open_positions:
            logger.warning("Risk block: max open positions reached")
            return False

        total_open_risk = np.sum([pos['max_loss'] for pos in open_positions])
        if (total_open_risk + candidate_max_loss) > self.max_total_risk_pct * account_equity:
            logger.warning("Risk block: aggregate open risk too high")
            return False

        if cash_available < candidate_max_loss:
            logger.warning("Risk block: insufficient cash for new trade")
            return False

        if (cash_available / account_equity) < self.min_cash_reserve_pct:
            logger.warning("Risk block: cash reserve rule violated")
            return False

        return True

    def validate_entry(
        self,
        spread: Dict[str, Any],
        open_positions: List[Dict[str, Any]],
        account_equity: float,
        cash_available: float
    ) -> Dict[str, Any]:
        """
        Validate all risk rules for new trade entry.
        Args:
            spread: candidate spread (must include 'net_credit', 'max_loss', etc.)
            open_positions: current open positions
            account_equity: total portfolio equity
            cash_available: cash
        Returns:
            Dict with 'allowed', 'rationale', and recommended position size
        """
        msg = []

        # Check min credit per risk
        rr = spread['net_credit'] / spread['max_loss'] if spread['max_loss'] else 0
        if rr < self.min_credit_rr:
            msg.append(f"Risk/reward too low: {rr:.2f} (min {self.min_credit_rr:.2f})")

        contracts = self.position_size(account_equity, spread['max_loss'])
        if contracts < 1:
            msg.append("Max loss too large for account size (contracts=0)")

        if not self.can_open_new_position(open_positions, spread['max_loss'], account_equity, cash_available):
            msg.append("Portfolio/global risk constraint not met")

        allowed = len(msg) == 0
        rationale = " | ".join(msg) if msg else "All risk controls passed"
        logger.info(f"Risk validation: allowed={allowed} | {rationale}")

        return {
            "allowed": allowed,
            "rationale": rationale,
            "contracts": contracts if allowed else 0
        }

    def stop_loss_triggered(self, spread: Dict[str, Any], current_price: float) -> bool:
        """
        Check if stop loss should be triggered for an open spread.
        Args:
            spread: dict for the spread (must include 'entry_price', 'max_loss', 'direction')
            current_price: current mid or ask price to close the spread
        Returns:
            True if stop loss hit, else False
        """
        entry = spread['entry_price']
        max_loss = spread['max_loss']
        direction = spread.get('direction', 'credit')

        # For a credit spread, stop loss is typically a debit to close the spread
        loss = (current_price - entry) * 100 if direction == 'credit' else (entry - current_price) * 100
        trigger = loss > self.stop_loss_pct * max_loss * 100

        if trigger:
            logger.warning(f"Stop loss triggered: loss={loss:.2f}, threshold={self.stop_loss_pct * max_loss * 100:.2f}")
        return trigger

# Example usage
if __name__ == "__main__":
    risk = RiskManager()
    dummy_spread = {"net_credit": 0.35, "max_loss": 1.65, "entry_price": 0.35, "direction": "credit"}
    account_equity = 10000
    cash = 3000
    open_pos = [{"max_loss": 1.60}, {"max_loss": 1.25}]
    print(risk.validate_entry(dummy_spread, open_pos, account_equity, cash))
    print("Contracts allowed:", risk.position_size(account_equity, dummy_spread['max_loss']))
    print("Stop loss triggered?", risk.stop_loss_triggered(dummy_spread, current_price=1.7))
