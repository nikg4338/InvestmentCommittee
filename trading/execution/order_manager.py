# Order manager module
# Manages order placement, monitoring, and execution for bull put spreads # order_manager.py

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from trading.execution.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

class OrderManager:
    """
    Manages bull put spread order placement, monitoring, and execution via Alpaca.
    """

    def __init__(self, alpaca_client: Optional[AlpacaClient] = None):
        self.alpaca = alpaca_client or AlpacaClient()

    def place_bull_put_spread(
        self,
        symbol: str,
        sell_strike: float,
        buy_strike: float,
        expiration: str,
        quantity: int = 1,
        sell_price: Optional[float] = None,
        buy_price: Optional[float] = None,
        time_in_force: str = "day"
    ) -> Dict[str, Any]:
        """
        Place a bull put spread (sell higher-strike put, buy lower-strike put).
        Args:
            symbol: Underlying ticker (e.g. "AAPL")
            sell_strike: Strike price to sell (short put leg)
            buy_strike: Strike price to buy (long put leg)
            expiration: Expiration date (YYYY-MM-DD)
            quantity: Number of contracts (default 1)
            sell_price: Limit price for short put (optional)
            buy_price: Limit price for long put (optional)
            time_in_force: TIF for both legs
        Returns:
            Dict with order IDs, status, leg details
        """
        logger.info(
            f"Placing bull put spread: {symbol} {expiration} sell {sell_strike}P, buy {buy_strike}P, qty={quantity}"
        )

        # Compose option symbols per OCC symbology (may differ by broker)
        # Format: [Underlying][YYMMDD][C/P][Strike*1000] (e.g. AAPL250726P00170000)
        def make_option_symbol(sym, exp, strike, right):
            strike_int = int(float(strike) * 1000)
            return f"{sym}{exp.replace('-','')[2:]}{right}{strike_int:08d}"

        sell_option_symbol = make_option_symbol(symbol, expiration, sell_strike, "P")
        buy_option_symbol = make_option_symbol(symbol, expiration, buy_strike, "P")

        results = {"legs": []}

        # Place short put leg (sell)
        try:
            sell_order = self.alpaca.submit_order(
                symbol=sell_option_symbol,
                qty=quantity,
                side="sell",
                order_type="limit" if sell_price else "market",
                limit_price=sell_price,
                time_in_force=time_in_force
            )
            logger.info(f"Sell (short put) order placed: {sell_order['id']}")
            results["legs"].append(
                {
                    "side": "sell",
                    "symbol": sell_option_symbol,
                    "order_id": sell_order["id"],
                    "status": sell_order["status"],
                    "price": sell_order.get("limit_price") or sell_order.get("avg_fill_price"),
                }
            )
        except Exception as e:
            logger.error(f"Error placing short put leg: {e}")
            results["legs"].append(
                {
                    "side": "sell",
                    "symbol": sell_option_symbol,
                    "order_id": None,
                    "status": "error",
                    "error": str(e),
                }
            )
            return results

        # Place long put leg (buy)
        try:
            buy_order = self.alpaca.submit_order(
                symbol=buy_option_symbol,
                qty=quantity,
                side="buy",
                order_type="limit" if buy_price else "market",
                limit_price=buy_price,
                time_in_force=time_in_force
            )
            logger.info(f"Buy (long put) order placed: {buy_order['id']}")
            results["legs"].append(
                {
                    "side": "buy",
                    "symbol": buy_option_symbol,
                    "order_id": buy_order["id"],
                    "status": buy_order["status"],
                    "price": buy_order.get("limit_price") or buy_order.get("avg_fill_price"),
                }
            )
        except Exception as e:
            logger.error(f"Error placing long put leg: {e}")
            results["legs"].append(
                {
                    "side": "buy",
                    "symbol": buy_option_symbol,
                    "order_id": None,
                    "status": "error",
                    "error": str(e),
                }
            )
            # Optional: attempt to cancel short put if long fails
            short_leg = results["legs"][0]
            if short_leg.get("order_id"):
                self.cancel_leg(short_leg["order_id"])
            return results

        return results

    def monitor_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch and return order status by ID.
        """
        try:
            orders = self.alpaca.get_orders(status="all", limit=100)
            for order in orders:
                if order["id"] == order_id:
                    logger.info(f"Order status for {order_id}: {order['status']}")
                    return order
            logger.warning(f"Order ID {order_id} not found in recent orders")
            return None
        except Exception as e:
            logger.error(f"Error monitoring order status: {e}")
            return None

    def cancel_leg(self, order_id: str) -> bool:
        """
        Cancel a single order leg by ID.
        """
        return self.alpaca.cancel_order(order_id)

    def close_spread(self, leg_order_ids: List[str]) -> Dict[str, Any]:
        """
        Attempts to close both legs of an open spread.
        """
        results = {}
        for oid in leg_order_ids:
            try:
                result = self.alpaca.cancel_order(oid)
                results[oid] = result
            except Exception as e:
                results[oid] = f"Error: {e}"
        return results

# Example usage:
if __name__ == "__main__":
    # Example usage with dummy data (requires real option symbols for live use)
    manager = OrderManager()
    spread = manager.place_bull_put_spread(
        symbol="AAPL",
        sell_strike=170,
        buy_strike=165,
        expiration="2025-07-26",
        quantity=1,
        sell_price=2.50,
        buy_price=1.00,
    )
    print(spread)
