# Tests for trading modules
# Unit tests for strategy, execution, and portfolio management # test_trading.py
"""
Unit tests for strategy, execution, and portfolio management modules.
"""

import pandas as pd
import pytest

def test_bull_put_spread_strategy():
    from trading.strategy.bull_put_spread import BullPutSpreadStrategy
    df = pd.DataFrame({
        'symbol': ['AAPL']*3,
        'expiration': ['2025-07-26']*3,
        'strike': [170, 167.5, 165],
        'type': ['put']*3,
        'bid': [2.5, 1.6, 1.0],
        'ask': [2.65, 1.7, 1.1],
        'volume': [150, 140, 120],
        'open_interest': [200, 180, 150],
        'delta': [-0.22, -0.18, -0.12],
        'iv': [0.33, 0.31, 0.29]
    })
    strat = BullPutSpreadStrategy()
    spreads = strat.screen_candidates(df, underlying_price=173.0)
    assert isinstance(spreads, list)
    if spreads:
        best = strat.pick_best_spread(spreads)
        assert isinstance(best, dict)
        assert "net_credit" in best

def test_risk_management_entry_validation():
    from trading.strategy.risk_management import RiskManager
    dummy_spread = {"net_credit": 0.35, "max_loss": 1.65, "entry_price": 0.35, "direction": "credit"}
    account_equity = 10000
    cash = 3000
    open_pos = [{"max_loss": 1.60}, {"max_loss": 1.25}]
    risk = RiskManager()
    result = risk.validate_entry(dummy_spread, open_pos, account_equity, cash)
    assert isinstance(result, dict)
    assert "allowed" in result

def test_order_manager_place_spread(monkeypatch):
    from trading.execution.order_manager import OrderManager
    from trading.execution.alpaca_client import AlpacaClient
    
    class DummyAlpaca(AlpacaClient):
        def __init__(self):
            # Skip the parent __init__ to avoid API key validation
            pass
            
        def submit_order(self, **kwargs):
            return {"id": "dummy", "status": "accepted", "symbol": kwargs.get("symbol", "TEST"), "side": kwargs.get("side", "buy")}
            
        def cancel_order(self, order_id):
            return True
            
        def get_account_info(self):
            return {"buying_power": 10000, "cash": 10000}
            
        def is_market_open(self):
            return True
    
    manager = OrderManager(alpaca_client=DummyAlpaca())
    spread = manager.place_bull_put_spread(
        symbol="AAPL", sell_strike=170, buy_strike=165, expiration="2025-07-26", quantity=1
    )
    assert "legs" in spread and len(spread["legs"]) == 2
