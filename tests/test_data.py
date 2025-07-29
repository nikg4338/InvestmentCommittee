# test_data.py
"""
Unit tests for data collectors, processors, and storage modules.
"""

import os
import pytest
import pandas as pd
from storage.database import Database
from storage.cache import SimpleCache

def test_database_insert_and_fetch(tmp_path):
    db_path = os.path.join(tmp_path, "test_trades.db")
    db = Database(db_path=db_path)
    trade = {
        "entry_time": "2024-07-17T09:00:00",
        "exit_time": "2024-07-18T15:30:00",
        "symbol": "AAPL",
        "strategy": "bull_put",
        "trade_type": "options",
        "direction": "credit",
        "entry_price": 1.25,
        "exit_price": 0.60,
        "size": 1,
        "pnl": 65.0,
        "status": "closed",
        "assigned": 0
    }
    db.insert_trade(trade)
    trades = db.fetch_trades()
    assert not trades.empty
    assert trades.iloc[-1]["symbol"] == "AAPL"
    db.close()

def test_cache_set_and_get(tmp_path):
    cache_dir = os.path.join(tmp_path, "cache")
    cache = SimpleCache(cache_dir=cache_dir, default_expiry=3)
    cache.set("foo", {"bar": 42})
    value = cache.get("foo")
    assert isinstance(value, dict)
    assert value["bar"] == 42

def test_dataframe_validation():
    from utils.validators import validate_dataframe_columns
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6]
    })
    assert validate_dataframe_columns(df, ["a", "b"]) is True
    assert validate_dataframe_columns(df, ["a", "c"]) is False
