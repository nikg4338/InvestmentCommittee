# Validation utilities module
# Input validation and data quality checks # validators.py
"""
Validation Utilities Module
Input validation and data quality checks for the trading system.
"""

import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

def validate_not_none(val: Any, name: str = "") -> bool:
    """Check if a value is not None."""
    if val is None:
        print(f"Validation error: '{name}' is None.")
        return False
    return True

def validate_positive(val: float, name: str = "") -> bool:
    """Check if a value is positive (greater than zero)."""
    if val is None or val <= 0:
        print(f"Validation error: '{name}' must be positive. Got {val}.")
        return False
    return True

def validate_in_range(val: float, min_val: float, max_val: float, name: str = "") -> bool:
    """Check if a value is within a specified range."""
    if val is None or not (min_val <= val <= max_val):
        print(f"Validation error: '{name}'={val} not in range [{min_val}, {max_val}]")
        return False
    return True

def validate_dataframe_columns(df: pd.DataFrame, required_cols: List[str]) -> bool:
    """Check if DataFrame contains all required columns."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Validation error: DataFrame missing columns: {missing}")
        return False
    return True

def validate_no_nans(df: pd.DataFrame, cols: Optional[List[str]] = None) -> bool:
    """Check for NaNs in DataFrame (optionally restrict to columns)."""
    if cols is not None:
        subset = df[cols]
    else:
        subset = df
    nans = subset.isnull().sum()
    if nans.any():
        print(f"Validation error: Found NaNs:\n{nans[nans > 0]}")
        return False
    return True

def validate_option_chain_df(df: pd.DataFrame) -> bool:
    """Quick check for option chain DataFrame (minimal columns & types)."""
    required_cols = ['symbol', 'expiration', 'strike', 'type', 'bid', 'ask', 'volume', 'open_interest', 'delta', 'iv']
    if not validate_dataframe_columns(df, required_cols):
        return False
    if not validate_no_nans(df, required_cols):
        return False
    return True

def validate_trade_dict(trade: Dict[str, Any]) -> bool:
    """Check if a trade dict has all necessary fields."""
    required = ['symbol', 'entry_price', 'qty', 'direction', 'status']
    for key in required:
        if key not in trade:
            print(f"Validation error: Trade missing field '{key}'")
            return False
    return True

# Example usage/test
if __name__ == "__main__":
    # DataFrame column/NaN test
    df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "expiration": ["2025-07-26", "2025-07-26"],
        "strike": [170, 167.5],
        "type": ["put", "put"],
        "bid": [2.5, 1.6],
        "ask": [2.65, 1.7],
        "volume": [150, 140],
        "open_interest": [200, 180],
        "delta": [-0.22, -0.18],
        "iv": [0.33, 0.31],
    })
    print("Option chain valid?", validate_option_chain_df(df))
    print("Positive test:", validate_positive(1.0, "test_field"))
    print("In range test:", validate_in_range(0.25, 0.2, 0.7, "delta"))
    print("Trade dict test:", validate_trade_dict({
        "symbol": "AAPL",
        "entry_price": 1.25,
        "qty": 1,
        "direction": "credit",
        "status": "open"
    }))
