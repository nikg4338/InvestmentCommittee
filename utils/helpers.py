# Helper utilities module
# Common utility functions used across the trading system # helpers.py
"""
Helper Utilities Module
Common utility functions used across the trading system.
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, Optional, List, Union

logger = logging.getLogger(__name__)

def to_float(val: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, return default if fails."""
    try:
        return float(val)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert {val} to float, returning default={default}")
        return default

def to_int(val: Any, default: int = 0) -> int:
    """Safely convert a value to int, return default if fails."""
    try:
        return int(val)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert {val} to int, returning default={default}")
        return default

def parse_date(date_str: Union[str, pd.Timestamp, None], fallback: Optional[datetime] = None) -> Optional[datetime]:
    """Parse string or timestamp to datetime, or fallback if parsing fails."""
    if date_str is None:
        return fallback
    if isinstance(date_str, datetime):
        return date_str
    try:
        return pd.to_datetime(date_str)
    except Exception:
        logger.warning(f"Failed to parse date: {date_str}")
        return fallback

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Divide a by b, return default if b is zero."""
    try:
        return a / b if b != 0 else default
    except Exception:
        logger.warning(f"Safe divide failed for {a} / {b}, returning {default}")
        return default

def percent(a: float, b: float) -> float:
    """Return a as percent of b."""
    return safe_divide(a, b, default=0.0) * 100

def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """Flatten a list of lists into a single list."""
    return [item for sublist in nested_list for item in sublist]

def get_option_symbol(
    underlying: str, expiration: str, strike: float, right: str = "P"
) -> str:
    """
    Format an OCC-compliant option symbol (e.g. AAPL250726P00170000).
    """
    strike_int = int(round(strike * 1000))
    exp_fmt = expiration.replace("-", "")[2:]  # YYMMDD
    return f"{underlying.upper()}{exp_fmt}{right.upper()}{strike_int:08d}"

def log_trade(trade: Dict[str, Any]):
    """Quickly pretty-print a trade dict."""
    logger.info("TRADE: " + " | ".join(f"{k}: {v}" for k, v in trade.items()))

def filter_df_by_date(df: pd.DataFrame, date_col: str, start: str, end: str) -> pd.DataFrame:
    """Filter DataFrame rows to between start and end date."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    mask = (df[date_col] >= pd.to_datetime(start)) & (df[date_col] <= pd.to_datetime(end))
    return df.loc[mask]

def dict_diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Return key/value pairs where two dicts differ."""
    return {k: (a[k], b[k]) for k in set(a) | set(b) if a.get(k) != b.get(k)}

def load_iex_filtered_symbols(batch_file: str = "filtered_iex_batches.json") -> List[str]:
    """
    Load all symbols from IEX filtered batches.
    
    Args:
        batch_file (str): Path to the IEX batches JSON file
        
    Returns:
        List[str]: All symbols from all batches
    """
    try:
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
        
        all_symbols = []
        for batch_name, symbols in batch_data['batches'].items():
            all_symbols.extend(symbols)
        
        logger.info(f"Loaded {len(all_symbols)} IEX filtered symbols from {len(batch_data['batches'])} batches")
        return all_symbols
        
    except FileNotFoundError:
        logger.error(f"IEX batches file not found: {batch_file}")
        logger.info("Make sure filtered_iex_batches.json exists in the project root")
        raise
    except Exception as e:
        logger.error(f"Error loading IEX batches: {e}")
        raise

def load_iex_batch_symbols(batch_file: str, batch_name: str) -> List[str]:
    """
    Load symbols from a specific IEX batch.
    
    Args:
        batch_file (str): Path to the IEX batches JSON file
        batch_name (str): Name of the batch to load (e.g., "batch_1")
        
    Returns:
        List[str]: Symbols from the specified batch
    """
    try:
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
        
        if batch_name not in batch_data['batches']:
            available_batches = list(batch_data['batches'].keys())
            raise ValueError(f"Batch '{batch_name}' not found. Available batches: {available_batches}")
        
        symbols = batch_data['batches'][batch_name]
        logger.info(f"Loaded {len(symbols)} symbols from {batch_name}")
        return symbols
        
    except FileNotFoundError:
        logger.error(f"IEX batches file not found: {batch_file}")
        raise
    except Exception as e:
        logger.error(f"Error loading batch {batch_name}: {e}")
        raise

def get_iex_batch_info(batch_file: str = "filtered_iex_batches.json") -> Dict[str, Any]:
    """
    Get metadata about IEX batches.
    
    Args:
        batch_file (str): Path to the IEX batches JSON file
        
    Returns:
        Dict[str, Any]: Batch metadata including total symbols, batches, etc.
    """
    try:
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
        
        return {
            'total_batches': len(batch_data['batches']),
            'total_symbols': sum(len(symbols) for symbols in batch_data['batches'].values()),
            'batch_names': list(batch_data['batches'].keys()),
            'symbols_per_batch': {name: len(symbols) for name, symbols in batch_data['batches'].items()},
            'metadata': batch_data.get('metadata', {}),
            'filter_criteria': batch_data.get('metadata', {}).get('filter_criteria', {})
        }
        
    except FileNotFoundError:
        logger.error(f"IEX batches file not found: {batch_file}")
        raise
    except Exception as e:
        logger.error(f"Error loading batch info: {e}")
        raise

# Example/test block (remove in production)
if __name__ == "__main__":
    print(get_option_symbol("AAPL", "2025-07-26", 170))
    print(flatten_list([[1, 2], [3, 4], [5]]))
    print(to_float("nan"))
    print(parse_date("2024-01-01"))
    a = {"x": 1, "y": 2}
    b = {"x": 1, "y": 3, "z": 9}
    print(dict_diff(a, b))
