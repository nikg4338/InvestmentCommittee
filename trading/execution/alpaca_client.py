# Alpaca API client module
# Handles communication with Alpaca API for paper trading execution

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame
from alpaca_trade_api.common import URL

# Load environment variables
load_dotenv()

# Import config
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import get_alpaca_config, validate_api_keys

logger = logging.getLogger(__name__)


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float, handling None, NaN, and other types.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        float: Converted value or default
    """
    if value is None:
        return default
    
    # Handle pandas NaT (Not a Time) and similar types
    if hasattr(value, 'isnull') and value.isnull():
        return default
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


class AlpacaClient:
    """
    Alpaca API client for paper trading execution.
    Handles account management, market data, and order execution.
    """

    def __init__(self):
        """Initialize Alpaca client with API credentials."""
        self.config = get_alpaca_config()
        
        # Validate API keys
        if not validate_api_keys():
            raise ValueError(
                "Invalid or missing Alpaca API keys. "
                "Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file."
            )
        
        # Initialize Alpaca API client
        # Handle base_url - remove trailing /v2 if present since api_version='v2' will add it
        base_url = self.config["base_url"].rstrip('/v2').rstrip('/')
        
        self.api = tradeapi.REST(
            key_id=self.config["api_key"],
            secret_key=self.config["secret_key"],
            base_url=base_url,
            api_version='v2'
        )
        
        # Test connection
        try:
            account = self.api.get_account()
            logger.info(f"Connected to Alpaca API - Account: {account.id}")
            logger.info(f"Paper Trading: {self.config['paper_trading']}")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca API: {e}")
            raise

    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dict[str, Any]: Account information
        """
        try:
            account = self.api.get_account()
            return {
                'id': account.id,
                'account_number': account.account_number,
                'status': account.status,
                'currency': account.currency,
                'buying_power': safe_float(account.buying_power),
                'cash': safe_float(account.cash),
                'portfolio_value': safe_float(account.portfolio_value),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked,
                'account_blocked': account.account_blocked,
                'created_at': account.created_at,
                'trade_suspended_by_user': account.trade_suspended_by_user,
                'multiplier': safe_float(account.multiplier, 1.0),
                'shorting_enabled': account.shorting_enabled,
                'equity': safe_float(account.equity),
                'last_equity': safe_float(account.last_equity),
                'long_market_value': safe_float(account.long_market_value),
                'short_market_value': safe_float(account.short_market_value),
                'initial_margin': safe_float(account.initial_margin),
                'maintenance_margin': safe_float(account.maintenance_margin),
                'daytime_buying_power': safe_float(getattr(account, 'daytime_buying_power', account.buying_power)),
                'regt_buying_power': safe_float(getattr(account, 'regt_buying_power', account.buying_power))
            }
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            raise

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions.
        
        Returns:
            List[Dict[str, Any]]: List of positions
        """
        try:
            positions = self.api.list_positions()
            return [
                {
                    'symbol': pos.symbol,
                    'qty': safe_float(pos.qty),
                    'side': pos.side,
                    'market_value': safe_float(pos.market_value),
                    'cost_basis': safe_float(pos.cost_basis),
                    'unrealized_pl': safe_float(pos.unrealized_pl),
                    'unrealized_plpc': safe_float(pos.unrealized_plpc),
                    'current_price': safe_float(pos.current_price),
                    'lastday_price': safe_float(pos.lastday_price),
                    'change_today': safe_float(pos.change_today)
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            raise

    def get_orders(self, status: str = 'all', limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get orders.
        
        Args:
            status (str): Order status filter ('open', 'closed', 'all')
            limit (int): Maximum number of orders to return
            
        Returns:
            List[Dict[str, Any]]: List of orders
        """
        try:
            orders = self.api.list_orders(status=status, limit=limit)
            return [
                {
                    'id': order.id,
                    'symbol': order.symbol,
                    'asset_class': order.asset_class,
                    'side': order.side,
                    'order_type': order.order_type,
                    'qty': float(order.qty),
                    'filled_qty': float(order.filled_qty),
                    'status': order.status,
                    'time_in_force': order.time_in_force,
                    'limit_price': float(order.limit_price) if order.limit_price else None,
                    'stop_price': float(order.stop_price) if order.stop_price else None,
                    'avg_fill_price': float(order.avg_fill_price) if order.avg_fill_price else None,
                    'created_at': order.created_at,
                    'updated_at': order.updated_at,
                    'submitted_at': order.submitted_at,
                    'filled_at': order.filled_at,
                    'expired_at': order.expired_at,
                    'canceled_at': order.canceled_at,
                    'failed_at': order.failed_at,
                    'replaced_at': order.replaced_at
                }
                for order in orders
            ]
        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            raise

    def get_market_data(self, symbol: str, timeframe: str = '1Day', limit: int = 100, delayed: bool = True) -> Dict[str, Any]:
        """
        Get market data for a symbol using IEX feed (free tier).
        
        Args:
            symbol (str): Symbol to get data for
            timeframe (str): Timeframe ('1Min', '5Min', '15Min', '1Hour', '1Day')
            limit (int): Number of bars to return
            delayed (bool): Use delayed data (required for free tier)
            
        Returns:
            Dict[str, Any]: Market data
        """
        try:
            # Map timeframe string to TimeFrame enum
            timeframe_map = {
                '1Min': TimeFrame.Minute,
                '5Min': TimeFrame(5, TimeFrame.Minute),
                '15Min': TimeFrame(15, TimeFrame.Minute),
                '1Hour': TimeFrame.Hour,
                '1Day': TimeFrame.Day
            }
            
            tf = timeframe_map.get(timeframe, TimeFrame.Day)
            
            # For free tier with IEX, use data that's at least 15 minutes old
            if delayed:
                end_time = datetime.now() - timedelta(days=2)  # Go back 2 days to avoid recent data restrictions
                start_time = end_time - timedelta(days=limit + 2)
            else:
                end_time = datetime.now()
                start_time = end_time - timedelta(days=limit)
            
            # Use IEX feed for free tier instead of SIP
            bars = self.api.get_bars(
                symbol,
                tf,
                start=start_time.strftime('%Y-%m-%d'),
                end=end_time.strftime('%Y-%m-%d'),
                feed='iex'  # Use IEX feed (free tier) instead of SIP
            )
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'bars': [
                    {
                        'timestamp': bar.t,
                        'open': float(bar.o),
                        'high': float(bar.h),
                        'low': float(bar.l),
                        'close': float(bar.c),
                        'volume': int(bar.v)
                    }
                    for bar in bars
                ]
            }
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            raise

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get current quote for a symbol using IEX feed (free tier).
        
        Args:
            symbol (str): Symbol to get quote for
            
        Returns:
            Dict[str, Any]: Quote data
        """
        try:
            # Use IEX feed for free tier instead of SIP
            quote = self.api.get_latest_quote(symbol, feed='iex')
            return {
                'symbol': symbol,
                'bid_price': float(quote.bid_price),
                'bid_size': int(quote.bid_size),
                'ask_price': float(quote.ask_price),
                'ask_size': int(quote.ask_size),
                'timestamp': quote.timestamp
            }
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            raise

    def submit_order(self, symbol: str, qty: float, side: str, order_type: str = 'market', 
                     limit_price: Optional[float] = None, stop_price: Optional[float] = None,
                     time_in_force: str = 'day') -> Dict[str, Any]:
        """
        Submit an order.
        
        Args:
            symbol (str): Symbol to trade
            qty (float): Quantity to trade
            side (str): 'buy' or 'sell'
            order_type (str): 'market', 'limit', 'stop', 'stop_limit'
            limit_price (float, optional): Limit price for limit orders
            stop_price (float, optional): Stop price for stop orders
            time_in_force (str): 'day', 'gtc', 'ioc', 'fok'
            
        Returns:
            Dict[str, Any]: Order response
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=time_in_force
            )
            
            logger.info(f"Order submitted: {order.id} - {side} {qty} {symbol}")
            
            return {
                'id': order.id,
                'symbol': order.symbol,
                'side': order.side,
                'order_type': order.order_type,
                'qty': float(order.qty),
                'status': order.status,
                'time_in_force': order.time_in_force,
                'limit_price': float(order.limit_price) if order.limit_price else None,
                'stop_price': float(order.stop_price) if order.stop_price else None,
                'created_at': order.created_at,
                'submitted_at': order.submitted_at
            }
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id (str): Order ID to cancel
            
        Returns:
            bool: True if successful
        """
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def get_options_chain(self, symbol: str, expiration_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get options chain for a symbol.
        Note: This is a placeholder as Alpaca's options API might have different endpoints.
        
        Args:
            symbol (str): Underlying symbol
            expiration_date (str, optional): Expiration date filter
            
        Returns:
            List[Dict[str, Any]]: Options chain data
        """
        # This is a placeholder implementation
        # Alpaca's options API might have different endpoints
        logger.warning("Options chain functionality not yet implemented for Alpaca API")
        return []

    def is_market_open(self) -> bool:
        """
        Check if market is open.
        
        Returns:
            bool: True if market is open
        """
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False

    def get_market_calendar(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Get market calendar.
        
        Args:
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            
        Returns:
            List[Dict[str, Any]]: Market calendar
        """
        try:
            calendar = self.api.get_calendar(start=start_date, end=end_date)
            return [
                {
                    'date': str(day.date),
                    'open': str(day.open),
                    'close': str(day.close),
                    'session_open': str(day.session_open),
                    'session_close': str(day.session_close)
                }
                for day in calendar
            ]
        except Exception as e:
            logger.error(f"Error fetching market calendar: {e}")
            raise

    def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest quote for a symbol.
        
        Args:
            symbol (str): Symbol to get quote for
            
        Returns:
            Optional[Dict[str, Any]]: Latest quote data or None
        """
        try:
            quote = self.api.get_latest_quote(symbol)
            if quote:
                return {
                    'symbol': symbol,
                    'bid_price': safe_float(quote.bid_price),
                    'ask_price': safe_float(quote.ask_price),
                    'bid_size': safe_float(quote.bid_size),
                    'ask_size': safe_float(quote.ask_size),
                    'timestamp': quote.timestamp
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None

    def get_bars(self, symbol: str, start: str, end: str, timeframe: str = '1Day') -> List[Dict[str, Any]]:
        """
        Get historical bars for a symbol.
        
        Args:
            symbol (str): Symbol to get bars for
            start (str): Start date (YYYY-MM-DD)
            end (str): End date (YYYY-MM-DD)
            timeframe (str): Timeframe ('1Day', '1Hour', etc.)
            
        Returns:
            List[Dict[str, Any]]: List of bar data
        """
        try:
            # Map timeframe string to Alpaca TimeFrame
            if timeframe == '1Day':
                tf = TimeFrame.Day
            elif timeframe == '1Hour':
                tf = TimeFrame.Hour
            elif timeframe == '1Min':
                tf = TimeFrame.Minute
            else:
                tf = TimeFrame.Day  # Default
            
            bars = self.api.get_bars(symbol, tf, start=start, end=end, asof=None, feed='iex')
            
            result = []
            for bar in bars:
                result.append({
                    'timestamp': str(bar.timestamp) if hasattr(bar, 'timestamp') else bar.t,
                    'open': safe_float(bar.open if hasattr(bar, 'open') else bar.o),
                    'high': safe_float(bar.high if hasattr(bar, 'high') else bar.h),
                    'low': safe_float(bar.low if hasattr(bar, 'low') else bar.l),
                    'close': safe_float(bar.close if hasattr(bar, 'close') else bar.c),
                    'volume': safe_float(bar.volume if hasattr(bar, 'volume') else bar.v),
                    'trade_count': safe_float(getattr(bar, 'trade_count', getattr(bar, 'n', 0))),
                    'vwap': safe_float(getattr(bar, 'vwap', getattr(bar, 'vw', 0)))
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching bars for {symbol}: {e}")
            return [] 