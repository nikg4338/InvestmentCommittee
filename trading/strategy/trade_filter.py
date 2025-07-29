# Trade filter module for Investment Committee
# Determines whether a ticker qualifies for bull put spread trades

from datetime import datetime, timedelta
from typing import Dict, Any, Optional


def is_trade_eligible(ticker_data: dict) -> bool:
    """
    Determines if a ticker qualifies for a bull put spread trade.
    
    Args:
        ticker_data (dict): Dictionary containing ticker data with the following structure:
            {
                'ticker': str,
                'market_data': {
                    'vix': float,
                    'vvix': float,
                    'spy_trend': str  # 'up', 'down', 'sideways'
                },
                'ticker_data': {
                    'avg_daily_volume': int,
                    'iv_rank': float,  # 0-100 scale
                    'options_chain': {
                        'put_leg_1': {
                            'open_interest': int,
                            'bid_ask_spread': float
                        },
                        'put_leg_2': {
                            'open_interest': int,
                            'bid_ask_spread': float
                        }
                    }
                },
                'earnings': {
                    'next_earnings_date': str  # 'YYYY-MM-DD' format or None
                }
            }
    
    Returns:
        bool: True if all conditions are met for trade eligibility, False otherwise
    """
    
    # Check market filters
    if not _check_market_filters(ticker_data.get('market_data', {})):
        return False
    
    # Check ticker filters
    if not _check_ticker_filters(ticker_data.get('ticker_data', {})):
        return False
    
    # Check earnings filter
    if not _check_earnings_filter(ticker_data.get('earnings', {})):
        return False
    
    return True


def _check_market_filters(market_data: dict) -> bool:
    """
    Check market-wide conditions for trade eligibility.
    
    Args:
        market_data (dict): Market data containing VIX, VVIX, and SPY trend
    
    Returns:
        bool: True if market conditions are favorable
    """
    # VIX must be < 20 (low volatility environment)
    vix = market_data.get('vix', 0)
    if vix >= 20:
        return False
    
    # VVIX must be < 100 (volatility of volatility not too high)
    vvix = market_data.get('vvix', 0)
    if vvix >= 100:
        return False
    
    # SPY trend must be sideways or up (favorable for bull put spreads)
    spy_trend = market_data.get('spy_trend', '').lower()
    if spy_trend not in ['up', 'sideways']:
        return False
    
    return True


def _check_ticker_filters(ticker_data: dict) -> bool:
    """
    Check ticker-specific conditions for trade eligibility.
    
    Args:
        ticker_data (dict): Ticker data containing volume, IV rank, and options chain
    
    Returns:
        bool: True if ticker conditions are favorable
    """
    # Average daily volume must be > 1,000,000
    avg_daily_volume = ticker_data.get('avg_daily_volume', 0)
    if avg_daily_volume <= 1_000_000:
        return False
    
    # IV Rank must be between 30 and 70
    iv_rank = ticker_data.get('iv_rank', 0)
    if not (30 <= iv_rank <= 70):
        return False
    
    # Check options chain requirements
    options_chain = ticker_data.get('options_chain', {})
    
    # Both legs must have open interest > 500
    put_leg_1 = options_chain.get('put_leg_1', {})
    put_leg_2 = options_chain.get('put_leg_2', {})
    
    if put_leg_1.get('open_interest', 0) <= 500:
        return False
    
    if put_leg_2.get('open_interest', 0) <= 500:
        return False
    
    # Both legs must have tight bid-ask spreads (< $0.10)
    if put_leg_1.get('bid_ask_spread', 1.0) >= 0.10:
        return False
    
    if put_leg_2.get('bid_ask_spread', 1.0) >= 0.10:
        return False
    
    return True


def _check_earnings_filter(earnings_data: dict) -> bool:
    """
    Check earnings-related conditions for trade eligibility.
    
    Args:
        earnings_data (dict): Earnings data containing next earnings date
    
    Returns:
        bool: True if no earnings within 7 days
    """
    next_earnings_date = earnings_data.get('next_earnings_date')
    
    # If no earnings date provided, assume it's safe
    if not next_earnings_date:
        return True
    
    try:
        # Parse the earnings date
        earnings_date = datetime.strptime(next_earnings_date, '%Y-%m-%d')
        current_date = datetime.now()
        
        # Check if earnings are within the next 7 days
        days_until_earnings = (earnings_date - current_date).days
        
        # Return False if earnings are within 7 days
        return days_until_earnings > 7
    
    except (ValueError, TypeError):
        # If date parsing fails, assume it's not safe
        return False


# Example usage and testing functions
def create_sample_ticker_data() -> dict:
    """
    Create sample ticker data for testing purposes.
    
    Returns:
        dict: Sample ticker data structure
    """
    return {
        'ticker': 'AAPL',
        'market_data': {
            'vix': 18.5,  # Below 20 ✓
            'vvix': 85.0,  # Below 100 ✓
            'spy_trend': 'up'  # Up trend ✓
        },
        'ticker_data': {
            'avg_daily_volume': 50_000_000,  # Above 1M ✓
            'iv_rank': 45.0,  # Between 30-70 ✓
            'options_chain': {
                'put_leg_1': {
                    'open_interest': 1200,  # Above 500 ✓
                    'bid_ask_spread': 0.05  # Below 0.10 ✓
                },
                'put_leg_2': {
                    'open_interest': 800,  # Above 500 ✓
                    'bid_ask_spread': 0.08  # Below 0.10 ✓
                }
            }
        },
        'earnings': {
            'next_earnings_date': '2024-02-15'  # Assume this is more than 7 days away
        }
    }


def test_trade_filter():
    """
    Test the trade filter with sample data.
    """
    # Test with qualifying ticker
    sample_data = create_sample_ticker_data()
    result = is_trade_eligible(sample_data)
    print(f"Sample ticker eligibility: {result}")
    
    # Test with non-qualifying ticker (high VIX)
    sample_data['market_data']['vix'] = 25.0
    result = is_trade_eligible(sample_data)
    print(f"High VIX ticker eligibility: {result}")


if __name__ == "__main__":
    test_trade_filter() 