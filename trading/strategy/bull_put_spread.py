# Bull put spread strategy module
# Implements bull put spread strategy logic, strike selection, and entry criteria # bull_put_spread.py

import logging
from typing import Dict, Any, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class BullPutSpreadStrategy:
    """
    Implements bull put spread selection and entry logic.
    """

    def __init__(
        self,
        min_delta: float = -0.3,
        max_delta: float = -0.15,
        min_credit: float = 0.30,
        min_iv: float = 0.2,
        max_iv: float = 1.0,
        min_dte: int = 7,
        max_dte: int = 45,
        min_volume: int = 10,
        min_oi: int = 10,
        max_bid_ask: float = 0.10,
        earnings_buffer: int = 7
    ):
        self.min_delta = min_delta
        self.max_delta = max_delta
        self.min_credit = min_credit
        self.min_iv = min_iv
        self.max_iv = max_iv
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.min_volume = min_volume
        self.min_oi = min_oi
        self.max_bid_ask = max_bid_ask
        self.earnings_buffer = earnings_buffer

    def screen_candidates(
        self,
        options_chain: pd.DataFrame,
        underlying_price: float,
        next_earnings: Optional[str] = None,
        today: Optional[pd.Timestamp] = None
    ) -> List[Dict[str, Any]]:
        """
        Screen for eligible bull put spread candidates from an options chain.

        Args:
            options_chain: DataFrame with all puts, columns: ['expiration','strike','bid','ask','volume','open_interest','delta','iv',...]
            underlying_price: Current price of underlying
            next_earnings: Date of next earnings (YYYY-MM-DD)
            today: The current date (if None, uses pd.Timestamp.now())

        Returns:
            List of dicts, one per eligible spread, with rationale.
        """
        today = today or pd.Timestamp.now()
        eligible_spreads = []

        for expiry in options_chain['expiration'].unique():
            dte = (pd.to_datetime(expiry) - today).days

            # Earnings filter: skip expiries that fall too close to earnings
            if next_earnings:
                earnings_day = pd.to_datetime(next_earnings)
                if 0 <= (earnings_day - pd.to_datetime(expiry)).days <= self.earnings_buffer:
                    logger.info(f"Skipping {expiry} (too close to earnings: {next_earnings})")
                    continue

            if dte < self.min_dte or dte > self.max_dte:
                continue

            puts = options_chain[(options_chain['expiration'] == expiry) & (options_chain['type'] == 'put')]
            puts = puts.sort_values('strike')

            for idx, sell_leg in puts.iterrows():
                # Only OTM
                if sell_leg['strike'] >= underlying_price:
                    continue
                # Liquidity
                if (sell_leg['volume'] < self.min_volume or sell_leg['open_interest'] < self.min_oi):
                    continue
                # Greeks and IV filters
                if not (self.min_delta <= sell_leg['delta'] <= self.max_delta):
                    continue
                if not (self.min_iv <= sell_leg['iv'] <= self.max_iv):
                    continue
                # Spread filter
                if (sell_leg['ask'] - sell_leg['bid']) > self.max_bid_ask:
                    continue

                # Now find a buy leg (lower strike, same expiry)
                possible_buys = puts[puts['strike'] < sell_leg['strike']]
                for bidx, buy_leg in possible_buys.iterrows():
                    if (buy_leg['volume'] < self.min_volume or buy_leg['open_interest'] < self.min_oi):
                        continue
                    if (buy_leg['ask'] - buy_leg['bid']) > self.max_bid_ask:
                        continue

                    net_credit = sell_leg['bid'] - buy_leg['ask']
                    width = sell_leg['strike'] - buy_leg['strike']
                    max_loss = width - net_credit
                    rr = net_credit / max_loss if max_loss > 0 else None

                    # Only accept spreads that meet minimum credit and positive RR
                    if net_credit < self.min_credit or max_loss <= 0:
                        continue

                    spread = {
                        'symbol': sell_leg.get('symbol', ''),
                        'expiration': expiry,
                        'sell_strike': sell_leg['strike'],
                        'buy_strike': buy_leg['strike'],
                        'sell_bid': sell_leg['bid'],
                        'buy_ask': buy_leg['ask'],
                        'net_credit': net_credit,
                        'max_loss': max_loss,
                        'risk_reward': rr,
                        'dte': dte,
                        'sell_delta': sell_leg['delta'],
                        'sell_iv': sell_leg['iv'],
                        'sell_oi': sell_leg['open_interest'],
                        'buy_oi': buy_leg['open_interest'],
                        'spread_desc': f"Sell {sell_leg['strike']}P / Buy {buy_leg['strike']}P exp {expiry}",
                        'rationale': (
                            f"Delta {sell_leg['delta']:.2f}, Credit {net_credit:.2f}, IV {sell_leg['iv']:.2f}, "
                            f"RR {rr:.2f}, DTE {dte}, OI {sell_leg['open_interest']}/{buy_leg['open_interest']}"
                        ),
                    }
                    eligible_spreads.append(spread)

        # Sort by risk/reward, net_credit, or custom logic
        eligible_spreads = sorted(eligible_spreads, key=lambda s: (-s['risk_reward'], -s['net_credit']))
        return eligible_spreads

    def pick_best_spread(
        self, spreads: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Select the single best spread by your preferred ranking (risk/reward, net credit, etc.).
        """
        if not spreads:
            logger.info("No eligible spreads found.")
            return None
        return spreads[0]

    def entry_decision(
        self,
        options_chain: pd.DataFrame,
        underlying_price: float,
        next_earnings: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Run full selection and entry logic. Returns the recommended spread and rationale, or None.
        """
        candidates = self.screen_candidates(options_chain, underlying_price, next_earnings)
        best = self.pick_best_spread(candidates)
        if best:
            logger.info(f"Selected bull put spread: {best['spread_desc']} | {best['rationale']}")
        else:
            logger.info("No valid spread found for entry.")
        return best

# Example usage:
if __name__ == "__main__":
    # Dummy option chain for testing (should use real data in prod)
    import numpy as np
    df = pd.DataFrame({
        'symbol': ['AAPL']*6,
        'expiration': ['2025-07-26']*6,
        'strike': [170, 167.5, 165, 162.5, 160, 157.5],
        'type': ['put']*6,
        'bid': [2.5, 1.6, 1.0, 0.7, 0.4, 0.25],
        'ask': [2.65, 1.7, 1.1, 0.8, 0.45, 0.3],
        'volume': [150, 140, 120, 100, 80, 60],
        'open_interest': [200, 180, 150, 120, 100, 80],
        'delta': [-0.22, -0.18, -0.12, -0.08, -0.05, -0.03],
        'iv': [0.33, 0.31, 0.29, 0.28, 0.26, 0.24]
    })
    strat = BullPutSpreadStrategy()
    spreads = strat.screen_candidates(df, underlying_price=173.0, next_earnings="2025-08-01")
    best = strat.pick_best_spread(spreads)
    print("Best Spread:", best)
    print("All Candidates:")
    for s in spreads:
        print(s['spread_desc'], "|", s['rationale'])
