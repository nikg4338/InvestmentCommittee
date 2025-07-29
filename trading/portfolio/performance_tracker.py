# Performance tracker module
# Tracks trading performance, calculates metrics, and generates reports # performance_tracker.py
"""
Performance Tracker Module
Tracks trading performance, calculates metrics, and generates reports.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

class PerformanceTracker:
    """
    Tracks and reports on trading/portfolio performance.
    """

    def __init__(self, trade_log: pd.DataFrame):
        """
        Args:
            trade_log: DataFrame of all trades with required columns:
              ['entry_time', 'exit_time', 'symbol', 'strategy', 'trade_type',
               'direction', 'entry_price', 'exit_price', 'size', 'pnl', 'status', ...]
        """
        self.log = trade_log.copy()
        self._preprocess()

    def _preprocess(self):
        # Convert times
        self.log['entry_time'] = pd.to_datetime(self.log['entry_time'])
        self.log['exit_time'] = pd.to_datetime(self.log['exit_time'])
        # Compute holding period (days)
        self.log['hold_days'] = (self.log['exit_time'] - self.log['entry_time']).dt.total_seconds() / 86400
        # Compute percent return per trade
        self.log['pct_return'] = (self.log['exit_price'] - self.log['entry_price']) / self.log['entry_price']
        # If P&L not provided, estimate
        if 'pnl' not in self.log.columns:
            self.log['pnl'] = self.log['pct_return'] * self.log['entry_price'] * self.log['size']

    def win_rate(self) -> float:
        wins = (self.log['pnl'] > 0).sum()
        total = len(self.log)
        return wins / total if total > 0 else 0

    def avg_win(self) -> float:
        return self.log[self.log['pnl'] > 0]['pnl'].mean() if not self.log[self.log['pnl'] > 0].empty else 0

    def avg_loss(self) -> float:
        return self.log[self.log['pnl'] < 0]['pnl'].mean() if not self.log[self.log['pnl'] < 0].empty else 0

    def expectancy(self) -> float:
        win = self.avg_win()
        loss = self.avg_loss()
        winr = self.win_rate()
        return win * winr + loss * (1 - winr)

    def total_pnl(self) -> float:
        return self.log['pnl'].sum()

    def total_return(self) -> float:
        initial = self.log['entry_price'].iloc[0] * self.log['size'].iloc[0] if not self.log.empty else 1
        final = initial + self.total_pnl()
        return (final - initial) / initial if initial != 0 else 0

    def sharpe_ratio(self, risk_free_rate: float = 0.01) -> float:
        daily_returns = self.log['pct_return']
        excess = daily_returns - risk_free_rate / 252  # annualized risk free
        if excess.std() == 0:
            return 0
        return np.sqrt(252) * excess.mean() / excess.std()

    def max_drawdown(self) -> float:
        cum_pnl = self.log['pnl'].cumsum()
        high = cum_pnl.cummax()
        drawdown = cum_pnl - high
        return drawdown.min()

    def avg_hold_days(self) -> float:
        return self.log['hold_days'].mean() if not self.log['hold_days'].empty else 0

    def assignment_rate(self) -> float:
        if 'assigned' not in self.log.columns:
            return 0
        assigned = self.log['assigned'].sum()
        return assigned / len(self.log) if len(self.log) > 0 else 0

    def strategy_breakdown(self) -> pd.DataFrame:
        return self.log.groupby('strategy')['pnl'].agg(['count', 'sum', 'mean'])

    def as_report(self) -> Dict[str, Any]:
        return {
            'num_trades': len(self.log),
            'win_rate': round(100 * self.win_rate(), 2),
            'avg_win': round(self.avg_win(), 2),
            'avg_loss': round(self.avg_loss(), 2),
            'expectancy': round(self.expectancy(), 2),
            'total_pnl': round(self.total_pnl(), 2),
            'total_return': round(100 * self.total_return(), 2),
            'sharpe_ratio': round(self.sharpe_ratio(), 2),
            'max_drawdown': round(self.max_drawdown(), 2),
            'avg_hold_days': round(self.avg_hold_days(), 2),
            'assignment_rate': round(100 * self.assignment_rate(), 2),
        }

    def report_df(self) -> pd.DataFrame:
        report = self.as_report()
        return pd.DataFrame(report, index=[0])

    def to_csv(self, path: str):
        self.log.to_csv(path, index=False)

# Example usage
if __name__ == "__main__":
    # Dummy trade log for demo
    df = pd.DataFrame({
        'entry_time': pd.date_range('2024-07-01', periods=10, freq='D'),
        'exit_time': pd.date_range('2024-07-02', periods=10, freq='D'),
        'symbol': ['AAPL']*10,
        'strategy': ['bull_put']*10,
        'trade_type': ['options']*10,
        'direction': ['credit']*10,
        'entry_price': np.linspace(1.0, 2.0, 10),
        'exit_price': np.linspace(1.05, 1.90, 10),
        'size': [1]*10,
        'pnl': np.random.uniform(-30, 40, 10),
        'status': ['closed']*10,
        'assigned': [False]*9 + [True]
    })
    tracker = PerformanceTracker(df)
    print(tracker.as_report())
    print(tracker.strategy_breakdown())
