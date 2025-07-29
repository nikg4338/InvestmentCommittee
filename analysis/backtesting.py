# Backtesting pipeline for Investment Committee
# Tests bull put spread strategy with historical data and mock predictions

import logging
import sys
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import math

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from trading.strategy.trade_filter import is_trade_eligible
from models.model_predictor import ModelPredictor
from models.neural_predictor import NeuralPredictor
from models.meta_model import MetaModel, ModelInput, TradeDecision, TradeSignal
from utils.trade_logger import TradeLogger

logger = logging.getLogger(__name__)


@dataclass
class BacktestPosition:
    """Backtesting position record."""
    symbol: str
    entry_date: str
    exit_date: Optional[str]
    entry_price: float
    exit_price: Optional[float]
    position_size: int
    strategy: str
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    max_drawdown: Optional[float] = None
    hold_days: Optional[int] = None
    trade_id: str = ""


@dataclass
class BacktestResult:
    """Backtesting results."""
    start_date: str
    end_date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    avg_hold_days: float
    positions: List[BacktestPosition]
    performance_by_symbol: Dict[str, Any]
    monthly_returns: Dict[str, float]


class BacktestEngine:
    """
    Backtesting engine for Investment Committee strategy.
    Simulates bull put spread trades using historical data.
    """
    
    def __init__(self, initial_capital: float = 100000, max_positions: int = 10):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital (float): Starting capital
            max_positions (int): Maximum concurrent positions
        """
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.current_capital = initial_capital
        
        # Initialize models
        self.model_predictor = ModelPredictor()
        self.neural_mlp = NeuralPredictor(model_type='mlp')
        self.meta_model = MetaModel()
        
        # Track positions and results
        self.positions: List[BacktestPosition] = []
        self.closed_positions: List[BacktestPosition] = []
        self.open_positions: Dict[str, BacktestPosition] = {}
        
        # Logging
        self.logger = TradeLogger(log_dir="backtest_logs")
        
        logger.info(f"Backtest engine initialized with ${initial_capital:,.2f}")
    
    def run_backtest(self, historical_data: Dict[str, List[Dict[str, Any]]], 
                    start_date: str, end_date: str) -> BacktestResult:
        """
        Run backtesting simulation.
        
        Args:
            historical_data (Dict[str, List[Dict[str, Any]]]): Historical data by symbol
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            
        Returns:
            BacktestResult: Backtesting results
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Generate date range
        date_range = self._generate_date_range(start_date, end_date)
        
        # Process each trading day
        for current_date in date_range:
            self._process_trading_day(current_date, historical_data)
        
        # Close any remaining positions
        self._close_all_positions(end_date)
        
        # Calculate results
        result = self._calculate_results(start_date, end_date)
        
        logger.info(f"Backtest completed: {result.total_trades} trades, {result.win_rate:.1%} win rate")
        return result
    
    def _generate_date_range(self, start_date: str, end_date: str) -> List[str]:
        """Generate trading date range."""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        dates = []
        current = start
        while current <= end:
            # Skip weekends (basic approach)
            if current.weekday() < 5:  # Monday = 0, Friday = 4
                dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        
        return dates
    
    def _process_trading_day(self, date: str, historical_data: Dict[str, List[Dict[str, Any]]]):
        """Process a single trading day."""
        # Check for exit conditions first
        self._check_exit_conditions(date, historical_data)
        
        # Look for new entry opportunities
        if len(self.open_positions) < self.max_positions:
            self._look_for_entries(date, historical_data)
    
    def _check_exit_conditions(self, date: str, historical_data: Dict[str, List[Dict[str, Any]]]):
        """Check exit conditions for open positions."""
        positions_to_close = []
        
        for symbol, position in self.open_positions.items():
            if symbol not in historical_data:
                continue
            
            # Get current price
            current_price = self._get_price_for_date(historical_data[symbol], date)
            if current_price is None:
                continue
            
            # Calculate days held
            entry_date = datetime.strptime(position.entry_date, '%Y-%m-%d')
            current_date = datetime.strptime(date, '%Y-%m-%d')
            days_held = (current_date - entry_date).days
            
            # Exit conditions for bull put spread
            should_exit = False
            
            # Time-based exit (30 days max)
            if days_held >= 30:
                should_exit = True
            
            # Profit target (50% of max profit)
            max_profit = position.position_size * 100  # Assume $100 max profit per contract
            current_profit = self._calculate_bull_put_pnl(position, current_price)
            if current_profit >= max_profit * 0.5:
                should_exit = True
            
            # Stop loss (close if underlying drops below short strike)
            short_strike = position.entry_price * 0.95  # Assume short strike 5% below entry
            if current_price < short_strike:
                should_exit = True
            
            if should_exit:
                positions_to_close.append((symbol, position, current_price, date))
        
        # Close positions
        for symbol, position, exit_price, exit_date in positions_to_close:
            self._close_position(symbol, position, exit_price, exit_date)
    
    def _look_for_entries(self, date: str, historical_data: Dict[str, List[Dict[str, Any]]]):
        """Look for new entry opportunities."""
        for symbol, data in historical_data.items():
            # Skip if already have position
            if symbol in self.open_positions:
                continue
            
            # Get current price
            current_price = self._get_price_for_date(data, date)
            if current_price is None:
                continue
            
            # Generate trading candidate
            candidate = self._create_trading_candidate(symbol, data, date, current_price)
            
            # Run through models
            trade_decision = self._evaluate_candidate(candidate)
            
            # Enter position if signal is BUY
            if trade_decision.signal == TradeSignal.BUY:
                self._enter_position(symbol, candidate, trade_decision, date, current_price)
    
    def _create_trading_candidate(self, symbol: str, data: List[Dict[str, Any]], 
                                date: str, current_price: float) -> Dict[str, Any]:
        """Create trading candidate from historical data."""
        # Get historical prices leading up to this date
        historical_prices = []
        historical_volumes = []
        
        for record in data:
            if record['date'] <= date:
                historical_prices.append(record['close'])
                historical_volumes.append(record['volume'])
        
        # Take last 20 days
        historical_prices = historical_prices[-20:]
        historical_volumes = historical_volumes[-20:]
        
        # Generate mock technical indicators
        technicals = self._generate_mock_technicals(historical_prices, current_price)
        
        return {
            'symbol': symbol,
            'historical_data': {
                'prices': historical_prices,
                'volumes': historical_volumes,
                'current_price': current_price
            },
            'technicals': technicals,
            'date': date
        }
    
    def _generate_mock_technicals(self, prices: List[float], current_price: float) -> Dict[str, float]:
        """Generate mock technical indicators from price history."""
        if len(prices) < 10:
            # Return neutral values if insufficient data
            return {
                'rsi': 50.0,
                'macd_signal': 0.0,
                'bollinger_position': 0.5,
                'volume_ratio': 1.0,
                'price_momentum': 0.0,
                'volatility_rank': 50.0,
                'vix_level': 20.0,
                'market_trend': 0.0
            }
        
        # Calculate simple RSI approximation
        gains = []
        losses = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else sum(gains) / len(gains)
        avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else sum(losses) / len(losses)
        
        rs = avg_gain / avg_loss if avg_loss != 0 else 100
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate momentum
        momentum = (current_price - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
        
        # Calculate volatility
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        volatility = math.sqrt(sum(r*r for r in returns) / len(returns)) if returns else 0
        
        # Generate other indicators
        return {
            'rsi': max(0, min(100, rsi)),
            'macd_signal': max(-1, min(1, momentum)),
            'bollinger_position': max(0, min(1, 0.5 + momentum)),
            'volume_ratio': 1.0 + (momentum * 0.5),
            'price_momentum': max(-1, min(1, momentum)),
            'volatility_rank': max(0, min(100, volatility * 1000)),
            'vix_level': 20.0 + (volatility * 100),
            'market_trend': max(-1, min(1, momentum * 0.8))
        }
    
    def _evaluate_candidate(self, candidate: Dict[str, Any]) -> TradeDecision:
        """Evaluate trading candidate using models."""
        try:
            # Generate model inputs
            model_inputs = []
            
            # Enhanced model predictor
            try:
                direction, confidence, metadata = self.model_predictor.predict_trade_signal(
                    candidate['symbol'], candidate['historical_data'], candidate['technicals']
                )
                model_inputs.append(ModelInput(
                    model_name='xgboost',
                    direction=direction,
                    confidence=confidence,
                    metadata=metadata
                ))
            except Exception as e:
                logger.warning(f"Model predictor error for {candidate['symbol']}: {e}")
            
            # Neural MLP
            try:
                neural_features = {
                    'technicals': candidate['technicals'],
                    'sequence': [[p, 1000000, p, p, p] for p in candidate['historical_data']['prices']]
                }
                direction, confidence, metadata = self.neural_mlp.predict_nn_signal(neural_features)
                model_inputs.append(ModelInput(
                    model_name='neural_mlp',
                    direction=direction,
                    confidence=confidence,
                    metadata=metadata
                ))
            except Exception as e:
                logger.warning(f"Neural MLP error for {candidate['symbol']}: {e}")
            
            # Meta-model decision
            if model_inputs:
                return self.meta_model.predict_trade_signal(model_inputs)
            else:
                return TradeDecision(
                    signal=TradeSignal.PASS,
                    confidence=0.0,
                    reasoning=["No model inputs available"],
                    context={},
                    model_inputs=[]
                )
        
        except Exception as e:
            logger.error(f"Error evaluating candidate {candidate['symbol']}: {e}")
            return TradeDecision(
                signal=TradeSignal.PASS,
                confidence=0.0,
                reasoning=[f"Evaluation error: {str(e)}"],
                context={},
                model_inputs=[]
            )
    
    def _enter_position(self, symbol: str, candidate: Dict[str, Any], 
                       decision: TradeDecision, date: str, price: float):
        """Enter a new position."""
        try:
            # Calculate position size (2% of capital)
            risk_amount = self.current_capital * 0.02
            contracts = max(1, int(risk_amount / (price * 100)))
            
            # Create position
            position = BacktestPosition(
                symbol=symbol,
                entry_date=date,
                exit_date=None,
                entry_price=price,
                exit_price=None,
                position_size=contracts,
                strategy="bull_put_spread",
                trade_id=f"BT_{symbol}_{date}"
            )
            
            # Track position
            self.open_positions[symbol] = position
            self.positions.append(position)
            
            # Log entry
            logger.info(f"Entered position: {symbol} at ${price:.2f} on {date}")
            
        except Exception as e:
            logger.error(f"Error entering position for {symbol}: {e}")
    
    def _close_position(self, symbol: str, position: BacktestPosition, 
                       exit_price: float, exit_date: str):
        """Close an existing position."""
        try:
            # Calculate PnL for bull put spread
            pnl = self._calculate_bull_put_pnl(position, exit_price)
            
            # Update position
            position.exit_date = exit_date
            position.exit_price = exit_price
            position.pnl = pnl
            position.pnl_percent = (pnl / (position.entry_price * position.position_size * 100)) * 100
            
            # Calculate hold days
            entry_date = datetime.strptime(position.entry_date, '%Y-%m-%d')
            exit_date_obj = datetime.strptime(exit_date, '%Y-%m-%d')
            position.hold_days = (exit_date_obj - entry_date).days
            
            # Update capital
            self.current_capital += pnl
            
            # Move to closed positions
            self.closed_positions.append(position)
            del self.open_positions[symbol]
            
            logger.info(f"Closed position: {symbol} P&L: ${pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
    
    def _calculate_bull_put_pnl(self, position: BacktestPosition, current_price: float) -> float:
        """Calculate P&L for bull put spread."""
        # Simplified bull put spread P&L calculation
        # Assumes credit spread with strikes 5% and 10% below entry price
        
        short_strike = position.entry_price * 0.95
        long_strike = position.entry_price * 0.90
        
        # Credit received (estimated)
        credit_received = (short_strike - long_strike) * 0.3 * position.position_size * 100
        
        # Calculate current value
        if current_price >= short_strike:
            # Both options expire worthless - keep full credit
            current_value = 0
        elif current_price <= long_strike:
            # Maximum loss
            current_value = (short_strike - long_strike) * position.position_size * 100
        else:
            # Partial loss
            current_value = (short_strike - current_price) * position.position_size * 100
        
        # P&L = Credit received - Current value
        pnl = credit_received - current_value
        
        return pnl
    
    def _get_price_for_date(self, data: List[Dict[str, Any]], date: str) -> Optional[float]:
        """Get price for specific date."""
        for record in data:
            if record['date'] == date:
                return record['close']
        return None
    
    def _close_all_positions(self, final_date: str):
        """Close all remaining positions at end of backtest."""
        for symbol, position in list(self.open_positions.items()):
            # Use entry price as exit price if no data available
            exit_price = position.entry_price
            self._close_position(symbol, position, exit_price, final_date)
    
    def _calculate_results(self, start_date: str, end_date: str) -> BacktestResult:
        """Calculate backtesting results."""
        total_trades = len(self.closed_positions)
        winning_trades = len([p for p in self.closed_positions if p.pnl and p.pnl > 0])
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(p.pnl for p in self.closed_positions if p.pnl)
        
        # Calculate other metrics
        returns = [p.pnl / (p.entry_price * p.position_size * 100) for p in self.closed_positions if p.pnl]
        avg_return = sum(returns) / len(returns) if returns else 0
        volatility = math.sqrt(sum((r - avg_return)**2 for r in returns) / len(returns)) if len(returns) > 1 else 0
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0
        
        avg_hold_days = sum(p.hold_days for p in self.closed_positions if p.hold_days) / total_trades if total_trades > 0 else 0
        
        # Performance by symbol
        performance_by_symbol = {}
        for position in self.closed_positions:
            symbol = position.symbol
            if symbol not in performance_by_symbol:
                performance_by_symbol[symbol] = {'trades': 0, 'pnl': 0, 'win_rate': 0}
            
            performance_by_symbol[symbol]['trades'] += 1
            performance_by_symbol[symbol]['pnl'] += position.pnl or 0
        
        # Calculate win rates by symbol
        for symbol in performance_by_symbol:
            symbol_positions = [p for p in self.closed_positions if p.symbol == symbol]
            wins = len([p for p in symbol_positions if p.pnl and p.pnl > 0])
            performance_by_symbol[symbol]['win_rate'] = wins / len(symbol_positions)
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            max_drawdown=0.0,  # TODO: Calculate actual max drawdown
            sharpe_ratio=sharpe_ratio,
            avg_hold_days=avg_hold_days,
            positions=self.closed_positions,
            performance_by_symbol=performance_by_symbol,
            monthly_returns={}  # TODO: Calculate monthly returns
        )


def create_sample_historical_data() -> Dict[str, List[Dict[str, Any]]]:
    """Create sample historical data for backtesting."""
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    data = {}
    
    for symbol in symbols:
        prices = []
        base_price = 150.0
        
        # Generate 100 days of data
        for i in range(100):
            date = (datetime.now() - timedelta(days=100-i)).strftime('%Y-%m-%d')
            
            # Random walk with slight upward bias
            change = (hash(f"{symbol}_{i}") % 1000 - 500) / 10000
            base_price *= (1 + change)
            
            prices.append({
                'date': date,
                'open': base_price * 0.99,
                'high': base_price * 1.02,
                'low': base_price * 0.98,
                'close': base_price,
                'volume': 50000000 + (hash(f"{symbol}_{i}") % 10000000)
            })
        
        data[symbol] = prices
    
    return data


def test_backtesting():
    """Test the backtesting engine."""
    print("Testing Backtesting Engine...")
    
    # Create sample data
    historical_data = create_sample_historical_data()
    
    # Initialize backtest engine
    engine = BacktestEngine(initial_capital=100000)
    
    # Run backtest
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    result = engine.run_backtest(historical_data, start_date, end_date)
    
    # Print results
    print(f"\nBacktest Results ({start_date} to {end_date}):")
    print(f"Total Trades: {result.total_trades}")
    print(f"Winning Trades: {result.winning_trades}")
    print(f"Win Rate: {result.win_rate:.1%}")
    print(f"Total P&L: ${result.total_pnl:,.2f}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Average Hold Days: {result.avg_hold_days:.1f}")
    
    print(f"\nPerformance by Symbol:")
    for symbol, perf in result.performance_by_symbol.items():
        print(f"  {symbol}: {perf['trades']} trades, ${perf['pnl']:,.2f} P&L, {perf['win_rate']:.1%} win rate")
    
    print(f"\nSample Positions:")
    for position in result.positions[:5]:  # Show first 5 positions
        print(f"  {position.symbol}: {position.entry_date} to {position.exit_date}, P&L: ${position.pnl:.2f}")


if __name__ == "__main__":
    test_backtesting() 