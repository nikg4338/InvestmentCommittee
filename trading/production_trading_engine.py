#!/usr/bin/env python3
"""
Production Trading Engine
========================

Integrates optimized models with real-time trading decisions using Gemini LLM analysis.
Updated to use our production-ready models from models/production/.
"""

import logging
import sys
import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import our production modules
from models.llm_analyzer import GeminiAnalyzer, AnalysisType, AnalysisResult
from trading.execution.alpaca_client import AlpacaClient
from trading.portfolio.position_manager import PositionManager
from trading.portfolio.performance_tracker import PerformanceTracker
from trading.strategy.risk_management import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class ProductionModelConfig:
    """Configuration for production models."""
    primary_model_path: str = "models/production/optimized_catboost.pkl"
    backup_models: List[str] = field(default_factory=lambda: [
        "models/production/random_forest.pkl",
        "models/production/svm.pkl"
    ])
    threshold_config_path: str = "config/optimized_thresholds_batch_1.json"
    feature_columns: List[str] = field(default_factory=list)
    portfolio_size: int = 20
    risk_tolerance: str = "moderate"


@dataclass
class TradingSignal:
    """Enhanced trading signal with LLM analysis."""
    symbol: str
    signal_strength: float  # 0.0 to 1.0
    confidence: float      # 0.0 to 1.0
    model_predictions: Dict[str, float]
    llm_analysis: Optional[AnalysisResult] = None
    technical_indicators: Dict[str, float] = field(default_factory=dict)
    risk_score: float = 0.5
    position_size: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    action: str = "HOLD"  # BUY, SELL, HOLD


@dataclass
class MarketContext:
    """Market context for LLM analysis."""
    market_sentiment: str = "NEUTRAL"
    vix_level: float = 20.0
    spy_change: float = 0.0
    sector_rotation: Dict[str, float] = field(default_factory=dict)
    economic_indicators: Dict[str, Any] = field(default_factory=dict)
    news_sentiment: float = 0.0


class ProductionModelEnsemble:
    """Production model ensemble for trading decisions."""
    
    def __init__(self, config: ProductionModelConfig):
        """Initialize production model ensemble."""
        self.config = config
        self.models = {}
        self.thresholds = {}
        self.feature_columns = []
        self.load_models()
        self.load_thresholds()
        
    def load_models(self):
        """Load production models."""
        logger.info("Loading production models...")
        
        # Load primary model
        try:
            primary_model = joblib.load(self.config.primary_model_path)
            model_name = Path(self.config.primary_model_path).stem
            self.models[model_name] = primary_model
            logger.info(f"Loaded primary model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load primary model: {e}")
            raise
        
        # Load backup models
        for backup_path in self.config.backup_models:
            try:
                if os.path.exists(backup_path):
                    backup_model = joblib.load(backup_path)
                    model_name = Path(backup_path).stem
                    self.models[model_name] = backup_model
                    logger.info(f"Loaded backup model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load backup model {backup_path}: {e}")
        
        logger.info(f"Loaded {len(self.models)} models total")
    
    def load_thresholds(self):
        """Load universal production thresholds."""
        threshold_files = [
            "config/production_thresholds.json",  # Universal production thresholds
            "config/model_thresholds.json",       # Fallback
            "config/optimized_thresholds_batch_1.json"  # Legacy fallback
        ]
        
        for threshold_file in threshold_files:
            if os.path.exists(threshold_file):
                try:
                    with open(threshold_file, 'r') as f:
                        threshold_data = json.load(f)
                    
                    # Extract model thresholds
                    if 'model_thresholds' in threshold_data:
                        for model_name, threshold_info in threshold_data['model_thresholds'].items():
                            if isinstance(threshold_info, dict) and 'threshold' in threshold_info:
                                self.thresholds[model_name] = threshold_info['threshold']
                            elif isinstance(threshold_info, (int, float)):
                                self.thresholds[model_name] = threshold_info
                    
                    logger.info(f"Loaded thresholds for {len(self.thresholds)} models from {threshold_file}")
                    break
                    
                except Exception as e:
                    logger.error(f"Failed to load thresholds from {threshold_file}: {e}")
        else:
            logger.warning("No threshold config found, using defaults")
            for model_name in self.models.keys():
                self.thresholds[model_name] = 0.5
    
    def prepare_features(self, market_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare features for model prediction."""
        # Standard feature columns based on our training data
        feature_columns = [
            'price_change_1d', 'price_change_5d', 'price_change_10d', 'price_change_20d',
            'price_vs_sma5', 'price_vs_sma10', 'price_vs_sma20',
            'volatility_5d', 'volatility_10d', 'volatility_20d',
            'volume_ratio', 'hl_ratio', 'hl_ratio_5d',
            'rsi_14', 'macd', 'macd_signal', 'macd_histogram', 'bb_position'
        ]
        
        # Extract features from market data
        features = {}
        for col in feature_columns:
            features[col] = market_data.get(col, 0.0)
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Fill any missing values
        df = df.fillna(df.median())
        
        return df
    
    def predict(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Generate predictions from all models."""
        predictions = {}
        
        try:
            # Prepare features
            features = self.prepare_features(market_data)
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        # Classification model
                        proba = model.predict_proba(features)[:, 1]
                        predictions[model_name] = float(proba[0])
                    elif hasattr(model, 'predict'):
                        # Regression model - convert to probability
                        raw_pred = model.predict(features)
                        # Apply sigmoid to convert to [0,1] range
                        proba = 1 / (1 + np.exp(-raw_pred[0]))
                        predictions[model_name] = float(proba)
                    else:
                        logger.warning(f"Model {model_name} has no predict method")
                        predictions[model_name] = 0.5
                        
                except Exception as e:
                    logger.error(f"Prediction failed for {model_name}: {e}")
                    predictions[model_name] = 0.5
                    
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            # Return default predictions
            for model_name in self.models.keys():
                predictions[model_name] = 0.5
        
        return predictions
    
    def ensemble_prediction(self, predictions: Dict[str, float]) -> Tuple[float, float]:
        """Create ensemble prediction with confidence."""
        if not predictions:
            return 0.5, 0.0
        
        # Use weighted average (can be enhanced with dynamic weights)
        weights = {
            'optimized_catboost': 0.5,  # Primary model gets higher weight
            'random_forest': 0.3,
            'svm': 0.2
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model_name, prediction in predictions.items():
            weight = weights.get(model_name, 0.1)  # Default weight for unknown models
            weighted_sum += prediction * weight
            total_weight += weight
        
        ensemble_pred = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Calculate confidence based on agreement between models and prediction strength
        pred_values = list(predictions.values())
        
        if len(pred_values) == 0:
            confidence = 0.0
        else:
            # Method 1: Agreement-based confidence (inverse of standard deviation)
            agreement_confidence = 1.0 - min(np.std(pred_values), 0.5) * 2
            
            # Method 2: Extremity-based confidence (how far from 0.5)
            extremity_confidence = abs(ensemble_pred - 0.5) * 2
            
            # Method 3: Threshold-based confidence (how far above/below threshold)
            primary_threshold = self.thresholds.get('optimized_catboost', 0.6)
            if ensemble_pred > primary_threshold:
                threshold_confidence = min((ensemble_pred - primary_threshold) / (1.0 - primary_threshold), 1.0)
            elif ensemble_pred < (1.0 - primary_threshold):
                threshold_confidence = min((1.0 - primary_threshold - ensemble_pred) / (1.0 - primary_threshold), 1.0)
            else:
                threshold_confidence = 0.0
            
            # Combine all confidence measures
            confidence = (agreement_confidence * 0.3 + extremity_confidence * 0.4 + threshold_confidence * 0.3)
            confidence = max(0.0, min(1.0, confidence))
        
        return ensemble_pred, confidence


class EnhancedTradingEngine:
    """Enhanced trading engine with LLM integration."""
    
    def __init__(self, 
                 alpaca_client: AlpacaClient,
                 model_config: ProductionModelConfig = None,
                 gemini_api_key: str = None):
        """Initialize enhanced trading engine."""
        
        self.alpaca_client = alpaca_client
        self.model_config = model_config or ProductionModelConfig()
        
        # Initialize model ensemble
        self.model_ensemble = ProductionModelEnsemble(self.model_config)
        
        # Initialize LLM analyzer
        try:
            from models.enhanced_llm_analyzer import create_gemini_analyzer
            self.llm_analyzer = create_gemini_analyzer(api_key=gemini_api_key)
            logger.info("Gemini LLM analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini analyzer: {e}")
            self.llm_analyzer = None
        
        # Initialize trading components
        self.position_manager = PositionManager(alpaca_client)
        
        # Create an empty trade log DataFrame for performance tracker
        import pandas as pd
        trade_log_columns = ['entry_time', 'exit_time', 'symbol', 'strategy', 'trade_type',
                           'direction', 'entry_price', 'exit_price', 'size', 'pnl', 'status']
        trade_log = pd.DataFrame(columns=trade_log_columns)
        self.performance_tracker = PerformanceTracker(trade_log)
        
        self.risk_manager = RiskManager()
        
        # Trading state
        self.market_context = MarketContext()
        self.active_signals: List[TradingSignal] = []
        self.last_analysis_time = None
        
    def update_market_context(self) -> MarketContext:
        """Update market context for LLM analysis."""
        try:
            # Get market data from Alpaca
            account_info = self.alpaca_client.get_account_info()
            
            # Get SPY data for market sentiment
            spy_data = self.alpaca_client.get_latest_quote("SPY")
            if spy_data:
                # Calculate basic market metrics
                spy_change = ((spy_data.get('ask_price', 0) - spy_data.get('bid_price', 0)) / 
                             spy_data.get('bid_price', 1)) * 100
                self.market_context.spy_change = spy_change
            
            # Get VIX if available
            try:
                vix_data = self.alpaca_client.get_latest_quote("VIX")
                if vix_data:
                    self.market_context.vix_level = vix_data.get('ask_price', 20.0)
            except:
                self.market_context.vix_level = 20.0  # Default
            
            # Determine market sentiment
            if self.market_context.spy_change > 1.0:
                self.market_context.market_sentiment = "BULLISH"
            elif self.market_context.spy_change < -1.0:
                self.market_context.market_sentiment = "BEARISH"
            else:
                self.market_context.market_sentiment = "NEUTRAL"
                
        except Exception as e:
            logger.error(f"Failed to update market context: {e}")
        
        return self.market_context
    
    def get_llm_analysis(self, symbol: str, market_data: Dict[str, Any]) -> Optional[AnalysisResult]:
        """Get LLM analysis for a symbol (optional if rate limited)."""
        if not self.llm_analyzer:
            logger.warning("LLM analyzer not available - using models only")
            return None
        
        try:
            # Prepare analysis prompt
            analysis_data = {
                'symbol': symbol,
                'market_context': asdict(self.market_context),
                'technical_data': market_data,
                'analysis_type': 'TRADE_DECISION'
            }
            
            # Get LLM analysis
            result = self.llm_analyzer.analyze_trade_opportunity(
                symbol=symbol,
                market_data=analysis_data,
                analysis_type='TRADE_DECISION'  # Use string instead of enum
            )
            
            logger.info(f"LLM analysis completed for {symbol}")
            return result
            
        except Exception as e:
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                logger.warning(f"LLM rate limited for {symbol} - continuing with ML models only")
                return None
            else:
                logger.error(f"LLM analysis failed for {symbol}: {e}")
                return None
            
        except Exception as e:
            logger.error(f"LLM analysis failed for {symbol}: {e}")
            return None
    
    def generate_trading_signal(self, symbol: str, market_data: Dict[str, Any]) -> TradingSignal:
        """Generate comprehensive trading signal with smart LLM consultation."""
        
        # Get model predictions first
        predictions = self.model_ensemble.predict(market_data)
        ensemble_pred, confidence = self.model_ensemble.ensemble_prediction(predictions)
        
        # Smart LLM consultation: Only consult LLM if ML models show promise
        llm_consultation_threshold = 0.4  # Only consult if ensemble prediction > 0.4 or < 0.6
        should_consult_llm = (ensemble_pred > llm_consultation_threshold and ensemble_pred < (1.0 - llm_consultation_threshold)) or confidence > 0.5
        
        llm_analysis = None
        if should_consult_llm and self.llm_analyzer:
            logger.info(f"Consulting LLM for {symbol} (ensemble: {ensemble_pred:.3f}, confidence: {confidence:.3f})")
            llm_analysis = self.get_llm_analysis(symbol, market_data)
        else:
            logger.info(f"⚡ Skipping LLM for {symbol} - clear ML signal (ensemble: {ensemble_pred:.3f}, confidence: {confidence:.3f})")
        
        # Calculate signal strength
        signal_strength = ensemble_pred
        
        # Integrate LLM analysis into final decision
        final_confidence = confidence
        if llm_analysis and llm_analysis.confidence > 0.6:
            logger.info(f"LLM analysis for {symbol}: {llm_analysis.recommendation} (confidence: {llm_analysis.confidence:.2f})")
            
            # Boost confidence if LLM agrees with ML models
            if llm_analysis.recommendation == "BUY" and signal_strength > 0.5:
                signal_strength = min(1.0, signal_strength * 1.1)
                final_confidence = min(1.0, confidence + 0.1)
            elif llm_analysis.recommendation == "SELL" and signal_strength < 0.5:
                signal_strength = max(0.0, signal_strength * 0.9)
                final_confidence = min(1.0, confidence + 0.1)
            elif llm_analysis.recommendation != "HOLD":
                # LLM disagrees - be more conservative
                final_confidence = max(0.0, confidence - 0.05)
        
        # Determine action based on thresholds
        primary_model = 'optimized_catboost'
        threshold = self.model_ensemble.thresholds.get(primary_model, 0.6)
        
        # Use lower confidence threshold for testing (0.3)
        min_confidence_threshold = 0.3
        
        if signal_strength >= threshold and final_confidence > min_confidence_threshold:
            action = "BUY"
        elif signal_strength < 0.3 and final_confidence > min_confidence_threshold:
            action = "SELL"
        else:
            action = "HOLD"
        
        # Calculate position size
        position_size = self.calculate_position_size(signal_strength, final_confidence)
        
        # Create trading signal
        signal = TradingSignal(
            symbol=symbol,
            signal_strength=signal_strength,
            confidence=final_confidence,
            model_predictions=predictions,
            llm_analysis=llm_analysis,
            technical_indicators=market_data,
            risk_score=1.0 - final_confidence,
            position_size=position_size,
            action=action
        )
        
        logger.info(f"{symbol}: {action} - Strength: {signal_strength:.3f}, Confidence: {final_confidence:.3f}")
        return signal
    
    def calculate_position_size(self, signal_strength: float, confidence: float) -> float:
        """Calculate position size based on signal strength and confidence."""
        base_size = 1.0 / self.model_config.portfolio_size  # Equal weight baseline
        
        # Adjust based on signal strength and confidence
        strength_multiplier = signal_strength * 2.0  # 0-2x multiplier
        confidence_multiplier = confidence  # 0-1x multiplier
        
        position_size = base_size * strength_multiplier * confidence_multiplier
        
        # Apply risk limits
        max_position = 0.1  # 10% max per position
        min_position = 0.01  # 1% minimum
        
        return max(min_position, min(max_position, position_size))
    
    def process_trading_universe(self, symbols: List[str]) -> List[TradingSignal]:
        """Process a universe of symbols for trading signals."""
        signals = []
        
        logger.info(f"Processing {len(symbols)} symbols for trading signals...")
        
        for symbol in symbols:
            try:
                # Get market data for symbol
                market_data = self.get_symbol_market_data(symbol)
                
                if market_data:
                    # Generate trading signal
                    signal = self.generate_trading_signal(symbol, market_data)
                    signals.append(signal)
                    
                    logger.info(f"{symbol}: {signal.action} - "
                               f"Strength: {signal.signal_strength:.3f}, "
                               f"Confidence: {signal.confidence:.3f}")
                else:
                    logger.warning(f"No market data available for {symbol}")
                    
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
        
        return signals
    
    def get_symbol_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get market data for a symbol."""
        try:
            # Get historical data for technical indicators
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            # Get bars from Alpaca
            bars = self.alpaca_client.get_bars(
                symbol, 
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                timeframe='1Day'
            )
            
            if not bars or len(bars) < 20:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Calculate technical indicators
            market_data = self.calculate_technical_indicators(bars)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, bars: List[Dict]) -> Dict[str, float]:
        """Calculate technical indicators from price bars."""
        try:
            # Convert to DataFrame for easier calculation
            df = pd.DataFrame(bars)
            df['close'] = pd.to_numeric(df['close'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['volume'] = pd.to_numeric(df['volume'])
            
            # Calculate indicators
            indicators = {}
            
            # Price changes
            indicators['price_change_1d'] = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) if len(df) >= 2 else 0
            indicators['price_change_5d'] = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) if len(df) >= 6 else 0
            indicators['price_change_10d'] = (df['close'].iloc[-1] / df['close'].iloc[-11] - 1) if len(df) >= 11 else 0
            indicators['price_change_20d'] = (df['close'].iloc[-1] / df['close'].iloc[-21] - 1) if len(df) >= 21 else 0
            
            # Moving averages
            sma5 = df['close'].rolling(5).mean().iloc[-1] if len(df) >= 5 else df['close'].iloc[-1]
            sma10 = df['close'].rolling(10).mean().iloc[-1] if len(df) >= 10 else df['close'].iloc[-1]
            sma20 = df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['close'].iloc[-1]
            
            indicators['price_vs_sma5'] = (df['close'].iloc[-1] / sma5 - 1)
            indicators['price_vs_sma10'] = (df['close'].iloc[-1] / sma10 - 1)
            indicators['price_vs_sma20'] = (df['close'].iloc[-1] / sma20 - 1)
            
            # Volatility
            returns = df['close'].pct_change()
            indicators['volatility_5d'] = returns.rolling(5).std().iloc[-1] * np.sqrt(252) if len(df) >= 5 else 0
            indicators['volatility_10d'] = returns.rolling(10).std().iloc[-1] * np.sqrt(252) if len(df) >= 10 else 0
            indicators['volatility_20d'] = returns.rolling(20).std().iloc[-1] * np.sqrt(252) if len(df) >= 20 else 0
            
            # Volume
            avg_volume = df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['volume'].mean()
            indicators['volume_ratio'] = df['volume'].iloc[-1] / avg_volume if avg_volume > 0 else 1
            
            # High-Low ratios
            indicators['hl_ratio'] = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1] if df['close'].iloc[-1] > 0 else 0
            if len(df) >= 5:
                hl_5d = ((df['high'].rolling(5).max() - df['low'].rolling(5).min()) / df['close']).iloc[-1]
                indicators['hl_ratio_5d'] = hl_5d
            else:
                indicators['hl_ratio_5d'] = indicators['hl_ratio']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            indicators['rsi_14'] = (100 - (100 / (1 + rs))).iloc[-1] if len(df) >= 14 else 50
            
            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            
            indicators['macd'] = macd_line.iloc[-1] if len(df) >= 26 else 0
            indicators['macd_signal'] = signal_line.iloc[-1] if len(df) >= 26 else 0
            indicators['macd_histogram'] = (macd_line - signal_line).iloc[-1] if len(df) >= 26 else 0
            
            # Bollinger Bands
            sma20_bb = df['close'].rolling(20).mean()
            std20 = df['close'].rolling(20).std()
            bb_upper = sma20_bb + (std20 * 2)
            bb_lower = sma20_bb - (std20 * 2)
            indicators['bb_position'] = ((df['close'].iloc[-1] - bb_lower.iloc[-1]) / 
                                       (bb_upper.iloc[-1] - bb_lower.iloc[-1])) if len(df) >= 20 else 0.5
            
            # Clean up any NaN values
            for key, value in indicators.items():
                if pd.isna(value):
                    indicators[key] = 0.0
                else:
                    indicators[key] = float(value)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Failed to calculate technical indicators: {e}")
            return {}
    
    def execute_trading_signals(self, signals: List[TradingSignal]) -> Dict[str, Any]:
        """Execute trading signals."""
        results = {
            'executed_trades': [],
            'failed_trades': [],
            'portfolio_summary': {}
        }
        
        # Filter signals for actionable trades
        buy_signals = [s for s in signals if s.action == "BUY" and s.confidence > 0.6]
        sell_signals = [s for s in signals if s.action == "SELL" and s.confidence > 0.6]
        
        # Sort by signal strength
        buy_signals.sort(key=lambda x: x.signal_strength * x.confidence, reverse=True)
        sell_signals.sort(key=lambda x: x.signal_strength * x.confidence, reverse=True)
        
        # Execute BUY trades
        for signal in buy_signals[:self.model_config.portfolio_size]:
            try:
                success = self.position_manager.execute_trade(
                    symbol=signal.symbol,
                    action=signal.action,
                    position_size=signal.position_size,
                    signal_metadata=asdict(signal)
                )
                
                if success:
                    results['executed_trades'].append(signal.symbol)
                    logger.info(f"Executed {signal.action} for {signal.symbol}")
                else:
                    results['failed_trades'].append(signal.symbol)
                    logger.error(f"Failed to execute {signal.action} for {signal.symbol}")
                    
            except Exception as e:
                logger.error(f"Trade execution failed for {signal.symbol}: {e}")
                results['failed_trades'].append(signal.symbol)
        
        # Execute SELL trades (for existing positions)
        for signal in sell_signals:
            try:
                # Check if we have a position to sell
                current_positions = self.position_manager.get_current_positions()
                if signal.symbol in current_positions:
                    success = self.position_manager.execute_trade(
                        symbol=signal.symbol,
                        action=signal.action,
                        position_size=signal.position_size,
                        signal_metadata=asdict(signal)
                    )
                    
                    if success:
                        results['executed_trades'].append(signal.symbol)
                        logger.info(f"Executed {signal.action} for {signal.symbol}")
                    else:
                        results['failed_trades'].append(signal.symbol)
                        logger.error(f"Failed to execute {signal.action} for {signal.symbol}")
                else:
                    logger.info(f"No position to sell for {signal.symbol}, skipping SELL signal")
                    
            except Exception as e:
                logger.error(f"Trade execution failed for {signal.symbol}: {e}")
                results['failed_trades'].append(signal.symbol)
        
        # Update portfolio summary
        results['portfolio_summary'] = self.position_manager.get_portfolio_summary()
        
        return results


def create_production_trading_engine(
    alpaca_key_id: str = None,
    alpaca_secret: str = None,
    gemini_api_key: str = None,
    paper_trading: bool = True
) -> EnhancedTradingEngine:
    """Create production trading engine with all components."""
    
    # Initialize Alpaca client
    alpaca_client = AlpacaClient()
    
    # Create model configuration
    model_config = ProductionModelConfig(
        portfolio_size=20,
        risk_tolerance="moderate"
    )
    
    # Initialize trading engine
    trading_engine = EnhancedTradingEngine(
        alpaca_client=alpaca_client,
        model_config=model_config,
        gemini_api_key=gemini_api_key
    )
    
    logger.info("Production trading engine initialized")
    
    return trading_engine


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create trading engine
    engine = create_production_trading_engine(paper_trading=True)
    
    # Test with sample symbols
    test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    
    # Update market context
    engine.update_market_context()
    
    # Generate signals
    signals = engine.process_trading_universe(test_symbols)
    
    # Execute trades
    results = engine.execute_trading_signals(signals)
    
    print("\n" + "="*60)
    print("TRADING ENGINE TEST RESULTS")
    print("="*60)
    print(f"Signals Generated: {len(signals)}")
    print(f"Trades Executed: {len(results['executed_trades'])}")
    print(f"Failed Trades: {len(results['failed_trades'])}")
    print("="*60)
