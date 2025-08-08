#!/usr/bin/env python3
"""
Alpaca Data Collection for Investment Committee Training
========================================================

This module integrates the Alpaca API with the Committee of Five training pipeline.
It fetches market data for stocks in filtered_iex_batches.json and engineers
features relevant to bull put spread trading strategies.

Features engineered:
- Price momentum indicators
- Volatility measures
- Volume analysis
- Technical indicators
- Market context features

Target variable: Bull put spread trade eligibility based on trade_filter criteria
"""

import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

# Import Alpaca client and trade filter
from trading.execution.alpaca_client import AlpacaClient
from trading.strategy.trade_filter import is_trade_eligible

logger = logging.getLogger(__name__)


class AlpacaDataCollector:
    """
    Collects and engineers training data from Alpaca API for the Committee of Five models.
    """
    
    def __init__(self):
        """Initialize the data collector with Alpaca client."""
        try:
            self.alpaca_client = AlpacaClient()
            logger.info("✅ Alpaca client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Alpaca client: {e}")
            raise
            
    def load_stock_batches(self, batch_file: str = "filtered_iex_batches.json") -> Dict[str, List[str]]:
        """
        Load stock batches from JSON file.
        
        Args:
            batch_file: Path to the batch file
            
        Returns:
            Dict mapping batch names to stock symbols
        """
        try:
            with open(batch_file, 'r') as f:
                data = json.load(f)
            
            batches = data.get('batches', {})
            total_symbols = sum(len(symbols) for symbols in batches.values())
            
            logger.info(f"✅ Loaded {len(batches)} batches with {total_symbols} total symbols")
            return batches
        except Exception as e:
            logger.error(f"❌ Failed to load batch file {batch_file}: {e}")
            raise
    
    def get_historical_data(self, symbol: str, days: int = 730) -> Optional[pd.DataFrame]:
        """
        Get historical price data for a symbol.
        
        Args:
            symbol: Stock symbol
            days: Number of days of historical data (default: 730 = ~24 months)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Get historical bars with extended look-back (24 months)
            market_data = self.alpaca_client.get_market_data(
                symbol=symbol,
                timeframe='1Day',
                limit=days
            )
            
            if not market_data or 'bars' not in market_data:
                logger.warning(f"⚠️  No market data for {symbol}")
                return None
                
            bars = market_data['bars']
            if not bars:
                logger.warning(f"⚠️  Empty bars for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'timestamp': bar['timestamp'],
                    'open': float(bar['open']),
                    'high': float(bar['high']),
                    'low': float(bar['low']),
                    'close': float(bar['close']),
                    'volume': int(bar['volume'])
                }
                for bar in bars
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to get historical data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for feature engineering, including regime-aware features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicator columns
        """
        if len(df) < 20:
            logger.warning("⚠️  Insufficient data for technical indicators")
            return df
            
        # Price-based features
        df['price_change_1d'] = df['close'].pct_change(1)
        df['price_change_5d'] = df['close'].pct_change(5) 
        df['price_change_10d'] = df['close'].pct_change(10)
        df['price_change_20d'] = df['close'].pct_change(20)
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean() if len(df) >= 50 else np.nan
        df['sma_200'] = df['close'].rolling(200).mean() if len(df) >= 200 else np.nan
        
        # Price relative to moving averages
        df['price_vs_sma5'] = df['close'] / df['sma_5'] - 1
        df['price_vs_sma10'] = df['close'] / df['sma_10'] - 1
        df['price_vs_sma20'] = df['close'] / df['sma_20'] - 1
        if 'sma_50' in df.columns:
            df['price_vs_sma50'] = df['close'] / df['sma_50'] - 1
        if 'sma_200' in df.columns:
            df['price_vs_sma200'] = df['close'] / df['sma_200'] - 1
        
        # REGIME-AWARE FEATURES (Improvement #8)
        
        # 1. Trend Regime Detection
        df['trend_regime'] = 0  # Default: sideways
        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            # Bull regime: price > SMA50 > SMA200 and rising
            bull_condition = (
                (df['close'] > df['sma_50']) & 
                (df['sma_50'] > df['sma_200']) &
                (df['sma_50'] > df['sma_50'].shift(5))  # 50 SMA rising
            )
            df.loc[bull_condition, 'trend_regime'] = 1
            
            # Bear regime: price < SMA50 < SMA200 and falling  
            bear_condition = (
                (df['close'] < df['sma_50']) & 
                (df['sma_50'] < df['sma_200']) &
                (df['sma_50'] < df['sma_50'].shift(5))  # 50 SMA falling
            )
            df.loc[bear_condition, 'trend_regime'] = -1
        
        # 2. Volatility Regime Detection
        df['volatility_5d'] = df['price_change_1d'].rolling(5).std()
        df['volatility_10d'] = df['price_change_1d'].rolling(10).std()
        df['volatility_20d'] = df['price_change_1d'].rolling(20).std()
        df['volatility_50d'] = df['price_change_1d'].rolling(50).std() if len(df) >= 50 else np.nan
        
        # Volatility regime (relative to 50-day average)
        if 'volatility_50d' in df.columns:
            vol_percentile = df['volatility_20d'].rolling(50).rank(pct=True)
            df['volatility_regime'] = 0  # Normal
            df.loc[vol_percentile > 0.8, 'volatility_regime'] = 1   # High vol
            df.loc[vol_percentile < 0.2, 'volatility_regime'] = -1  # Low vol
        else:
            df['volatility_regime'] = 0
        
        # 3. Volume Regime Detection
        df['volume_sma_10'] = df['volume'].rolling(10).mean()
        df['volume_sma_50'] = df['volume'].rolling(50).mean() if len(df) >= 50 else np.nan
        df['volume_ratio'] = df['volume'] / df['volume_sma_10']
        
        # Volume regime
        if 'volume_sma_50' in df.columns:
            volume_percentile = df['volume'].rolling(50).rank(pct=True)
            df['volume_regime'] = 0  # Normal
            df.loc[volume_percentile > 0.8, 'volume_regime'] = 1   # High volume
            df.loc[volume_percentile < 0.2, 'volume_regime'] = -1  # Low volume
        else:
            df['volume_regime'] = 0
        
        # 4. Market Microstructure Regime
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['hl_ratio_5d'] = df['hl_ratio'].rolling(5).mean()
        
        # Gap detection (regime change signals)
        df['gap_up'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)) > 0.02
        df['gap_down'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)) < -0.02
        
        # 5. Momentum Regime Features
        # RSI (simplified)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # RSI-based momentum regime
        df['momentum_regime'] = 0  # Neutral
        df.loc[df['rsi_14'] > 70, 'momentum_regime'] = 1   # Overbought
        df.loc[df['rsi_14'] < 30, 'momentum_regime'] = -1  # Oversold
        
        # 6. MACD Regime
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # MACD regime
        df['macd_regime'] = 0
        df.loc[(df['macd'] > df['macd_signal']) & (df['macd_histogram'] > 0), 'macd_regime'] = 1   # Bullish
        df.loc[(df['macd'] < df['macd_signal']) & (df['macd_histogram'] < 0), 'macd_regime'] = -1  # Bearish
        
        # 7. Bollinger Bands and Mean Reversion Regime
        bb_sma = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = bb_sma + (bb_std * 2)
        df['bb_lower'] = bb_sma - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Mean reversion regime
        df['mean_reversion_regime'] = 0
        df.loc[df['bb_position'] > 0.8, 'mean_reversion_regime'] = 1   # Near upper band (potential reversion)
        df.loc[df['bb_position'] < 0.2, 'mean_reversion_regime'] = -1  # Near lower band (potential reversion)
        
        # 8. Composite Regime Score
        # Combine multiple regime indicators for a comprehensive score
        regime_components = ['trend_regime', 'volatility_regime', 'momentum_regime', 'macd_regime']
        df['composite_regime_score'] = df[regime_components].sum(axis=1)
        
        # 9. Regime Stability (how long in current regime)
        df['trend_regime_changes'] = (df['trend_regime'] != df['trend_regime'].shift(1)).astype(int)
        df['trend_regime_stability'] = (~df['trend_regime_changes'].astype(bool)).groupby(df['trend_regime_changes'].cumsum()).cumsum()
        
        # 10. Cross-Regime Interactions
        df['bull_low_vol'] = ((df['trend_regime'] == 1) & (df['volatility_regime'] <= 0)).astype(int)
        df['bear_high_vol'] = ((df['trend_regime'] == -1) & (df['volatility_regime'] == 1)).astype(int)
        df['sideways_low_vol'] = ((df['trend_regime'] == 0) & (df['volatility_regime'] == -1)).astype(int)
        
        logger.info(f"📊 Enhanced with regime-aware features:")
        logger.info(f"   Trend regime distribution: {df['trend_regime'].value_counts().to_dict()}")
        logger.info(f"   Volatility regime distribution: {df['volatility_regime'].value_counts().to_dict()}")
        logger.info(f"   Composite regime score range: [{df['composite_regime_score'].min():.1f}, {df['composite_regime_score'].max():.1f}]")
        
        # === ENHANCED FEATURE ENGINEERING ===
        
        # 11. TIME-DECAY AND GREEKS-INSPIRED FEATURES
        logger.info("📈 Adding time-decay and Greeks-inspired features...")
        
        # Theta-like feature: rate of price change acceleration
        df['price_acceleration'] = df['close'].diff().diff()
        df['price_theta'] = df['price_acceleration'] / df['close']
        
        # Vega-like feature: volatility sensitivity
        df['vol_sensitivity'] = df['volatility_20d'].diff() / df['volatility_20d'].shift(1)
        df['price_vol_correlation'] = df['close'].rolling(20).corr(df['volatility_20d'])
        
        # Delta-like feature: price momentum sensitivity
        df['delta_proxy'] = df['price_change_1d'] / df['volatility_20d']
        
        # Gamma-like feature: acceleration of momentum
        df['momentum_acceleration'] = df['delta_proxy'].diff()
        
        # Implied volatility proxy using high-low ranges
        df['implied_vol_proxy'] = (df['high'] - df['low']) / df['close']
        df['implied_vol_change'] = df['implied_vol_proxy'].diff()
        df['implied_vol_percentile'] = df['implied_vol_proxy'].rolling(50).rank(pct=True)
        
        # 12. MARKET REGIME AND MACRO FEATURES
        logger.info("🌍 Adding market regime and macro context features...")
        
        # VIX-like volatility regime (using rolling volatility percentiles)
        df['vol_percentile_50d'] = df['volatility_20d'].rolling(50).rank(pct=True)
        df['vol_regime_high'] = (df['vol_percentile_50d'] > 0.8).astype(int)
        df['vol_regime_low'] = (df['vol_percentile_50d'] < 0.2).astype(int)
        
        # ENHANCED: VIX top 10 percentile flag (extreme volatility periods)
        df['vix_top_10pct'] = (df['vol_percentile_50d'] > 0.9).astype(int)
        df['vix_bottom_10pct'] = (df['vol_percentile_50d'] < 0.1).astype(int)
        
        # ENHANCED: Time decay features for options-like strategies
        df['time_to_expiry_proxy'] = np.arange(len(df)) % 21  # Proxy for days until monthly expiry
        df['theta_decay'] = df['implied_vol_proxy'] * np.sqrt(df['time_to_expiry_proxy'] / 21.0)  # Time decay simulation
        df['theta_acceleration'] = df['theta_decay'].diff()  # Rate of time decay change
        
        # ENHANCED: Spread width features (distance of strikes / underlying move)
        df['atr_20d'] = df['hl_ratio'].rolling(20).mean()  # Average True Range proxy
        df['spread_width_proxy'] = df['atr_20d'] / df['close']  # Normalized spread width
        df['move_vs_spread'] = abs(df['price_change_1d']) / (df['spread_width_proxy'] + 1e-8)  # How much moved vs expected
        df['spread_efficiency'] = df['spread_width_proxy'] / (df['volatility_20d'] + 1e-8)  # Spread width efficiency
        
        # Market trend strength (proxy using price vs moving averages)
        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            df['market_trend_strength'] = df['close'].rolling(20).corr(df['sma_50'])
            df['long_term_trend'] = (df['sma_50'] > df['sma_200']).astype(int)
        else:
            df['market_trend_strength'] = df['close'].rolling(20).corr(df['sma_20'])
            df['long_term_trend'] = (df['sma_20'] > df['sma_10']).astype(int)
        
        # Sector momentum proxy (relative to own history)
        df['relative_strength'] = df['close'] / df['close'].rolling(252).mean() if len(df) >= 252 else df['close'] / df['close'].rolling(50).mean()
        df['momentum_percentile'] = df['price_change_20d'].rolling(100).rank(pct=True)
        
        # 13. EARNINGS AND FUNDAMENTAL PROXIES
        logger.info("💼 Adding earnings and fundamental proxies...")
        
        # Earnings cycle proxies (quarterly patterns)
        df['quarter'] = pd.to_datetime(df.index).quarter if hasattr(df.index, 'quarter') else 1
        df['month'] = pd.to_datetime(df.index).month if hasattr(df.index, 'month') else 1
        df['earnings_season'] = df['month'].isin([1, 4, 7, 10]).astype(int)  # Earnings months
        
        # Volume-price divergence (fundamental strength indicator)
        df['volume_price_divergence'] = df['volume_ratio'] * df['price_change_1d']
        df['accumulation_distribution'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-8) * df['volume']
        df['accumulation_distribution_sma'] = df['accumulation_distribution'].rolling(20).mean()
        
        # 14. TECHNICAL PATTERN RECOGNITION
        logger.info("🔍 Adding technical pattern recognition features...")
        
        # Candlestick patterns
        df['doji'] = (abs(df['open'] - df['close']) / (df['high'] - df['low'] + 1e-8) < 0.1).astype(int)
        df['hammer'] = ((df['close'] > df['open']) & ((df['open'] - df['low']) > 2 * (df['close'] - df['open'])) & ((df['high'] - df['close']) < 0.1 * (df['close'] - df['open']))).astype(int)
        df['shooting_star'] = ((df['open'] > df['close']) & ((df['high'] - df['open']) > 2 * (df['open'] - df['close'])) & ((df['close'] - df['low']) < 0.1 * (df['open'] - df['close']))).astype(int)
        
        # Support/Resistance levels
        df['resistance_level'] = df['high'].rolling(20).max()
        df['support_level'] = df['low'].rolling(20).min()
        df['distance_to_resistance'] = (df['resistance_level'] - df['close']) / df['close']
        df['distance_to_support'] = (df['close'] - df['support_level']) / df['close']
        
        # 15. VOLATILITY CLUSTERING AND ARCH EFFECTS
        logger.info("📊 Adding volatility clustering features...")
        
        # GARCH-like features
        df['vol_clustering'] = df['volatility_5d'].rolling(10).std()
        df['vol_persistence'] = df['volatility_20d'].rolling(5).mean() / df['volatility_20d'].rolling(20).mean()
        
        # Volatility skew
        df['vol_skew'] = df['price_change_1d'].rolling(20).skew()
        df['vol_kurtosis'] = df['price_change_1d'].rolling(20).apply(lambda x: x.kurtosis())
        
        # 16. LIQUIDITY AND MICROSTRUCTURE FEATURES
        logger.info("💧 Adding liquidity and microstructure features...")
        
        # Bid-ask spread proxy
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        df['spread_volatility'] = df['spread_proxy'].rolling(10).std()
        
        # Price impact proxy
        df['price_impact'] = abs(df['price_change_1d']) / (df['volume'] / df['volume'].rolling(20).mean() + 1e-8)
        
        # Amihud illiquidity measure proxy
        df['illiquidity_proxy'] = abs(df['price_change_1d']) / (df['volume'] * df['close'] + 1e-8)
        
        # 17. MOMENTUM AND MEAN REVERSION FEATURES
        logger.info("🔄 Adding momentum and mean reversion features...")
        
        # Multiple timeframe momentum
        df['momentum_3d'] = df['close'].pct_change(3)
        df['momentum_7d'] = df['close'].pct_change(7)
        df['momentum_14d'] = df['close'].pct_change(14)
        df['momentum_21d'] = df['close'].pct_change(21)
        
        # Mean reversion indicators
        df['mean_reversion_5d'] = df['close'] / df['close'].rolling(5).mean() - 1
        df['mean_reversion_20d'] = df['close'] / df['close'].rolling(20).mean() - 1
        
        # Momentum consistency
        momentum_cols = ['momentum_3d', 'momentum_7d', 'momentum_14d', 'momentum_21d']
        df['momentum_consistency'] = df[momentum_cols].apply(lambda x: (x > 0).sum(), axis=1)
        
        # 18. CORRELATION AND BETA FEATURES
        logger.info("🔗 Adding correlation and systematic risk features...")
        
        # Beta to market proxy (using own price as market proxy - simplified)
        market_returns = df['price_change_1d'].rolling(252).mean() if len(df) >= 252 else df['price_change_1d'].rolling(50).mean()
        df['beta_proxy'] = df['price_change_1d'].rolling(60).cov(market_returns) / market_returns.rolling(60).var()
        
        # Correlation stability
        df['correlation_stability'] = df['price_change_1d'].rolling(20).corr(df['price_change_1d'].shift(1))
        
        # 19. NEWS AND SENTIMENT PROXIES
        logger.info("📰 Adding news and sentiment proxies...")
        
        # Extreme price movements (news proxy)
        df['extreme_move_up'] = (df['price_change_1d'] > df['price_change_1d'].rolling(60).quantile(0.95)).astype(int)
        df['extreme_move_down'] = (df['price_change_1d'] < df['price_change_1d'].rolling(60).quantile(0.05)).astype(int)
        
        # Gap analysis (potential news events)
        df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['gap_magnitude'] = abs(df['overnight_gap'])
        df['gap_follow_through'] = df['overnight_gap'] * df['price_change_1d']  # Gap continuation
        
        # 20. COMPOSITE SIGNAL FEATURES
        logger.info("🎯 Creating composite signal features...")
        
        # Technical strength composite
        technical_signals = ['momentum_regime', 'macd_regime', 'trend_regime']
        df['technical_strength'] = df[technical_signals].sum(axis=1)
        
        # Risk-adjusted returns
        df['risk_adjusted_return_5d'] = df['momentum_3d'] / (df['volatility_5d'] + 1e-8)
        df['risk_adjusted_return_20d'] = df['momentum_7d'] / (df['volatility_20d'] + 1e-8)
        
        # Quality score (combination of multiple factors)
        quality_factors = ['vol_regime_low', 'momentum_consistency', 'trend_regime_stability']
        df['quality_score'] = df[quality_factors].sum(axis=1)
        
        logger.info(f"🎉 Enhanced feature engineering complete!")
        logger.info(f"   📊 Total features: {len(df.columns)}")
        logger.info(f"   🔧 Greeks-inspired features: price_theta, vol_sensitivity, delta_proxy, momentum_acceleration")
        logger.info(f"   🌍 Market regime features: vol_percentile_50d, market_trend_strength, relative_strength")
        logger.info(f"   💼 Fundamental proxies: earnings_season, accumulation_distribution, volume_price_divergence")
        logger.info(f"   🔍 Pattern recognition: candlestick patterns, support/resistance levels")
        logger.info(f"   📊 Advanced analytics: volatility clustering, liquidity proxies, correlation features")
        
        return df
    
    def create_target_variable(self, df: pd.DataFrame, symbol: str, use_regression: bool = True, 
                             target_horizon: int = 3, 
                             create_all_horizons: bool = True,
                             target_strategy: str = 'top_percentile') -> Union[pd.Series, pd.DataFrame]:
        """
        Create enhanced target variable(s) with improved positive sample rate and richer patterns.
        
        Args:
            df: DataFrame with price data and indicators
            symbol: Stock symbol
            use_regression: If True, use returns; if False, use enhanced classification
            target_horizon: Primary target horizon in days (default: 3 days)
            create_all_horizons: If True, create multiple horizons including 7, 14, 21 days
            target_strategy: Strategy for target creation ('top_percentile', 'multi_class', 'quantile_buckets')
            
        Returns:
            Series with single target OR DataFrame with multiple target columns
        """
        if use_regression:
            # Keep regression approach but add enhanced sampling support
            if create_all_horizons:
                # Extended horizons including longer holding periods
                horizons = [1, 3, 5, 7, 10, 14, 21]  # Added 7, 14, 21-day horizons
                target_columns = {}
                
                for h in horizons:
                    return_col = f'target_{h}d_return'
                    target_columns[return_col] = self._create_regression_target(df, symbol, h)
                
                # Add enhanced binary targets for ensemble diversity
                for h in horizons:
                    binary_col = f'target_{h}d_enhanced'
                    target_columns[binary_col] = self._create_enhanced_target(df, symbol, h, target_strategy)
                
                # Create multi-target DataFrame
                target_df = pd.DataFrame(target_columns, index=df.index)
                
                logger.info(f"📊 Created {len(horizons)} regression + {len(horizons)} enhanced targets for {symbol}")
                for col in target_df.columns:
                    if 'return' in col:
                        valid_values = target_df[col].dropna()
                        if len(valid_values) > 0:
                            logger.info(f"   {col}: mean={valid_values.mean():.4f}, std={valid_values.std():.4f}")
                
                return target_df
            
            else:
                # Single target horizon
                target_series = self._create_regression_target(df, symbol, target_horizon)
                
                logger.info(f"📊 Created regression target: {target_horizon}-day daily returns for {symbol}")
                valid_values = target_series.dropna()
                if len(valid_values) > 0:
                    logger.info(f"   Target stats: mean={valid_values.mean():.4f}, std={valid_values.std():.4f}")
                
                return target_series
        
        else:
            # Use enhanced classification strategies
            if create_all_horizons:
                # Extended horizons including longer holding periods
                horizons = [1, 3, 5, 7, 10, 14, 21]
                target_columns = {}
                
                for h in horizons:
                    enhanced_col = f'target_{h}d_enhanced'
                    target_columns[enhanced_col] = self._create_enhanced_target(df, symbol, h, target_strategy)
                
                target_df = pd.DataFrame(target_columns, index=df.index)
                logger.info(f"📊 Created {len(horizons)} enhanced classification targets for {symbol}")
                
                return target_df
            else:
                return self._create_enhanced_target(df, symbol, target_horizon, target_strategy)
                return self._create_binary_target(df, symbol, target_horizon)
    
    def _create_enhanced_target(self, df: pd.DataFrame, symbol: str, target_horizon: int = 3, 
                              target_strategy: str = 'top_percentile') -> pd.Series:
        """
        Create enhanced target variable using various strategies for better model learning.
        
        Args:
            df: DataFrame with price data
            symbol: Stock symbol
            target_horizon: Target horizon in days
            target_strategy: Strategy to use ('top_percentile', 'multi_class', 'quantile_buckets')
            
        Returns:
            Enhanced target series
        """
        # Calculate forward returns
        forward_returns = (df['close'].shift(-target_horizon) / df['close']) - 1
        forward_returns = forward_returns.dropna()
        
        if target_strategy == 'top_percentile':
            # Top 25% labeling strategy for more generous positive identification
            top_percentile = 75  # Top 25% as positive (more generous than 80/90)
            
            if len(forward_returns) > 0:
                top_threshold = forward_returns.quantile(top_percentile / 100)
                
                targets = []
                for i in range(len(df)):
                    if i >= len(df) - target_horizon:
                        targets.append(0)
                        continue
                        
                    future_return = forward_returns.iloc[i] if i < len(forward_returns) else np.nan
                    
                    if pd.isna(future_return):
                        targets.append(0)
                    elif future_return >= top_threshold:
                        targets.append(1)  # Top 25% performers
                    else:
                        targets.append(0)  # Bottom 75% as negatives (use all remaining data)
                
                positive_rate = sum(targets) / len(targets) if len(targets) > 0 else 0
                logger.info(f"📊 Top {100-top_percentile}% strategy for {symbol}: {sum(targets)}/{len(targets)} positives ({100*positive_rate:.1f}%)")
                logger.info(f"   Threshold: top={top_threshold:.4f} (25% positive, 75% negative)")
                logger.info(f"   📈 Using ALL data: no samples discarded, robust negative class with full spectrum")
        
        elif target_strategy == 'multi_class':
            # Multi-class strategy: strong loss / neutral / strong gain
            if len(forward_returns) > 0:
                strong_gain_threshold = forward_returns.quantile(0.80)  # Top 20%
                strong_loss_threshold = forward_returns.quantile(0.20)  # Bottom 20%
                
                targets = []
                for i in range(len(df)):
                    if i >= len(df) - target_horizon:
                        targets.append(1)  # Neutral default
                        continue
                        
                    future_return = forward_returns.iloc[i] if i < len(forward_returns) else np.nan
                    
                    if pd.isna(future_return):
                        targets.append(1)  # Neutral
                    elif future_return >= strong_gain_threshold:
                        targets.append(2)  # Strong gain
                    elif future_return <= strong_loss_threshold:
                        targets.append(0)  # Strong loss
                    else:
                        targets.append(1)  # Neutral
                
                class_counts = pd.Series(targets).value_counts().sort_index()
                logger.info(f"📊 Multi-class for {symbol}: Loss={class_counts.get(0,0)}, Neutral={class_counts.get(1,0)}, Gain={class_counts.get(2,0)}")
        
        elif target_strategy == 'quantile_buckets':
            # Quantile buckets strategy for richer patterns
            if len(forward_returns) > 0:
                # Create 5 quantile buckets
                targets = []
                for i in range(len(df)):
                    if i >= len(df) - target_horizon:
                        targets.append(2)  # Middle bucket default
                        continue
                        
                    future_return = forward_returns.iloc[i] if i < len(forward_returns) else np.nan
                    
                    if pd.isna(future_return):
                        targets.append(2)  # Middle bucket
                    else:
                        # Assign to quantile bucket (0-4)
                        percentile = (forward_returns <= future_return).mean()
                        if percentile <= 0.2:
                            targets.append(0)  # Bottom quintile
                        elif percentile <= 0.4:
                            targets.append(1)  # Second quintile
                        elif percentile <= 0.6:
                            targets.append(2)  # Middle quintile
                        elif percentile <= 0.8:
                            targets.append(3)  # Fourth quintile
                        else:
                            targets.append(4)  # Top quintile
                
                bucket_counts = pd.Series(targets).value_counts().sort_index()
                logger.info(f"📊 Quantile buckets for {symbol}: {dict(bucket_counts)}")
        
        return pd.Series(targets, index=df.index)
    
    def _create_regression_target(self, df: pd.DataFrame, symbol: str, target_horizon: int = 3) -> pd.Series:
        """Create regression target based on daily returns normalized by holding period."""
        targets = []
        
        # Calculate forward returns (equivalent to pnl_ratio)
        forward_returns = (df['close'].shift(-target_horizon) / df['close']) - 1
        
        for i in range(len(df)):
            if i >= len(df) - target_horizon:
                # Not enough future data - use NaN for proper handling
                targets.append(np.nan)
                continue
                
            future_return = forward_returns.iloc[i]
            
            if not pd.isna(future_return):
                # Daily return: equivalent to pnl_ratio / holding_days
                # This is the normalized daily return target
                daily_return = future_return / target_horizon
                targets.append(daily_return)
            else:
                targets.append(np.nan)
        
        # Remove NaN values for statistics
        valid_targets = [t for t in targets if not pd.isna(t)]
        logger.info(f"📊 Created daily_return target for {symbol}: {len(valid_targets)}/{len(targets)} valid samples, "
                   f"mean={np.mean(valid_targets):.4f}, std={np.std(valid_targets):.4f}")
        return pd.Series(targets, index=df.index)
    
    def _create_binary_target(self, df: pd.DataFrame, symbol: str, target_horizon: int = 3) -> pd.Series:
        """Create binary classification target based on future returns threshold."""
        targets = []
        
        # Calculate forward returns for thresholding
        forward_returns = (df['close'].shift(-target_horizon) / df['close']) - 1
        
        # Define positive threshold (e.g., >2% return over target_horizon days)
        positive_threshold = 0.02 if target_horizon <= 3 else 0.03
        
        for i in range(len(df)):
            if i >= len(df) - target_horizon:
                # Not enough future data
                targets.append(0)
                continue
                
            future_return = forward_returns.iloc[i]
            
            # Simple threshold-based approach (faster than trade filter)
            if not pd.isna(future_return):
                # Positive if future return > threshold
                is_positive = future_return > positive_threshold
                targets.append(1 if is_positive else 0)
            else:
                targets.append(0)
        
        logger.info(f"📊 Created binary target for {symbol}: {sum(targets)}/{len(targets)} positives ({100*sum(targets)/len(targets):.1f}%)")
        return pd.Series(targets, index=df.index)
    
    def _add_daily_return_columns(self, df: pd.DataFrame, target_horizon: int = 3) -> pd.DataFrame:
        """
        Add pnl_ratio and daily_return columns as specified in the migration instructions.
        
        This method simulates the calculation that would be done with actual backtesting data:
        df['pnl_ratio'] = df['exit_pnl'] / df['capital_required']
        df['daily_return'] = df['pnl_ratio'] / df['holding_days']
        
        Args:
            df: DataFrame with price data
            target_horizon: Holding period in days (simulated holding_days)
            
        Returns:
            DataFrame with additional pnl_ratio and daily_return columns
        """
        # Calculate forward returns to simulate pnl_ratio
        df['pnl_ratio'] = (df['close'].shift(-target_horizon) / df['close']) - 1
        
        # Simulate holding_days (constant for this example)
        df['holding_days'] = target_horizon
        
        # Calculate daily_return as specified in the instructions
        df['daily_return'] = df['pnl_ratio'] / df['holding_days']
        
        logger.info(f"📊 Added daily_return columns: pnl_ratio mean={df['pnl_ratio'].mean():.4f}, "
                   f"daily_return mean={df['daily_return'].mean():.4f}")
        
        return df
    
    def engineer_features_for_symbol(self, symbol: str, days: int = 730, 
                                    use_enhanced_targets: bool = True,
                                    target_strategy: str = 'top_percentile') -> Optional[pd.DataFrame]:
        """
        Engineer complete feature set for a single symbol with enhanced targets and features.
        
        Args:
            symbol: Stock symbol
            days: Days of historical data (extended to 730 for 24-month lookback)
            use_enhanced_targets: Whether to use enhanced target strategies
            target_strategy: Target strategy ('top_percentile', 'multi_class', 'quantile_buckets')
            
        Returns:
            DataFrame with engineered features or None if failed
        """
        try:
            # Get extended historical data (24-month lookback)
            logger.info(f"🔄 Processing {symbol} with {days}-day extended lookback...")
            df = self.get_historical_data(symbol, days)
            
            if df is None or len(df) < 50:  # Need minimum data for technical indicators
                logger.warning(f"❌ Insufficient data for {symbol}")
                return None
            
            # Calculate enhanced technical indicators and features
            df = self.calculate_technical_indicators(df)
            logger.info(f"✅ Enhanced features calculated for {symbol}: {len(df.columns)} total features")
            
            # Create enhanced target variables with multiple horizons
            if use_enhanced_targets:
                target_results = self.create_target_variable(
                    df, symbol, 
                    use_regression=False,  # Use classification for generous labeling strategy
                    create_all_horizons=True,  # Create multiple horizons including 7, 14, 21 days
                    target_strategy=target_strategy
                )
                
                if isinstance(target_results, pd.DataFrame):
                    # Multiple targets created - use enhanced binary target with generous labeling
                    enhanced_target = 'target_3d_enhanced'  # Enhanced binary target with 25% positive rate
                    
                    if enhanced_target in target_results.columns:
                        df['target'] = target_results[enhanced_target]  # Use generous labeling as primary target
                        df['target_enhanced'] = target_results[enhanced_target]  # Backup reference
                        logger.info(f"✅ Using enhanced target with generous labeling as primary target for {symbol}")
                    else:
                        # Fallback to any enhanced target
                        enhanced_cols = [col for col in target_results.columns if 'enhanced' in col]
                        if enhanced_cols:
                            df['target'] = target_results[enhanced_cols[0]]
                            df['target_enhanced'] = target_results[enhanced_cols[0]]
                            logger.info(f"✅ Using fallback enhanced target {enhanced_cols[0]} for {symbol}")
                        else:
                            logger.warning(f"No enhanced columns found for {symbol}")
                            return None
                    
                    # Add all horizon targets for multi-horizon ensemble
                    for col in target_results.columns:
                        if col not in df.columns:
                            df[col] = target_results[col]
                
                else:
                    # Single target
                    df['target'] = target_results
            
            else:
                # Use generous labeling classification strategy
                df['target'] = self.create_target_variable(
                    df, symbol, 
                    use_regression=False,  # Use classification for generous labeling
                    create_all_horizons=False,  # Single target
                    target_strategy='top_percentile'  # 25% positive labeling
                )
            
            # Add daily return columns for compatibility
            df = self._add_daily_return_columns(df, target_horizon=3)
            
            # Add symbol identifier
            df['ticker'] = symbol
            
            # Select all feature columns (include all engineered features)
            exclude_cols = ['ticker', 'target', 'target_enhanced', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
            feature_columns = [col for col in df.columns if col not in exclude_cols]
            
            # Add back essential columns
            essential_cols = ['ticker', 'target']
            if 'target_enhanced' in df.columns:
                essential_cols.append('target_enhanced')
            
            # Filter to available columns and drop rows with NaN in key features
            all_columns = essential_cols + feature_columns
            available_columns = [col for col in all_columns if col in df.columns]
            result_df = df[available_columns].copy()
            
            # Keep only rows with non-null target and sufficient feature data
            result_df = result_df.dropna(subset=['target'])
            result_df = result_df.dropna(thresh=len(available_columns) * 0.7)  # At least 70% features non-null
            
            if len(result_df) == 0:
                logger.warning(f"❌ No valid rows after feature engineering for {symbol}")
                return None
            
            # Final data quality check
            valid_samples = len(result_df)
            total_features = len(feature_columns)
            
            logger.info(f"✅ Feature engineering complete for {symbol}:")
            logger.info(f"   📊 Valid samples: {valid_samples}")
            logger.info(f"   📋 Total features: {total_features}")
            
            # Log target statistics
            target_values = result_df['target'].dropna()
            if len(target_values) > 0:
                logger.info(f"   🎯 Target stats: mean={target_values.mean():.4f}, std={target_values.std():.4f}")
                
                # Check if enhanced target was created
                if 'target_enhanced' in result_df.columns:
                    enhanced_values = result_df['target_enhanced'].dropna()
                    if len(enhanced_values) > 0:
                        positive_rate = enhanced_values.mean() * 100
                        logger.info(f"   🎯 Enhanced target positive rate: {positive_rate:.1f}%")
            
            return result_df
            
        except Exception as e:
            logger.error(f"❌ Feature engineering failed for {symbol}: {e}")
            return None
    
    def collect_batch_data(self, batch_name: str, symbols: List[str], 
                         max_symbols: Optional[int] = None,
                         use_enhanced_targets: bool = True,
                         target_strategy: str = 'top_percentile',
                         days: int = 730) -> pd.DataFrame:
        """
        Collect and engineer features for a batch of symbols with enhanced targets.
        
        Args:
            batch_name: Name of the batch
            symbols: List of symbols in the batch
            max_symbols: Optional limit on number of symbols to process
            use_enhanced_targets: Whether to use enhanced target strategies
            target_strategy: Target strategy ('top_percentile', 'multi_class', 'quantile_buckets')
            days: Days of historical data (default 730 for 24-month lookback)
            
        Returns:
            Combined DataFrame with all symbols' data
        """
        logger.info(f"� Processing batch {batch_name} with {len(symbols)} symbols...")
        logger.info(f"   📊 Enhanced targets: {use_enhanced_targets} ({target_strategy})")
        logger.info(f"   📅 Lookback period: {days} days")
        
        if max_symbols:
            symbols = symbols[:max_symbols]
            logger.info(f"🔄 Limited to first {max_symbols} symbols")
        
        batch_dataframes = []
        successful_symbols = 0
        total_samples = 0
        enhanced_target_stats = []
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"🔄 Processing {symbol} ({i}/{len(symbols)})...")
            
            try:
                symbol_df = self.engineer_features_for_symbol(
                    symbol, 
                    days=days,
                    use_enhanced_targets=use_enhanced_targets,
                    target_strategy=target_strategy
                )
                
                if symbol_df is not None and len(symbol_df) > 0:
                    batch_dataframes.append(symbol_df)
                    successful_symbols += 1
                    total_samples += len(symbol_df)
                    
                    # Track enhanced target statistics
                    if 'target_enhanced' in symbol_df.columns:
                        pos_rate = symbol_df['target_enhanced'].mean() * 100
                        enhanced_target_stats.append({
                            'symbol': symbol,
                            'samples': len(symbol_df),
                            'positive_rate': pos_rate
                        })
                    
                    logger.info(f"✅ {symbol}: {len(symbol_df)} samples collected")
                else:
                    logger.warning(f"❌ {symbol}: No data collected")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
        
        if not batch_dataframes:
            logger.error(f"❌ No data collected for batch {batch_name}")
            return pd.DataFrame()
        
        # Combine all symbol data
        combined_df = pd.concat(batch_dataframes, ignore_index=True)
        
        # Log comprehensive batch statistics
        logger.info(f"🎉 Batch {batch_name} collection complete:")
        logger.info(f"   ✅ Successful symbols: {successful_symbols}/{len(symbols)}")
        logger.info(f"   📊 Total samples: {total_samples:,}")
        logger.info(f"   📋 Features per sample: {len(combined_df.columns) - 2}")  # Exclude ticker and target
        
        # Enhanced target statistics
        if enhanced_target_stats:
            avg_pos_rate = np.mean([s['positive_rate'] for s in enhanced_target_stats])
            logger.info(f"   🎯 Average enhanced target positive rate: {avg_pos_rate:.1f}%")
            
            # Show distribution of positive rates
            pos_rates = [s['positive_rate'] for s in enhanced_target_stats]
            logger.info(f"   📈 Enhanced target rate range: {min(pos_rates):.1f}% - {max(pos_rates):.1f}%")
        
        # Overall target statistics
        if 'target' in combined_df.columns:
            target_stats = combined_df['target'].describe()
            logger.info(f"   📊 Target statistics:")
            logger.info(f"      Mean: {target_stats['mean']:.4f}")
            logger.info(f"      Std:  {target_stats['std']:.4f}")
            logger.info(f"      Range: {target_stats['min']:.4f} to {target_stats['max']:.4f}")
        
        return combined_df
    
    def collect_training_data(self, batch_numbers: List[int], 
                            max_symbols_per_batch: Optional[int] = 10,
                            use_enhanced_targets: bool = True,
                            target_strategy: str = 'top_percentile',
                            days: int = 730) -> pd.DataFrame:
        """
        Collect training data for specified batches with enhanced targets.
        
        Args:
            batch_numbers: List of batch numbers to process
            max_symbols_per_batch: Optional limit on symbols per batch
            use_enhanced_targets: Whether to use enhanced target strategies
            target_strategy: Target strategy ('top_percentile', 'multi_class', 'quantile_buckets')
            days: Days of historical data (default 730 for 24-month lookback)
            
        Returns:
            Combined DataFrame ready for training
        """
        logger.info(f"🚀 Starting enhanced data collection for batches: {batch_numbers}")
        logger.info(f"   📊 Enhanced targets: {use_enhanced_targets} ({target_strategy})")
        logger.info(f"   📅 Lookback period: {days} days")
        
        # Load stock batches
        batches = self.load_stock_batches()
        
        all_dataframes = []
        total_symbols_processed = 0
        
        for batch_num in batch_numbers:
            batch_name = f"batch_{batch_num}"
            
            if batch_name not in batches:
                logger.warning(f"⚠️  Batch {batch_name} not found in batch file")
                continue
            
            symbols = batches[batch_name]
            logger.info(f"🔄 Processing {batch_name} with {len(symbols)} symbols...")
            
            batch_df = self.collect_batch_data(
                batch_name, symbols, 
                max_symbols=max_symbols_per_batch,
                use_enhanced_targets=use_enhanced_targets,
                target_strategy=target_strategy,
                days=days
            )
            
            if len(batch_df) > 0:
                all_dataframes.append(batch_df)
                total_symbols_processed += batch_df['ticker'].nunique()
                logger.info(f"✅ {batch_name}: {len(batch_df)} samples from {batch_df['ticker'].nunique()} symbols")
            else:
                logger.warning(f"❌ {batch_name}: No data collected")
        
        if not all_dataframes:
            logger.error("❌ No training data collected")
            return pd.DataFrame()
        
        # Combine all batch data
        final_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Final cleanup - use newer pandas syntax
        final_df = final_df.bfill().ffill()
        final_df = final_df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Log final statistics
        logger.info(f"🎉 Enhanced data collection complete:")
        logger.info(f"   📊 Total samples: {len(final_df):,}")
        logger.info(f"   🏢 Total symbols: {final_df['ticker'].nunique()}")
        logger.info(f"   📋 Total features: {len(final_df.columns) - 2}")  # Exclude ticker and target
        
        # Enhanced target statistics
        if use_enhanced_targets and 'target_enhanced' in final_df.columns:
            enhanced_pos_rate = final_df['target_enhanced'].mean() * 100
            logger.info(f"   🎯 Enhanced target positive rate: {enhanced_pos_rate:.1f}%")
        
        # Standard target statistics
        if 'target' in final_df.columns:
            target_stats = final_df['target'].describe()
            logger.info(f"   📈 Target statistics:")
            logger.info(f"      Mean: {target_stats['mean']:.4f}")
            logger.info(f"      Std:  {target_stats['std']:.4f}")
            logger.info(f"      Range: {target_stats['min']:.4f} to {target_stats['max']:.4f}")
        
        return final_df
        
        # Print summary
        target_dist = final_df['target'].value_counts().to_dict()
        unique_tickers = final_df['ticker'].nunique()
        
        logger.info(f"🎉 Data collection complete!")
        logger.info(f"   📊 Total samples: {len(final_df)}")
        logger.info(f"   🏢 Unique tickers: {unique_tickers}")
        logger.info(f"   🎯 Target distribution: {target_dist}")
        logger.info(f"   📋 Features: {len([col for col in final_df.columns if col not in ['ticker', 'target']])}")
        
        return final_df


def main():
    """
    Command-line interface for the AlpacaDataCollector.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect training data from Alpaca API')
    parser.add_argument('--batches', type=str, help='Comma-separated batch numbers (e.g., "1,2,3") or single batch number')
    parser.add_argument('--max-symbols', type=int, default=50, help='Maximum symbols per batch')
    parser.add_argument('--output-file', type=str, default='alpaca_training_data.csv', help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Parse batch numbers
    if args.batches:
        try:
            if ',' in args.batches:
                batch_numbers = [int(b.strip()) for b in args.batches.split(',')]
            else:
                batch_numbers = [int(args.batches)]
        except ValueError:
            logger.error("❌ Invalid batch numbers format. Use comma-separated integers (e.g., '1,2,3')")
            return
    else:
        batch_numbers = [1, 2]  # Default for backward compatibility
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize collector
        collector = AlpacaDataCollector()
        
        # Collect data for specified batches
        training_data = collector.collect_training_data(
            batch_numbers=batch_numbers,
            max_symbols_per_batch=args.max_symbols
        )
        
        if len(training_data) > 0:
            # Save to specified output file
            training_data.to_csv(args.output_file, index=False)
            logger.info(f"💾 Training data saved to {args.output_file}")
            
            # Print sample data
            print(f"\n📋 Sample of collected data:")
            print(training_data.head())
            print(f"\n📊 Data shape: {training_data.shape}")
            print(f"🎯 Target distribution: {training_data['target'].value_counts().to_dict()}")
            
        else:
            logger.error("❌ No training data collected")
    
    except Exception as e:
        logger.error(f"❌ Data collection failed: {e}")


if __name__ == "__main__":
    main()
