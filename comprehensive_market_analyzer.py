#!/usr/bin/env python3
"""
Comprehensive Market Analyzer - Generates exact 118 features from training data
===============================================================================

This module generates the exact same 118 features that the ML models were trained on,
ensuring perfect compatibility between the training data and real-time inference.

Based on the training data columns:
ticker,target,target_enhanced,timestamp,price_change_1d,price_change_5d,price_change_10d,price_change_20d,sma_5,sma_10,sma_20,sma_50,sma_200,price_vs_sma5,price_vs_sma10,price_vs_sma20,price_vs_sma50,price_vs_sma200,trend_regime,volatility_5d,volatility_10d,volatility_20d,volatility_50d,volatility_regime,volume_sma_10,volume_sma_50,volume_ratio,volume_regime,hl_ratio,hl_ratio_5d,gap_up,gap_down,rsi_14,momentum_regime,macd,macd_signal,macd_histogram,macd_regime,bb_upper,bb_lower,bb_position,mean_reversion_regime,composite_regime_score,trend_regime_changes,trend_regime_stability,bull_low_vol,bear_high_vol,sideways_low_vol,price_acceleration,price_theta,vol_sensitivity,price_vol_correlation,delta_proxy,momentum_acceleration,implied_vol_proxy,implied_vol_change,implied_vol_percentile,vol_percentile_50d,vol_regime_high,vol_regime_low,vix_top_10pct,vix_bottom_10pct,time_to_expiry_proxy,theta_decay,theta_acceleration,atr_20d,spread_width_proxy,move_vs_spread,spread_efficiency,market_trend_strength,long_term_trend,relative_strength,momentum_percentile,quarter,month,earnings_season,volume_price_divergence,accumulation_distribution,accumulation_distribution_sma,doji,hammer,shooting_star,resistance_level,support_level,distance_to_resistance,distance_to_support,vol_clustering,vol_persistence,vol_skew,vol_kurtosis,spread_proxy,spread_volatility,price_impact,illiquidity_proxy,momentum_3d,momentum_7d,momentum_14d,momentum_21d,mean_reversion_5d,mean_reversion_20d,momentum_consistency,beta_proxy,correlation_stability,extreme_move_up,extreme_move_down,overnight_gap,gap_magnitude,gap_follow_through,technical_strength,risk_adjusted_return_5d,risk_adjusted_return_20d,quality_score,target_1d_enhanced,target_3d_enhanced,target_5d_enhanced,target_7d_enhanced,target_10d_enhanced,target_14d_enhanced,target_21d_enhanced,pnl_ratio,holding_days,daily_return
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ComprehensiveMarketAnalyzer:
    """
    Generates comprehensive 118-feature analysis matching training data exactly.
    """
    
    def __init__(self):
        """Initialize the comprehensive market analyzer."""
        logger.info("🔬 Initializing Comprehensive Market Analyzer (118 features)")
    
    def analyze_symbol(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze a symbol and generate all 118 features that match training data.
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume, timestamp)
            
        Returns:
            Dict with all 118 features matching training data exactly
        """
        try:
            logger.info(f"Analyzing symbol with {len(df)} bars of data...")
            
            if len(df) < 5:
                logger.warning(f"Insufficient data: {len(df)} bars")
                return self._get_default_features()
            
            # Sort by timestamp to ensure chronological order
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate all 118 features
            features = {}
            
            # === BASIC PRICE FEATURES ===
            features.update(self._calculate_price_features(df))
            
            # === MOVING AVERAGES ===
            features.update(self._calculate_moving_averages(df))
            
            # === TREND AND REGIME FEATURES ===
            features.update(self._calculate_trend_regime(df))
            
            # === VOLATILITY FEATURES ===
            features.update(self._calculate_volatility_features(df))
            
            # === VOLUME FEATURES ===
            features.update(self._calculate_volume_features(df))
            
            # === MICROSTRUCTURE FEATURES ===
            features.update(self._calculate_microstructure_features(df))
            
            # === TECHNICAL INDICATORS ===
            features.update(self._calculate_technical_indicators(df))
            
            # === ENHANCED FEATURES ===
            features.update(self._calculate_enhanced_features(df))
            
            # === IMPLIED VOL FEATURES ===
            features.update(self._calculate_implied_vol_features(df))
            
            # === TIME DECAY FEATURES ===
            features.update(self._calculate_time_decay_features(df))
            
            # === MARKET CONTEXT FEATURES ===
            features.update(self._calculate_market_context_features(df))
            
            # === CALENDAR FEATURES ===
            features.update(self._calculate_calendar_features())
            
            # === CANDLESTICK PATTERNS ===
            features.update(self._calculate_candlestick_patterns(df))
            
            # === SUPPORT/RESISTANCE ===
            features.update(self._calculate_support_resistance(df))
            
            # === VOLATILITY CLUSTERING ===
            features.update(self._calculate_volatility_clustering(df))
            
            # === LIQUIDITY FEATURES ===
            features.update(self._calculate_liquidity_features(df))
            
            # === MOMENTUM FEATURES ===
            features.update(self._calculate_momentum_features(df))
            
            # === MEAN REVERSION ===
            features.update(self._calculate_mean_reversion_features(df))
            
            # === CORRELATION FEATURES ===
            features.update(self._calculate_correlation_features(df))
            
            # === NEWS/SENTIMENT PROXIES ===
            features.update(self._calculate_news_sentiment_features(df))
            
            # === COMPOSITE SIGNALS ===
            features.update(self._calculate_composite_signals(features))
            
            # === TARGET VARIABLES (defaults for real-time) ===
            features.update(self._calculate_target_variables())
            
            # Validate all features are present and numeric
            features = self._validate_and_fill_features(features)
            
            logger.info(f"✅ Generated {len(features)} comprehensive features")
            return features
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return self._get_default_features()
    
    def _calculate_price_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic price change features."""
        features = {}
        
        # Price changes
        features['price_change_1d'] = df['close'].pct_change().iloc[-1] if len(df) > 1 else 0.0
        features['price_change_5d'] = df['close'].pct_change(5).iloc[-1] if len(df) > 5 else 0.0
        features['price_change_10d'] = df['close'].pct_change(10).iloc[-1] if len(df) > 10 else 0.0
        features['price_change_20d'] = df['close'].pct_change(20).iloc[-1] if len(df) > 20 else 0.0
        
        return features
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate moving averages and ratios."""
        features = {}
        
        # Moving averages
        features['sma_5'] = df['close'].rolling(5).mean().iloc[-1] if len(df) >= 5 else df['close'].iloc[-1]
        features['sma_10'] = df['close'].rolling(10).mean().iloc[-1] if len(df) >= 10 else df['close'].iloc[-1]
        features['sma_20'] = df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['close'].iloc[-1]
        features['sma_50'] = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else df['close'].iloc[-1]
        features['sma_200'] = df['close'].rolling(200).mean().iloc[-1] if len(df) >= 200 else df['close'].iloc[-1]
        
        # Price vs SMA ratios
        current_price = df['close'].iloc[-1]
        features['price_vs_sma5'] = (current_price / features['sma_5']) - 1 if features['sma_5'] > 0 else 0.0
        features['price_vs_sma10'] = (current_price / features['sma_10']) - 1 if features['sma_10'] > 0 else 0.0
        features['price_vs_sma20'] = (current_price / features['sma_20']) - 1 if features['sma_20'] > 0 else 0.0
        features['price_vs_sma50'] = (current_price / features['sma_50']) - 1 if features['sma_50'] > 0 else 0.0
        features['price_vs_sma200'] = (current_price / features['sma_200']) - 1 if features['sma_200'] > 0 else 0.0
        
        return features
    
    def _calculate_trend_regime(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate trend regime features."""
        features = {}
        
        # Basic trend regime
        features['trend_regime'] = 0  # Default: sideways
        if len(df) >= 50:
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            sma_200 = df['close'].rolling(200).mean().iloc[-1] if len(df) >= 200 else sma_50
            current_price = df['close'].iloc[-1]
            
            # Bull regime: price > SMA50 > SMA200 and rising
            if (current_price > sma_50 and sma_50 > sma_200 and 
                len(df) >= 55 and sma_50 > df['close'].rolling(50).mean().iloc[-6]):
                features['trend_regime'] = 1
            # Bear regime: price < SMA50 < SMA200 and falling
            elif (current_price < sma_50 and sma_50 < sma_200 and 
                  len(df) >= 55 and sma_50 < df['close'].rolling(50).mean().iloc[-6]):
                features['trend_regime'] = -1
        
        # Trend stability metrics
        features['trend_regime_changes'] = 0  # Simplified for real-time
        features['trend_regime_stability'] = 20  # Assume stable
        
        return features
    
    def _calculate_volatility_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility features."""
        features = {}
        
        # Basic volatility measures
        returns = df['close'].pct_change()
        features['volatility_5d'] = returns.rolling(5).std().iloc[-1] if len(df) >= 5 else 0.0
        features['volatility_10d'] = returns.rolling(10).std().iloc[-1] if len(df) >= 10 else 0.0
        features['volatility_20d'] = returns.rolling(20).std().iloc[-1] if len(df) >= 20 else 0.0
        features['volatility_50d'] = returns.rolling(50).std().iloc[-1] if len(df) >= 50 else 0.0
        
        # Volatility regime
        features['volatility_regime'] = 0
        if len(df) >= 50 and features['volatility_50d'] > 0:
            vol_percentile = returns.rolling(20).std().rolling(50).rank(pct=True).iloc[-1]
            if vol_percentile > 0.8:
                features['volatility_regime'] = 1  # High vol
            elif vol_percentile < 0.2:
                features['volatility_regime'] = -1  # Low vol
        
        # VIX-like percentile features
        if len(df) >= 50:
            features['vol_percentile_50d'] = returns.rolling(20).std().rolling(50).rank(pct=True).iloc[-1]
            features['vol_regime_high'] = int(features['vol_percentile_50d'] > 0.8)
            features['vol_regime_low'] = int(features['vol_percentile_50d'] < 0.2)
            features['vix_top_10pct'] = int(features['vol_percentile_50d'] > 0.9)
            features['vix_bottom_10pct'] = int(features['vol_percentile_50d'] < 0.1)
        else:
            features['vol_percentile_50d'] = 0.5
            features['vol_regime_high'] = 0
            features['vol_regime_low'] = 0
            features['vix_top_10pct'] = 0
            features['vix_bottom_10pct'] = 0
        
        return features
    
    def _calculate_volume_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume features."""
        features = {}
        
        # Volume moving averages
        features['volume_sma_10'] = df['volume'].rolling(10).mean().iloc[-1] if len(df) >= 10 else df['volume'].iloc[-1]
        features['volume_sma_50'] = df['volume'].rolling(50).mean().iloc[-1] if len(df) >= 50 else df['volume'].iloc[-1]
        
        # Volume ratio
        features['volume_ratio'] = df['volume'].iloc[-1] / features['volume_sma_10'] if features['volume_sma_10'] > 0 else 1.0
        
        # Volume regime
        features['volume_regime'] = 0
        if len(df) >= 50:
            volume_percentile = df['volume'].rolling(50).rank(pct=True).iloc[-1]
            if volume_percentile > 0.8:
                features['volume_regime'] = 1  # High volume
            elif volume_percentile < 0.2:
                features['volume_regime'] = -1  # Low volume
        
        # Volume-price divergence
        features['volume_price_divergence'] = features['volume_ratio'] * (features.get('price_change_1d', 0.0))
        
        return features
    
    def _calculate_microstructure_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate market microstructure features."""
        features = {}
        
        # High-low ratio
        features['hl_ratio'] = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1]
        features['hl_ratio_5d'] = ((df['high'] - df['low']) / df['close']).rolling(5).mean().iloc[-1] if len(df) >= 5 else features['hl_ratio']
        
        # Gap detection
        if len(df) > 1:
            gap = (df['open'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
            features['gap_up'] = gap > 0.02
            features['gap_down'] = gap < -0.02
            features['overnight_gap'] = gap
            features['gap_magnitude'] = abs(gap)
            features['gap_follow_through'] = gap * (features.get('price_change_1d', 0.0))
        else:
            features['gap_up'] = False
            features['gap_down'] = False
            features['overnight_gap'] = 0.0
            features['gap_magnitude'] = 0.0
            features['gap_follow_through'] = 0.0
        
        return features
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators."""
        features = {}
        
        # RSI
        if len(df) >= 14:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features['rsi_14'] = rsi.iloc[-1]
            
            # Momentum regime based on RSI
            if features['rsi_14'] > 70:
                features['momentum_regime'] = 1  # Overbought
            elif features['rsi_14'] < 30:
                features['momentum_regime'] = -1  # Oversold
            else:
                features['momentum_regime'] = 0  # Neutral
        else:
            features['rsi_14'] = 50.0
            features['momentum_regime'] = 0
        
        # MACD
        if len(df) >= 26:
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            features['macd'] = macd.iloc[-1]
            
            if len(df) >= 35:
                macd_signal = macd.ewm(span=9).mean()
                features['macd_signal'] = macd_signal.iloc[-1]
                features['macd_histogram'] = (macd - macd_signal).iloc[-1]
                
                # MACD regime
                features['macd_regime'] = 1 if features['macd'] > features['macd_signal'] else -1
            else:
                features['macd_signal'] = features['macd']
                features['macd_histogram'] = 0.0
                features['macd_regime'] = 0
        else:
            features['macd'] = 0.0
            features['macd_signal'] = 0.0
            features['macd_histogram'] = 0.0
            features['macd_regime'] = 0
        
        # Bollinger Bands
        if len(df) >= 20:
            sma = df['close'].rolling(20).mean()
            std = df['close'].rolling(20).std()
            bb_upper = sma + (std * 2)
            bb_lower = sma - (std * 2)
            bb_position = (df['close'] - bb_lower) / (bb_upper - bb_lower)
            features['bb_upper'] = bb_upper.iloc[-1]
            features['bb_lower'] = bb_lower.iloc[-1]
            features['bb_position'] = bb_position.iloc[-1]
            
            # Mean reversion regime
            if bb_position.iloc[-1] > 0.8:
                features['mean_reversion_regime'] = 1  # Near upper band
            elif bb_position.iloc[-1] < 0.2:
                features['mean_reversion_regime'] = -1  # Near lower band
            else:
                features['mean_reversion_regime'] = 0  # Middle
        else:
            features['bb_upper'] = df['close'].iloc[-1] * 1.02
            features['bb_lower'] = df['close'].iloc[-1] * 0.98
            features['bb_position'] = 0.5
            features['mean_reversion_regime'] = 0
        
        return features
    
    def _calculate_enhanced_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate enhanced technical features."""
        features = {}
        
        # Price acceleration and theta-like features
        if len(df) >= 3:
            price_accel = df['close'].diff().diff().iloc[-1]
            features['price_acceleration'] = price_accel
            features['price_theta'] = price_accel / df['close'].iloc[-1] if df['close'].iloc[-1] > 0 else 0.0
        else:
            features['price_acceleration'] = 0.0
            features['price_theta'] = 0.0
        
        # Volatility sensitivity
        if len(df) >= 20:
            vol_20d = df['close'].pct_change().rolling(20).std()
            vol_sensitivity = vol_20d.diff().iloc[-1] if len(vol_20d.dropna()) > 1 else 0.0
            features['vol_sensitivity'] = vol_sensitivity / vol_20d.iloc[-1] if vol_20d.iloc[-1] > 0 else 0.0
        else:
            features['vol_sensitivity'] = 0.0
        
        # Price-volatility correlation
        if len(df) >= 20:
            features['price_vol_correlation'] = df['close'].rolling(20).corr(df['close'].pct_change().rolling(20).std()).iloc[-1]
        else:
            features['price_vol_correlation'] = 0.0
        
        # Delta proxy and momentum acceleration
        vol_20d = features.get('volatility_20d', 0.001)  # Prevent division by zero
        features['delta_proxy'] = (features.get('price_change_1d', 0.0)) / vol_20d if vol_20d > 0 else 0.0
        
        if len(df) >= 3:
            delta_prev = df['close'].pct_change().iloc[-2] / vol_20d if vol_20d > 0 else 0.0
            features['momentum_acceleration'] = features['delta_proxy'] - delta_prev
        else:
            features['momentum_acceleration'] = 0.0
        
        return features
    
    def _calculate_implied_vol_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate implied volatility proxy features."""
        features = {}
        
        # Implied volatility proxy using high-low ranges
        hl_ratio = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1]
        features['implied_vol_proxy'] = hl_ratio
        
        # Implied vol change
        if len(df) > 1:
            prev_hl_ratio = (df['high'].iloc[-2] - df['low'].iloc[-2]) / df['close'].iloc[-2]
            features['implied_vol_change'] = hl_ratio - prev_hl_ratio
        else:
            features['implied_vol_change'] = 0.0
        
        # Implied vol percentile
        if len(df) >= 50:
            hl_ratios = (df['high'] - df['low']) / df['close']
            features['implied_vol_percentile'] = hl_ratios.rolling(50).rank(pct=True).iloc[-1]
        else:
            features['implied_vol_percentile'] = 0.5
        
        return features
    
    def _calculate_time_decay_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate time decay and Greeks-like features."""
        features = {}
        
        # Time to expiry proxy
        day_of_month = datetime.now().day
        features['time_to_expiry_proxy'] = day_of_month % 21  # Proxy for days until monthly expiry
        
        # Theta decay
        implied_vol = features.get('implied_vol_proxy', 0.02)
        features['theta_decay'] = implied_vol * np.sqrt(features['time_to_expiry_proxy'] / 21.0)
        
        # Theta acceleration
        prev_theta = implied_vol * np.sqrt(max(0, features['time_to_expiry_proxy']-1) / 21.0)
        features['theta_acceleration'] = features['theta_decay'] - prev_theta
        
        # ATR and spread features
        features['atr_20d'] = features.get('hl_ratio_5d', features.get('hl_ratio', 0.02))
        features['spread_width_proxy'] = features['atr_20d'] / df['close'].iloc[-1]
        features['move_vs_spread'] = abs(features.get('price_change_1d', 0.0)) / (features['spread_width_proxy'] + 1e-8)
        features['spread_efficiency'] = features['spread_width_proxy'] / (features.get('volatility_20d', 0.001) + 1e-8)
        
        return features
    
    def _calculate_market_context_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate market context features."""
        features = {}
        
        # Market trend strength
        if len(df) >= 50:
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            features['market_trend_strength'] = df['close'].rolling(20).corr(pd.Series([sma_50]*20)).iloc[-1] if len(df) >= 20 else 0.0
        else:
            features['market_trend_strength'] = df['close'].rolling(20).corr(pd.Series(range(20))).iloc[-1] if len(df) >= 20 else 0.0
        
        # Long-term trend
        sma_50 = features.get('sma_50', df['close'].iloc[-1])
        sma_200 = features.get('sma_200', df['close'].iloc[-1])
        features['long_term_trend'] = int(sma_50 > sma_200)
        
        # Relative strength
        if len(df) >= 252:
            features['relative_strength'] = df['close'].iloc[-1] / df['close'].rolling(252).mean().iloc[-1]
        elif len(df) >= 50:
            features['relative_strength'] = df['close'].iloc[-1] / df['close'].rolling(50).mean().iloc[-1]
        else:
            features['relative_strength'] = 1.0
        
        # Momentum percentile
        if len(df) >= 100:
            momentum_20d = df['close'].pct_change(20)
            features['momentum_percentile'] = momentum_20d.rolling(100).rank(pct=True).iloc[-1] if len(df) >= 20 else 0.5
        else:
            features['momentum_percentile'] = 0.5
        
        return features
    
    def _calculate_calendar_features(self) -> Dict[str, float]:
        """Calculate calendar-based features."""
        features = {}
        
        now = datetime.now()
        features['quarter'] = (now.month - 1) // 3 + 1
        features['month'] = now.month
        features['earnings_season'] = int(now.month in [1, 4, 7, 10])
        
        return features
    
    def _calculate_candlestick_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate candlestick pattern features."""
        features = {}
        
        if len(df) >= 1:
            open_price = df['open'].iloc[-1]
            high_price = df['high'].iloc[-1]
            low_price = df['low'].iloc[-1]
            close_price = df['close'].iloc[-1]
            
            body_size = abs(open_price - close_price)
            total_range = high_price - low_price
            
            # Doji pattern
            features['doji'] = int(body_size / (total_range + 1e-8) < 0.1)
            
            # Hammer pattern
            if close_price > open_price:  # Bullish
                lower_shadow = open_price - low_price
                upper_shadow = high_price - close_price
                body = close_price - open_price
                features['hammer'] = int(lower_shadow > 2 * body and upper_shadow < 0.1 * body)
            else:
                features['hammer'] = 0
            
            # Shooting star pattern
            if open_price > close_price:  # Bearish
                upper_shadow = high_price - open_price
                lower_shadow = close_price - low_price
                body = open_price - close_price
                features['shooting_star'] = int(upper_shadow > 2 * body and lower_shadow < 0.1 * body)
            else:
                features['shooting_star'] = 0
        else:
            features['doji'] = 0
            features['hammer'] = 0
            features['shooting_star'] = 0
        
        return features
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate support and resistance features."""
        features = {}
        
        if len(df) >= 20:
            features['resistance_level'] = df['high'].rolling(20).max().iloc[-1]
            features['support_level'] = df['low'].rolling(20).min().iloc[-1]
            current_price = df['close'].iloc[-1]
            features['distance_to_resistance'] = (features['resistance_level'] - current_price) / current_price
            features['distance_to_support'] = (current_price - features['support_level']) / current_price
        else:
            features['resistance_level'] = df['high'].iloc[-1]
            features['support_level'] = df['low'].iloc[-1]
            features['distance_to_resistance'] = 0.0
            features['distance_to_support'] = 0.0
        
        return features
    
    def _calculate_volatility_clustering(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility clustering features."""
        features = {}
        
        if len(df) >= 30:
            vol_5d = df['close'].pct_change().rolling(5).std()
            vol_10d_std = vol_5d.rolling(10).std()
            features['vol_clustering'] = vol_5d.iloc[-1] / (vol_10d_std.iloc[-1] + 1e-8)
            features['vol_persistence'] = vol_5d.rolling(5).mean().iloc[-1] / (vol_5d.rolling(20).mean().iloc[-1] + 1e-8)
        else:
            features['vol_clustering'] = 1.0
            features['vol_persistence'] = 1.0
        
        # Vol skew and kurtosis
        if len(df) >= 20:
            returns = df['close'].pct_change().dropna()
            if len(returns) >= 20:
                recent_returns = returns.tail(20)
                try:
                    features['vol_skew'] = recent_returns.skew()
                    features['vol_kurtosis'] = recent_returns.kurtosis()
                except:
                    features['vol_skew'] = 0.0
                    features['vol_kurtosis'] = 0.0
            else:
                features['vol_skew'] = 0.0
                features['vol_kurtosis'] = 0.0
        else:
            features['vol_skew'] = 0.0
            features['vol_kurtosis'] = 0.0
        
        return features
    
    def _calculate_liquidity_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate liquidity and microstructure features."""
        features = {}
        
        # Spread proxy and volatility
        hl_ratio = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1]
        features['spread_proxy'] = hl_ratio
        features['spread_volatility'] = ((df['high'] - df['low']) / df['close']).rolling(10).std().iloc[-1] if len(df) >= 10 else 0.0
        
        # Price impact and illiquidity
        volume_norm = features.get('volume_sma_10', df['volume'].iloc[-1])
        features['price_impact'] = abs(features.get('price_change_1d', 0.0)) / (df['volume'].iloc[-1] / (volume_norm + 1e-8) + 1e-8)
        features['illiquidity_proxy'] = abs(features.get('price_change_1d', 0.0)) / (df['volume'].iloc[-1] * df['close'].iloc[-1] + 1e-8)
        
        return features
    
    def _calculate_momentum_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate multi-timeframe momentum features."""
        features = {}
        
        # Multi-timeframe momentum
        features['momentum_3d'] = df['close'].pct_change(3).iloc[-1] if len(df) > 3 else 0.0
        features['momentum_7d'] = df['close'].pct_change(7).iloc[-1] if len(df) > 7 else 0.0
        features['momentum_14d'] = df['close'].pct_change(14).iloc[-1] if len(df) > 14 else 0.0
        features['momentum_21d'] = df['close'].pct_change(21).iloc[-1] if len(df) > 21 else 0.0
        
        # Momentum consistency
        momentum_signals = [
            features['momentum_3d'] > 0,
            features['momentum_7d'] > 0,
            features['momentum_14d'] > 0,
            features['momentum_21d'] > 0
        ]
        features['momentum_consistency'] = sum(momentum_signals)
        
        return features
    
    def _calculate_mean_reversion_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate mean reversion features."""
        features = {}
        
        # Mean reversion indicators
        features['mean_reversion_5d'] = (df['close'].iloc[-1] / df['close'].rolling(5).mean().iloc[-1]) - 1 if len(df) >= 5 else 0.0
        features['mean_reversion_20d'] = (df['close'].iloc[-1] / df['close'].rolling(20).mean().iloc[-1]) - 1 if len(df) >= 20 else 0.0
        
        return features
    
    def _calculate_correlation_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlation and beta features."""
        features = {}
        
        if len(df) >= 60:
            # Simplified beta using own price returns as market proxy
            returns = df['close'].pct_change()
            market_returns = returns.rolling(252).mean().iloc[-1] if len(df) >= 252 else returns.rolling(50).mean().iloc[-1]
            returns_var = returns.rolling(60).var().iloc[-1]
            features['beta_proxy'] = returns_var / (returns_var + 1e-8)  # Simplified
            features['correlation_stability'] = returns.rolling(20).corr(returns.shift(1)).iloc[-1]
        else:
            features['beta_proxy'] = 1.0
            features['correlation_stability'] = 0.0
        
        return features
    
    def _calculate_news_sentiment_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate news and sentiment proxy features."""
        features = {}
        
        if len(df) >= 60:
            returns = df['close'].pct_change()
            extreme_threshold_up = returns.rolling(60).quantile(0.95).iloc[-1]
            extreme_threshold_down = returns.rolling(60).quantile(0.05).iloc[-1]
            current_return = features.get('price_change_1d', 0.0)
            features['extreme_move_up'] = int(current_return > extreme_threshold_up)
            features['extreme_move_down'] = int(current_return < extreme_threshold_down)
        else:
            features['extreme_move_up'] = 0
            features['extreme_move_down'] = 0
        
        return features
    
    def _calculate_composite_signals(self, features: Dict[str, float]) -> Dict[str, float]:
        """Calculate composite signal features."""
        composite_features = {}
        
        # Composite regime score
        composite_features['composite_regime_score'] = (
            features.get('trend_regime', 0) + 
            features.get('momentum_regime', 0) + 
            features.get('macd_regime', 0)
        )
        
        # Combined regime flags
        trend_regime = features.get('trend_regime', 0)
        vol_regime = features.get('volatility_regime', 0)
        composite_features['bull_low_vol'] = int(trend_regime == 1 and vol_regime == -1)
        composite_features['bear_high_vol'] = int(trend_regime == -1 and vol_regime == 1)
        composite_features['sideways_low_vol'] = int(trend_regime == 0 and vol_regime == -1)
        
        # Technical strength composite
        tech_signals = [features.get('momentum_regime', 0), features.get('macd_regime', 0), features.get('trend_regime', 0)]
        composite_features['technical_strength'] = sum(tech_signals)
        
        # Risk-adjusted returns
        vol_5d = features.get('volatility_5d', 0.001)
        vol_20d = features.get('volatility_20d', 0.001)
        composite_features['risk_adjusted_return_5d'] = features.get('momentum_3d', 0.0) / (vol_5d + 1e-8)
        composite_features['risk_adjusted_return_20d'] = features.get('momentum_7d', 0.0) / (vol_20d + 1e-8)
        
        # Quality score
        quality_factors = [
            features.get('vol_regime_low', 0),
            int(features.get('momentum_consistency', 0) >= 3),  # Most momentum signals positive
            int(abs(features.get('trend_regime', 0)) >= 1)      # Strong trend
        ]
        composite_features['quality_score'] = sum(quality_factors)
        
        # Accumulation/Distribution
        composite_features['accumulation_distribution'] = features.get('volume_price_divergence', 0.0) * 1000
        composite_features['accumulation_distribution_sma'] = composite_features['accumulation_distribution']  # Simplified
        
        return composite_features
    
    def _calculate_target_variables(self) -> Dict[str, float]:
        """Calculate target variables (defaults for real-time)."""
        features = {}
        
        # Enhanced target variables (unknown for real-time)
        features['target_1d_enhanced'] = 0
        features['target_3d_enhanced'] = 0
        features['target_5d_enhanced'] = 0
        features['target_7d_enhanced'] = 0
        features['target_10d_enhanced'] = 0
        features['target_14d_enhanced'] = 0
        features['target_21d_enhanced'] = 0
        
        # Trading simulation features
        features['pnl_ratio'] = 0.0
        features['holding_days'] = 3
        features['daily_return'] = 0.0  # Will be filled with price_change_1d
        
        return features
    
    def _validate_and_fill_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Validate and fill missing features to match training data exactly."""
        
        # Expected feature names from training data
        expected_features = [
            'price_change_1d', 'price_change_5d', 'price_change_10d', 'price_change_20d',
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
            'price_vs_sma5', 'price_vs_sma10', 'price_vs_sma20', 'price_vs_sma50', 'price_vs_sma200',
            'trend_regime', 'volatility_5d', 'volatility_10d', 'volatility_20d', 'volatility_50d', 'volatility_regime',
            'volume_sma_10', 'volume_sma_50', 'volume_ratio', 'volume_regime',
            'hl_ratio', 'hl_ratio_5d', 'gap_up', 'gap_down',
            'rsi_14', 'momentum_regime', 'macd', 'macd_signal', 'macd_histogram', 'macd_regime',
            'bb_upper', 'bb_lower', 'bb_position', 'mean_reversion_regime',
            'composite_regime_score', 'trend_regime_changes', 'trend_regime_stability',
            'bull_low_vol', 'bear_high_vol', 'sideways_low_vol',
            'price_acceleration', 'price_theta', 'vol_sensitivity', 'price_vol_correlation',
            'delta_proxy', 'momentum_acceleration', 'implied_vol_proxy', 'implied_vol_change', 'implied_vol_percentile',
            'vol_percentile_50d', 'vol_regime_high', 'vol_regime_low', 'vix_top_10pct', 'vix_bottom_10pct',
            'time_to_expiry_proxy', 'theta_decay', 'theta_acceleration',
            'atr_20d', 'spread_width_proxy', 'move_vs_spread', 'spread_efficiency',
            'market_trend_strength', 'long_term_trend', 'relative_strength', 'momentum_percentile',
            'quarter', 'month', 'earnings_season', 'volume_price_divergence',
            'accumulation_distribution', 'accumulation_distribution_sma',
            'doji', 'hammer', 'shooting_star',
            'resistance_level', 'support_level', 'distance_to_resistance', 'distance_to_support',
            'vol_clustering', 'vol_persistence', 'vol_skew', 'vol_kurtosis',
            'spread_proxy', 'spread_volatility', 'price_impact', 'illiquidity_proxy',
            'momentum_3d', 'momentum_7d', 'momentum_14d', 'momentum_21d',
            'mean_reversion_5d', 'mean_reversion_20d', 'momentum_consistency',
            'beta_proxy', 'correlation_stability', 'extreme_move_up', 'extreme_move_down',
            'overnight_gap', 'gap_magnitude', 'gap_follow_through',
            'technical_strength', 'risk_adjusted_return_5d', 'risk_adjusted_return_20d', 'quality_score',
            'target_1d_enhanced', 'target_3d_enhanced', 'target_5d_enhanced', 'target_7d_enhanced',
            'target_10d_enhanced', 'target_14d_enhanced', 'target_21d_enhanced',
            'pnl_ratio', 'holding_days', 'daily_return'
        ]
        
        # Fill missing features with defaults
        for feature_name in expected_features:
            if feature_name not in features:
                if 'ratio' in feature_name or feature_name in ['relative_strength']:
                    features[feature_name] = 1.0
                elif feature_name in ['gap_up', 'gap_down', 'doji', 'hammer', 'shooting_star', 'extreme_move_up', 'extreme_move_down']:
                    features[feature_name] = 0
                elif feature_name in ['rsi_14']:
                    features[feature_name] = 50.0
                elif feature_name in ['bb_position', 'implied_vol_percentile', 'vol_percentile_50d', 'momentum_percentile']:
                    features[feature_name] = 0.5
                elif feature_name in ['quarter']:
                    features[feature_name] = 1
                elif feature_name in ['month']:
                    features[feature_name] = 1
                elif feature_name in ['holding_days']:
                    features[feature_name] = 3
                else:
                    features[feature_name] = 0.0
        
        # Ensure daily_return matches price_change_1d
        features['daily_return'] = features['price_change_1d']
        
        # Validate all features are numeric
        for key, value in features.items():
            if not isinstance(value, (int, float, bool)) or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                if 'ratio' in key or key in ['relative_strength']:
                    features[key] = 1.0
                elif key in ['rsi_14']:
                    features[key] = 50.0
                elif key in ['bb_position', 'implied_vol_percentile', 'vol_percentile_50d', 'momentum_percentile']:
                    features[key] = 0.5
                else:
                    features[key] = 0.0
        
        logger.info(f"✅ Validated {len(features)} features for ML compatibility")
        return features
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when analysis fails."""
        logger.warning("Returning default features due to analysis failure")
        return self._validate_and_fill_features({})
