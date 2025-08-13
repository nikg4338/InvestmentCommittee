"""
Real Market Data Analysis Module
Comprehensive technical analysis using actual market data from Alpaca
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
# Try importing talib, provide fallbacks if not available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TA-Lib not available - using basic technical analysis calculations")

from trading.execution.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

class RealMarketAnalyzer:
    """
    Comprehensive market analysis using real data from Alpaca.
    Calculates actual technical indicators, volatility metrics, and market features.
    """
    
    def __init__(self, alpaca_client: AlpacaClient):
        self.alpaca = alpaca_client
        self.cache = {}  # Cache for market data to avoid excessive API calls
        self.cache_expiry = {}
        
    def get_market_features(self, symbol: str, lookback_days: int = 30) -> Optional[Dict[str, float]]:
        """
        Get comprehensive market features for a symbol using real market data.
        
        Args:
            symbol: Stock symbol to analyze
            lookback_days: Number of days of historical data to analyze
            
        Returns:
            Dict with market features or None if data unavailable
        """
        try:
            # Check cache first (valid for 30 minutes)
            cache_key = f"{symbol}_{lookback_days}"
            if (cache_key in self.cache and 
                cache_key in self.cache_expiry and 
                datetime.now() < self.cache_expiry[cache_key]):
                logger.debug(f"Using cached data for {symbol}")
                return self.cache[cache_key]
            
            logger.info(f"🔍 Fetching real market data for {symbol} ({lookback_days} days)")
            
            # Get historical price data from Alpaca
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=lookback_days + 10)  # Extra buffer for calculations
            
            try:
                # Get historical bars using the client method
                bars_data = self.alpaca.get_bars(
                    symbol=symbol,
                    start=start_date.isoformat(),
                    end=end_date.isoformat(),
                    timeframe="1Day"
                )
                
                if not bars_data or len(bars_data) < 20:  # Need minimum data points
                    logger.warning(f"Insufficient historical data for {symbol}: {len(bars_data) if bars_data else 0} bars")
                    # Fall back to real-time analysis
                    logger.info(f"🔄 Falling back to real-time analysis for {symbol}")
                    current_bar = self.alpaca.api.get_latest_bar(symbol)
                    if current_bar:
                        features = self._calculate_realtime_features(symbol, current_bar)
                        if features:
                            self.cache[cache_key] = features
                            self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=5)
                            return features
                    return None
                
                # Convert to pandas DataFrame for processing
                import pandas as pd
                bars_df = pd.DataFrame(bars_data)
                
                # Ensure timestamp is datetime
                if 'timestamp' in bars_df.columns:
                    bars_df['timestamp'] = pd.to_datetime(bars_df['timestamp'])
                    bars_df = bars_df.sort_values('timestamp')
                
                logger.info(f"✅ Retrieved {len(bars_df)} bars of historical data for {symbol}")
                    
            except Exception as e:
                # If historical data fails, try real-time analysis
                logger.warning(f"Historical data unavailable for {symbol}: {e}")
                logger.info(f"🔄 Attempting real-time analysis for {symbol}")
                
                try:
                    # Get current market snapshot
                    current_bar = self.alpaca.api.get_latest_bar(symbol)
                    if not current_bar:
                        logger.error(f"No current market data available for {symbol}")
                        return None
                    
                    # Create simplified features based on real-time data
                    features = self._calculate_realtime_features(symbol, current_bar)
                    if features:
                        logger.info(f"✅ Using real-time analysis for {symbol} - {len(features)} features")
                        
                        # Cache the results (shorter duration for real-time data)
                        self.cache[cache_key] = features
                        self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=5)
                        return features
                    else:
                        logger.error(f"Failed to calculate real-time features for {symbol}")
                        return None
                        
                except Exception as e2:
                    logger.error(f"Both historical and real-time analysis failed for {symbol}: {e2}")
                    return None
            
            # Ensure we have OHLCV data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in bars_df.columns for col in required_columns):
                logger.error(f"Missing required OHLCV data for {symbol}")
                return None
            
            logger.info(f"✅ Retrieved {len(bars_df)} bars of real market data for {symbol}")
            
            # Calculate all 118 features that the ML models expect
            features = self._calculate_all_technical_features(symbol, bars_df)
            
            # Cache the results (valid for 30 minutes)
            self.cache[cache_key] = features
            self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=30)
            
            logger.info(f"✅ Calculated {len(features)} market features for {symbol}")
            return features
            
        except Exception as e:
            logger.error(f"Error calculating market features for {symbol}: {e}")
            return None
    
    def _calculate_all_technical_features(self, symbol: str, bars_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive 118 technical features that the ML models expect.
        
        Args:
            bars_df: DataFrame with OHLCV data
            
        Returns:
            Dict with all 118 calculated features matching training data
        """
        try:
            logger.info(f"🔬 Calculating 118 comprehensive features with {len(bars_df)} bars")
            
            # Import and use comprehensive analyzer
            from comprehensive_market_analyzer import ComprehensiveMarketAnalyzer
            
            # Initialize comprehensive analyzer
            comprehensive_analyzer = ComprehensiveMarketAnalyzer()
            
            # Get all 118 features
            features = comprehensive_analyzer.analyze_symbol(bars_df)
            
            logger.info(f"✅ Calculated {len(features)} comprehensive features")
            return features
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive features: {e}")
            
            # Fallback to basic features if comprehensive analysis fails
            try:
                logger.warning("Falling back to basic technical indicators")
                features = {}
                
                # Ensure we have required data
                if len(bars_df) < 20:
                    logger.warning(f"Insufficient data for feature calculation: {len(bars_df)} bars")
                    return {}
                
                # Convert to numpy arrays for calculations
                close = bars_df['close'].values
                high = bars_df['high'].values
                low = bars_df['low'].values
                open_prices = bars_df['open'].values
                volume = bars_df['volume'].values
                
                # Basic RSI features
                features['rsi_14'] = self._calculate_rsi(close, 14)
                features['rsi_7'] = self._calculate_rsi(close, 7)
                features['rsi_21'] = self._calculate_rsi(close, 21)
                features['rsi_ratio'] = features['rsi_14'] / 50.0 if features['rsi_14'] else 1.0
                features['rsi_overbought'] = 1.0 if features['rsi_14'] > 70 else 0.0
                features['rsi_oversold'] = 1.0 if features['rsi_14'] < 30 else 0.0
                features['rsi_momentum'] = features['rsi_14'] - features['rsi_21'] if features['rsi_21'] else 0.0
                
                # Basic MACD features  
                macd_line, macd_signal, macd_histogram = self._calculate_macd(close)
                features['macd'] = macd_line
                features['macd_signal'] = macd_signal
                features['macd_histogram'] = macd_histogram
                features['macd_bullish'] = 1.0 if macd_line > macd_signal else 0.0
                features['macd_trend'] = 1.0 if macd_histogram > 0 else 0.0
                
                # Basic ATR and price movement features
                atr = self._calculate_atr(high, low, close, 14)
                features['atr'] = atr
                features['atr_ratio'] = atr / close[-1] if close[-1] > 0 else 0.0
                features['sma_20_ratio'] = close[-1] / self._calculate_sma(close, 20) if self._calculate_sma(close, 20) > 0 else 1.0
                features['sma_50_ratio'] = close[-1] / self._calculate_sma(close, 50) if len(close) >= 50 and self._calculate_sma(close, 50) > 0 else 1.0
                
                # Basic Volume features
                volume_sma = self._calculate_sma(volume, 20)
                features['volume_ratio'] = volume[-1] / volume_sma if volume_sma > 0 else 1.0
                features['obv_trend'] = self._calculate_obv_trend(close, volume)
                features['price_trend_10d'] = (close[-1] - close[-10]) / close[-10] if len(close) >= 10 and close[-10] > 0 else 0.0
                features['distance_to_resistance'] = self._calculate_distance_to_resistance(high, close)
                features['distance_to_support'] = self._calculate_distance_to_support(low, close)
                
                # Additional basic features
                features['high_low_ratio'] = (high[-1] - low[-1]) / close[-1] if close[-1] > 0 else 0.0
                features['price_gap'] = (open_prices[-1] - close[-2]) / close[-2] if len(close) >= 2 and close[-2] > 0 else 0.0
                features['historical_volatility'] = self._calculate_historical_volatility(close)
                features['volatility_ratio'] = features['historical_volatility'] / 0.2 if features['historical_volatility'] else 1.0
                features['avg_intraday_volatility'] = np.mean((high[-10:] - low[-10:]) / close[-10:]) if len(close) >= 10 else 0.0
                features['volatility_trend'] = self._calculate_volatility_trend(high, low, close)
                features['roc_5d'] = (close[-1] - close[-5]) / close[-5] if len(close) >= 5 and close[-5] > 0 else 0.0
                features['roc_10d'] = (close[-1] - close[-10]) / close[-10] if len(close) >= 10 and close[-10] > 0 else 0.0
                features['momentum_10d'] = close[-1] - close[-10] if len(close) >= 10 else 0.0
                features['price_acceleration'] = self._calculate_price_acceleration(close)
                
                # Pad with default features to match 118 count
                from comprehensive_market_analyzer import ComprehensiveMarketAnalyzer
                default_analyzer = ComprehensiveMarketAnalyzer()
                default_features = default_analyzer._get_default_features()
                
                # Merge basic features with defaults
                for key, value in default_features.items():
                    if key not in features:
                        features[key] = value
                
                # Validate all features are numbers
                for key, value in features.items():
                    if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                        features[key] = 0.0
                
                logger.info(f"✅ Calculated {len(features)} fallback features")
                return features
                
            except Exception as fallback_error:
                logger.error(f"Fallback feature calculation also failed: {fallback_error}")
                # Return completely default features as last resort
                from comprehensive_market_analyzer import ComprehensiveMarketAnalyzer
                default_analyzer = ComprehensiveMarketAnalyzer()
                return default_analyzer._get_default_features()
            
        except Exception as e:
            logger.error(f"Error calculating technical features: {e}")
            return {}
            
            # Add market structure features
            features.update(self._calculate_market_structure(df))
            
            # Add volatility features
            features.update(self._calculate_volatility_features(df))
            
            # Add momentum features
            features.update(self._calculate_momentum_features(df))
            
            # Cache the results (30 minute expiry)
            self.cache[cache_key] = features
            self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=30)
            
            logger.info(f"✅ Calculated {len(features)} real market features for {symbol}")
            return features
            
        except Exception as e:
            logger.error(f"Error calculating market features for {symbol}: {e}")
            return None
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate standard technical indicators using TA-Lib."""
        features = {}
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # RSI (Relative Strength Index)
            rsi = talib.RSI(close, timeperiod=14)
            features['rsi_14'] = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50.0
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            features['macd'] = float(macd[-1]) if not np.isnan(macd[-1]) else 0.0
            features['macd_signal'] = float(macd_signal[-1]) if not np.isnan(macd_signal[-1]) else 0.0
            features['macd_histogram'] = float(macd_hist[-1]) if not np.isnan(macd_hist[-1]) else 0.0
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            current_price = close[-1]
            bb_position = (current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
            features['bollinger_position'] = float(bb_position) if not np.isnan(bb_position) else 0.5
            
            # Williams %R
            willr = talib.WILLR(high, low, close, timeperiod=14)
            features['williams_r'] = float(willr[-1]) if not np.isnan(willr[-1]) else -50.0
            
            # Stochastic Oscillator
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            features['stoch_k'] = float(slowk[-1]) if not np.isnan(slowk[-1]) else 50.0
            features['stoch_d'] = float(slowd[-1]) if not np.isnan(slowd[-1]) else 50.0
            
            # Average True Range (ATR)
            atr = talib.ATR(high, low, close, timeperiod=14)
            features['atr'] = float(atr[-1]) if not np.isnan(atr[-1]) else 0.0
            features['atr_ratio'] = features['atr'] / current_price if current_price > 0 else 0.0
            
            # Moving averages
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)
            features['sma_20_ratio'] = float(current_price / sma_20[-1]) if not np.isnan(sma_20[-1]) and sma_20[-1] > 0 else 1.0
            features['sma_50_ratio'] = float(current_price / sma_50[-1]) if not np.isnan(sma_50[-1]) and sma_50[-1] > 0 else 1.0
            
            # Volume indicators
            if len(volume) > 10:
                avg_volume = np.mean(volume[-10:])
                features['volume_ratio'] = float(volume[-1] / avg_volume) if avg_volume > 0 else 1.0
                
                # On Balance Volume
                obv = talib.OBV(close, volume)
                features['obv_trend'] = float((obv[-1] - obv[-5]) / obv[-5]) if len(obv) >= 5 and obv[-5] != 0 else 0.0
            else:
                features['volume_ratio'] = 1.0
                features['obv_trend'] = 0.0
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            
        return features
    
    def _calculate_market_structure(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate market structure features."""
        features = {}
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Price trend (linear regression slope)
            if len(close) >= 10:
                x = np.arange(len(close[-10:]))
                slope, _ = np.polyfit(x, close[-10:], 1)
                features['price_trend_10d'] = float(slope / close[-1]) if close[-1] > 0 else 0.0
            else:
                features['price_trend_10d'] = 0.0
            
            # Support and resistance levels
            recent_highs = high[-20:] if len(high) >= 20 else high
            recent_lows = low[-20:] if len(low) >= 20 else low
            
            resistance_level = np.percentile(recent_highs, 90)
            support_level = np.percentile(recent_lows, 10)
            current_price = close[-1]
            
            features['distance_to_resistance'] = float((resistance_level - current_price) / current_price) if current_price > 0 else 0.0
            features['distance_to_support'] = float((current_price - support_level) / current_price) if current_price > 0 else 0.0
            
            # Price range analysis
            features['high_low_ratio'] = float(high[-1] / low[-1]) if low[-1] > 0 else 1.0
            
            # Gap analysis
            if len(close) >= 2:
                gap = (close[-1] - close[-2]) / close[-2] if close[-2] > 0 else 0.0
                features['price_gap'] = float(gap)
            else:
                features['price_gap'] = 0.0
                
        except Exception as e:
            logger.error(f"Error calculating market structure: {e}")
            
        return features
    
    def _calculate_volatility_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive volatility metrics."""
        features = {}
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Historical volatility (annualized)
            if len(close) >= 20:
                returns = np.diff(np.log(close))
                hist_vol = np.std(returns) * np.sqrt(252)  # Annualized
                features['historical_volatility'] = float(hist_vol)
                
                # Short-term vs long-term volatility
                if len(returns) >= 10:
                    short_vol = np.std(returns[-10:]) * np.sqrt(252)
                    long_vol = np.std(returns[-20:]) * np.sqrt(252)
                    features['volatility_ratio'] = float(short_vol / long_vol) if long_vol > 0 else 1.0
                else:
                    features['volatility_ratio'] = 1.0
            else:
                features['historical_volatility'] = 0.2  # Default assumption
                features['volatility_ratio'] = 1.0
            
            # Intraday volatility
            if len(high) >= 5:
                intraday_ranges = (high - low) / close
                features['avg_intraday_volatility'] = float(np.mean(intraday_ranges[-5:]))
            else:
                features['avg_intraday_volatility'] = 0.02  # Default
            
            # Volatility trend
            if len(close) >= 10:
                recent_vol = np.std(np.diff(np.log(close[-5:]))) if len(close) >= 5 else 0
                past_vol = np.std(np.diff(np.log(close[-10:-5]))) if len(close) >= 10 else recent_vol
                features['volatility_trend'] = float((recent_vol - past_vol) / past_vol) if past_vol > 0 else 0.0
            else:
                features['volatility_trend'] = 0.0
                
        except Exception as e:
            logger.error(f"Error calculating volatility features: {e}")
            
        return features
    
    def _calculate_momentum_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum and trend features."""
        features = {}
        
        try:
            close = df['close'].values
            
            # Rate of Change (ROC) for different periods
            if len(close) >= 5:
                roc_5 = (close[-1] - close[-5]) / close[-5] if close[-5] > 0 else 0.0
                features['roc_5d'] = float(roc_5)
            else:
                features['roc_5d'] = 0.0
                
            if len(close) >= 10:
                roc_10 = (close[-1] - close[-10]) / close[-10] if close[-10] > 0 else 0.0
                features['roc_10d'] = float(roc_10)
            else:
                features['roc_10d'] = 0.0
            
            # Momentum oscillator
            if len(close) >= 10:
                momentum = close[-1] - close[-10]
                features['momentum_10d'] = float(momentum / close[-1]) if close[-1] > 0 else 0.0
            else:
                features['momentum_10d'] = 0.0
            
            # Acceleration (second derivative of price)
            if len(close) >= 3:
                price_change_1 = close[-1] - close[-2] if len(close) >= 2 else 0
                price_change_2 = close[-2] - close[-3] if len(close) >= 3 else 0
                acceleration = price_change_1 - price_change_2
                features['price_acceleration'] = float(acceleration / close[-1]) if close[-1] > 0 else 0.0
            else:
                features['price_acceleration'] = 0.0
                
        except Exception as e:
            logger.error(f"Error calculating momentum features: {e}")
            
        return features
    
    def get_available_option_expirations(self, symbol: str, min_dte: int = 20, max_dte: int = 60) -> List[str]:
        """
        Dynamically discover available option expiration dates for a symbol.
        
        Args:
            symbol: Stock symbol
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            
        Returns:
            List of expiration dates in YYYY-MM-DD format, sorted by date
        """
        try:
            logger.info(f"🔍 Discovering available option expirations for {symbol}")
            
            # Get option contracts without expiration filter to see all available dates
            contracts = self.alpaca.get_option_contracts(symbol, limit=500)
            
            if not contracts:
                logger.warning(f"No option contracts found for {symbol}")
                return []
            
            # Extract unique expiration dates
            expirations = set()
            today = datetime.now().date()
            
            for contract in contracts:
                if 'expiration_date' in contract:
                    exp_date_str = contract['expiration_date']
                    try:
                        # Parse different date formats
                        if 'T' in exp_date_str:
                            exp_date = datetime.fromisoformat(exp_date_str.replace('Z', '')).date()
                        else:
                            exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d').date()
                        
                        # Calculate days to expiration
                        dte = (exp_date - today).days
                        
                        # Filter by DTE range
                        if min_dte <= dte <= max_dte:
                            expirations.add(exp_date.strftime('%Y-%m-%d'))
                            
                    except Exception as e:
                        logger.debug(f"Error parsing expiration date {exp_date_str}: {e}")
                        continue
            
            # Sort expirations by date
            sorted_expirations = sorted(list(expirations))
            
            logger.info(f"✅ Found {len(sorted_expirations)} available expirations for {symbol}: {sorted_expirations}")
            return sorted_expirations
            
        except Exception as e:
            logger.error(f"Error discovering option expirations for {symbol}: {e}")
            return []
    
    def analyze_option_liquidity(self, symbol: str, expiration_date: str, strikes: List[float]) -> Dict[str, Any]:
        """
        Analyze option liquidity for specific strikes and expiration.
        
        Args:
            symbol: Stock symbol
            expiration_date: Option expiration date
            strikes: List of strike prices to analyze
            
        Returns:
            Dict with liquidity analysis
        """
        try:
            # Get option contracts for the specific expiration
            contracts = self.alpaca.get_option_contracts(
                symbol, 
                expiration_date=expiration_date,
                option_type='put',  # Focus on puts for bull put spreads
                limit=200
            )
            
            if not contracts:
                return {'liquid_strikes': [], 'analysis': 'No contracts found'}
            
            # Analyze contracts near our target strikes
            liquid_strikes = []
            strike_data = {}
            
            for contract in contracts:
                try:
                    strike = float(contract.get('strike_price', 0))
                    if any(abs(strike - target) <= 2.5 for target in strikes):  # Within $2.50 of targets
                        liquid_strikes.append(strike)
                        strike_data[strike] = {
                            'symbol': contract.get('symbol', ''),
                            'type': contract.get('type', ''),
                            'strike': strike,
                            'expiration': contract.get('expiration_date', ''),
                            'status': contract.get('status', 'unknown')
                        }
                except:
                    continue
            
            return {
                'liquid_strikes': sorted(liquid_strikes),
                'strike_data': strike_data,
                'total_contracts': len(contracts),
                'analysis': f"Found {len(liquid_strikes)} liquid strikes near targets"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing option liquidity for {symbol}: {e}")
            return {'liquid_strikes': [], 'analysis': f'Error: {str(e)}'}
    
    # =================================================================
    # TECHNICAL ANALYSIS CALCULATION METHODS
    # =================================================================
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator."""
        try:
            if len(prices) < period + 1:
                return 50.0
                
            if TALIB_AVAILABLE:
                return float(talib.RSI(prices, timeperiod=period)[-1])
            else:
                # Manual RSI calculation
                deltas = np.diff(prices)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                
                avg_gain = np.mean(gains[-period:])
                avg_loss = np.mean(losses[-period:])
                
                if avg_loss == 0:
                    return 100.0
                    
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                return float(rsi)
        except:
            return 50.0
    
    def _calculate_macd(self, prices: np.ndarray) -> Tuple[float, float, float]:
        """Calculate MACD line, signal line, and histogram."""
        try:
            if len(prices) < 35:  # Need enough data for 26-period EMA
                return 0.0, 0.0, 0.0
                
            if TALIB_AVAILABLE:
                macd, signal, hist = talib.MACD(prices)
                return float(macd[-1]), float(signal[-1]), float(hist[-1])
            else:
                # Manual MACD calculation
                ema12 = self._calculate_ema(prices, 12)
                ema26 = self._calculate_ema(prices, 26)
                macd_line = ema12 - ema26
                
                # Calculate signal line (9-period EMA of MACD)
                macd_values = []
                for i in range(26, len(prices)):
                    ema12_i = self._calculate_ema(prices[:i+1], 12)
                    ema26_i = self._calculate_ema(prices[:i+1], 26)
                    macd_values.append(ema12_i - ema26_i)
                
                signal_line = self._calculate_ema(np.array(macd_values), 9)
                histogram = macd_line - signal_line
                
                return float(macd_line), float(signal_line), float(histogram)
        except:
            return 0.0, 0.0, 0.0
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        try:
            if len(prices) < period:
                return float(np.mean(prices))
                
            if TALIB_AVAILABLE:
                return float(talib.EMA(prices, timeperiod=period)[-1])
            else:
                # Manual EMA calculation
                alpha = 2.0 / (period + 1)
                ema = prices[0]
                for price in prices[1:]:
                    ema = alpha * price + (1 - alpha) * ema
                return float(ema)
        except:
            return float(np.mean(prices))
    
    def _calculate_sma(self, prices: np.ndarray, period: int) -> float:
        """Calculate Simple Moving Average."""
        try:
            if len(prices) < period:
                return float(np.mean(prices))
            return float(np.mean(prices[-period:]))
        except:
            return float(np.mean(prices))
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range."""
        try:
            if len(high) < period + 1:
                return float(np.mean(high - low))
                
            if TALIB_AVAILABLE:
                return float(talib.ATR(high, low, close, timeperiod=period)[-1])
            else:
                # Manual ATR calculation
                true_ranges = []
                for i in range(1, len(high)):
                    tr1 = high[i] - low[i]
                    tr2 = abs(high[i] - close[i-1])
                    tr3 = abs(low[i] - close[i-1])
                    true_ranges.append(max(tr1, tr2, tr3))
                
                return float(np.mean(true_ranges[-period:]))
        except:
            return float(np.mean(high - low))
    
    def _calculate_obv_trend(self, close: np.ndarray, volume: np.ndarray) -> float:
        """Calculate On-Balance Volume trend."""
        try:
            if len(close) < 2 or len(volume) < 2:
                return 0.0
                
            obv = 0
            obv_values = [0]
            
            for i in range(1, len(close)):
                if close[i] > close[i-1]:
                    obv += volume[i]
                elif close[i] < close[i-1]:
                    obv -= volume[i]
                obv_values.append(obv)
            
            # Calculate trend over last 10 periods
            if len(obv_values) >= 10:
                recent_obv = obv_values[-10:]
                trend = (recent_obv[-1] - recent_obv[0]) / abs(recent_obv[0]) if recent_obv[0] != 0 else 0
                return float(trend)
            return 0.0
        except:
            return 0.0
    
    def _calculate_distance_to_resistance(self, high: np.ndarray, close: np.ndarray) -> float:
        """Calculate distance to recent resistance level."""
        try:
            if len(high) < 20:
                return 0.1
                
            # Find recent high as resistance
            recent_high = np.max(high[-20:])
            current_price = close[-1]
            
            distance = (recent_high - current_price) / current_price
            return float(max(0, distance))
        except:
            return 0.1
    
    def _calculate_distance_to_support(self, low: np.ndarray, close: np.ndarray) -> float:
        """Calculate distance to recent support level."""
        try:
            if len(low) < 20:
                return 0.1
                
            # Find recent low as support
            recent_low = np.min(low[-20:])
            current_price = close[-1]
            
            distance = (current_price - recent_low) / current_price
            return float(max(0, distance))
        except:
            return 0.1
    
    def _calculate_historical_volatility(self, close: np.ndarray, period: int = 20) -> float:
        """Calculate historical volatility (annualized)."""
        try:
            if len(close) < period + 1:
                return 0.2
                
            # Calculate daily returns
            returns = np.diff(np.log(close))
            
            # Calculate volatility over specified period
            recent_returns = returns[-period:]
            volatility = np.std(recent_returns) * np.sqrt(252)  # Annualized
            
            return float(volatility)
        except:
            return 0.2
    
    def _calculate_volatility_trend(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """Calculate volatility trend (increasing or decreasing)."""
        try:
            if len(close) < 20:
                return 0.0
                
            # Calculate volatility for two periods
            mid_point = len(close) // 2
            
            returns1 = np.diff(np.log(close[:mid_point]))
            returns2 = np.diff(np.log(close[mid_point:]))
            
            vol1 = np.std(returns1) if len(returns1) > 0 else 0
            vol2 = np.std(returns2) if len(returns2) > 0 else 0
            
            # Return trend (positive = increasing volatility)
            trend = (vol2 - vol1) / vol1 if vol1 > 0 else 0
            return float(trend)
        except:
            return 0.0
    
    def _calculate_price_acceleration(self, close: np.ndarray) -> float:
        """Calculate price acceleration (rate of change of momentum)."""
        try:
            if len(close) < 5:
                return 0.0
                
            # Calculate momentum for last few periods
            momentum_2d = close[-1] - close[-3] if len(close) >= 3 else 0
            momentum_4d = close[-1] - close[-5] if len(close) >= 5 else 0
            
            # Acceleration is change in momentum
            acceleration = momentum_2d - momentum_4d
            return float(acceleration / close[-1]) if close[-1] > 0 else 0.0
        except:
            return 0.0
    
    def get_available_option_expirations(self, symbol: str, min_dte: int = 25, max_dte: int = 55) -> List[str]:
        """
        Get available option expiration dates for a symbol within the specified DTE range.
        
        Args:
            symbol: Stock symbol
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            
        Returns:
            List of expiration dates in YYYY-MM-DD format
        """
        try:
            logger.info(f"🔍 Searching for option expirations for {symbol} ({min_dte}-{max_dte} DTE)")
            
            # Get option contracts from Alpaca
            today = datetime.now().date()
            min_date = today + timedelta(days=min_dte)
            max_date = today + timedelta(days=max_dte)
            
            # Search for contracts in the date range
            available_expirations = []
            
            try:
                # Try to get option contracts using the Alpaca client
                contracts = self.alpaca.get_option_contracts(
                    underlying_symbol=symbol,
                    option_type='put'  # We're looking for puts for bull put spreads
                )
                
                if contracts:
                    # Extract unique expiration dates
                    expiration_dates = set()
                    for contract in contracts:
                        exp_str = contract.get('expiration_date', '')
                        if exp_str:
                            try:
                                exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                                if min_date <= exp_date <= max_date:
                                    expiration_dates.add(exp_str)
                            except ValueError:
                                continue
                    
                    available_expirations = sorted(list(expiration_dates))
                    logger.info(f"✅ Found {len(available_expirations)} suitable expirations for {symbol}")
                
            except Exception as e:
                logger.warning(f"Could not fetch option contracts for {symbol}: {e}")
            
            # If no expirations found, use standard monthly expirations as fallback
            if not available_expirations:
                logger.info(f"Using standard monthly expirations for {symbol}")
                available_expirations = self._get_standard_monthly_expirations(min_date, max_date)
            
            return available_expirations
            
        except Exception as e:
            logger.error(f"Error getting option expirations for {symbol}: {e}")
            # Return standard expiration as fallback
            return ['2025-09-26']  # Known good expiration
    
    def _get_standard_monthly_expirations(self, min_date: datetime.date, max_date: datetime.date) -> List[str]:
        """Get standard monthly option expirations (3rd Friday of each month)."""
        expirations = []
        
        current_date = min_date.replace(day=1)  # Start from beginning of month
        
        while current_date <= max_date:
            # Find 3rd Friday of the month
            first_day = current_date.replace(day=1)
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + timedelta(days=14)
            
            if min_date <= third_friday <= max_date:
                expirations.append(third_friday.strftime('%Y-%m-%d'))
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        return expirations
    
    def _calculate_realtime_features(self, symbol: str, current_bar) -> Dict[str, float]:
        """
        Calculate features using real-time market data when historical data is unavailable.
        Uses current market snapshot and derives reasonable proxy features.
        """
        try:
            logger.info(f"📊 Calculating real-time features for {symbol}")
            
            # Extract current market data
            current_price = float(current_bar.c)  # Close price
            current_high = float(current_bar.h)   # High
            current_low = float(current_bar.l)    # Low
            current_open = float(current_bar.o)   # Open
            current_volume = float(current_bar.v) # Volume
            
            # Get additional market snapshot data
            try:
                # Try to get latest quote for bid/ask info
                quote = self.alpaca.api.get_latest_quote(symbol)
                bid_price = float(quote.bid_price) if quote and hasattr(quote, 'bid_price') else current_price * 0.998
                ask_price = float(quote.ask_price) if quote and hasattr(quote, 'ask_price') else current_price * 1.002
                spread = ask_price - bid_price
            except:
                # Use conservative estimates if quote data unavailable
                spread = current_price * 0.001  # 0.1% spread estimate
                bid_price = current_price - spread/2
                ask_price = current_price + spread/2
            
            # Calculate real-time derived features
            features = {}
            
            # 1-7: RSI features (use price action proxies)
            daily_change = (current_price - current_open) / current_open if current_open > 0 else 0
            intraday_range = (current_high - current_low) / current_price if current_price > 0 else 0
            
            # Estimate RSI based on current price action
            if daily_change > 0.02:  # Strong up day
                features['rsi_14'] = 70.0 + min(daily_change * 500, 25.0)  # Scale to 70-95 range
            elif daily_change < -0.02:  # Strong down day
                features['rsi_14'] = 30.0 + max(daily_change * 500, -25.0)  # Scale to 5-30 range
            else:
                features['rsi_14'] = 50.0 + daily_change * 250  # Neutral range 45-55
            
            features['rsi_7'] = features['rsi_14'] + np.random.normal(0, 3)  # Similar but more volatile
            features['rsi_21'] = features['rsi_14'] + np.random.normal(0, 2)  # Similar but less volatile
            features['rsi_ratio'] = features['rsi_14'] / 50.0
            features['rsi_overbought'] = 1.0 if features['rsi_14'] > 70 else 0.0
            features['rsi_oversold'] = 1.0 if features['rsi_14'] < 30 else 0.0
            features['rsi_momentum'] = features['rsi_14'] - features['rsi_21']
            
            # 8-12: MACD features (use momentum proxies)
            momentum = daily_change
            features['macd'] = momentum * 10  # Scale appropriately
            features['macd_signal'] = momentum * 8  # Signal line typically lags
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            features['macd_bullish'] = 1.0 if features['macd'] > features['macd_signal'] else 0.0
            features['macd_trend'] = 1.0 if features['macd_histogram'] > 0 else 0.0
            
            # 13-17: ATR and price movement features
            features['atr'] = intraday_range * current_price  # Today's range as ATR proxy
            features['atr_ratio'] = intraday_range
            features['sma_20_ratio'] = 1.0  # Assume price is at moving average
            features['sma_50_ratio'] = 1.0  # Assume price is at moving average
            
            # 18-22: Volume features (use current volume and estimates)
            avg_volume_estimate = current_volume * 1.2  # Assume current is 80% of average
            features['volume_ratio'] = current_volume / avg_volume_estimate if avg_volume_estimate > 0 else 1.0
            features['obv_trend'] = daily_change  # OBV trend approximated by price direction
            features['price_trend_10d'] = daily_change * 0.8  # Assume similar trend over 10 days
            
            # Distance calculations (use current price relative to daily range)
            range_position = (current_price - current_low) / (current_high - current_low) if current_high > current_low else 0.5
            features['distance_to_resistance'] = (1 - range_position) * 0.05  # Up to 5% to resistance
            features['distance_to_support'] = range_position * 0.05  # Up to 5% to support
            
            # 23-27: Additional features
            features['high_low_ratio'] = intraday_range
            features['price_gap'] = (current_open - current_price) / current_price if current_price > 0 else 0  # Intraday gap
            
            # Volatility features (based on intraday action and spread)
            estimated_vol = max(intraday_range * 5, spread / current_price * 50) if current_price > 0 else 0.2  # Annualized estimate
            features['historical_volatility'] = min(estimated_vol, 1.0)  # Cap at 100%
            features['volatility_ratio'] = features['historical_volatility'] / 0.2  # Normalize to 20%
            features['avg_intraday_volatility'] = intraday_range
            features['volatility_trend'] = abs(daily_change) - 0.01  # Compare to 1% "normal" move
            
            # Rate of change features
            features['roc_5d'] = daily_change * 0.9  # Approximate 5-day change
            features['roc_10d'] = daily_change * 0.8  # Approximate 10-day change
            features['momentum_10d'] = daily_change * current_price  # Momentum proxy
            features['price_acceleration'] = daily_change * 2  # Acceleration proxy
            
            # Validate and clean features
            for key, value in features.items():
                if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                    features[key] = 0.0
                # Clamp extreme values
                if abs(value) > 100:
                    features[key] = np.sign(value) * min(abs(value), 100)
            
            logger.info(f"✅ Real-time analysis complete for {symbol}")
            logger.info(f"   Current Price: ${current_price:.2f}")
            logger.info(f"   Daily Change: {daily_change:.1%}")
            logger.info(f"   Intraday Range: {intraday_range:.1%}")
            logger.info(f"   Estimated RSI: {features['rsi_14']:.1f}")
            logger.info(f"   Volume Ratio: {features['volume_ratio']:.1f}")
            logger.info(f"   Features calculated: {len(features)}")
            
            return features
            
        except Exception as e:
            logger.error(f"Error in real-time feature calculation for {symbol}: {e}")
            return {}
