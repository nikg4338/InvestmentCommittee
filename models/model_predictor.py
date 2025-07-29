# Model predictor module for Investment Committee
# Central interface for machine learning predictions for bull put spread trading

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pickle
import os
import math

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Central model predictor for bull put spread trading signals.
    Coordinates between XGBoost, Neural Network, and LLM models.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the model predictor.
        
        Args:
            model_path (str, optional): Path to saved model files
        """
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), 'saved_models')
        self.xgboost_model = None
        self.neural_network = None
        self.scaler = None
        self.feature_columns = None
        self.model_loaded = False
        
        # Initialize with dummy model for now
        self._initialize_dummy_model()
    
    def _initialize_dummy_model(self):
        """Initialize dummy model parameters for testing."""
        # Feature importance weights (dummy values)
        self.feature_weights = {
            'rsi': 0.15,
            'macd_signal': 0.12,
            'bollinger_position': 0.10,
            'volume_ratio': 0.08,
            'price_momentum': 0.15,
            'volatility_rank': 0.20,
            'vix_level': 0.10,
            'market_trend': 0.10
        }
        
        # Expected feature columns
        self.feature_columns = list(self.feature_weights.keys())
        logger.info("Dummy model initialized with feature weights")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load pre-trained model from file.
        
        Args:
            model_path (str, optional): Path to model file
            
        Returns:
            bool: True if model loaded successfully
        """
        if model_path:
            self.model_path = model_path
            
        try:
            # Placeholder for actual model loading
            # In real implementation, this would load:
            # - self.xgboost_model = pickle.load(open(model_path, 'rb'))
            # - self.scaler = pickle.load(open(scaler_path, 'rb'))
            # - self.feature_columns = pickle.load(open(features_path, 'rb'))
            
            logger.info(f"Model loading from {self.model_path} - PLACEHOLDER")
            self.model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict_trade_signal(self, symbol: str, historical_data: Dict[str, Any], 
                           technicals: Dict[str, float]) -> Tuple[str, float, Dict[str, Any]]:
        """
        Predict trade signal for bull put spread with enhanced inputs.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'MSFT')
            historical_data (Dict[str, Any]): Historical price data
                {
                    'prices': List[float],  # Historical close prices
                    'volumes': List[float],  # Historical volumes
                    'highs': List[float],   # Historical high prices
                    'lows': List[float],    # Historical low prices
                    'dates': List[str],     # Dates in 'YYYY-MM-DD' format
                    'current_price': float  # Current stock price
                }
            technicals (Dict[str, float]): Technical indicators
                {
                    'rsi': float,              # RSI indicator (0-100)
                    'macd_signal': float,      # MACD signal strength (-1 to 1)
                    'bollinger_position': float, # Position within Bollinger Bands (0-1)
                    'volume_ratio': float,     # Volume vs average (0-5+)
                    'price_momentum': float,   # Price momentum indicator (-1 to 1)
                    'volatility_rank': float,  # IV rank (0-100)
                    'vix_level': float,        # VIX level (0-100)
                    'market_trend': float      # Market trend strength (-1 to 1)
                }
        
        Returns:
            Tuple[str, float, Dict[str, Any]]: (direction, confidence, metadata)
                - direction: 'BULLISH', 'BEARISH', or 'NEUTRAL'
                - confidence: Confidence score from 0.0 to 1.0
                - metadata: Additional prediction information
        """
        try:
            # Validate inputs
            if not self._validate_symbol(symbol):
                logger.warning(f"Invalid symbol: {symbol}")
                return 'NEUTRAL', 0.5, {'error': 'Invalid symbol'}
            
            if not self._validate_historical_data(historical_data):
                logger.warning("Invalid historical data provided")
                return 'NEUTRAL', 0.5, {'error': 'Invalid historical data'}
            
            if not self._validate_features(technicals):
                logger.warning("Invalid technical indicators provided")
                return 'NEUTRAL', 0.5, {'error': 'Invalid technicals'}
            
            # Process historical data for additional features
            enhanced_features = self._enhance_features_with_history(historical_data, technicals)
            
            # Use dummy prediction logic for now
            confidence = self._calculate_dummy_prediction(enhanced_features)
            
            # Determine direction based on confidence
            if confidence > 0.6:
                direction = 'BULLISH'
            elif confidence < 0.4:
                direction = 'BEARISH'
            else:
                direction = 'NEUTRAL'
            
            # Ensure confidence is within bounds
            confidence = max(0.0, min(1.0, confidence))
            
            # Create metadata
            metadata = {
                'symbol': symbol,
                'prediction_time': datetime.now().isoformat(),
                'model_version': '1.0.0',
                'feature_importance': self._get_feature_contributions(enhanced_features),
                'historical_analysis': self._analyze_historical_pattern(historical_data),
                'risk_factors': self._identify_risk_factors(enhanced_features),
                'current_price': historical_data.get('current_price', 0.0)
            }
            
            logger.info(f"Prediction for {symbol}: {direction} with {confidence:.3f} confidence")
            return direction, confidence, metadata
            
        except Exception as e:
            logger.error(f"Error in prediction for {symbol}: {e}")
            return 'NEUTRAL', 0.5, {'error': str(e)}

    def predict_trade_signal_legacy(self, features: Dict[str, float]) -> float:
        """
        Legacy predict trade signal function for backward compatibility.
        
        Args:
            features (Dict[str, float]): Dictionary of market and stock features
        
        Returns:
            float: Confidence score from 0.0 to 1.0
        """
        try:
            # Validate input features
            if not self._validate_features(features):
                logger.warning("Invalid features provided, using default prediction")
                return 0.5
            
            # Use dummy prediction logic for now
            confidence = self._calculate_dummy_prediction(features)
            
            # Ensure confidence is within bounds
            confidence = max(0.0, min(1.0, confidence))
            
            logger.info(f"Prediction confidence: {confidence:.3f}")
            return confidence
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return 0.5  # Default neutral prediction
    
    def _validate_features(self, features: Dict[str, float]) -> bool:
        """
        Validate input features.
        
        Args:
            features (Dict[str, float]): Features to validate
            
        Returns:
            bool: True if features are valid
        """
        if not isinstance(features, dict):
            return False
            
        # Check for required features
        required_features = ['rsi', 'volatility_rank', 'vix_level']
        for feature in required_features:
            if feature not in features:
                logger.warning(f"Missing required feature: {feature}")
                return False
                
        # Check feature ranges
        validations = {
            'rsi': (0, 100),
            'volatility_rank': (0, 100),
            'vix_level': (0, 100),
            'macd_signal': (-2, 2),
            'bollinger_position': (0, 1),
            'volume_ratio': (0, 10),
            'price_momentum': (-2, 2),
            'market_trend': (-1, 1)
        }
        
        for feature, (min_val, max_val) in validations.items():
            if feature in features:
                value = features[feature]
                if not (min_val <= value <= max_val):
                    logger.warning(f"Feature {feature} out of range: {value}")
                    return False
        
        return True
    
    def _validate_symbol(self, symbol: str) -> bool:
        """
        Validate stock symbol format.
        
        Args:
            symbol (str): Stock symbol to validate
            
        Returns:
            bool: True if symbol is valid
        """
        if not symbol or not isinstance(symbol, str):
            return False
        
        # Basic symbol validation (alphanumeric, 1-5 characters)
        if not symbol.isalnum() or len(symbol) < 1 or len(symbol) > 5:
            return False
        
        return True
    
    def _validate_historical_data(self, historical_data: Dict[str, Any]) -> bool:
        """
        Validate historical data structure.
        
        Args:
            historical_data (Dict[str, Any]): Historical data to validate
            
        Returns:
            bool: True if data is valid
        """
        if not isinstance(historical_data, dict):
            return False
        
        required_fields = ['prices', 'volumes', 'current_price']
        for field in required_fields:
            if field not in historical_data:
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Check that prices and volumes are lists
        if not isinstance(historical_data['prices'], list):
            return False
        if not isinstance(historical_data['volumes'], list):
            return False
        
        # Check that current_price is a number
        try:
            float(historical_data['current_price'])
        except (ValueError, TypeError):
            return False
        
        # Check that prices and volumes have the same length
        if len(historical_data['prices']) != len(historical_data['volumes']):
            return False
        
        return True
    
    def _enhance_features_with_history(self, historical_data: Dict[str, Any], 
                                     technicals: Dict[str, float]) -> Dict[str, float]:
        """
        Enhance technical features with historical data analysis.
        
        Args:
            historical_data (Dict[str, Any]): Historical price data
            technicals (Dict[str, float]): Technical indicators
            
        Returns:
            Dict[str, float]: Enhanced features
        """
        enhanced = technicals.copy()
        
        try:
            prices = historical_data['prices']
            volumes = historical_data['volumes']
            current_price = historical_data['current_price']
            
            if len(prices) >= 20:
                # Calculate price volatility (20-day)
                recent_prices = prices[-20:]
                price_changes = [abs(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] 
                               for i in range(1, len(recent_prices))]
                enhanced['price_volatility'] = sum(price_changes) / len(price_changes)
                
                # Calculate support/resistance levels
                recent_highs = max(recent_prices)
                recent_lows = min(recent_prices)
                enhanced['support_distance'] = (current_price - recent_lows) / recent_lows
                enhanced['resistance_distance'] = (recent_highs - current_price) / current_price
                
                # Calculate trend strength
                if len(prices) >= 10:
                    short_avg = sum(prices[-10:]) / 10
                    long_avg = sum(prices[-20:]) / 20
                    enhanced['trend_strength'] = (short_avg - long_avg) / long_avg
            
            # Volume analysis
            if len(volumes) >= 20:
                avg_volume = sum(volumes[-20:]) / 20
                current_volume = volumes[-1] if volumes else avg_volume
                enhanced['volume_spike'] = current_volume / avg_volume if avg_volume > 0 else 1.0
            
        except Exception as e:
            logger.warning(f"Error enhancing features: {e}")
        
        return enhanced
    
    def _get_feature_contributions(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate feature contributions to the prediction.
        
        Args:
            features (Dict[str, float]): Input features
            
        Returns:
            Dict[str, float]: Feature contributions
        """
        contributions = {}
        
        # Calculate relative contributions based on feature weights
        for feature, weight in self.feature_weights.items():
            if feature in features:
                value = features[feature]
                
                # Normalize contribution based on feature type
                if feature == 'rsi':
                    # RSI: oversold (low) is positive for puts, overbought (high) is negative
                    contribution = weight * (50 - value) / 50
                elif feature == 'vix_level':
                    # VIX: low is positive for bull put spreads
                    contribution = weight * (30 - value) / 30
                elif feature == 'volatility_rank':
                    # IV Rank: moderate levels are best
                    optimal_iv = 50
                    contribution = weight * (1 - abs(value - optimal_iv) / optimal_iv)
                else:
                    # Generic normalization
                    contribution = weight * (value / 100 if value > 1 else value)
                
                contributions[feature] = contribution
        
        return contributions
    
    def _analyze_historical_pattern(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze historical price patterns.
        
        Args:
            historical_data (Dict[str, Any]): Historical data
            
        Returns:
            Dict[str, Any]: Pattern analysis
        """
        analysis = {}
        
        try:
            prices = historical_data['prices']
            current_price = historical_data['current_price']
            
            if len(prices) >= 20:
                # Calculate recent performance
                analysis['1_day_return'] = (current_price - prices[-1]) / prices[-1] if prices else 0.0
                analysis['5_day_return'] = (current_price - prices[-5]) / prices[-5] if len(prices) >= 5 else 0.0
                analysis['20_day_return'] = (current_price - prices[-20]) / prices[-20] if len(prices) >= 20 else 0.0
                
                # Volatility analysis
                recent_prices = prices[-20:]
                returns = [(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] 
                          for i in range(1, len(recent_prices))]
                if len(returns) > 1:
                    mean_return = sum(returns) / len(returns)
                    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
                    analysis['volatility'] = variance ** 0.5  # Standard deviation
                else:
                    analysis['volatility'] = 0.0
                
                # Support/resistance analysis
                analysis['near_support'] = current_price <= min(prices[-10:]) * 1.02
                analysis['near_resistance'] = current_price >= max(prices[-10:]) * 0.98
                
        except Exception as e:
            logger.warning(f"Error analyzing historical pattern: {e}")
            analysis = {'error': str(e)}
        
        return analysis
    
    def _identify_risk_factors(self, features: Dict[str, float]) -> List[str]:
        """
        Identify risk factors based on current features.
        
        Args:
            features (Dict[str, float]): Input features
            
        Returns:
            List[str]: List of identified risk factors
        """
        risk_factors = []
        
        # High volatility risk
        if features.get('vix_level', 0) > 25:
            risk_factors.append("High market volatility (VIX > 25)")
        
        # Overbought conditions
        if features.get('rsi', 50) > 70:
            risk_factors.append("Overbought conditions (RSI > 70)")
        
        # High IV risk
        if features.get('volatility_rank', 50) > 80:
            risk_factors.append("High implied volatility (IV Rank > 80)")
        
        # Bearish momentum
        if features.get('price_momentum', 0) < -0.3:
            risk_factors.append("Strong bearish momentum")
        
        # Market downtrend
        if features.get('market_trend', 0) < -0.2:
            risk_factors.append("Market downtrend")
        
        # Low volume
        if features.get('volume_ratio', 1.0) < 0.7:
            risk_factors.append("Below average volume")
        
        return risk_factors
    
    def _calculate_dummy_prediction(self, features: Dict[str, float]) -> float:
        """
        Calculate dummy prediction using heuristic logic.
        This simulates what a trained XGBoost model might output.
        
        Args:
            features (Dict[str, float]): Input features
            
        Returns:
            float: Prediction confidence
        """
        # Start with base confidence
        base_confidence = 0.5
        
        # RSI analysis (oversold = bullish for put spreads)
        rsi = features.get('rsi', 50)
        if rsi < 30:  # Oversold
            base_confidence += 0.2
        elif rsi > 70:  # Overbought
            base_confidence -= 0.15
        elif 40 <= rsi <= 60:  # Neutral
            base_confidence += 0.05
        
        # Volatility analysis (moderate IV is good for put spreads)
        iv_rank = features.get('volatility_rank', 50)
        if 30 <= iv_rank <= 70:  # Sweet spot
            base_confidence += 0.15
        elif iv_rank > 80:  # Too high
            base_confidence -= 0.1
        elif iv_rank < 20:  # Too low
            base_confidence -= 0.05
        
        # VIX analysis (low VIX is good for bull put spreads)
        vix = features.get('vix_level', 20)
        if vix < 15:  # Very low volatility
            base_confidence += 0.15
        elif vix < 20:  # Low volatility
            base_confidence += 0.1
        elif vix > 30:  # High volatility
            base_confidence -= 0.2
        
        # Market trend analysis
        market_trend = features.get('market_trend', 0)
        if market_trend > 0.3:  # Strong uptrend
            base_confidence += 0.1
        elif market_trend < -0.3:  # Strong downtrend
            base_confidence -= 0.15
        
        # Price momentum
        momentum = features.get('price_momentum', 0)
        if momentum > 0.5:  # Strong positive momentum
            base_confidence += 0.08
        elif momentum < -0.5:  # Strong negative momentum
            base_confidence -= 0.1
        
        # Volume analysis
        volume_ratio = features.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:  # High volume
            base_confidence += 0.05
        elif volume_ratio < 0.5:  # Low volume
            base_confidence -= 0.03
        
        # MACD signal
        macd = features.get('macd_signal', 0)
        if macd > 0.5:  # Strong bullish signal
            base_confidence += 0.08
        elif macd < -0.5:  # Strong bearish signal
            base_confidence -= 0.1
        
        # Bollinger Band position
        bb_position = features.get('bollinger_position', 0.5)
        if bb_position < 0.2:  # Near lower band (oversold)
            base_confidence += 0.1
        elif bb_position > 0.8:  # Near upper band (overbought)
            base_confidence -= 0.08
        
        # Add some randomness to simulate model uncertainty
        noise = np.random.normal(0, 0.02)  # Small random noise
        base_confidence += noise
        
        # Apply logistic transformation for smoother output
        logistic_confidence = 1 / (1 + math.exp(-4 * (base_confidence - 0.5)))
        
        return logistic_confidence
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the model.
        
        Returns:
            Dict[str, float]: Feature importance scores
        """
        if self.model_loaded:
            # In real implementation, this would return actual feature importance
            # from the trained XGBoost model
            pass
        
        return self.feature_weights
    
    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[float]:
        """
        Predict for multiple feature sets (legacy batch processing).
        
        Args:
            features_list (List[Dict[str, float]]): List of feature dictionaries
            
        Returns:
            List[float]: List of confidence scores
        """
        predictions = []
        for features in features_list:
            prediction = self.predict_trade_signal_legacy(features)
            predictions.append(prediction)
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            'model_loaded': self.model_loaded,
            'model_path': self.model_path,
            'feature_columns': self.feature_columns,
            'model_type': 'XGBoost (Dummy)',
            'version': '1.0.0',
            'last_updated': datetime.now().isoformat()
        }


# Convenience functions for easy usage
def predict_trade_signal(symbol: str, historical_data: Dict[str, Any], 
                        technicals: Dict[str, float]) -> Tuple[str, float, Dict[str, Any]]:
    """
    Convenience function for single prediction.
    
    Args:
        symbol (str): Stock symbol
        historical_data (Dict[str, Any]): Historical price data
        technicals (Dict[str, float]): Technical indicators
        
    Returns:
        Tuple[str, float, Dict[str, Any]]: (direction, confidence, metadata)
    """
    predictor = ModelPredictor()
    return predictor.predict_trade_signal(symbol, historical_data, technicals)

def predict_trade_signal_legacy(features: Dict[str, float]) -> float:
    """
    Legacy convenience function for backward compatibility.
    
    Args:
        features (Dict[str, float]): Market and stock features
        
    Returns:
        float: Confidence score from 0.0 to 1.0
    """
    predictor = ModelPredictor()
    return predictor.predict_trade_signal_legacy(features)


def create_sample_data() -> Tuple[str, Dict[str, Any], Dict[str, float]]:
    """
    Create sample data for testing.
    
    Returns:
        Tuple[str, Dict[str, Any], Dict[str, float]]: (symbol, historical_data, technicals)
    """
    symbol = "AAPL"
    
    # Sample historical data
    historical_data = {
        'prices': [150.0, 152.0, 148.0, 151.0, 153.0, 150.0, 149.0, 152.0, 154.0, 151.0,
                  148.0, 150.0, 152.0, 155.0, 153.0, 151.0, 149.0, 150.0, 152.0, 154.0],
        'volumes': [50000000, 45000000, 55000000, 48000000, 52000000, 47000000, 49000000,
                   51000000, 46000000, 53000000, 48000000, 50000000, 47000000, 52000000,
                   49000000, 51000000, 48000000, 50000000, 49000000, 52000000],
        'highs': [152.0, 154.0, 150.0, 153.0, 155.0, 152.0, 151.0, 154.0, 156.0, 153.0,
                 150.0, 152.0, 154.0, 157.0, 155.0, 153.0, 151.0, 152.0, 154.0, 156.0],
        'lows': [148.0, 150.0, 146.0, 149.0, 151.0, 148.0, 147.0, 150.0, 152.0, 149.0,
                146.0, 148.0, 150.0, 153.0, 151.0, 149.0, 147.0, 148.0, 150.0, 152.0],
        'dates': [f"2024-01-{i:02d}" for i in range(1, 21)],
        'current_price': 155.0
    }
    
    # Sample technical indicators
    technicals = {
        'rsi': 45.0,  # Neutral RSI
        'macd_signal': 0.2,  # Slightly bullish MACD
        'bollinger_position': 0.4,  # Below middle band
        'volume_ratio': 1.2,  # 20% above average volume
        'price_momentum': 0.1,  # Slight positive momentum
        'volatility_rank': 50.0,  # Medium IV rank
        'vix_level': 18.0,  # Low VIX
        'market_trend': 0.3  # Moderate uptrend
    }
    
    return symbol, historical_data, technicals

def create_sample_features() -> Dict[str, float]:
    """
    Create sample features for testing legacy function.
    
    Returns:
        Dict[str, float]: Sample feature dictionary
    """
    return {
        'rsi': 45.0,  # Neutral RSI
        'macd_signal': 0.2,  # Slightly bullish MACD
        'bollinger_position': 0.4,  # Below middle band
        'volume_ratio': 1.2,  # 20% above average volume
        'price_momentum': 0.1,  # Slight positive momentum
        'volatility_rank': 50.0,  # Medium IV rank
        'vix_level': 18.0,  # Low VIX
        'market_trend': 0.3  # Moderate uptrend
    }


def test_model_predictor():
    """Test the model predictor with sample data."""
    print("Testing Model Predictor...")
    
    # Create sample data
    symbol, historical_data, technicals = create_sample_data()
    print(f"Testing symbol: {symbol}")
    print(f"Historical data points: {len(historical_data['prices'])}")
    print(f"Technical indicators: {list(technicals.keys())}")
    
    # Make prediction
    direction, confidence, metadata = predict_trade_signal(symbol, historical_data, technicals)
    print(f"\nPrediction Results:")
    print(f"  Direction: {direction}")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Current Price: ${metadata['current_price']:.2f}")
    
    # Test with different scenarios
    print("\nTesting different scenarios:")
    
    # Bullish scenario
    bullish_technicals = technicals.copy()
    bullish_technicals.update({
        'rsi': 25.0,  # Oversold
        'vix_level': 12.0,  # Very low VIX
        'market_trend': 0.7,  # Strong uptrend
        'volatility_rank': 45.0  # Good IV level
    })
    direction, confidence, metadata = predict_trade_signal(symbol, historical_data, bullish_technicals)
    print(f"Bullish scenario - Direction: {direction}, Confidence: {confidence:.3f}")
    
    # Bearish scenario
    bearish_technicals = technicals.copy()
    bearish_technicals.update({
        'rsi': 75.0,  # Overbought
        'vix_level': 35.0,  # High VIX
        'market_trend': -0.5,  # Downtrend
        'volatility_rank': 85.0  # High IV
    })
    direction, confidence, metadata = predict_trade_signal(symbol, historical_data, bearish_technicals)
    print(f"Bearish scenario - Direction: {direction}, Confidence: {confidence:.3f}")
    
    # Test legacy function
    print("\nTesting legacy function:")
    legacy_confidence = predict_trade_signal_legacy(technicals)
    print(f"Legacy confidence: {legacy_confidence:.3f}")


if __name__ == "__main__":
    test_model_predictor() 