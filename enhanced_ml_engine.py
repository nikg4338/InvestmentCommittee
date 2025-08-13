"""
Enhanced ML Decision Engine
Properly evaluates stocks using trained ML models with real market features
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import os
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ModelPrediction:
    """Individual model prediction result."""
    model_name: str
    prediction: int  # 0 or 1
    confidence: float  # 0.0 to 1.0
    probabilities: List[float]  # [prob_class_0, prob_class_1]
    features_used: int

@dataclass
class EnsemblePrediction:
    """Ensemble prediction combining multiple models."""
    final_prediction: int
    ensemble_confidence: float
    consensus_ratio: float  # What fraction of models agree
    individual_predictions: List[ModelPrediction]
    feature_importance_summary: Dict[str, float]
    decision_reasoning: List[str]

class EnhancedMLDecisionEngine:
    """
    Advanced ML decision engine that properly evaluates stocks using:
    1. Real market features from RealMarketAnalyzer
    2. Properly trained ML models from the models/ directory
    3. Ensemble voting with confidence weighting
    4. Feature importance analysis
    5. Strict quality controls
    """
    
    def __init__(self, model_directory: str = "models/production"):
        self.model_directory = Path(model_directory)
        self.models = {}
        self.feature_columns = None
        self.feature_scalers = {}
        self.load_models()
        
    def load_models(self) -> None:
        """Load all available trained models."""
        try:
            if not self.model_directory.exists():
                logger.warning(f"Model directory {self.model_directory} does not exist")
                return
            
            model_files = list(self.model_directory.glob("*.pkl"))
            if not model_files:
                logger.warning(f"No .pkl model files found in {self.model_directory}")
                return
            
            for model_file in model_files:
                try:
                    model_name = model_file.stem
                    
                    # Skip metadata and scaler files
                    if 'metadata' in model_name or 'scaler' in model_name:
                        continue
                    
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    
                    self.models[model_name] = model
                    logger.info(f"✅ Loaded model: {model_name}")
                    
                    # Try to load corresponding metadata
                    metadata_file = model_file.parent / f"{model_name}_metadata.json"
                    if metadata_file.exists():
                        import json
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            logger.info(f"  📊 Model {model_name} accuracy: {metadata.get('accuracy', 'unknown')}")
                    
                except Exception as e:
                    logger.error(f"Error loading model {model_file}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {len(self.models)} ML models")
            
            # Try to determine expected feature columns from a sample model
            self._determine_feature_structure()
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _determine_feature_structure(self) -> None:
        """Determine the expected feature structure from trained models."""
        try:
            # Check if training_summary.json has feature info
            summary_file = self.model_directory / "training_summary.json"
            if summary_file.exists():
                import json
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    
                # Extract feature count from summary
                feature_count = summary.get('feature_count', 27)
                logger.info(f"Found feature count from training summary: {feature_count}")
            
            # Use the exact 118 feature set that matches our training data
            # This should match exactly what the models were trained on
            if not self.feature_columns:
                self.feature_columns = [
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
                logger.info(f"Using comprehensive feature set: {len(self.feature_columns)} features")
                
        except Exception as e:
            logger.error(f"Error determining feature structure: {e}")
    
    def evaluate_symbol(self, market_features: Dict[str, float]) -> Optional[EnsemblePrediction]:
        """
        Evaluate a symbol using the ensemble of trained ML models.
        
        Args:
            market_features: Real market features from RealMarketAnalyzer
            
        Returns:
            EnsemblePrediction with ensemble results, or None if evaluation fails
        """
        try:
            if not self.models:
                logger.warning("No models loaded for evaluation")
                return None
            
            if not market_features:
                logger.warning("No market features provided")
                return None
            
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(market_features)
            if feature_vector is None:
                return None
            
            # Get predictions from all models
            individual_predictions = []
            
            for model_name, model in self.models.items():
                try:
                    prediction = self._get_model_prediction(model, model_name, feature_vector, market_features)
                    if prediction:
                        individual_predictions.append(prediction)
                except Exception as e:
                    logger.error(f"Error getting prediction from {model_name}: {e}")
                    continue
            
            if not individual_predictions:
                logger.warning("No valid predictions obtained from models")
                return None
            
            # Create ensemble prediction
            ensemble_result = self._create_ensemble_prediction(individual_predictions, market_features)
            
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Error in ML evaluation: {e}")
            return None
    
    def _prepare_feature_vector(self, market_features: Dict[str, float]) -> Optional[np.ndarray]:
        """Prepare feature vector for model input."""
        try:
            if not self.feature_columns:
                logger.error("Feature columns not defined")
                return None
            
            # Create feature vector with proper ordering
            feature_values = []
            missing_features = []
            
            for feature_name in self.feature_columns:
                if feature_name in market_features:
                    value = market_features[feature_name]
                    # Handle NaN/infinite values
                    if np.isnan(value) or np.isinf(value):
                        value = 0.0
                    feature_values.append(value)
                else:
                    # Use reasonable default for missing features
                    default_value = self._get_default_feature_value(feature_name)
                    feature_values.append(default_value)
                    missing_features.append(feature_name)
            
            if missing_features:
                logger.warning(f"Missing features (using defaults): {missing_features}")
            
            return np.array(feature_values).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing feature vector: {e}")
            return None
    
    def _get_default_feature_value(self, feature_name: str) -> float:
        """Get reasonable default value for missing features."""
        defaults = {
            # Price changes
            'price_change_1d': 0.0, 'price_change_5d': 0.0, 'price_change_10d': 0.0, 'price_change_20d': 0.0,
            # Moving averages (absolute values)
            'sma_5': 100.0, 'sma_10': 100.0, 'sma_20': 100.0, 'sma_50': 100.0, 'sma_200': 100.0,
            # Price vs SMA ratios
            'price_vs_sma5': 0.0, 'price_vs_sma10': 0.0, 'price_vs_sma20': 0.0, 'price_vs_sma50': 0.0, 'price_vs_sma200': 0.0,
            # Regime indicators
            'trend_regime': 0, 'volatility_regime': 0, 'momentum_regime': 0, 'macd_regime': 0, 'mean_reversion_regime': 0,
            'volume_regime': 0, 'vol_regime_high': 0, 'vol_regime_low': 0,
            # Volatility features
            'volatility_5d': 0.02, 'volatility_10d': 0.02, 'volatility_20d': 0.02, 'volatility_50d': 0.02,
            'vol_sensitivity': 0.0, 'vol_clustering': 1.0, 'vol_persistence': 1.0, 'vol_skew': 0.0, 'vol_kurtosis': 0.0,
            'vol_percentile_50d': 0.5, 'vix_top_10pct': 0, 'vix_bottom_10pct': 0,
            # Volume features
            'volume_sma_10': 1000000.0, 'volume_sma_50': 1000000.0, 'volume_ratio': 1.0, 'volume_price_divergence': 0.0,
            # Microstructure
            'hl_ratio': 0.02, 'hl_ratio_5d': 0.02, 'gap_up': 0, 'gap_down': 0, 'overnight_gap': 0.0,
            'gap_magnitude': 0.0, 'gap_follow_through': 0.0,
            # Technical indicators
            'rsi_14': 50.0, 'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0,
            'bb_upper': 102.0, 'bb_lower': 98.0, 'bb_position': 0.5,
            # Composite features
            'composite_regime_score': 0, 'trend_regime_changes': 0, 'trend_regime_stability': 20,
            'bull_low_vol': 0, 'bear_high_vol': 0, 'sideways_low_vol': 0,
            # Price dynamics
            'price_acceleration': 0.0, 'price_theta': 0.0, 'price_vol_correlation': 0.0,
            'delta_proxy': 0.0, 'momentum_acceleration': 0.0,
            # Implied volatility
            'implied_vol_proxy': 0.02, 'implied_vol_change': 0.0, 'implied_vol_percentile': 0.5,
            # Time decay
            'time_to_expiry_proxy': 10, 'theta_decay': 0.0, 'theta_acceleration': 0.0,
            # Market structure
            'atr_20d': 0.02, 'spread_width_proxy': 0.001, 'move_vs_spread': 1.0, 'spread_efficiency': 1.0,
            'spread_proxy': 0.02, 'spread_volatility': 0.0, 'price_impact': 0.0, 'illiquidity_proxy': 0.0,
            # Market context
            'market_trend_strength': 0.0, 'long_term_trend': 0, 'relative_strength': 1.0, 'momentum_percentile': 0.5,
            # Calendar
            'quarter': 1, 'month': 1, 'earnings_season': 0,
            # Accumulation/Distribution
            'accumulation_distribution': 0.0, 'accumulation_distribution_sma': 0.0,
            # Candlestick patterns
            'doji': 0, 'hammer': 0, 'shooting_star': 0,
            # Support/Resistance
            'resistance_level': 105.0, 'support_level': 95.0, 'distance_to_resistance': 0.05, 'distance_to_support': 0.05,
            # Momentum multi-timeframe
            'momentum_3d': 0.0, 'momentum_7d': 0.0, 'momentum_14d': 0.0, 'momentum_21d': 0.0,
            'momentum_consistency': 2,
            # Mean reversion
            'mean_reversion_5d': 0.0, 'mean_reversion_20d': 0.0,
            # Correlation/Beta
            'beta_proxy': 1.0, 'correlation_stability': 0.0,
            # Extreme moves
            'extreme_move_up': 0, 'extreme_move_down': 0,
            # Technical composite
            'technical_strength': 0, 'risk_adjusted_return_5d': 0.0, 'risk_adjusted_return_20d': 0.0, 'quality_score': 1,
            # Target variables (unknown for real-time)
            'target_1d_enhanced': 0, 'target_3d_enhanced': 0, 'target_5d_enhanced': 0, 'target_7d_enhanced': 0,
            'target_10d_enhanced': 0, 'target_14d_enhanced': 0, 'target_21d_enhanced': 0,
            # Trading metrics
            'pnl_ratio': 0.0, 'holding_days': 3, 'daily_return': 0.0,
            
            # Legacy features for backward compatibility
            'bollinger_position': 0.5, 'williams_r': -50.0, 'stoch_k': 50.0, 'stoch_d': 50.0,
            'atr': 0.02, 'atr_ratio': 0.02, 'sma_20_ratio': 1.0, 'sma_50_ratio': 1.0,
            'obv_trend': 0.0, 'price_trend_10d': 0.0, 'high_low_ratio': 1.02, 'price_gap': 0.0,
            'historical_volatility': 0.2, 'volatility_ratio': 1.0, 'avg_intraday_volatility': 0.02,
            'volatility_trend': 0.0, 'roc_5d': 0.0, 'roc_10d': 0.0, 'momentum_10d': 0.0
        }
        
        return defaults.get(feature_name, 0.0)
    
    def _get_model_prediction(self, model: Any, model_name: str, feature_vector: np.ndarray, 
                            market_features: Dict[str, float]) -> Optional[ModelPrediction]:
        """Get prediction from a single model."""
        try:
            # Try predict_proba first (for probabilistic models)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(feature_vector)[0]
                
                if len(probabilities) >= 2:
                    confidence = max(probabilities)
                    prediction = 1 if probabilities[1] > probabilities[0] else 0
                else:
                    # Binary case with single probability
                    prob_positive = probabilities[0] if len(probabilities) == 1 else 0.5
                    confidence = max(prob_positive, 1 - prob_positive)
                    prediction = 1 if prob_positive > 0.5 else 0
                    probabilities = [1 - prob_positive, prob_positive]
            
            # Fallback to regular predict
            elif hasattr(model, 'predict'):
                prediction = int(model.predict(feature_vector)[0])
                confidence = 0.6  # Default confidence for non-probabilistic models
                probabilities = [0.4, 0.6] if prediction == 1 else [0.6, 0.4]
            
            else:
                logger.warning(f"Model {model_name} has no predict method")
                return None
            
            return ModelPrediction(
                model_name=model_name,
                prediction=prediction,
                confidence=confidence,
                probabilities=list(probabilities),
                features_used=len(feature_vector[0])
            )
            
        except Exception as e:
            logger.error(f"Error getting prediction from {model_name}: {e}")
            return None
    
    def _create_ensemble_prediction(self, individual_predictions: List[ModelPrediction], 
                                  market_features: Dict[str, float]) -> EnsemblePrediction:
        """Create ensemble prediction from individual model predictions."""
        try:
            # Calculate ensemble metrics
            total_models = len(individual_predictions)
            bullish_votes = sum(1 for pred in individual_predictions if pred.prediction == 1)
            consensus_ratio = bullish_votes / total_models
            
            # Weighted ensemble based on confidence
            weighted_score = sum(pred.confidence * pred.prediction for pred in individual_predictions)
            total_confidence = sum(pred.confidence for pred in individual_predictions)
            
            if total_confidence > 0:
                ensemble_confidence = weighted_score / total_confidence
            else:
                ensemble_confidence = 0.5
            
            # Final prediction based on weighted ensemble
            final_prediction = 1 if ensemble_confidence > 0.5 else 0
            
            # Generate decision reasoning
            reasoning = self._generate_decision_reasoning(
                individual_predictions, market_features, consensus_ratio, ensemble_confidence
            )
            
            # Calculate feature importance (simplified)
            feature_importance = self._calculate_feature_importance(market_features)
            
            return EnsemblePrediction(
                final_prediction=final_prediction,
                ensemble_confidence=ensemble_confidence,
                consensus_ratio=consensus_ratio,
                individual_predictions=individual_predictions,
                feature_importance_summary=feature_importance,
                decision_reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error creating ensemble prediction: {e}")
            # Return a conservative prediction
            return EnsemblePrediction(
                final_prediction=0,
                ensemble_confidence=0.3,
                consensus_ratio=0.0,
                individual_predictions=individual_predictions,
                feature_importance_summary={},
                decision_reasoning=["Error in ensemble calculation - defaulting to bearish"]
            )
    
    def _generate_decision_reasoning(self, predictions: List[ModelPrediction], 
                                   market_features: Dict[str, float], 
                                   consensus_ratio: float, 
                                   ensemble_confidence: float) -> List[str]:
        """Generate human-readable reasoning for the decision."""
        reasoning = []
        
        # Model consensus
        bullish_models = [p.model_name for p in predictions if p.prediction == 1]
        bearish_models = [p.model_name for p in predictions if p.prediction == 0]
        
        reasoning.append(f"Model Consensus: {len(bullish_models)}/{len(predictions)} models bullish ({consensus_ratio:.1%})")
        
        if bullish_models:
            reasoning.append(f"Bullish models: {', '.join(bullish_models)}")
        if bearish_models:
            reasoning.append(f"Bearish models: {', '.join(bearish_models)}")
        
        # Confidence analysis
        avg_confidence = np.mean([p.confidence for p in predictions])
        reasoning.append(f"Average model confidence: {avg_confidence:.1%}")
        reasoning.append(f"Ensemble confidence: {ensemble_confidence:.1%}")
        
        # Key market signals
        if 'rsi_14' in market_features:
            rsi = market_features['rsi_14']
            if rsi > 70:
                reasoning.append(f"⚠️ RSI overbought: {rsi:.1f}")
            elif rsi < 30:
                reasoning.append(f"✅ RSI oversold: {rsi:.1f}")
            else:
                reasoning.append(f"RSI neutral: {rsi:.1f}")
        
        if 'bollinger_position' in market_features:
            bb_pos = market_features['bollinger_position']
            if bb_pos > 0.8:
                reasoning.append("⚠️ Price near upper Bollinger Band")
            elif bb_pos < 0.2:
                reasoning.append("✅ Price near lower Bollinger Band")
        
        if 'historical_volatility' in market_features:
            vol = market_features['historical_volatility']
            if vol > 0.4:
                reasoning.append(f"⚠️ High volatility: {vol:.1%}")
            elif vol < 0.15:
                reasoning.append(f"✅ Low volatility: {vol:.1%}")
        
        return reasoning
    
    def _calculate_feature_importance(self, market_features: Dict[str, float]) -> Dict[str, float]:
        """Calculate simplified feature importance scores."""
        importance = {}
        
        try:
            # RSI importance
            if 'rsi_14' in market_features:
                rsi = market_features['rsi_14']
                # More extreme RSI values are more important
                importance['rsi_signal'] = abs(rsi - 50) / 50
            
            # Volatility importance
            if 'historical_volatility' in market_features:
                vol = market_features['historical_volatility']
                # Extreme volatility is important
                importance['volatility_signal'] = min(vol / 0.3, 1.0)
            
            # Momentum importance
            if 'roc_10d' in market_features:
                roc = abs(market_features['roc_10d'])
                importance['momentum_signal'] = min(roc / 0.1, 1.0)
            
            # Trend importance
            if 'price_trend_10d' in market_features:
                trend = abs(market_features['price_trend_10d'])
                importance['trend_signal'] = min(trend / 0.05, 1.0)
                
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
        
        return importance
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of loaded models and their capabilities."""
        return {
            'total_models': len(self.models),
            'model_names': list(self.models.keys()),
            'feature_count': len(self.feature_columns) if self.feature_columns else 0,
            'feature_columns': self.feature_columns,
            'model_directory': str(self.model_directory)
        }
