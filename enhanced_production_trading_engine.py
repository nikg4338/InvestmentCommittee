#!/usr/bin/env python3
"""
Enhanced Production Trading Engine
==================================

Production-ready trading engine with:
- Fixed feature ordering consistency
- Enhanced ensemble integration
- Real-time performance monitoring
- Uncertainty-aware decision making
- Comprehensive logging and error handling
"""

import json
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced components
from enhanced_ensemble_classifier import EnhancedEnsembleClassifier
from data_collection_alpaca import AlpacaDataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedProductionTradingEngine:
    """Production trading engine with enhanced ML capabilities."""
    
    def __init__(self, feature_manifest_path: str = "models/feature_order_manifest.json"):
        """Initialize enhanced trading engine."""
        self.feature_manifest_path = feature_manifest_path
        self.feature_order = self._load_feature_order()
        
        # Initialize components
        self.data_collector = AlpacaDataCollector()
        self.ensemble = EnhancedEnsembleClassifier()
        
        # Trading parameters
        self.confidence_threshold = 0.65
        self.uncertainty_threshold = 0.35
        self.min_prediction_confidence = 0.6
        
        # Performance tracking
        self.daily_predictions = []
        self.daily_performance = {}
        self.trading_session_stats = {
            'total_predictions': 0,
            'high_confidence_predictions': 0,
            'trades_executed': 0,
            'successful_trades': 0,
            'session_start': datetime.now()
        }
        
        logger.info(f"🚀 Enhanced Production Trading Engine initialized")
        
    def _load_feature_order(self) -> List[str]:
        """Load canonical feature ordering."""
        try:
            with open(self.feature_manifest_path, 'r') as f:
                manifest = json.load(f)
            return manifest['feature_order']
        except Exception as e:
            logger.error(f"❌ Failed to load feature manifest: {e}")
            return []
    
    def initialize_models(self) -> bool:
        """Initialize and load all ML models."""
        logger.info(f"🔧 Initializing ML models...")
        
        try:
            # Load ensemble models
            self.ensemble.load_models()
            
            if len(self.ensemble.models) == 0:
                logger.error("❌ No models loaded in ensemble")
                return False
            
            logger.info(f"✅ Models initialized successfully")
            logger.info(f"   Loaded models: {list(self.ensemble.models.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            return False
    
    def get_current_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current market data for a symbol."""
        try:
            # Get latest market data from Alpaca
            market_data = self.data_collector.alpaca_client.get_market_data(
                symbol=symbol,
                timeframe='1Day',
                limit=1
            )
            
            if not market_data or 'bars' not in market_data or not market_data['bars']:
                logger.warning(f"⚠️ No current market data for {symbol}")
                return None
            
            latest_bar = market_data['bars'][-1]
            
            # Convert to standard format
            current_data = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'price': float(latest_bar['close']),
                'open': float(latest_bar['open']),
                'high': float(latest_bar['high']),
                'low': float(latest_bar['low']),
                'close': float(latest_bar['close']),
                'volume': int(latest_bar['volume']),
                'bid': float(latest_bar['close']) * 0.999,  # Approximate bid
                'ask': float(latest_bar['close']) * 1.001   # Approximate ask
            }
            
            return current_data
            
        except Exception as e:
            logger.error(f"❌ Failed to get current market data for {symbol}: {e}")
            return None
        """Prepare features with exact ordering for ML prediction."""
        try:
            logger.debug(f"🔧 Preparing features for {symbol}")
            
            # Use data collector to generate features
            features = self.data_collector._calculate_features(symbol, current_data)
            
            if features is None:
                logger.warning(f"⚠️ Failed to calculate features for {symbol}")
                return None
            
            # Convert to DataFrame
            features_df = pd.DataFrame([features])
            
            # Ensure feature ordering matches training
            if not self.feature_order:
                logger.warning(f"⚠️ No feature order loaded, using calculated features")
                return features_df
            
            # Validate and reorder features
            available_features = [f for f in self.feature_order if f in features_df.columns]
            missing_features = [f for f in self.feature_order if f not in features_df.columns]
            
            if missing_features:
                logger.debug(f"   Missing {len(missing_features)} features for {symbol}")
                # Add missing features as zeros
                for feature in missing_features:
                    features_df[feature] = 0.0
            
            # Reorder to match training
            features_ordered = features_df[self.feature_order]
            
            # Handle any NaN values
            features_ordered = features_ordered.fillna(0.0)
            
            logger.debug(f"✅ Features prepared for {symbol}: {features_ordered.shape}")
            return features_ordered
            
        except Exception as e:
            logger.error(f"❌ Feature preparation failed for {symbol}: {e}")
            return None
    
    def make_enhanced_prediction(self, symbol: str, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make enhanced ML prediction with uncertainty quantification."""
        try:
            # Prepare features
            features = self.prepare_features_for_prediction(symbol, current_data)
            if features is None:
                return self._default_prediction(symbol, "Feature preparation failed")
            
            # Make ensemble prediction with uncertainty
            prediction_result = self.ensemble.predict_single(features, return_uncertainty=True)
            
            # Extract key metrics
            prediction = prediction_result['prediction']
            confidence = prediction_result['confidence']
            uncertainty = prediction_result.get('overall_uncertainty', 0.5)
            
            # Update session stats
            self.trading_session_stats['total_predictions'] += 1
            if confidence >= self.min_prediction_confidence:
                self.trading_session_stats['high_confidence_predictions'] += 1
            
            # Create comprehensive result
            result = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'prediction': prediction,
                'confidence': confidence,
                'uncertainty': uncertainty,
                'individual_predictions': prediction_result.get('individual_predictions', {}),
                'individual_confidences': prediction_result.get('individual_confidences', {}),
                'meta_prediction': prediction_result.get('meta_prediction'),
                'ensemble_prediction': prediction_result.get('ensemble_prediction'),
                'feature_count': len(features.columns),
                'data_quality': self._assess_data_quality(current_data),
                'recommendation': self._generate_recommendation(prediction, confidence, uncertainty)
            }
            
            # Store for daily tracking
            self.daily_predictions.append(result)
            
            logger.debug(f"✅ Enhanced prediction for {symbol}: "
                        f"pred={prediction:.3f}, conf={confidence:.3f}, unc={uncertainty:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Enhanced prediction failed for {symbol}: {e}")
            return self._default_prediction(symbol, f"Prediction error: {str(e)}")
    
    def _default_prediction(self, symbol: str, reason: str) -> Dict[str, Any]:
        """Generate default prediction when ML fails."""
        logger.warning(f"⚠️ Using default prediction for {symbol}: {reason}")
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'prediction': 0.5,
            'confidence': 0.0,
            'uncertainty': 1.0,
            'individual_predictions': {},
            'individual_confidences': {},
            'meta_prediction': None,
            'ensemble_prediction': 0.5,
            'feature_count': 0,
            'data_quality': 0.0,
            'recommendation': 'HOLD',
            'error': reason
        }
    
    def _assess_data_quality(self, current_data: Dict[str, Any]) -> float:
        """Assess quality of input data."""
        try:
            quality_score = 1.0
            
            # Check for missing critical data
            critical_fields = ['price', 'volume', 'bid', 'ask']
            for field in critical_fields:
                if field not in current_data or current_data[field] is None:
                    quality_score -= 0.2
            
            # Check data freshness (if timestamp available)
            if 'timestamp' in current_data:
                data_age = (datetime.now() - current_data['timestamp']).total_seconds()
                if data_age > 300:  # 5 minutes
                    quality_score -= 0.3
            
            # Check for reasonable values
            if 'price' in current_data and current_data['price'] > 0:
                if 'volume' in current_data and current_data['volume'] == 0:
                    quality_score -= 0.1
            
            return max(0.0, quality_score)
            
        except Exception as e:
            logger.debug(f"Data quality assessment failed: {e}")
            return 0.5
    
    def _generate_recommendation(self, prediction: float, confidence: float, uncertainty: float) -> str:
        """Generate trading recommendation based on prediction and uncertainty."""
        # High uncertainty or low confidence = HOLD
        if uncertainty > self.uncertainty_threshold or confidence < self.min_prediction_confidence:
            return 'HOLD'
        
        # Strong bullish signal
        if prediction > 0.7 and confidence > self.confidence_threshold:
            return 'STRONG_BUY'
        elif prediction > 0.6 and confidence > self.confidence_threshold:
            return 'BUY'
        
        # Strong bearish signal
        elif prediction < 0.3 and confidence > self.confidence_threshold:
            return 'STRONG_SELL'
        elif prediction < 0.4 and confidence > self.confidence_threshold:
            return 'SELL'
        
        # Neutral/uncertain
        else:
            return 'HOLD'
    
    def analyze_batch_symbols(self, symbols: List[str]) -> pd.DataFrame:
        """Analyze a batch of symbols with enhanced predictions."""
        logger.info(f"📊 Analyzing batch of {len(symbols)} symbols")
        
        results = []
        for symbol in symbols:
            try:
                # Get current market data
                current_data = self.data_collector.get_current_market_data(symbol)
                if current_data is None:
                    logger.warning(f"⚠️ No market data for {symbol}")
                    continue
                
                # Make enhanced prediction
                prediction_result = self.make_enhanced_prediction(symbol, current_data)
                results.append(prediction_result)
                
            except Exception as e:
                logger.error(f"❌ Analysis failed for {symbol}: {e}")
                continue
        
        if not results:
            logger.warning("⚠️ No successful predictions made")
            return pd.DataFrame()
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Add ranking based on confidence-adjusted prediction
        results_df['score'] = (
            results_df['prediction'] * results_df['confidence'] * 
            (1 - results_df['uncertainty'])
        )
        results_df = results_df.sort_values('score', ascending=False)
        
        logger.info(f"✅ Batch analysis complete: {len(results_df)} successful predictions")
        
        return results_df
    
    def get_top_recommendations(self, symbols: List[str], top_n: int = 10) -> pd.DataFrame:
        """Get top trading recommendations with enhanced filtering."""
        logger.info(f"🎯 Getting top {top_n} recommendations from {len(symbols)} symbols")
        
        # Analyze all symbols
        analysis_results = self.analyze_batch_symbols(symbols)
        
        if analysis_results.empty:
            logger.warning("⚠️ No analysis results available")
            return pd.DataFrame()
        
        # Filter for high-quality predictions
        high_quality = analysis_results[
            (analysis_results['confidence'] >= self.min_prediction_confidence) &
            (analysis_results['uncertainty'] <= self.uncertainty_threshold) &
            (analysis_results['data_quality'] >= 0.7)
        ]
        
        if high_quality.empty:
            logger.warning("⚠️ No high-quality predictions found, using all results")
            high_quality = analysis_results
        
        # Get top recommendations
        top_recommendations = high_quality.head(top_n)
        
        logger.info(f"✅ Generated {len(top_recommendations)} top recommendations")
        logger.info(f"   Best score: {top_recommendations['score'].max():.3f}")
        logger.info(f"   Avg confidence: {top_recommendations['confidence'].mean():.3f}")
        
        return top_recommendations
    
    def update_models_with_feedback(self, trading_results: List[Dict[str, Any]]) -> None:
        """Update ML models with actual trading outcomes."""
        logger.info(f"📈 Updating models with {len(trading_results)} trading results")
        
        try:
            # Prepare feedback data
            predictions = []
            actuals = []
            
            for result in trading_results:
                if 'prediction_data' in result and 'actual_outcome' in result:
                    predictions.append(result['prediction_data'])
                    actuals.append(result['actual_outcome'])
            
            if predictions and actuals:
                # Update ensemble with feedback
                self.ensemble.update_with_feedback(predictions, actuals)
                
                # Update session stats
                successful_trades = sum(1 for actual in actuals if actual > 0.5)
                self.trading_session_stats['successful_trades'] += successful_trades
                self.trading_session_stats['trades_executed'] += len(actuals)
                
                logger.info(f"✅ Models updated with feedback")
                logger.info(f"   Success rate: {successful_trades/len(actuals):.2%}")
            
        except Exception as e:
            logger.error(f"❌ Model feedback update failed: {e}")
    
    def get_session_performance(self) -> Dict[str, Any]:
        """Get current trading session performance metrics."""
        session_duration = (datetime.now() - self.trading_session_stats['session_start']).total_seconds() / 3600
        
        performance = {
            'session_duration_hours': session_duration,
            'total_predictions': self.trading_session_stats['total_predictions'],
            'high_confidence_predictions': self.trading_session_stats['high_confidence_predictions'],
            'trades_executed': self.trading_session_stats['trades_executed'],
            'successful_trades': self.trading_session_stats['successful_trades'],
            'prediction_rate': self.trading_session_stats['total_predictions'] / max(session_duration, 0.1),
            'confidence_rate': (
                self.trading_session_stats['high_confidence_predictions'] / 
                max(self.trading_session_stats['total_predictions'], 1)
            ),
            'trade_success_rate': (
                self.trading_session_stats['successful_trades'] / 
                max(self.trading_session_stats['trades_executed'], 1)
            )
        }
        
        # Add model performance summary
        performance['model_performance'] = self.ensemble.get_model_performance_summary()
        
        return performance
    
    def save_daily_summary(self, filepath: Optional[str] = None) -> None:
        """Save daily trading and prediction summary."""
        if filepath is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
            filepath = f"reports/daily_summary_{date_str}.json"
        
        # Create summary
        summary = {
            'date': datetime.now().strftime("%Y-%m-%d"),
            'session_performance': self.get_session_performance(),
            'total_daily_predictions': len(self.daily_predictions),
            'daily_prediction_stats': self._calculate_daily_stats(),
            'model_performance': self.ensemble.get_model_performance_summary()
        }
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(exist_ok=True)
        
        # Save summary
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"📊 Daily summary saved to {filepath}")
    
    def _calculate_daily_stats(self) -> Dict[str, Any]:
        """Calculate daily prediction statistics."""
        if not self.daily_predictions:
            return {}
        
        predictions_df = pd.DataFrame(self.daily_predictions)
        
        stats = {
            'mean_prediction': predictions_df['prediction'].mean(),
            'mean_confidence': predictions_df['confidence'].mean(),
            'mean_uncertainty': predictions_df['uncertainty'].mean(),
            'high_confidence_count': len(predictions_df[predictions_df['confidence'] >= self.min_prediction_confidence]),
            'low_uncertainty_count': len(predictions_df[predictions_df['uncertainty'] <= self.uncertainty_threshold]),
            'strong_buy_count': len(predictions_df[predictions_df['recommendation'] == 'STRONG_BUY']),
            'buy_count': len(predictions_df[predictions_df['recommendation'] == 'BUY']),
            'hold_count': len(predictions_df[predictions_df['recommendation'] == 'HOLD']),
            'sell_count': len(predictions_df[predictions_df['recommendation'] == 'SELL']),
            'strong_sell_count': len(predictions_df[predictions_df['recommendation'] == 'STRONG_SELL'])
        }
        
        return stats


def main():
    """Demo script for enhanced production trading engine."""
    logger.info("🚀 Enhanced Production Trading Engine Demo")
    
    # Initialize engine
    engine = EnhancedProductionTradingEngine()
    
    # Initialize models
    if not engine.initialize_models():
        logger.error("❌ Failed to initialize models")
        return
    
    # Demo with a few symbols
    demo_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    logger.info(f"📊 Analyzing demo symbols: {demo_symbols}")
    
    # Get recommendations
    recommendations = engine.get_top_recommendations(demo_symbols, top_n=3)
    
    if not recommendations.empty:
        logger.info("🎯 Top Recommendations:")
        for idx, row in recommendations.iterrows():
            logger.info(f"   {row['symbol']}: {row['recommendation']} "
                       f"(score={row['score']:.3f}, conf={row['confidence']:.3f})")
    
    # Show session performance
    performance = engine.get_session_performance()
    logger.info(f"📈 Session Performance:")
    logger.info(f"   Predictions: {performance['total_predictions']}")
    logger.info(f"   High confidence: {performance['high_confidence_predictions']}")
    logger.info(f"   Confidence rate: {performance['confidence_rate']:.2%}")
    
    logger.info("✅ Enhanced Production Trading Engine demo complete")


if __name__ == "__main__":
    main()
