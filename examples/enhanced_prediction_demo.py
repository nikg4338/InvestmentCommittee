# Enhanced Prediction Demo for Investment Committee
# Demonstrates the enhanced model_predictor.py and neural_predictor.py working together

import sys
import os
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Demonstrate the enhanced prediction system:
    1. Enhanced ModelPredictor with historical data
    2. Neural Predictor with MLP and LSTM
    3. Combined ensemble prediction
    """
    
    print("=== Enhanced Investment Committee Prediction Demo ===\n")
    
    try:
        # Import modules
        from models.model_predictor import ModelPredictor, create_sample_data
        from models.neural_predictor import NeuralPredictor, create_sample_neural_features
        
        print("✓ Models imported successfully")
        
        # Create sample data
        symbol, historical_data, technicals = create_sample_data()
        neural_features = create_sample_neural_features()
        
        print(f"✓ Sample data created for {symbol}")
        print(f"  Historical data points: {len(historical_data['prices'])}")
        print(f"  Technical indicators: {len(technicals)}")
        print(f"  Neural features: {len(neural_features['technicals'])}")
        
        # Initialize predictors
        model_predictor = ModelPredictor()
        mlp_predictor = NeuralPredictor(model_type='mlp')
        lstm_predictor = NeuralPredictor(model_type='lstm')
        
        print("\n✓ Predictors initialized")
        
        # Test scenarios
        scenarios = {
            "Bullish Market": {
                "technicals": {
                    'rsi': 25.0,              # Oversold
                    'macd_signal': 0.7,       # Strong bullish
                    'bollinger_position': 0.1, # Near lower band
                    'volume_ratio': 1.8,      # High volume
                    'price_momentum': 0.6,    # Strong positive momentum
                    'volatility_rank': 45.0,  # Good IV
                    'vix_level': 14.0,        # Low VIX
                    'market_trend': 0.7       # Strong uptrend
                }
            },
            "Bearish Market": {
                "technicals": {
                    'rsi': 80.0,              # Overbought
                    'macd_signal': -0.6,      # Bearish signal
                    'bollinger_position': 0.9, # Near upper band
                    'volume_ratio': 2.0,      # High volume
                    'price_momentum': -0.4,   # Negative momentum
                    'volatility_rank': 85.0,  # High IV
                    'vix_level': 35.0,        # High VIX
                    'market_trend': -0.5      # Downtrend
                }
            },
            "Neutral Market": {
                "technicals": {
                    'rsi': 52.0,              # Neutral
                    'macd_signal': 0.1,       # Weak signal
                    'bollinger_position': 0.5, # Middle band
                    'volume_ratio': 1.0,      # Average volume
                    'price_momentum': 0.05,   # Slight momentum
                    'volatility_rank': 45.0,  # Good IV
                    'vix_level': 18.0,        # Low VIX
                    'market_trend': 0.1       # Slight uptrend
                }
            }
        }
        
        # Test each scenario
        for scenario_name, scenario_data in scenarios.items():
            print(f"\n{'='*60}")
            print(f"SCENARIO: {scenario_name}")
            print(f"{'='*60}")
            
            # Update historical data with scenario technicals
            scenario_historical = historical_data.copy()
            scenario_technicals = scenario_data['technicals']
            
            # Update neural features
            scenario_neural_features = neural_features.copy()
            scenario_neural_features['technicals'].update(scenario_technicals)
            
            # 1. Enhanced Model Predictor
            print("\n1. ENHANCED MODEL PREDICTOR:")
            direction, confidence, metadata = model_predictor.predict_trade_signal(
                symbol, scenario_historical, scenario_technicals
            )
            print(f"   Direction: {direction}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Current Price: ${metadata['current_price']:.2f}")
            print(f"   Risk Factors: {len(metadata['risk_factors'])}")
            
            # 2. Neural MLP Predictor
            print("\n2. NEURAL MLP PREDICTOR:")
            mlp_direction, mlp_confidence, mlp_metadata = mlp_predictor.predict_nn_signal(
                scenario_neural_features
            )
            print(f"   Direction: {mlp_direction}")
            print(f"   Confidence: {mlp_confidence:.3f}")
            print(f"   Probabilities: {mlp_metadata['probabilities']}")
            
            # 3. Neural LSTM Predictor
            print("\n3. NEURAL LSTM PREDICTOR:")
            lstm_direction, lstm_confidence, lstm_metadata = lstm_predictor.predict_nn_signal(
                scenario_neural_features
            )
            print(f"   Direction: {lstm_direction}")
            print(f"   Confidence: {lstm_confidence:.3f}")
            print(f"   Probabilities: {lstm_metadata['probabilities']}")
            
            # 4. Ensemble Prediction
            print("\n4. ENSEMBLE PREDICTION:")
            ensemble_result = create_ensemble_prediction(
                (direction, confidence),
                (mlp_direction, mlp_confidence),
                (lstm_direction, lstm_confidence)
            )
            print(f"   Final Direction: {ensemble_result['direction']}")
            print(f"   Final Confidence: {ensemble_result['confidence']:.3f}")
            print(f"   Agreement Score: {ensemble_result['agreement']:.3f}")
            print(f"   Recommendation: {ensemble_result['recommendation']}")
            
            # 5. Feature Analysis
            print("\n5. KEY INSIGHTS:")
            feature_contributions = metadata['feature_importance']
            top_features = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            
            for feature, contribution in top_features:
                print(f"   {feature.replace('_', ' ').title()}: {contribution:.3f}")
            
            if metadata['risk_factors']:
                print(f"   Risk Factors: {', '.join(metadata['risk_factors'][:2])}")
            
            # Historical analysis
            hist_analysis = metadata['historical_analysis']
            if '5_day_return' in hist_analysis:
                print(f"   5-Day Return: {hist_analysis['5_day_return']:.2%}")
            
            print(f"\n   📊 Summary: {scenario_name}")
            print(f"   🎯 Enhanced Model: {direction} ({confidence:.1%})")
            print(f"   🧠 Neural MLP: {mlp_direction} ({mlp_confidence:.1%})")
            print(f"   📈 Neural LSTM: {lstm_direction} ({lstm_confidence:.1%})")
            print(f"   🔮 Ensemble: {ensemble_result['direction']} ({ensemble_result['confidence']:.1%})")
        
        print(f"\n{'='*60}")
        print("ADVANCED FEATURES DEMONSTRATED")
        print(f"{'='*60}")
        print("✓ Enhanced Model Predictor:")
        print("  - Historical data analysis")
        print("  - Feature engineering with price history")
        print("  - Risk factor identification")
        print("  - Support/resistance levels")
        print("  - Volatility analysis")
        print("\n✓ Neural Network Predictors:")
        print("  - MLP for structured data")
        print("  - LSTM for sequential data")
        print("  - Probability distributions")
        print("  - PyTorch architecture (with fallback)")
        print("\n✓ Ensemble Methods:")
        print("  - Multi-model consensus")
        print("  - Confidence weighting")
        print("  - Agreement scoring")
        print("  - Final recommendation logic")
        
        print(f"\n{'='*60}")
        print("NEXT STEPS")
        print(f"{'='*60}")
        print("1. Install PyTorch: pip install torch")
        print("2. Collect historical training data")
        print("3. Train neural networks on real data")
        print("4. Integrate with Alpaca API for live data")
        print("5. Add model versioning and A/B testing")
        print("6. Implement automated retraining pipeline")
        print("7. Add performance monitoring and alerts")
        
    except Exception as e:
        logger.error(f"Error in prediction demo: {e}")
        print(f"Error: {e}")
        print("\nMake sure to install dependencies:")
        print("pip install torch numpy pandas")


def create_ensemble_prediction(model_pred: tuple, mlp_pred: tuple, lstm_pred: tuple) -> dict:
    """
    Create ensemble prediction from multiple model outputs.
    
    Args:
        model_pred: (direction, confidence) from enhanced model
        mlp_pred: (direction, confidence) from MLP
        lstm_pred: (direction, confidence) from LSTM
        
    Returns:
        dict: Ensemble prediction results
    """
    predictions = [model_pred, mlp_pred, lstm_pred]
    directions = [pred[0] for pred in predictions]
    confidences = [pred[1] for pred in predictions]
    
    # Calculate weighted average confidence
    weights = [0.4, 0.3, 0.3]  # Enhanced model gets higher weight
    weighted_confidence = sum(w * c for w, c in zip(weights, confidences))
    
    # Determine consensus direction
    direction_counts = {}
    for direction in directions:
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
    
    # Find most common direction
    consensus_direction = max(direction_counts.items(), key=lambda x: x[1])[0]
    
    # Calculate agreement score
    agreement = direction_counts[consensus_direction] / len(directions)
    
    # Generate recommendation
    if agreement >= 0.67 and weighted_confidence > 0.6:
        recommendation = "STRONG BUY" if consensus_direction == "BULLISH" else "STRONG SELL" if consensus_direction == "BEARISH" else "HOLD"
    elif agreement >= 0.67:
        recommendation = "MODERATE BUY" if consensus_direction == "BULLISH" else "MODERATE SELL" if consensus_direction == "BEARISH" else "HOLD"
    else:
        recommendation = "HOLD - MIXED SIGNALS"
    
    return {
        'direction': consensus_direction,
        'confidence': weighted_confidence,
        'agreement': agreement,
        'recommendation': recommendation,
        'individual_predictions': {
            'enhanced_model': model_pred,
            'mlp': mlp_pred,
            'lstm': lstm_pred
        }
    }


if __name__ == "__main__":
    main() 