# Example of integrating ModelPredictor with TradeFilter
# This demonstrates the complete pipeline for bull put spread decision making

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
    Demonstrate the complete trade decision pipeline:
    1. Model Predictor generates confidence score
    2. Trade Filter validates eligibility
    3. Combined decision for bull put spread
    """
    
    print("=== Investment Committee Trade Decision Pipeline ===\n")
    
    try:
        # Import modules (with error handling since packages may not be installed)
        try:
            from models.model_predictor import ModelPredictor, create_sample_features
            from trading.strategy.trade_filter import is_trade_eligible, create_sample_ticker_data
            predictor_available = True
        except ImportError as e:
            print(f"Error importing modules: {e}")
            print("This is expected if dependencies aren't installed yet.")
            predictor_available = False
        
        if not predictor_available:
            print("Run 'pip install -r requirements.txt' to install dependencies")
            return
        
        # Initialize model predictor
        predictor = ModelPredictor()
        print("✓ ModelPredictor initialized")
        
        # Create sample ticker data for different scenarios
        scenarios = {
            "Bullish AAPL": {
                "features": {
                    'rsi': 28.0,           # Oversold (bullish)
                    'macd_signal': 0.6,    # Strong bullish signal
                    'bollinger_position': 0.15,  # Near lower band
                    'volume_ratio': 1.8,   # High volume
                    'price_momentum': 0.4, # Positive momentum
                    'volatility_rank': 45.0,  # Good IV level
                    'vix_level': 16.0,     # Low VIX
                    'market_trend': 0.5    # Uptrend
                },
                "ticker_data": {
                    'ticker': 'AAPL',
                    'market_data': {
                        'vix': 16.0,       # Below 20 ✓
                        'vvix': 80.0,      # Below 100 ✓
                        'spy_trend': 'up'   # Uptrend ✓
                    },
                    'ticker_data': {
                        'avg_daily_volume': 75_000_000,  # Above 1M ✓
                        'iv_rank': 45.0,    # Between 30-70 ✓
                        'options_chain': {
                            'put_leg_1': {
                                'open_interest': 1500,    # Above 500 ✓
                                'bid_ask_spread': 0.06    # Below 0.10 ✓
                            },
                            'put_leg_2': {
                                'open_interest': 1200,    # Above 500 ✓
                                'bid_ask_spread': 0.08    # Below 0.10 ✓
                            }
                        }
                    },
                    'earnings': {
                        'next_earnings_date': '2024-03-15'  # More than 7 days away
                    }
                }
            },
            
            "Bearish TSLA": {
                "features": {
                    'rsi': 78.0,           # Overbought (bearish)
                    'macd_signal': -0.4,   # Bearish signal
                    'bollinger_position': 0.85,  # Near upper band
                    'volume_ratio': 2.2,   # High volume
                    'price_momentum': -0.3, # Negative momentum
                    'volatility_rank': 88.0,  # High IV
                    'vix_level': 32.0,     # High VIX
                    'market_trend': -0.2   # Slight downtrend
                },
                "ticker_data": {
                    'ticker': 'TSLA',
                    'market_data': {
                        'vix': 32.0,       # Above 20 ✗
                        'vvix': 110.0,     # Above 100 ✗
                        'spy_trend': 'down' # Downtrend ✗
                    },
                    'ticker_data': {
                        'avg_daily_volume': 25_000_000,  # Above 1M ✓
                        'iv_rank': 88.0,    # Above 70 ✗
                        'options_chain': {
                            'put_leg_1': {
                                'open_interest': 800,     # Above 500 ✓
                                'bid_ask_spread': 0.12    # Above 0.10 ✗
                            },
                            'put_leg_2': {
                                'open_interest': 600,     # Above 500 ✓
                                'bid_ask_spread': 0.15    # Above 0.10 ✗
                            }
                        }
                    },
                    'earnings': {
                        'next_earnings_date': '2024-01-25'  # Within 7 days ✗
                    }
                }
            },
            
            "Neutral SPY": {
                "features": {
                    'rsi': 52.0,           # Neutral
                    'macd_signal': 0.1,    # Weak signal
                    'bollinger_position': 0.5,   # Middle band
                    'volume_ratio': 1.0,   # Average volume
                    'price_momentum': 0.05, # Slight positive momentum
                    'volatility_rank': 40.0,  # Good IV level
                    'vix_level': 18.0,     # Low VIX
                    'market_trend': 0.1    # Slight uptrend
                },
                "ticker_data": {
                    'ticker': 'SPY',
                    'market_data': {
                        'vix': 18.0,       # Below 20 ✓
                        'vvix': 85.0,      # Below 100 ✓
                        'spy_trend': 'sideways' # Sideways ✓
                    },
                    'ticker_data': {
                        'avg_daily_volume': 85_000_000,  # Above 1M ✓
                        'iv_rank': 40.0,    # Between 30-70 ✓
                        'options_chain': {
                            'put_leg_1': {
                                'open_interest': 2000,    # Above 500 ✓
                                'bid_ask_spread': 0.03    # Below 0.10 ✓
                            },
                            'put_leg_2': {
                                'open_interest': 1800,    # Above 500 ✓
                                'bid_ask_spread': 0.04    # Below 0.10 ✓
                            }
                        }
                    },
                    'earnings': {
                        'next_earnings_date': None  # No earnings ✓
                    }
                }
            }
        }
        
        # Process each scenario
        for scenario_name, scenario_data in scenarios.items():
            print(f"\n{'='*60}")
            print(f"SCENARIO: {scenario_name}")
            print(f"{'='*60}")
            
            features = scenario_data["features"]
            ticker_data = scenario_data["ticker_data"]
            
            # Step 1: Get ML prediction
            print("\n1. ML MODEL PREDICTION:")
            confidence = predictor.predict_trade_signal(features)
            print(f"   Confidence Score: {confidence:.3f}")
            
            confidence_level = "HIGH" if confidence > 0.65 else "MEDIUM" if confidence > 0.45 else "LOW"
            print(f"   Confidence Level: {confidence_level}")
            
            # Step 2: Check trade eligibility
            print("\n2. TRADE FILTER VALIDATION:")
            is_eligible = is_trade_eligible(ticker_data)
            print(f"   Trade Eligible: {'✓ YES' if is_eligible else '✗ NO'}")
            
            # Step 3: Final decision
            print("\n3. FINAL DECISION:")
            
            # Decision logic
            if is_eligible and confidence > 0.6:
                decision = "EXECUTE TRADE"
                risk_level = "LOW"
            elif is_eligible and confidence > 0.45:
                decision = "CONSIDER TRADE"
                risk_level = "MEDIUM"
            elif is_eligible:
                decision = "HOLD"
                risk_level = "HIGH"
            else:
                decision = "REJECT TRADE"
                risk_level = "HIGH"
            
            print(f"   Decision: {decision}")
            print(f"   Risk Level: {risk_level}")
            
            # Step 4: Feature analysis
            print("\n4. KEY FACTORS:")
            feature_importance = predictor.get_feature_importance()
            
            # Show top contributing factors
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            for feature, importance in sorted_features[:3]:
                if feature in features:
                    value = features[feature]
                    print(f"   {feature.replace('_', ' ').title()}: {value} (importance: {importance:.2f})")
            
            print(f"\n   📊 Summary: {scenario_name}")
            print(f"   🎯 ML Confidence: {confidence:.1%}")
            print(f"   ✅ Filter Passed: {is_eligible}")
            print(f"   📈 Recommendation: {decision}")
            
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"{'='*60}")
        print("\nNext Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Add your Alpaca API keys to the .env file")
        print("3. Train real models with historical data")
        print("4. Set up automated trading execution")
        print("5. Implement risk management and position sizing")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 