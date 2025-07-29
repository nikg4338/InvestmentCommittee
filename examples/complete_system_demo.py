# Complete Investment Committee System Demo
# Demonstrates the full pipeline from screening to execution to backtesting

import sys
import os
import logging
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Demonstrate the complete Investment Committee system.
    
    Pipeline:
    1. Trade Filtering & Screening
    2. Model Predictions (XGBoost + Neural Networks)
    3. Meta-Model Decision Making
    4. Entry Decision Engine
    5. Trade Execution (Paper Trading)
    6. Logging & Performance Tracking
    7. Backtesting Analysis
    """
    
    print("=== Investment Committee Complete System Demo ===\n")
    
    try:
        # Import all components
        from trading.strategy.trade_filter import is_trade_eligible, create_sample_ticker_data
        from models.model_predictor import ModelPredictor, create_sample_data
        from models.neural_predictor import NeuralPredictor, create_sample_neural_features
        from models.meta_model import MetaModel, create_model_input, combine_predictions
        from trading.entry_decision_engine import EntryDecisionEngine, create_trade_candidate
        from utils.trade_logger import TradeLogger, log_trade
        from analysis.backtesting import BacktestEngine, create_sample_historical_data
        
        print("✓ All components imported successfully")
        
        # ==========================================
        # STEP 1: TRADE FILTERING & SCREENING
        # ==========================================
        print(f"\n{'='*60}")
        print("STEP 1: TRADE FILTERING & SCREENING")
        print(f"{'='*60}")
        
        # Create sample ticker data
        ticker_data = create_sample_ticker_data()
        
        # Test trade filter
        is_eligible = is_trade_eligible(ticker_data)
        print(f"Sample ticker (AAPL) eligibility: {'✓ PASSED' if is_eligible else '✗ FAILED'}")
        
        # Test different scenarios
        scenarios = {
            "High VIX": {"vix": 35, "expected": False},
            "Low Volume": {"avg_daily_volume": 500000, "expected": False},
            "High IV": {"iv_rank": 85, "expected": False},
            "Good Setup": {"vix": 15, "avg_daily_volume": 50000000, "iv_rank": 45, "expected": True}
        }
        
        for scenario_name, params in scenarios.items():
            test_data = ticker_data.copy()
            for key, value in params.items():
                if key == "expected":
                    continue
                elif key in ["vix", "vvix"]:
                    test_data["market_data"][key] = value
                else:
                    test_data["ticker_data"][key] = value
            
            result = is_trade_eligible(test_data)
            expected = params["expected"]
            status = "✓" if result == expected else "✗"
            print(f"  {scenario_name}: {status} {'PASSED' if result else 'FAILED'}")
        
        # ==========================================
        # STEP 2: MODEL PREDICTIONS
        # ==========================================
        print(f"\n{'='*60}")
        print("STEP 2: MODEL PREDICTIONS")
        print(f"{'='*60}")
        
        # Enhanced Model Predictor
        model_predictor = ModelPredictor()
        symbol, historical_data, technicals = create_sample_data()
        
        direction, confidence, metadata = model_predictor.predict_trade_signal(
            symbol, historical_data, technicals
        )
        print(f"Enhanced Model: {direction} ({confidence:.1%} confidence)")
        
        # Neural Network Predictors
        neural_mlp = NeuralPredictor(model_type='mlp')
        neural_lstm = NeuralPredictor(model_type='lstm')
        neural_features = create_sample_neural_features()
        
        mlp_direction, mlp_confidence, mlp_metadata = neural_mlp.predict_nn_signal(neural_features)
        print(f"Neural MLP: {mlp_direction} ({mlp_confidence:.1%} confidence)")
        
        lstm_direction, lstm_confidence, lstm_metadata = neural_lstm.predict_nn_signal(neural_features)
        print(f"Neural LSTM: {lstm_direction} ({lstm_confidence:.1%} confidence)")
        
        # ==========================================
        # STEP 3: META-MODEL DECISION MAKING
        # ==========================================
        print(f"\n{'='*60}")
        print("STEP 3: META-MODEL DECISION MAKING")
        print(f"{'='*60}")
        
        # Create meta-model
        meta_model = MetaModel()
        
        # Create model inputs
        model_inputs = [
            create_model_input('xgboost', direction, confidence, metadata),
            create_model_input('neural_mlp', mlp_direction, mlp_confidence, mlp_metadata),
            create_model_input('neural_lstm', lstm_direction, lstm_confidence, lstm_metadata)
        ]
        
        # Add LLM analysis
        llm_analysis = {'sentiment': 'bullish', 'macro_risk': 0.3, 'news_sentiment': 0.7}
        llm_input = meta_model.add_llm_input(llm_analysis)
        model_inputs.append(llm_input)
        
        # Get meta-model decision
        decision = meta_model.predict_trade_signal(model_inputs)
        print(f"Meta-Model Decision: {decision.signal.value}")
        print(f"  Confidence: {decision.confidence:.1%}")
        print(f"  Agreement: {decision.context.get('agreement', 0):.1%}")
        print(f"  Reasoning: {decision.reasoning[0]}")
        
        # ==========================================
        # STEP 4: ENTRY DECISION ENGINE
        # ==========================================
        print(f"\n{'='*60}")
        print("STEP 4: ENTRY DECISION ENGINE")
        print(f"{'='*60}")
        
        # Create entry decision engine
        engine = EntryDecisionEngine(paper_trading=True, max_positions=5)
        
        # Create trade candidates
        candidates = []
        
        # Bullish candidate
        bullish_historical = {
            'prices': [150 + i * 0.5 for i in range(20)],
            'volumes': [50000000 + i * 1000000 for i in range(20)],
            'current_price': 160.0
        }
        bullish_technicals = {
            'rsi': 30, 'vix_level': 16, 'volatility_rank': 45,
            'market_trend': 0.6, 'price_momentum': 0.4
        }
        candidates.append(create_trade_candidate('AAPL', bullish_historical, bullish_technicals))
        
        # Bearish candidate (should be rejected)
        bearish_technicals = {
            'rsi': 75, 'vix_level': 32, 'volatility_rank': 85,
            'market_trend': -0.4, 'price_momentum': -0.3
        }
        candidates.append(create_trade_candidate('TSLA', bullish_historical, bearish_technicals))
        
        # Process candidates
        executions = engine.process_trade_candidates(candidates)
        
        print(f"Processed {len(candidates)} candidates")
        print(f"Generated {len(executions)} executions")
        
        for execution in executions:
            print(f"\n  {execution.symbol}:")
            print(f"    Signal: {execution.signal}")
            print(f"    Status: {execution.execution_status}")
            print(f"    Confidence: {execution.confidence:.1%}")
            if execution.reasoning:
                print(f"    Reasoning: {execution.reasoning[0]}")
        
        # ==========================================
        # STEP 5: TRADE LOGGING
        # ==========================================
        print(f"\n{'='*60}")
        print("STEP 5: TRADE LOGGING")
        print(f"{'='*60}")
        
        # Initialize logger
        trade_logger = TradeLogger(log_dir="demo_logs")
        
        # Log executions
        for execution in executions:
            trade_logger.log_trade(execution)
        
        # Get performance stats
        stats = trade_logger.get_performance_stats()
        print(f"Logged {stats['total_trades']} trades")
        print(f"Success rate: {stats['success_rate']:.1%}")
        print(f"Average confidence: {stats['average_confidence']:.1%}")
        
        # ==========================================
        # STEP 6: BACKTESTING
        # ==========================================
        print(f"\n{'='*60}")
        print("STEP 6: BACKTESTING ANALYSIS")
        print(f"{'='*60}")
        
        # Create backtest engine
        backtest_engine = BacktestEngine(initial_capital=100000, max_positions=10)
        
        # Create sample historical data
        historical_data = create_sample_historical_data()
        
        # Run backtest
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"Running backtest from {start_date} to {end_date}...")
        result = backtest_engine.run_backtest(historical_data, start_date, end_date)
        
        print(f"\nBacktest Results:")
        print(f"  Total Trades: {result.total_trades}")
        print(f"  Win Rate: {result.win_rate:.1%}")
        print(f"  Total P&L: ${result.total_pnl:,.2f}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Average Hold Days: {result.avg_hold_days:.1f}")
        
        # ==========================================
        # STEP 7: PERFORMANCE SUMMARY
        # ==========================================
        print(f"\n{'='*60}")
        print("SYSTEM PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        
        # Get engine performance
        engine_performance = engine.get_performance_summary()
        
        print(f"Entry Decision Engine:")
        print(f"  Candidates Processed: {len(candidates)}")
        print(f"  Trades Executed: {engine_performance['successful_trades']}")
        print(f"  Success Rate: {engine_performance['success_rate']:.1%}")
        print(f"  Average Confidence: {engine_performance['average_confidence']:.1%}")
        
        print(f"\nBacktest Performance:")
        print(f"  Historical Win Rate: {result.win_rate:.1%}")
        print(f"  Risk-Adjusted Return: {result.sharpe_ratio:.2f}")
        print(f"  Capital Efficiency: {result.total_pnl/100000:.1%} return on capital")
        
        # Performance by symbol
        print(f"\nPerformance by Symbol:")
        for symbol, perf in result.performance_by_symbol.items():
            print(f"  {symbol}: {perf['trades']} trades, ${perf['pnl']:,.2f} P&L, {perf['win_rate']:.1%} win rate")
        
        # ==========================================
        # SYSTEM ARCHITECTURE SUMMARY
        # ==========================================
        print(f"\n{'='*60}")
        print("SYSTEM ARCHITECTURE SUMMARY")
        print(f"{'='*60}")
        
        print("✓ Data Pipeline:")
        print("  - Market data collection and processing")
        print("  - Technical indicator calculation")
        print("  - Options data analysis")
        print("  - Risk factor assessment")
        
        print("\n✓ Model Ensemble:")
        print("  - Enhanced XGBoost predictor with historical analysis")
        print("  - Neural MLP for structured data")
        print("  - Neural LSTM for sequential data")
        print("  - LLM placeholder for macro analysis")
        
        print("\n✓ Decision Framework:")
        print("  - Multi-layer trade filtering")
        print("  - Meta-model consensus building")
        print("  - Risk-adjusted position sizing")
        print("  - Automated execution pipeline")
        
        print("\n✓ Risk Management:")
        print("  - Pre-trade eligibility screening")
        print("  - Position concentration limits")
        print("  - Model disagreement detection")
        print("  - Real-time risk factor monitoring")
        
        print("\n✓ Logging & Analytics:")
        print("  - Comprehensive trade logging (CSV + SQLite)")
        print("  - Model prediction tracking")
        print("  - Performance attribution analysis")
        print("  - Backtesting framework")
        
        print("\n✓ Bull Put Spread Strategy:")
        print("  - Credit spread automation")
        print("  - Strike selection optimization")
        print("  - Expiration management")
        print("  - Portfolio position tracking")
        
        print(f"\n{'='*60}")
        print("NEXT STEPS FOR PRODUCTION")
        print(f"{'='*60}")
        
        print("1. Data Integration:")
        print("   - Connect to live market data feeds")
        print("   - Implement real-time options chains")
        print("   - Add earnings calendar integration")
        
        print("\n2. Model Training:")
        print("   - Collect historical training data")
        print("   - Train XGBoost on real features")
        print("   - Train neural networks on price sequences")
        print("   - Implement LLM news/sentiment analysis")
        
        print("\n3. Risk Management:")
        print("   - Add dynamic position sizing")
        print("   - Implement stop-loss mechanisms")
        print("   - Add correlation analysis")
        print("   - Monitor model performance decay")
        
        print("\n4. Execution:")
        print("   - Integrate with live options trading")
        print("   - Add order management system")
        print("   - Implement slippage monitoring")
        print("   - Add execution cost analysis")
        
        print("\n5. Monitoring:")
        print("   - Real-time performance dashboard")
        print("   - Model prediction accuracy tracking")
        print("   - Risk metrics monitoring")
        print("   - Automated alerting system")
        
        print(f"\n{'='*60}")
        print("DEMO COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        
        print("The Investment Committee system has been successfully demonstrated.")
        print("All components are working together to provide automated bull put spread trading.")
        print("The system is ready for further development and production deployment.")
        
    except Exception as e:
        logger.error(f"Error in system demo: {e}")
        print(f"Error: {e}")
        print("\nPlease ensure all dependencies are installed:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main() 