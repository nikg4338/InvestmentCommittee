#!/usr/bin/env python3
"""
Simple LLM Trade Pipeline Demo
Demonstrates the core LLM integration for Investment Committee trade analysis

This script shows:
1. Basic LLM analysis integration
2. Macro, sentiment, and volatility analysis
3. Trade logging with LLM insights
4. Human-like risk commentary
"""

import sys
import os
from typing import Dict, List, Any
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.llm_analyzer import create_gemini_analyzer, AnalysisType
    from config.settings import validate_gemini_api_key
    from utils.trade_logger import TradeLogger
    
    # Simple mock trade execution for demonstration
    class MockTradeExecution:
        def __init__(self, trade_id, symbol, signal, confidence, reasoning, 
                     execution_status, order_details, meta_decision, llm_analysis=None):
            self.trade_id = trade_id
            self.symbol = symbol
            self.signal = signal
            self.confidence = confidence
            self.reasoning = reasoning
            self.execution_time = datetime.now().isoformat()
            self.execution_status = execution_status
            self.order_details = order_details
            self.meta_decision = meta_decision
            self.llm_analysis = llm_analysis
    
    def create_trade_candidate_data(symbol, scenario):
        """Create sample trade candidate data."""
        if scenario == "bullish":
            return {
                "symbol": symbol,
                "historical_data": {
                    "current_price": 150.0,
                    "prices": [145 + i * 0.5 for i in range(20)],
                    "volumes": [50_000_000 + i * 1_000_000 for i in range(20)]
                },
                "technicals": {
                    "rsi": 35,
                    "vix_level": 16,
                    "volatility_rank": 40,
                    "market_trend": 0.8,
                    "price_momentum": 0.6,
                    "iv_rank": 45,
                    "beta": 1.2,
                    "fed_funds_rate": 5.25,
                    "gdp_growth": 2.5,
                    "inflation_rate": 2.8,
                    "unemployment_rate": 3.7,
                    "consumer_sentiment": 65,
                    "sector_rotation": "tech_leadership",
                    "economic_cycle": "expansion"
                }
            }
        elif scenario == "bearish":
            return {
                "symbol": symbol,
                "historical_data": {
                    "current_price": 120.0,
                    "prices": [140 - i * 1.0 for i in range(20)],
                    "volumes": [30_000_000 + i * 500_000 for i in range(20)]
                },
                "technicals": {
                    "rsi": 75,
                    "vix_level": 28,
                    "volatility_rank": 85,
                    "market_trend": -0.6,
                    "price_momentum": -0.8,
                    "iv_rank": 90,
                    "beta": 1.8,
                    "fed_funds_rate": 5.25,
                    "gdp_growth": 1.8,
                    "inflation_rate": 3.5,
                    "unemployment_rate": 4.2,
                    "consumer_sentiment": 35,
                    "sector_rotation": "defensive_rotation",
                    "economic_cycle": "late_cycle"
                }
            }
        else:  # neutral
            return {
                "symbol": symbol,
                "historical_data": {
                    "current_price": 135.0,
                    "prices": [134 + (i % 3 - 1) * 0.5 for i in range(20)],
                    "volumes": [40_000_000 + i * 200_000 for i in range(20)]
                },
                "technicals": {
                    "rsi": 52,
                    "vix_level": 19,
                    "volatility_rank": 55,
                    "market_trend": 0.1,
                    "price_momentum": 0.0,
                    "iv_rank": 50,
                    "beta": 0.9,
                    "fed_funds_rate": 5.25,
                    "gdp_growth": 2.0,
                    "inflation_rate": 3.0,
                    "unemployment_rate": 3.9,
                    "consumer_sentiment": 50,
                    "sector_rotation": "balanced",
                    "economic_cycle": "mid_cycle"
                }
            }
    
    def prepare_economic_data(candidate_data):
        """Prepare economic data for LLM analysis."""
        technicals = candidate_data["technicals"]
        
        return {
            "symbol": candidate_data["symbol"],
            "market_conditions": {
                "vix_level": technicals.get('vix_level', 20),
                "market_trend": technicals.get('market_trend', 0),
                "sector_rotation": technicals.get('sector_rotation', 'neutral'),
                "economic_cycle": technicals.get('economic_cycle', 'expansion')
            },
            "interest_rates": {
                "federal_funds_rate": technicals.get('fed_funds_rate', 5.25),
                "yield_curve": 'normal',
                "rate_expectations": 'stable'
            },
            "indicators": {
                "gdp_growth": technicals.get('gdp_growth', 2.0),
                "inflation_rate": technicals.get('inflation_rate', 3.0),
                "employment_data": technicals.get('unemployment_rate', 4.0),
                "consumer_sentiment": technicals.get('consumer_sentiment', 50)
            },
            "trade_context": {
                "strategy": "bull_put_spread",
                "current_price": candidate_data["historical_data"]["current_price"],
                "implied_volatility": technicals.get('iv_rank', 50)
            }
        }
    
    def prepare_news_data(candidate_data):
        """Prepare news data for LLM analysis."""
        technicals = candidate_data["technicals"]
        symbol = candidate_data["symbol"]
        
        news_items = []
        
        # Market sentiment based on indicators
        if technicals.get('market_trend', 0) > 0.2:
            news_items.append({
                "headline": f"Market rallies as {symbol} shows strong momentum",
                "source": "Market News",
                "sentiment": "positive",
                "timestamp": datetime.now().isoformat(),
                "relevance": "high"
            })
        elif technicals.get('market_trend', 0) < -0.2:
            news_items.append({
                "headline": f"Market concerns weigh on {symbol} outlook",
                "source": "Market News",
                "sentiment": "negative",
                "timestamp": datetime.now().isoformat(),
                "relevance": "high"
            })
        
        # VIX-based sentiment
        if technicals.get('vix_level', 20) > 25:
            news_items.append({
                "headline": "Volatility spikes as market uncertainty increases",
                "source": "Financial Times",
                "sentiment": "negative",
                "timestamp": datetime.now().isoformat(),
                "relevance": "medium"
            })
        elif technicals.get('vix_level', 20) < 15:
            news_items.append({
                "headline": "Low volatility environment supports bullish sentiment",
                "source": "Wall Street Journal",
                "sentiment": "positive",
                "timestamp": datetime.now().isoformat(),
                "relevance": "medium"
            })
        
        # Add generic news
        news_items.append({
            "headline": f"Options activity increases in {symbol}",
            "source": "Options Monitor",
            "sentiment": "neutral",
            "timestamp": datetime.now().isoformat(),
            "relevance": "medium"
        })
        
        return news_items
    
    def prepare_volatility_data(candidate_data):
        """Prepare volatility data for LLM analysis."""
        technicals = candidate_data["technicals"]
        
        return {
            "symbol": candidate_data["symbol"],
            "volatility_metrics": {
                "implied_volatility": technicals.get('iv_rank', 50),
                "historical_volatility": 20,
                "vix_level": technicals.get('vix_level', 20),
                "volatility_skew": 0,
                "term_structure": 'normal'
            },
            "risk_metrics": {
                "beta": technicals.get('beta', 1.0),
                "correlation_spy": 0.7,
                "max_drawdown": -0.1,
                "sharpe_ratio": 1.0
            },
            "options_data": {
                "iv_rank": technicals.get('iv_rank', 50),
                "iv_percentile": 50,
                "put_call_ratio": 0.8,
                "options_volume": 1000000
            },
            "trade_context": {
                "strategy": "bull_put_spread",
                "position_size": "moderate",
                "risk_tolerance": "conservative"
            }
        }
    
    def prepare_market_data(candidate_data):
        """Prepare market data for LLM analysis."""
        return {
            "symbol": candidate_data["symbol"],
            "strategy": "bull_put_spread",
            "market_data": {
                "current_price": candidate_data["historical_data"]["current_price"],
                "price_trend": candidate_data["technicals"]["market_trend"],
                "volume_profile": 0,
                "support_levels": [],
                "resistance_levels": []
            },
            "technical_indicators": candidate_data["technicals"],
            "options_metrics": {
                "iv_rank": candidate_data["technicals"]["iv_rank"],
                "open_interest": 1000,
                "put_call_ratio": 0.8
            },
            "risk_assessment": {
                "market_risk": 0.5,
                "earnings_risk": 0.3,
                "volatility_risk": 0.4
            }
        }
    
    def perform_llm_analysis(analyzer, candidate_data):
        """Perform comprehensive LLM analysis."""
        print(f"\n🤖 Performing LLM Analysis for {candidate_data['symbol']}...")
        
        # Prepare data
        economic_data = prepare_economic_data(candidate_data)
        news_data = prepare_news_data(candidate_data)
        volatility_data = prepare_volatility_data(candidate_data)
        market_data = prepare_market_data(candidate_data)
        
        results = {}
        
        # 1. Macro Analysis
        print("   📊 Running macro economic analysis...")
        try:
            macro_result = analyzer.analyze_macro_conditions(economic_data)
            results['macro_analysis'] = macro_result
            print(f"   ✅ Macro analysis completed ({macro_result.processing_time:.1f}s)")
        except Exception as e:
            print(f"   ❌ Macro analysis failed: {e}")
            results['macro_analysis'] = None
        
        # 2. Sentiment Analysis
        print("   📰 Running sentiment analysis...")
        try:
            sentiment_result = analyzer.analyze_sentiment(news_data)
            results['sentiment_analysis'] = sentiment_result
            print(f"   ✅ Sentiment analysis completed ({sentiment_result.processing_time:.1f}s)")
        except Exception as e:
            print(f"   ❌ Sentiment analysis failed: {e}")
            results['sentiment_analysis'] = None
        
        # 3. Volatility Analysis
        print("   📈 Running volatility analysis...")
        try:
            volatility_result = analyzer.analyze_volatility_risks(volatility_data)
            results['volatility_analysis'] = volatility_result
            print(f"   ✅ Volatility analysis completed ({volatility_result.processing_time:.1f}s)")
        except Exception as e:
            print(f"   ❌ Volatility analysis failed: {e}")
            results['volatility_analysis'] = None
        
        # 4. Market Summary
        print("   📋 Running market summary...")
        try:
            market_result = analyzer.generate_market_summary(market_data)
            results['market_summary'] = market_result
            print(f"   ✅ Market summary completed ({market_result.processing_time:.1f}s)")
        except Exception as e:
            print(f"   ❌ Market summary failed: {e}")
            results['market_summary'] = None
        
        return results
    
    def display_analysis_results(symbol, results):
        """Display LLM analysis results."""
        print(f"\n{'='*60}")
        print(f"🤖 LLM ANALYSIS RESULTS FOR {symbol}")
        print(f"{'='*60}")
        
        # Macro Analysis
        if results.get('macro_analysis'):
            result = results['macro_analysis']
            print(f"\n📊 MACRO ANALYSIS (Confidence: {result.confidence:.0%}):")
            print(f"   📝 Preview: {result.content[:300]}...")
        
        # Sentiment Analysis
        if results.get('sentiment_analysis'):
            result = results['sentiment_analysis']
            print(f"\n📰 SENTIMENT ANALYSIS (Confidence: {result.confidence:.0%}):")
            print(f"   📝 Preview: {result.content[:300]}...")
        
        # Volatility Analysis
        if results.get('volatility_analysis'):
            result = results['volatility_analysis']
            print(f"\n📈 VOLATILITY ANALYSIS (Confidence: {result.confidence:.0%}):")
            print(f"   📝 Preview: {result.content[:300]}...")
        
        # Market Summary
        if results.get('market_summary'):
            result = results['market_summary']
            print(f"\n📋 MARKET SUMMARY (Confidence: {result.confidence:.0%}):")
            print(f"   📝 Preview: {result.content[:300]}...")
    
    def create_mock_trade_execution(symbol, llm_results):
        """Create a mock trade execution with LLM analysis."""
        
        # Synthesize LLM recommendation
        recommendations = []
        if llm_results.get('macro_analysis'):
            content = llm_results['macro_analysis'].content.lower()
            if 'bullish' in content or 'positive' in content:
                recommendations.append('BULLISH')
            elif 'bearish' in content or 'negative' in content:
                recommendations.append('BEARISH')
        
        if llm_results.get('sentiment_analysis'):
            content = llm_results['sentiment_analysis'].content.lower()
            if 'bullish' in content or 'positive' in content:
                recommendations.append('BULLISH')
            elif 'bearish' in content or 'negative' in content:
                recommendations.append('BEARISH')
        
        # Determine overall recommendation
        if recommendations.count('BULLISH') > recommendations.count('BEARISH'):
            signal = 'BUY'
            confidence = 0.8
        elif recommendations.count('BEARISH') > recommendations.count('BULLISH'):
            signal = 'SELL'
            confidence = 0.7
        else:
            signal = 'HOLD'
            confidence = 0.5
        
        # Create LLM analysis summary
        llm_analysis = {
            'overall_recommendation': 'BULLISH' if signal == 'BUY' else 'BEARISH' if signal == 'SELL' else 'NEUTRAL',
            'risk_score': 0.3 if signal == 'BUY' else 0.7 if signal == 'SELL' else 0.5,
            'confidence_score': confidence,
            'analysis_time': sum(r.processing_time for r in llm_results.values() if r),
            'macro_analysis': llm_results.get('macro_analysis', {}).content[:500] if llm_results.get('macro_analysis') else None,
            'sentiment_analysis': llm_results.get('sentiment_analysis', {}).content[:500] if llm_results.get('sentiment_analysis') else None,
            'volatility_analysis': llm_results.get('volatility_analysis', {}).content[:500] if llm_results.get('volatility_analysis') else None,
            'market_summary': llm_results.get('market_summary', {}).content[:500] if llm_results.get('market_summary') else None,
            'error_messages': []
        }
        
        # Create trade execution
        trade_id = f"BPS_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return MockTradeExecution(
            trade_id=trade_id,
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            reasoning=[f"LLM analysis: {llm_analysis['overall_recommendation']}", f"Risk score: {llm_analysis['risk_score']:.2f}"],
            execution_status='FILLED',
            order_details={
                'strategy': 'bull_put_spread',
                'contracts': 5,
                'credit_received': 250,
                'max_loss': 2250,
                'llm_risk_adjustment': llm_analysis['risk_score']
            },
            meta_decision={'llm_enabled': True, 'llm_weight': 0.3},
            llm_analysis=llm_analysis
        )
    
    def main():
        """Main demonstration function."""
        print("="*80)
        print("🚀 SIMPLE LLM TRADE PIPELINE DEMO")
        print("="*80)
        
        # Check if Gemini API is configured
        if not validate_gemini_api_key():
            print("❌ Gemini API key not configured!")
            print("Please set GEMINI_API_KEY in your .env file")
            return
        
        print("✅ Gemini API key is configured")
        
        # Initialize LLM analyzer
        print("\n🤖 Initializing Gemini LLM Analyzer...")
        try:
            analyzer = create_gemini_analyzer()
            print("✅ LLM analyzer initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize LLM analyzer: {e}")
            return
        
        # Show usage stats
        usage_stats = analyzer.get_usage_stats()
        print(f"📊 Usage: {usage_stats['requests_this_minute']}/{usage_stats['minute_limit']} requests this minute")
        
        # Initialize trade logger
        print("\n📝 Initializing Trade Logger...")
        trade_logger = TradeLogger()
        print("✅ Trade logger initialized")
        
        # Create sample trade candidates
        candidates = [
            ("AAPL", "bullish"),
            ("TSLA", "bearish"),
            ("MSFT", "neutral")
        ]
        
        print(f"\n📈 Processing {len(candidates)} trade candidates...")
        
        for symbol, scenario in candidates:
            print(f"\n{'-'*60}")
            print(f"PROCESSING {symbol} ({scenario.upper()} SCENARIO)")
            print(f"{'-'*60}")
            
            # Create candidate data
            candidate_data = create_trade_candidate_data(symbol, scenario)
            
            # Perform LLM analysis
            llm_results = perform_llm_analysis(analyzer, candidate_data)
            
            # Display results
            display_analysis_results(symbol, llm_results)
            
            # Create mock trade execution
            trade_execution = create_mock_trade_execution(symbol, llm_results)
            
            # Log trade
            trade_logger.log_trade(trade_execution)
            
            print(f"\n✅ Trade logged: {trade_execution.trade_id}")
            print(f"   📊 Signal: {trade_execution.signal}")
            print(f"   🎯 Confidence: {trade_execution.confidence:.0%}")
            print(f"   🤖 LLM Recommendation: {trade_execution.llm_analysis['overall_recommendation']}")
            print(f"   ⚠️  Risk Score: {trade_execution.llm_analysis['risk_score']:.2f}")
        
        # Show final statistics
        print(f"\n{'='*80}")
        print("📊 FINAL STATISTICS")
        print(f"{'='*80}")
        
        # LLM usage stats
        final_usage = analyzer.get_usage_stats()
        print(f"🤖 LLM Usage:")
        print(f"   📈 Requests this minute: {final_usage['requests_this_minute']}/{final_usage['minute_limit']}")
        print(f"   📊 Requests today: {final_usage['requests_this_day']}/{final_usage['day_limit']}")
        
        # Trade logger stats
        trade_stats = trade_logger.get_performance_stats()
        print(f"\n📝 Trade Statistics:")
        print(f"   📈 Total trades: {trade_stats['total_trades']}")
        print(f"   📊 Success rate: {trade_stats['success_rate']:.1%}")
        print(f"   🎯 Average confidence: {trade_stats['average_confidence']:.1%}")
        
        print(f"\n🎉 LLM Trade Pipeline Demo Complete!")
        print("✅ All trades logged with comprehensive LLM analysis")
        print("📊 View trade logs in: logs/trades.db and logs/trades.csv")
        print("="*80)
    
    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all required modules are available")
    print("Required: models.llm_analyzer, config.settings, utils.trade_logger") 