#!/usr/bin/env python3
"""
LLM-Enhanced Trade Pipeline Demo
Demonstrates the complete integration of Gemini API into the Investment Committee trade pipeline

This script shows:
1. How LLM analysis is integrated into the trade decision process
2. Comprehensive macro, sentiment, and volatility analysis
3. Enhanced trade logging with LLM insights
4. Risk-adjusted position sizing based on LLM analysis
5. Complete trade execution with human-like risk commentary
"""

import sys
import os
from typing import Dict, List, Any
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.entry_decision_engine import EntryDecisionEngine, create_trade_candidate
from config.settings import validate_gemini_api_key
from utils.trade_logger import TradeLogger


def create_sample_trade_candidates() -> List[Dict[str, Any]]:
    """Create sample trade candidates with varying market conditions."""
    
    candidates_data = [
        {
            "symbol": "AAPL",
            "scenario": "Strong Bullish",
            "historical_data": {
                "prices": [150 + i * 1.2 for i in range(20)],
                "volumes": [60_000_000 + i * 2_000_000 for i in range(20)],
                "highs": [151 + i * 1.2 for i in range(20)],
                "lows": [149 + i * 1.2 for i in range(20)],
                "current_price": 174.0
            },
            "technicals": {
                "rsi": 35,
                "vix_level": 16,
                "volatility_rank": 40,
                "market_trend": 0.8,
                "price_momentum": 0.6,
                "iv_rank": 45,
                "beta": 1.2,
                "spy_correlation": 0.8,
                "max_drawdown": -0.05,
                "sharpe_ratio": 1.5,
                "fed_funds_rate": 5.25,
                "gdp_growth": 2.5,
                "inflation_rate": 2.8,
                "unemployment_rate": 3.7,
                "consumer_sentiment": 65,
                "sector_rotation": "tech_leadership",
                "economic_cycle": "expansion"
            }
        },
        {
            "symbol": "TSLA",
            "scenario": "High Volatility Bearish",
            "historical_data": {
                "prices": [200 - i * 2.5 for i in range(20)],
                "volumes": [40_000_000 + i * 1_000_000 for i in range(20)],
                "highs": [202 - i * 2.5 for i in range(20)],
                "lows": [198 - i * 2.5 for i in range(20)],
                "current_price": 150.0
            },
            "technicals": {
                "rsi": 75,
                "vix_level": 28,
                "volatility_rank": 85,
                "market_trend": -0.6,
                "price_momentum": -0.8,
                "iv_rank": 90,
                "beta": 1.8,
                "spy_correlation": 0.6,
                "max_drawdown": -0.25,
                "sharpe_ratio": 0.3,
                "fed_funds_rate": 5.25,
                "gdp_growth": 1.8,
                "inflation_rate": 3.5,
                "unemployment_rate": 4.2,
                "consumer_sentiment": 35,
                "sector_rotation": "defensive_rotation",
                "economic_cycle": "late_cycle"
            }
        },
        {
            "symbol": "MSFT",
            "scenario": "Moderate Neutral",
            "historical_data": {
                "prices": [300 + (i % 5 - 2) * 0.8 for i in range(20)],
                "volumes": [30_000_000 + i * 500_000 for i in range(20)],
                "highs": [301 + (i % 5 - 2) * 0.8 for i in range(20)],
                "lows": [299 + (i % 5 - 2) * 0.8 for i in range(20)],
                "current_price": 298.5
            },
            "technicals": {
                "rsi": 52,
                "vix_level": 19,
                "volatility_rank": 55,
                "market_trend": 0.1,
                "price_momentum": 0.0,
                "iv_rank": 50,
                "beta": 0.9,
                "spy_correlation": 0.7,
                "max_drawdown": -0.08,
                "sharpe_ratio": 1.1,
                "fed_funds_rate": 5.25,
                "gdp_growth": 2.0,
                "inflation_rate": 3.0,
                "unemployment_rate": 3.9,
                "consumer_sentiment": 50,
                "sector_rotation": "balanced",
                "economic_cycle": "mid_cycle"
            }
        }
    ]
    
    return candidates_data


def display_llm_analysis_summary(candidate, llm_analysis):
    """Display a summary of LLM analysis for a trade candidate."""
    print(f"\n{'='*60}")
    print(f"LLM ANALYSIS SUMMARY FOR {candidate.symbol}")
    print(f"{'='*60}")
    
    if not llm_analysis:
        print("❌ No LLM analysis available")
        return
    
    print(f"📊 Overall Recommendation: {llm_analysis.overall_recommendation}")
    print(f"⚠️  Risk Score: {llm_analysis.risk_score:.2f}")
    print(f"🎯 Confidence Score: {llm_analysis.confidence_score:.2f}")
    print(f"⏱️  Analysis Time: {llm_analysis.analysis_time:.1f} seconds")
    
    if llm_analysis.error_messages:
        print(f"❌ Errors: {len(llm_analysis.error_messages)}")
        for error in llm_analysis.error_messages:
            print(f"   - {error}")
    
    # Macro Analysis
    if llm_analysis.macro_analysis:
        print(f"\n📈 MACRO ANALYSIS:")
        print(f"   Confidence: {llm_analysis.macro_analysis.confidence:.0%}")
        print(f"   Preview: {llm_analysis.macro_analysis.content[:200]}...")
    
    # Sentiment Analysis
    if llm_analysis.sentiment_analysis:
        print(f"\n📰 SENTIMENT ANALYSIS:")
        print(f"   Confidence: {llm_analysis.sentiment_analysis.confidence:.0%}")
        print(f"   Preview: {llm_analysis.sentiment_analysis.content[:200]}...")
    
    # Volatility Analysis
    if llm_analysis.volatility_analysis:
        print(f"\n📊 VOLATILITY ANALYSIS:")
        print(f"   Confidence: {llm_analysis.volatility_analysis.confidence:.0%}")
        print(f"   Preview: {llm_analysis.volatility_analysis.content[:200]}...")
    
    # Trade Summary
    if llm_analysis.trade_summary:
        print(f"\n📋 TRADE SUMMARY:")
        print(f"   Confidence: {llm_analysis.trade_summary.confidence:.0%}")
        print(f"   Preview: {llm_analysis.trade_summary.content[:200]}...")


def display_trade_execution_summary(execution):
    """Display a summary of trade execution with LLM insights."""
    print(f"\n{'='*60}")
    print(f"TRADE EXECUTION SUMMARY")
    print(f"{'='*60}")
    
    print(f"🎯 Trade ID: {execution.trade_id}")
    print(f"📈 Symbol: {execution.symbol}")
    print(f"🚦 Signal: {execution.signal}")
    print(f"📊 Confidence: {execution.confidence:.0%}")
    print(f"⏱️  Execution Time: {execution.execution_time}")
    print(f"✅ Status: {execution.execution_status}")
    
    if execution.reasoning:
        print(f"\n🧠 Reasoning:")
        for reason in execution.reasoning:
            print(f"   - {reason}")
    
    # Display LLM Analysis Impact
    if execution.llm_analysis:
        print(f"\n🤖 LLM ANALYSIS IMPACT:")
        llm_data = execution.llm_analysis
        print(f"   📊 Recommendation: {llm_data.get('overall_recommendation', 'N/A')}")
        print(f"   ⚠️  Risk Score: {llm_data.get('risk_score', 'N/A')}")
        print(f"   🎯 Confidence: {llm_data.get('confidence_score', 'N/A')}")
        print(f"   ⏱️  Analysis Time: {llm_data.get('analysis_time', 'N/A')} seconds")
        
        if llm_data.get('error_messages'):
            print(f"   ❌ Errors: {len(llm_data['error_messages'])}")
    
    # Display Order Details
    if execution.order_details:
        print(f"\n📝 ORDER DETAILS:")
        order_data = execution.order_details
        print(f"   💰 Strategy: {order_data.get('strategy', 'N/A')}")
        print(f"   📊 Contracts: {order_data.get('contracts', 'N/A')}")
        print(f"   💵 Credit Received: ${order_data.get('credit_received', 'N/A')}")
        print(f"   ⚠️  Max Loss: ${order_data.get('max_loss', 'N/A')}")
        
        if 'llm_risk_adjustment' in order_data:
            print(f"   🤖 LLM Risk Adjustment: {order_data['llm_risk_adjustment']:.2f}")
            print(f"   📈 Adjusted Risk %: {order_data['adjusted_risk_percent']:.1%}")


def main():
    """Main demonstration function."""
    print("="*80)
    print("🚀 LLM-ENHANCED TRADE PIPELINE DEMONSTRATION")
    print("="*80)
    
    # Check if Gemini API is configured
    if not validate_gemini_api_key():
        print("❌ Gemini API key not configured!")
        print("Please set GEMINI_API_KEY in your .env file")
        return
    
    print("✅ Gemini API key is configured")
    
    # Initialize the LLM-enhanced entry decision engine
    print("\n📊 Initializing LLM-Enhanced Entry Decision Engine...")
    engine = EntryDecisionEngine(
        paper_trading=True,
        max_positions=5,
        enable_llm_analysis=True
    )
    
    # Check LLM availability
    if not engine.llm_available:
        print("❌ LLM analyzer is not available")
        return
    
    print("✅ LLM-Enhanced Entry Decision Engine initialized")
    
    # Show initial LLM usage stats
    llm_stats = engine.get_llm_usage_stats()
    print(f"📊 Initial LLM Usage: {llm_stats['requests_this_minute']}/{llm_stats['minute_limit']} requests this minute")
    
    # Create sample trade candidates
    print("\n📈 Creating Sample Trade Candidates...")
    candidates_data = create_sample_trade_candidates()
    
    trade_candidates = []
    for candidate_data in candidates_data:
        candidate = create_trade_candidate(
            candidate_data["symbol"],
            candidate_data["historical_data"],
            candidate_data["technicals"]
        )
        trade_candidates.append(candidate)
        print(f"✅ Created candidate: {candidate.symbol} ({candidate_data['scenario']})")
    
    # Process candidates through the LLM-enhanced pipeline
    print(f"\n🔄 Processing {len(trade_candidates)} candidates through LLM-Enhanced Pipeline...")
    print("This will perform comprehensive analysis for each candidate:")
    print("   📊 Macro Economic Analysis")
    print("   📰 News Sentiment Analysis")
    print("   📈 Volatility Risk Analysis")
    print("   📋 Market Summary & Recommendations")
    print("   🎯 Risk-Adjusted Position Sizing")
    
    executions = engine.process_trade_candidates(trade_candidates)
    
    # Display detailed results
    print(f"\n🎯 PIPELINE RESULTS: {len(executions)} executions generated")
    
    for i, execution in enumerate(executions):
        print(f"\n{'-'*60}")
        print(f"EXECUTION {i+1}/{len(executions)}")
        print(f"{'-'*60}")
        
        # Find the corresponding candidate for LLM analysis display
        candidate = next((c for c in trade_candidates if c.symbol == execution.symbol), None)
        
        if candidate and candidate.llm_analysis:
            display_llm_analysis_summary(candidate, candidate.llm_analysis)
        
        display_trade_execution_summary(execution)
    
    # Show final performance summary
    print(f"\n{'='*80}")
    print("📊 FINAL PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    performance = engine.get_performance_summary()
    print(f"📈 Total Trades: {performance['total_trades']}")
    print(f"✅ Successful Trades: {performance['successful_trades']}")
    print(f"📊 Success Rate: {performance['success_rate']:.1%}")
    print(f"🎯 Average Confidence: {performance['average_confidence']:.1%}")
    print(f"💼 Active Positions: {performance['active_positions']}")
    
    # LLM-specific statistics
    print(f"\n🤖 LLM ANALYSIS STATISTICS:")
    print(f"   ✅ LLM Enabled: {performance['llm_enabled']}")
    print(f"   📊 LLM Success Rate: {performance['llm_success_rate']:.1%}")
    print(f"   ⚠️  Average Risk Score: {performance['average_risk_score']:.2f}")
    
    # Final LLM usage stats
    final_llm_stats = performance['llm_usage_stats']
    if final_llm_stats.get('llm_available', False):
        print(f"   📈 Requests this minute: {final_llm_stats['requests_this_minute']}/{final_llm_stats['minute_limit']}")
        print(f"   📊 Requests today: {final_llm_stats['requests_this_day']}/{final_llm_stats['day_limit']}")
        print(f"   ⏱️  Current backoff: {final_llm_stats['current_backoff_time']} seconds")
    
    # Show trade log information
    print(f"\n📝 TRADE LOGGING:")
    print(f"   ✅ All trades logged with LLM analysis")
    print(f"   📊 Database: trades.db")
    print(f"   📄 CSV: trades.csv")
    print(f"   🔍 View logs in: logs/ directory")
    
    print(f"\n🎉 LLM-Enhanced Trade Pipeline Demo Complete!")
    print("="*80)


if __name__ == "__main__":
    main() 