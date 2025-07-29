#!/usr/bin/env python3
"""
Example script demonstrating Gemini API integration
Shows how to use the GeminiAnalyzer for various financial analysis tasks
"""

import sys
import os
import asyncio
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llm_analyzer import GeminiAnalyzer, AnalysisType, create_gemini_analyzer, quick_analysis
from config.settings import validate_gemini_api_key

def main():
    """Main example function"""
    print("=== Gemini API Integration Example ===\n")
    
    # Check if API key is configured
    if not validate_gemini_api_key():
        print("❌ Gemini API key not configured!")
        print("Please set GEMINI_API_KEY in your .env file")
        return
    
    print("✅ Gemini API key is configured")
    
    try:
        # Create analyzer
        analyzer = create_gemini_analyzer()
        print(f"✅ Created Gemini analyzer with model: {analyzer.model_name}")
        
        # Show usage stats
        stats = analyzer.get_usage_stats()
        print(f"📊 Usage stats: {stats['requests_this_minute']}/{stats['minute_limit']} requests this minute")
        
        # Example 1: Macro Analysis
        print("\n1. 📈 Macro Economic Analysis")
        print("-" * 40)
        
        economic_data = {
            "gdp_growth": 2.1,
            "inflation_rate": 3.2,
            "unemployment_rate": 3.8,
            "federal_funds_rate": 5.25,
            "10_year_treasury_yield": 4.5,
            "dollar_index": 103.2,
            "recent_fed_actions": "Held rates steady in latest meeting",
            "market_sentiment": "Mixed with recession concerns"
        }
        
        result = analyzer.analyze_macro_conditions(economic_data)
        print(f"Analysis completed in {result.processing_time:.2f} seconds")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Analysis preview: {result.content[:200]}...")
        
        # Example 2: Sentiment Analysis
        print("\n2. 📰 News Sentiment Analysis")
        print("-" * 40)
        
        news_data = [
            {
                "headline": "Tech stocks rally on AI optimism",
                "source": "Financial Times",
                "sentiment": "positive",
                "timestamp": "2024-01-15T10:00:00"
            },
            {
                "headline": "Fed hints at potential rate cuts later this year",
                "source": "Wall Street Journal",
                "sentiment": "positive",
                "timestamp": "2024-01-15T11:30:00"
            },
            {
                "headline": "Geopolitical tensions rise in Middle East",
                "source": "Reuters",
                "sentiment": "negative",
                "timestamp": "2024-01-15T12:00:00"
            }
        ]
        
        result = analyzer.analyze_sentiment(news_data)
        print(f"Analysis completed in {result.processing_time:.2f} seconds")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Analysis preview: {result.content[:200]}...")
        
        # Example 3: Volatility Analysis
        print("\n3. 📊 Volatility Risk Analysis")
        print("-" * 40)
        
        volatility_data = {
            "vix_current": 16.5,
            "vix_average_30d": 18.2,
            "realized_volatility_spx": 14.8,
            "volatility_regime": "low_to_moderate",
            "volatility_term_structure": "normal_backwardation",
            "risk_metrics": {
                "max_drawdown": -0.08,
                "sharpe_ratio": 1.2,
                "var_95": -0.025
            }
        }
        
        result = analyzer.analyze_volatility_risks(volatility_data)
        print(f"Analysis completed in {result.processing_time:.2f} seconds")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Analysis preview: {result.content[:200]}...")
        
        # Example 4: Market Summary
        print("\n4. 📋 Market Summary")
        print("-" * 40)
        
        market_data = {
            "spy_price": 485.20,
            "spy_change_pct": 0.8,
            "nasdaq_change_pct": 1.2,
            "dow_change_pct": 0.5,
            "treasury_10y": 4.45,
            "dollar_index": 103.1,
            "oil_price": 72.5,
            "gold_price": 2045.0,
            "major_movers": {
                "winners": ["NVDA", "MSFT", "GOOGL"],
                "losers": ["XOM", "CVX", "PG"]
            }
        }
        
        result = analyzer.generate_market_summary(market_data)
        print(f"Analysis completed in {result.processing_time:.2f} seconds")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Analysis preview: {result.content[:200]}...")
        
        # Example 5: Quick Custom Analysis
        print("\n5. ⚡ Quick Custom Analysis")
        print("-" * 40)
        
        custom_prompt = """
        Analyze the current options market for SPY. The put/call ratio is 0.95, 
        implied volatility is 16.5%, and we're seeing increased activity in 
        weekly options. What does this suggest about market sentiment and 
        potential trading opportunities?
        """
        
        result = quick_analysis(custom_prompt, AnalysisType.RISK_ASSESSMENT)
        print(f"Analysis completed in {result.processing_time:.2f} seconds")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Analysis preview: {result.content[:200]}...")
        
        # Final usage stats
        print("\n📊 Final Usage Statistics")
        print("-" * 40)
        final_stats = analyzer.get_usage_stats()
        print(f"Requests this minute: {final_stats['requests_this_minute']}/{final_stats['minute_limit']}")
        print(f"Requests today: {final_stats['requests_this_day']}/{final_stats['day_limit']}")
        print(f"Current backoff time: {final_stats['current_backoff_time']} seconds")
        
        print("\n✅ All examples completed successfully!")
        print("\nNote: The Gemini API has rate limits for the free tier:")
        print("- 15 requests per minute")
        print("- 1,500 requests per day")
        print("- The rate limiter automatically handles these limits with exponential backoff")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your GEMINI_API_KEY is set in .env file")
        print("2. Check your internet connection")
        print("3. Verify your API key has proper permissions")
        print("4. Check if you've exceeded rate limits")

if __name__ == "__main__":
    main() 