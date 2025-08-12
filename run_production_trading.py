#!/usr/bin/env python3
"""
Production Trading System Runner
===============================

Main script to run the production trading system with optimized models and LLM integration.
"""

import os
import sys
import logging
import argparse
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import production components
from config.production_config import ProductionConfig, get_production_config, load_production_config
from trading.production_trading_engine import EnhancedTradingEngine, create_production_trading_engine
from trading.execution.alpaca_client import AlpacaClient
from models.enhanced_llm_analyzer import GeminiAnalyzer, create_gemini_analyzer

# Set up logging
def setup_logging(config: ProductionConfig):
    """Set up logging configuration."""
    log_level = getattr(logging, config.log_level.upper())
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(config.log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {config.log_level}, File: {config.log_file}")
    return logger


def validate_environment(config: ProductionConfig) -> bool:
    """Validate environment and configuration."""
    logger = logging.getLogger(__name__)
    
    logger.info("Validating environment...")
    
    # Validate configuration
    issues = config.validate_config()
    if issues:
        logger.error("Configuration validation failed:")
        for issue in issues:
            logger.error(f"  • {issue}")
        return False
    
    logger.info("Configuration is valid")
    
    # Test API connections
    try:
        # Test Alpaca connection
        alpaca_client = AlpacaClient()
        account_info = alpaca_client.get_account_info()
        logger.info(f"Alpaca connection successful - Account: {account_info.get('id', 'Unknown')}")
        
        # Test Gemini connection if enabled
        if config.llm_config.enable_llm:
            try:
                gemini_analyzer = create_gemini_analyzer(api_key=config.llm_config.api_key)
                logger.info("Gemini LLM connection successful")
            except Exception as e:
                logger.warning(f"Gemini LLM connection failed: {e}")
        
    except Exception as e:
        logger.error(f"API connection failed: {e}")
        return False
    
    return True


def run_trading_session(config: ProductionConfig, 
                       symbols: List[str] = None,
                       dry_run: bool = False) -> Dict[str, Any]:
    """Run a complete trading session."""
    logger = logging.getLogger(__name__)
    
    session_start = time.time()
    logger.info(f"Starting trading session - Environment: {config.environment}")
    
    # Use default trading universe if no symbols provided
    if symbols is None:
        symbols = config.trading_config.trading_universe
    
    logger.info(f"Processing {len(symbols)} symbols: {symbols[:5]}..." + 
                (f" and {len(symbols) - 5} more" if len(symbols) > 5 else ""))
    
    try:
        # Initialize trading engine
        trading_engine = create_production_trading_engine(
            gemini_api_key=config.llm_config.api_key,
            paper_trading=config.trading_config.paper_trading
        )
        
        # Update market context
        logger.info("Updating market context...")
        market_context = trading_engine.update_market_context()
        logger.info(f"Market sentiment: {market_context.market_sentiment}, "
                   f"SPY change: {market_context.spy_change:.2f}%, "
                   f"VIX: {market_context.vix_level:.1f}")
        
        # Generate trading signals
        logger.info("Generating trading signals...")
        signals = trading_engine.process_trading_universe(symbols)
        
        # Filter signals by confidence
        min_confidence = config.trading_config.min_signal_confidence
        high_confidence_signals = [s for s in signals if s.confidence >= min_confidence]
        
        logger.info(f"Generated {len(signals)} signals, {len(high_confidence_signals)} above confidence threshold")
        
        # Log top signals
        buy_signals = [s for s in high_confidence_signals if s.action == "BUY"]
        sell_signals = [s for s in high_confidence_signals if s.action == "SELL"]
        
        logger.info(f"BUY signals: {len(buy_signals)}, SELL signals: {len(sell_signals)}")
        
        if buy_signals:
            buy_signals.sort(key=lambda x: x.signal_strength * x.confidence, reverse=True)
            logger.info("Top BUY signals:")
            for signal in buy_signals[:5]:
                logger.info(f"  {signal.symbol}: {signal.signal_strength:.3f} strength, "
                           f"{signal.confidence:.3f} confidence, "
                           f"LLM: {signal.llm_analysis.recommendation if signal.llm_analysis else 'N/A'}")
        
        # Execute trades if not in dry run mode
        execution_results = {"executed_trades": [], "failed_trades": [], "skipped_trades": []}
        
        if not dry_run:
            logger.info("Executing trades...")
            execution_results = trading_engine.execute_trading_signals(high_confidence_signals)
            
            logger.info(f"Executed: {len(execution_results['executed_trades'])}")
            logger.info(f"Failed: {len(execution_results['failed_trades'])}")
            
        else:
            logger.info("Dry run mode - No trades executed")
            execution_results["skipped_trades"] = [s.symbol for s in high_confidence_signals]
        
        # Session summary
        session_time = time.time() - session_start
        
        session_summary = {
            "session_start": datetime.now().isoformat(),
            "session_duration": session_time,
            "symbols_processed": len(symbols),
            "signals_generated": len(signals),
            "high_confidence_signals": len(high_confidence_signals),
            "buy_signals": len(buy_signals),
            "sell_signals": len(sell_signals),
            "trades_executed": len(execution_results["executed_trades"]),
            "trades_failed": len(execution_results["failed_trades"]),
            "market_context": {
                "sentiment": market_context.market_sentiment,
                "spy_change": market_context.spy_change,
                "vix_level": market_context.vix_level
            },
            "top_signals": [
                {
                    "symbol": s.symbol,
                    "action": s.action,
                    "strength": s.signal_strength,
                    "confidence": s.confidence,
                    "llm_recommendation": s.llm_analysis.recommendation if s.llm_analysis else None
                }
                for s in (buy_signals + sell_signals)[:10]
            ],
            "execution_results": execution_results
        }
        
        logger.info(f"Trading session completed in {session_time:.1f} seconds")
        
        return session_summary
        
    except Exception as e:
        logger.error(f"Trading session failed: {e}")
        raise


def save_session_results(session_summary: Dict[str, Any], 
                        config: ProductionConfig):
    """Save session results to file."""
    logger = logging.getLogger(__name__)
    
    try:
        # Create results directory
        results_dir = "results/trading_sessions"
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{results_dir}/session_{timestamp}.json"
        
        # Save results
        with open(results_file, 'w') as f:
            json.dump(session_summary, f, indent=2, default=str)
        
        logger.info(f"Session results saved to {results_file}")
        
        # Also save to latest results file
        latest_file = f"{results_dir}/latest_session.json"
        with open(latest_file, 'w') as f:
            json.dump(session_summary, f, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Failed to save session results: {e}")


def main():
    """Main entry point for production trading system."""
    
    parser = argparse.ArgumentParser(description="Production Trading System")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--symbols", type=str, nargs="+", help="Symbols to trade (overrides config)")
    parser.add_argument("--dry-run", action="store_true", help="Run without executing trades")
    parser.add_argument("--validate-only", action="store_true", help="Only validate configuration")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_production_config(args.config)
    else:
        config = get_production_config()
    
    # Override log level if specified
    config.log_level = args.log_level
    
    # Set up logging
    logger = setup_logging(config)
    
    logger.info("=" * 80)
    logger.info("PRODUCTION TRADING SYSTEM STARTING")
    logger.info("=" * 80)
    
    # Show configuration summary
    summary = config.get_summary()
    logger.info("Configuration Summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
    
    try:
        # Validate environment
        if not validate_environment(config):
            logger.error("Environment validation failed")
            sys.exit(1)
        
        if args.validate_only:
            logger.info("Validation completed successfully")
            sys.exit(0)
        
        # Run trading session
        symbols = args.symbols if args.symbols else None
        session_summary = run_trading_session(config, symbols, args.dry_run)
        
        # Save results
        save_session_results(session_summary, config)
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("SESSION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Duration: {session_summary['session_duration']:.1f} seconds")
        logger.info(f"Symbols processed: {session_summary['symbols_processed']}")
        logger.info(f"Signals generated: {session_summary['signals_generated']}")
        logger.info(f"High confidence signals: {session_summary['high_confidence_signals']}")
        logger.info(f"Trades executed: {session_summary['trades_executed']}")
        logger.info(f"Market sentiment: {session_summary['market_context']['sentiment']}")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("Trading session interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
