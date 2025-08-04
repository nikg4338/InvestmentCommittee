#!/usr/bin/env python3
"""
Alpaca Data Collection for Investment Committee Training
========================================================

This module integrates the Alpaca API with the Committee of Five training pipeline.
It fetches market data for stocks in filtered_iex_batches.json and engineers
features relevant to bull put spread trading strategies.

Features engineered:
- Price momentum indicators
- Volatility measures
- Volume analysis
- Technical indicators
- Market context features

Target variable: Bull put spread trade eligibility based on trade_filter criteria
"""

import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import Alpaca client and trade filter
from trading.execution.alpaca_client import AlpacaClient
from trading.strategy.trade_filter import is_trade_eligible

logger = logging.getLogger(__name__)


class AlpacaDataCollector:
    """
    Collects and engineers training data from Alpaca API for the Committee of Five models.
    """
    
    def __init__(self):
        """Initialize the data collector with Alpaca client."""
        try:
            self.alpaca_client = AlpacaClient()
            logger.info("✅ Alpaca client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Alpaca client: {e}")
            raise
            
    def load_stock_batches(self, batch_file: str = "filtered_iex_batches.json") -> Dict[str, List[str]]:
        """
        Load stock batches from JSON file.
        
        Args:
            batch_file: Path to the batch file
            
        Returns:
            Dict mapping batch names to stock symbols
        """
        try:
            with open(batch_file, 'r') as f:
                data = json.load(f)
            
            batches = data.get('batches', {})
            total_symbols = sum(len(symbols) for symbols in batches.values())
            
            logger.info(f"✅ Loaded {len(batches)} batches with {total_symbols} total symbols")
            return batches
        except Exception as e:
            logger.error(f"❌ Failed to load batch file {batch_file}: {e}")
            raise
    
    def get_historical_data(self, symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
        """
        Get historical price data for a symbol.
        
        Args:
            symbol: Stock symbol
            days: Number of days of historical data
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Get historical bars
            market_data = self.alpaca_client.get_market_data(
                symbol=symbol,
                timeframe='1Day',
                limit=days
            )
            
            if not market_data or 'bars' not in market_data:
                logger.warning(f"⚠️  No market data for {symbol}")
                return None
                
            bars = market_data['bars']
            if not bars:
                logger.warning(f"⚠️  Empty bars for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'timestamp': bar['timestamp'],
                    'open': float(bar['open']),
                    'high': float(bar['high']),
                    'low': float(bar['low']),
                    'close': float(bar['close']),
                    'volume': int(bar['volume'])
                }
                for bar in bars
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to get historical data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for feature engineering.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicator columns
        """
        if len(df) < 20:
            logger.warning("⚠️  Insufficient data for technical indicators")
            return df
            
        # Price-based features
        df['price_change_1d'] = df['close'].pct_change(1)
        df['price_change_5d'] = df['close'].pct_change(5) 
        df['price_change_10d'] = df['close'].pct_change(10)
        df['price_change_20d'] = df['close'].pct_change(20)
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean() if len(df) >= 50 else np.nan
        
        # Price relative to moving averages
        df['price_vs_sma5'] = df['close'] / df['sma_5'] - 1
        df['price_vs_sma10'] = df['close'] / df['sma_10'] - 1
        df['price_vs_sma20'] = df['close'] / df['sma_20'] - 1
        
        # Volatility measures
        df['volatility_5d'] = df['price_change_1d'].rolling(5).std()
        df['volatility_10d'] = df['price_change_1d'].rolling(10).std()
        df['volatility_20d'] = df['price_change_1d'].rolling(20).std()
        
        # Volume indicators
        df['volume_sma_10'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_10']
        
        # High-Low range
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['hl_ratio_5d'] = df['hl_ratio'].rolling(5).mean()
        
        # RSI (simplified)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD (simplified)
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        bb_sma = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = bb_sma + (bb_std * 2)
        df['bb_lower'] = bb_sma - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def create_target_variable(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        """
        Create target variable based on bull put spread trade eligibility criteria.
        
        This is a simplified version - in practice you'd want to incorporate
        real-time options data, VIX, etc.
        
        Args:
            df: DataFrame with price data and indicators
            symbol: Stock symbol
            
        Returns:
            Series with binary target (1 = eligible for bull put spread, 0 = not eligible)
        """
        targets = []
        
        for i in range(len(df)):
            # Create mock ticker data for trade filter
            # In production, you'd fetch real VIX, IV rank, options data
            mock_ticker_data = {
                'ticker': symbol,
                'market_data': {
                    'vix': np.random.uniform(15, 25),  # Mock VIX - replace with real data
                    'vvix': np.random.uniform(80, 120),  # Mock VVIX
                    'spy_trend': 'up' if df.iloc[i]['price_change_10d'] > 0 else 'sideways'
                },
                'ticker_data': {
                    'avg_daily_volume': df.iloc[i]['volume'] * 10 if not pd.isna(df.iloc[i]['volume']) else 2_000_000,
                    'iv_rank': np.random.uniform(20, 80),  # Mock IV rank - replace with real data
                    'options_chain': {
                        'put_leg_1': {
                            'open_interest': np.random.randint(200, 2000),
                            'bid_ask_spread': np.random.uniform(0.02, 0.15)
                        },
                        'put_leg_2': {
                            'open_interest': np.random.randint(200, 2000), 
                            'bid_ask_spread': np.random.uniform(0.02, 0.15)
                        }
                    }
                },
                'earnings': {
                    'next_earnings_date': (datetime.now() + timedelta(days=np.random.randint(8, 60))).strftime('%Y-%m-%d')
                }
            }
            
            # Use trade filter to determine eligibility
            is_eligible = is_trade_eligible(mock_ticker_data)
            targets.append(1 if is_eligible else 0)
        
        return pd.Series(targets, index=df.index)
    
    def engineer_features_for_symbol(self, symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
        """
        Engineer complete feature set for a single symbol.
        
        Args:
            symbol: Stock symbol
            days: Days of historical data
            
        Returns:
            DataFrame with engineered features or None if failed
        """
        try:
            # Get historical data
            df = self.get_historical_data(symbol, days)
            if df is None or len(df) < 20:
                logger.warning(f"⚠️  Insufficient data for {symbol}")
                return None
            
            # Calculate technical indicators  
            df = self.calculate_technical_indicators(df)
            
            # Create target variable
            df['target'] = self.create_target_variable(df, symbol)
            
            # Add symbol column
            df['ticker'] = symbol
            
            # Select feature columns (exclude raw OHLCV, keep engineered features)
            feature_columns = [
                'ticker', 'target',
                'price_change_1d', 'price_change_5d', 'price_change_10d', 'price_change_20d',
                'price_vs_sma5', 'price_vs_sma10', 'price_vs_sma20',
                'volatility_5d', 'volatility_10d', 'volatility_20d',
                'volume_ratio', 'hl_ratio', 'hl_ratio_5d',
                'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
                'bb_position'
            ]
            
            # Filter to available columns and drop rows with NaN in key features
            available_features = [col for col in feature_columns if col in df.columns]
            result_df = df[available_features].copy()
            
            # Keep only rows with non-null target and sufficient feature data
            result_df = result_df.dropna(subset=['target'])
            result_df = result_df.dropna(thresh=len(available_features) * 0.8)  # At least 80% features non-null
            
            if len(result_df) == 0:
                logger.warning(f"⚠️  No valid rows after feature engineering for {symbol}")
                return None
                
            logger.info(f"✅ Engineered {len(result_df)} samples for {symbol}")
            return result_df
            
        except Exception as e:
            logger.error(f"❌ Failed to engineer features for {symbol}: {e}")
            return None
    
    def collect_batch_data(self, batch_name: str, symbols: List[str], max_symbols: Optional[int] = None) -> pd.DataFrame:
        """
        Collect and engineer features for a batch of symbols.
        
        Args:
            batch_name: Name of the batch
            symbols: List of symbols in the batch
            max_symbols: Optional limit on number of symbols to process
            
        Returns:
            Combined DataFrame with all symbols' data
        """
        logger.info(f"🔄 Processing batch {batch_name} with {len(symbols)} symbols...")
        
        if max_symbols:
            symbols = symbols[:max_symbols]
            logger.info(f"🔄 Limited to first {max_symbols} symbols")
        
        batch_dataframes = []
        successful_symbols = 0
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"🔄 Processing {symbol} ({i}/{len(symbols)})...")
            
            try:
                symbol_df = self.engineer_features_for_symbol(symbol)
                if symbol_df is not None and len(symbol_df) > 0:
                    batch_dataframes.append(symbol_df)
                    successful_symbols += 1
                    logger.info(f"✅ {symbol}: {len(symbol_df)} samples")
                else:
                    logger.warning(f"⚠️  {symbol}: No data")
                    
            except Exception as e:
                logger.error(f"❌ {symbol}: {e}")
                continue
        
        if not batch_dataframes:
            logger.error(f"❌ No data collected for batch {batch_name}")
            return pd.DataFrame()
        
        # Combine all symbol data
        combined_df = pd.concat(batch_dataframes, ignore_index=True)
        
        logger.info(f"✅ Batch {batch_name} complete: {successful_symbols}/{len(symbols)} symbols, {len(combined_df)} total samples")
        
        return combined_df
    
    def collect_training_data(self, batch_numbers: List[int], max_symbols_per_batch: Optional[int] = 10) -> pd.DataFrame:
        """
        Collect training data for specified batches.
        
        Args:
            batch_numbers: List of batch numbers to process
            max_symbols_per_batch: Optional limit on symbols per batch
            
        Returns:
            Combined DataFrame ready for training
        """
        logger.info(f"🚀 Starting data collection for batches: {batch_numbers}")
        
        # Load stock batches
        batches = self.load_stock_batches()
        
        all_dataframes = []
        
        for batch_num in batch_numbers:
            batch_name = f"batch_{batch_num}"
            
            if batch_name not in batches:
                logger.warning(f"⚠️  Batch {batch_name} not found in batch file")
                continue
            
            symbols = batches[batch_name]
            batch_df = self.collect_batch_data(batch_name, symbols, max_symbols_per_batch)
            
            if len(batch_df) > 0:
                all_dataframes.append(batch_df)
        
        if not all_dataframes:
            logger.error("❌ No training data collected")
            return pd.DataFrame()
        
        # Combine all batch data
        final_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Final cleanup - use newer pandas syntax
        final_df = final_df.bfill().ffill()
        final_df = final_df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Print summary
        target_dist = final_df['target'].value_counts().to_dict()
        unique_tickers = final_df['ticker'].nunique()
        
        logger.info(f"🎉 Data collection complete!")
        logger.info(f"   📊 Total samples: {len(final_df)}")
        logger.info(f"   🏢 Unique tickers: {unique_tickers}")
        logger.info(f"   🎯 Target distribution: {target_dist}")
        logger.info(f"   📋 Features: {len([col for col in final_df.columns if col not in ['ticker', 'target']])}")
        
        return final_df


def main():
    """
    Command-line interface for the AlpacaDataCollector.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect training data from Alpaca API')
    parser.add_argument('--batches', type=str, help='Comma-separated batch numbers (e.g., "1,2,3") or single batch number')
    parser.add_argument('--max-symbols', type=int, default=50, help='Maximum symbols per batch')
    parser.add_argument('--output-file', type=str, default='alpaca_training_data.csv', help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Parse batch numbers
    if args.batches:
        try:
            if ',' in args.batches:
                batch_numbers = [int(b.strip()) for b in args.batches.split(',')]
            else:
                batch_numbers = [int(args.batches)]
        except ValueError:
            logger.error("❌ Invalid batch numbers format. Use comma-separated integers (e.g., '1,2,3')")
            return
    else:
        batch_numbers = [1, 2]  # Default for backward compatibility
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize collector
        collector = AlpacaDataCollector()
        
        # Collect data for specified batches
        training_data = collector.collect_training_data(
            batch_numbers=batch_numbers,
            max_symbols_per_batch=args.max_symbols
        )
        
        if len(training_data) > 0:
            # Save to specified output file
            training_data.to_csv(args.output_file, index=False)
            logger.info(f"💾 Training data saved to {args.output_file}")
            
            # Print sample data
            print(f"\n📋 Sample of collected data:")
            print(training_data.head())
            print(f"\n📊 Data shape: {training_data.shape}")
            print(f"🎯 Target distribution: {training_data['target'].value_counts().to_dict()}")
            
        else:
            logger.error("❌ No training data collected")
    
    except Exception as e:
        logger.error(f"❌ Data collection failed: {e}")


if __name__ == "__main__":
    main()
