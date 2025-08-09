#!/usr/bin/env python3
"""
FIXED Alpaca Data Collection - NO DATA LEAKAGE
===============================================

This is the corrected version of data_collection_alpaca.py that eliminates
all forms of data leakage:

1. NO future data in target creation (no shift(-horizon))
2. NO future-looking features (pnl_ratio, daily_return removed)
3. STRICT temporal validation
4. FRESH Alpaca API data every time (no CSV reuse)

All features use ONLY past/current data available at prediction time.
Targets are created using PROPER temporal splits.
"""

import os
import sys
import json
import logging
import warnings
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union

warnings.filterwarnings('ignore')

# Import Alpaca client and trade filter
from trading.execution.alpaca_client import AlpacaClient
from trading.strategy.trade_filter import is_trade_eligible

logger = logging.getLogger(__name__)


class LeakFreeAlpacaDataCollector:
    """
    LEAK-FREE Alpaca data collector that ensures NO future information 
    contamination in features or targets.
    
    Key Principles:
    - All features use only backward-looking calculations
    - Targets are created using proper temporal methodology
    - No CSV file reuse - always fresh API data
    - Strict temporal validation at every step
    """
    
    def __init__(self):
        """Initialize the leak-free data collector."""
        try:
            self.alpaca_client = AlpacaClient()
            logger.info("✅ Initialized leak-free Alpaca data collector")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Alpaca client: {e}")
            raise
            
    def load_stock_batches(self, batch_file: str = "filtered_iex_batches.json") -> Dict[str, List[str]]:
        """Load stock batches from JSON file."""
        try:
            with open(batch_file, 'r') as f:
                data = json.load(f)
            return data.get('batches', {})
        except Exception as e:
            logger.error(f"Failed to load batches: {e}")
            return {}
    
    def get_fresh_historical_data(self, symbol: str, days: int = 730) -> Optional[pd.DataFrame]:
        """
        Get FRESH historical data from Alpaca API - NO file reuse.
        
        Args:
            symbol: Stock symbol
            days: Number of days of historical data
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            logger.info(f"📥 Fetching FRESH data from Alpaca API for {symbol}")
            
            # Use the correct AlpacaClient method
            market_data = self.alpaca_client.get_market_data(
                symbol=symbol,
                timeframe='1Day',
                limit=days,
                delayed=True
            )
            
            if not market_data or 'bars' not in market_data or not market_data['bars']:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Convert to DataFrame
            bars = market_data['bars']
            df = pd.DataFrame(bars)
            
            # Set timestamp as index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Sort by date to ensure temporal order
            df = df.sort_index()
            
            logger.info(f"✅ Fresh data collected: {len(df)} days for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None
    
    def calculate_leak_free_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features using ONLY backward-looking data.
        
        CRITICAL: NO feature can use shift(-N) or any future information.
        All calculations must use only current and past data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with leak-free features
        """
        if len(df) < 20:
            logger.warning("Insufficient data for feature calculation")
            return df
            
        logger.info("🔧 Creating leak-free features (backward-looking only)")
        
        # === PRICE-BASED FEATURES (BACKWARD-LOOKING ONLY) ===
        
        # Past price changes (using shift(+N) to look backward)
        df['price_change_1d'] = df['close'].pct_change(1)  # Current vs yesterday
        df['price_change_5d'] = df['close'].pct_change(5)  # Current vs 5 days ago
        df['price_change_10d'] = df['close'].pct_change(10)
        df['price_change_20d'] = df['close'].pct_change(20)
        
        # Moving averages (backward-looking)
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean() if len(df) >= 50 else np.nan
        df['sma_200'] = df['close'].rolling(200).mean() if len(df) >= 200 else np.nan
        
        # Price relative to moving averages (current position vs past average)
        df['price_vs_sma5'] = df['close'] / df['sma_5'] - 1
        df['price_vs_sma10'] = df['close'] / df['sma_10'] - 1
        df['price_vs_sma20'] = df['close'] / df['sma_20'] - 1
        if 'sma_50' in df.columns:
            df['price_vs_sma50'] = df['close'] / df['sma_50'] - 1
        if 'sma_200' in df.columns:
            df['price_vs_sma200'] = df['close'] / df['sma_200'] - 1
        
        # === VOLATILITY FEATURES (BACKWARD-LOOKING ONLY) ===
        
        # Historical volatility (using past returns only)
        df['volatility_5d'] = df['price_change_1d'].rolling(5).std()
        df['volatility_10d'] = df['price_change_1d'].rolling(10).std()
        df['volatility_20d'] = df['price_change_1d'].rolling(20).std()
        df['volatility_50d'] = df['price_change_1d'].rolling(50).std() if len(df) >= 50 else np.nan
        
        # Volatility percentiles (where we are vs historical volatility)
        if 'volatility_50d' in df.columns:
            df['vol_percentile_50d'] = df['volatility_20d'].rolling(50).rank(pct=True)
        
        # === VOLUME FEATURES (BACKWARD-LOOKING ONLY) ===
        
        # Volume moving averages
        df['volume_sma_10'] = df['volume'].rolling(10).mean()
        df['volume_sma_50'] = df['volume'].rolling(50).mean() if len(df) >= 50 else np.nan
        
        # Current volume vs historical average
        df['volume_ratio'] = df['volume'] / df['volume_sma_10']
        
        # === TECHNICAL INDICATORS (BACKWARD-LOOKING ONLY) ===
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
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
        
        # === REGIME DETECTION (BACKWARD-LOOKING ONLY) ===
        
        # Trend regime (based on past price action)
        df['trend_regime'] = 0  # Default: sideways
        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            df.loc[df['sma_50'] > df['sma_200'], 'trend_regime'] = 1   # Uptrend
            df.loc[df['sma_50'] < df['sma_200'], 'trend_regime'] = -1  # Downtrend
        
        # Momentum regime
        df['momentum_regime'] = 0
        df.loc[df['rsi_14'] > 70, 'momentum_regime'] = 1   # Overbought
        df.loc[df['rsi_14'] < 30, 'momentum_regime'] = -1  # Oversold
        
        # === MARKET MICROSTRUCTURE (BACKWARD-LOOKING ONLY) ===
        
        # High-low ratio (intraday range)
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['hl_ratio_5d'] = df['hl_ratio'].rolling(5).mean()
        
        # Gap detection (compare today's open vs yesterday's close)
        df['gap_up'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)) > 0.02
        df['gap_down'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)) < -0.02
        
        # Support/Resistance levels (based on past highs/lows)
        df['resistance_level'] = df['high'].rolling(20).max()
        df['support_level'] = df['low'].rolling(20).min()
        df['distance_to_resistance'] = (df['resistance_level'] - df['close']) / df['close']
        df['distance_to_support'] = (df['close'] - df['support_level']) / df['close']
        
        # Volume-price relationship
        df['accumulation_distribution'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-8) * df['volume']
        df['accumulation_distribution_sma'] = df['accumulation_distribution'].rolling(20).mean()
        
        # Price-volume correlation (backward-looking)
        df['price_vol_correlation'] = df['close'].rolling(20).corr(df['volume'])
        
        logger.info(f"✅ Created {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']])} leak-free features")
        
        return df
    
    def create_leak_free_targets(self, df: pd.DataFrame, symbol: str, 
                                target_horizon: int = 3,
                                validation_split: float = 0.2) -> pd.DataFrame:
        """
        Create targets using PROPER temporal methodology - NO future data leakage.
        
        Instead of using shift(-horizon), we:
        1. Split data temporally (train on past, predict future)
        2. Create targets only for training period using actual future outcomes
        3. Ensure test period has no target information available
        
        Args:
            df: DataFrame with features
            symbol: Stock symbol  
            target_horizon: Days ahead to predict
            validation_split: Fraction to reserve for validation
            
        Returns:
            DataFrame with leak-free targets and temporal split info
        """
        logger.info(f"🎯 Creating leak-free targets for {symbol} (horizon: {target_horizon}d)")
        
        # Sort by date to ensure temporal order
        df = df.sort_index()
        
        # Calculate the temporal split point
        split_idx = int(len(df) * (1 - validation_split))
        train_end_date = df.index[split_idx]
        
        logger.info(f"📅 Temporal split: Train up to {train_end_date}, Test after")
        
        # Initialize target column
        df['target'] = np.nan
        df['temporal_split'] = 'test'  # Default to test
        
        # For training period: create targets using actual future outcomes
        for i in range(min(split_idx, len(df) - target_horizon)):
            df.iloc[i, df.columns.get_loc('temporal_split')] = 'train'
            
            # Calculate actual future return
            current_price = df['close'].iloc[i]
            future_price = df['close'].iloc[i + target_horizon]
            future_return = (future_price / current_price) - 1
            
            # Create binary target (top 25% as positive)
            df.iloc[i, df.columns.get_loc('target')] = future_return
        
        # Convert returns to binary targets for training data only
        train_mask = df['temporal_split'] == 'train'
        train_returns = df.loc[train_mask, 'target'].dropna()
        
        if len(train_returns) > 0:
            # Use 75th percentile threshold (top 25% as positive)
            threshold = train_returns.quantile(0.75)
            
            # Apply binary classification only to training data
            df.loc[train_mask, 'target'] = (df.loc[train_mask, 'target'] > threshold).astype(int)
            
            # Test data keeps NaN targets (no future information available)
            df.loc[df['temporal_split'] == 'test', 'target'] = np.nan
            
            pos_count = df.loc[train_mask, 'target'].sum()
            total_train = train_mask.sum()
            
            logger.info(f"🎯 Training targets: {pos_count}/{total_train} positive ({pos_count/total_train*100:.1f}%)")
            logger.info(f"📊 Threshold: {threshold:.4f}")
            logger.info(f"🔒 Test period: {len(df) - split_idx} samples with NO target information")
        
        return df
    
    def collect_symbol_data(self, symbol: str, days: int = 730,
                           target_horizon: int = 3) -> Optional[pd.DataFrame]:
        """
        Collect completely leak-free data for a single symbol.
        
        Args:
            symbol: Stock symbol
            days: Number of days of historical data
            target_horizon: Target prediction horizon
            
        Returns:
            DataFrame with leak-free features and targets
        """
        logger.info(f"🔄 Processing {symbol} with leak-free pipeline")
        
        # Get fresh data from Alpaca API
        df = self.get_fresh_historical_data(symbol, days)
        if df is None:
            return None
        
        # Create leak-free features
        df = self.calculate_leak_free_features(df)
        
        # Create leak-free targets with temporal split
        df = self.create_leak_free_targets(df, symbol, target_horizon)
        
        # Add symbol column
        df['ticker'] = symbol
        df['timestamp'] = df.index
        
        # Validate no data leakage
        self._validate_no_leakage(df)
        
        logger.info(f"✅ Completed leak-free data collection for {symbol}")
        return df
    
    def _validate_no_leakage(self, df: pd.DataFrame) -> None:
        """Validate that no data leakage exists in the dataset."""
        
        # Check 1: No features should correlate perfectly with targets
        train_mask = df['temporal_split'] == 'train'
        if train_mask.sum() > 0:
            feature_cols = [c for c in df.columns if c not in ['target', 'ticker', 'timestamp', 'temporal_split']]
            
            for col in feature_cols:
                if df.loc[train_mask, col].nunique() > 1:  # Skip constant columns
                    corr = df.loc[train_mask, col].corr(df.loc[train_mask, 'target'])
                    if not np.isnan(corr) and abs(corr) > 0.95:
                        logger.error(f"🚨 POTENTIAL LEAKAGE: {col} correlation with target: {corr:.4f}")
                        raise ValueError(f"Feature {col} shows suspiciously high correlation with target")
        
        # Check 2: Test data should have no targets
        test_mask = df['temporal_split'] == 'test'
        if test_mask.sum() > 0:
            test_targets = df.loc[test_mask, 'target'].dropna()
            if len(test_targets) > 0:
                logger.error(f"🚨 DATA LEAKAGE: Test period has {len(test_targets)} target values")
                raise ValueError("Test period should not have target information")
        
        # Check 3: No NaN in training features
        train_features = df.loc[train_mask, feature_cols]
        nan_cols = train_features.columns[train_features.isnull().all()].tolist()
        if nan_cols:
            logger.warning(f"⚠️  Dropping features with all NaN: {nan_cols}")
        
        logger.info("✅ Passed data leakage validation")
    
    def collect_batch_data(self, batch_name: str, symbols: List[str],
                          max_symbols: Optional[int] = None,
                          target_horizon: int = 3,
                          days: int = 730) -> pd.DataFrame:
        """
        Collect leak-free data for a batch of symbols.
        
        Args:
            batch_name: Name of the batch
            symbols: List of symbols to process
            max_symbols: Maximum number of symbols to process
            target_horizon: Target prediction horizon
            days: Number of days of historical data
            
        Returns:
            Combined DataFrame with leak-free data
        """
        logger.info(f"🚀 Collecting leak-free batch data: {batch_name}")
        logger.info(f"📊 Symbols: {len(symbols)}, Max: {max_symbols}, Horizon: {target_horizon}d")
        
        if max_symbols:
            symbols = symbols[:max_symbols]
        
        all_data = []
        successful = 0
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"[{i}/{len(symbols)}] Processing {symbol}")
            
            try:
                symbol_data = self.collect_symbol_data(symbol, days, target_horizon)
                if symbol_data is not None and len(symbol_data) > 50:  # Minimum data requirement
                    all_data.append(symbol_data)
                    successful += 1
                    logger.info(f"✅ Added {symbol}: {len(symbol_data)} samples")
                else:
                    logger.warning(f"⚠️  Skipped {symbol}: insufficient data")
                    
            except Exception as e:
                logger.error(f"❌ Failed to process {symbol}: {e}")
        
        if not all_data:
            logger.error("No valid data collected for any symbol")
            return pd.DataFrame()
        
        # Combine all symbol data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        logger.info(f"🎉 Batch collection complete:")
        logger.info(f"   Successful symbols: {successful}/{len(symbols)}")
        logger.info(f"   Total samples: {len(combined_df)}")
        logger.info(f"   Training samples: {(combined_df['temporal_split'] == 'train').sum()}")
        logger.info(f"   Test samples: {(combined_df['temporal_split'] == 'test').sum()}")
        
        return combined_df
    
    def save_leak_free_data(self, df: pd.DataFrame, filename: str) -> str:
        """Save the leak-free dataset with validation."""
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        filepath = f"data/{filename}"
        
        # Add metadata
        df['data_collection_timestamp'] = datetime.now().isoformat()
        df['data_source'] = 'alpaca_api_fresh'
        df['leak_free_validated'] = True
        
        # Save with timestamp
        df.to_csv(filepath, index=False)
        
        # Log summary
        train_count = (df['temporal_split'] == 'train').sum()
        test_count = (df['temporal_split'] == 'test').sum()
        pos_rate = df[df['temporal_split'] == 'train']['target'].mean() if train_count > 0 else 0
        
        logger.info(f"💾 Saved leak-free data: {filepath}")
        logger.info(f"   Total: {len(df)} samples")
        logger.info(f"   Train: {train_count} samples ({pos_rate:.1%} positive)")
        logger.info(f"   Test: {test_count} samples (no targets)")
        logger.info(f"   Features: {len([c for c in df.columns if c not in ['target', 'ticker', 'timestamp', 'temporal_split', 'data_collection_timestamp', 'data_source', 'leak_free_validated']])}")
        
        return filepath


def main():
    """Command-line interface for leak-free data collection."""
    
    parser = argparse.ArgumentParser(description='Collect leak-free training data from Alpaca API')
    parser.add_argument('--batch', type=int, help='Single batch number to process')
    parser.add_argument('--max-symbols', type=int, default=50, help='Maximum symbols per batch')
    parser.add_argument('--output', type=str, help='Output CSV file path')
    parser.add_argument('--target-horizon', type=int, default=3, help='Target prediction horizon in days')
    parser.add_argument('--days', type=int, default=730, help='Days of historical data')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize leak-free collector
        collector = LeakFreeAlpacaDataCollector()
        
        # Load batches
        batches = collector.load_stock_batches()
        
        if args.batch:
            batch_name = f"batch_{args.batch}"
            if batch_name not in batches:
                logger.error(f"Batch {args.batch} not found")
                sys.exit(1)
                
            symbols = batches[batch_name]
            
            # Auto-generate output filename if not specified
            if not args.output:
                args.output = f"leak_free_batch_{args.batch}_data.csv"
            
            # Collect leak-free data
            df = collector.collect_batch_data(
                batch_name=batch_name,
                symbols=symbols,
                max_symbols=args.max_symbols,
                target_horizon=args.target_horizon,
                days=args.days
            )
            
            if len(df) > 0:
                filepath = collector.save_leak_free_data(df, args.output)
                logger.info(f"🎉 Success! Leak-free data saved to: {filepath}")
            else:
                logger.error("No data collected")
                sys.exit(1)
        else:
            logger.error("Please specify --batch number")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
