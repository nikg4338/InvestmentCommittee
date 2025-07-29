#!/usr/bin/env python3
"""
Model Training Script for Investment Committee
==============================================

This script trains both XGBoost and Neural Network models on multiple S&P 500 stocks
using real market data from Alpaca API. Features include RSI, Z-score, and short-term 
returns for binary classification with comprehensive visualizations.

Usage: 
  python train_models.py                    # Train on S&P 500 subset
  python train_models.py --batch 1         # Train on IEX batch 1
  python train_models.py --batch 2 --batch-file custom_batches.json
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import RandomOverSampler
from typing import List
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon


# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our models and data collectors
from models.xgboost_model import XGBoostModel
from models.neural_network import optuna_search_mlp
from models.neural_predictor import NeuralPredictor
from data.collectors.market_data import MarketDataCollector
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
# Visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Install with: pip install matplotlib seaborn")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log',encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load IEX filtered batches as default symbol universe
def load_iex_batches_all_symbols(batch_file: str = "filtered_iex_batches.json") -> list:
    """Load all symbols from all IEX filtered batches."""
    with open(batch_file, 'r') as f:
        batch_data = json.load(f)
    
    all_symbols = []
    for batch_name, symbols in batch_data['batches'].items():
        all_symbols.extend(symbols)
    
    logger.info(f"Loaded {len(all_symbols)} symbols from {len(batch_data['batches'])} IEX batches")
    return all_symbols

# IEX filtered symbols - our default universe (replaces S&P 500)
IEX_SYMBOLS = load_iex_batches_all_symbols()

def fetch_market_data_multi_ticker_yfinance(symbols: list, limit_tickers: int = 50) -> pd.DataFrame:
    """
    Fetch real market data for multiple tickers using yfinance API as fallback.
    
    Args:
        symbols (list): List of stock symbols
        limit_tickers (int): Maximum number of tickers to process
        
    Returns:
        pd.DataFrame: Combined market data for all tickers
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required for market data. Install with: pip install yfinance")
    
    logger.info(f"Fetching market data for {len(symbols[:limit_tickers])} tickers using yfinance API...")
    
    all_data = []
    failed_tickers = []
    
    for i, symbol in enumerate(symbols[:limit_tickers]):
        try:
            logger.info(f"Fetching data for {symbol} ({i+1}/{min(len(symbols), limit_tickers)})")
            
            # Fetch 1 year of daily data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            
            if len(hist) < 100:  # Skip tickers with insufficient data
                logger.warning(f"Skipping {symbol}: only {len(hist)} bars available (minimum 100 required)")
                failed_tickers.append(symbol)
                continue
            
            # Convert to our format
            df = pd.DataFrame({
                'Date': hist.index,
                'Open': hist['Open'].values,
                'High': hist['High'].values, 
                'Low': hist['Low'].values,
                'Close': hist['Close'].values,
                'Volume': hist['Volume'].values
            })
            
            df['ticker'] = symbol
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.reset_index(drop=True)
            
            # Select and reorder columns
            df = df[['Date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df = df.sort_values('Date').reset_index(drop=True)
            
            all_data.append(df)
            logger.info(f"Successfully loaded {len(df)} rows for {symbol}")
            
            # Rate limiting - be respectful to API
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            failed_tickers.append(symbol)
            continue
    
    if not all_data:
        raise ValueError("No market data was successfully fetched for any ticker")
        
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    logger.info(f"Successfully fetched data for {len(all_data)} tickers")
    logger.info(f"Failed tickers ({len(failed_tickers)}): {failed_tickers}")
    logger.info(f"Combined dataset shape: {combined_df.shape}")
    logger.info(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
    
    return combined_df

def fetch_market_data_multi_ticker(symbols: list, limit_tickers: int = 50, alpaca_only: bool = True, timeframe: str = "1Day") -> pd.DataFrame:
    """
    Fetch real market data for multiple tickers using Alpaca API exclusively.
    
    Args:
        symbols (list): List of stock symbols
        limit_tickers (int): Maximum number of tickers to process  
        alpaca_only (bool): Use only Alpaca API (no yfinance fallback)
        timeframe (str): Bar timeframe for historical data
        
    Returns:
        pd.DataFrame: Combined market data for all tickers
    """
    logger.info(f"Fetching market data for {len(symbols[:limit_tickers])} tickers using Alpaca API only...")
    
    if alpaca_only:
        logger.info("Using Alpaca API exclusively for IEX filtered symbols...")
        return fetch_market_data_alpaca(symbols, limit_tickers, timeframe)
    else:
        # Legacy fallback logic (deprecated)
        logger.warning("Alpaca-only mode disabled - this is not recommended for IEX filtered symbols")
        try:
            logger.info("Attempting to use Alpaca API...")
            collector = MarketDataCollector()
            
            # Test with one symbol first
            test_bars = collector.fetch_recent_ohlcv(symbols[0], timeframe="1Day", limit=5)
            if len(test_bars) > 0:
                logger.info("Alpaca API is working, proceeding with Alpaca...")
                return fetch_market_data_alpaca(symbols, limit_tickers)
        except Exception as e:
            logger.warning(f"Alpaca API failed: {e}")
        
        # Fallback to yfinance (not recommended for intraday trading)
        logger.info("Falling back to yfinance API...")
        return fetch_market_data_multi_ticker_yfinance(symbols, limit_tickers)

def fetch_market_data_alpaca(symbols: list, limit_tickers: int = 50, timeframe: str = "1Day") -> pd.DataFrame:
    """
    Fetch real market data for multiple tickers using Alpaca API.
    
    Args:
        symbols (list): List of stock symbols
        limit_tickers (int): Maximum number of tickers to process
        timeframe (str): Bar timeframe for historical data
        
    Returns:
        pd.DataFrame: Combined market data for all tickers
    """
    import time
    import random
    
    logger.info(f"Fetching market data for {len(symbols[:limit_tickers])} tickers from Alpaca API...")
    logger.info(f"Using timeframe: {timeframe}")
    
    collector = MarketDataCollector()
    all_data = []
    failed_tickers = []
    
    for i, symbol in enumerate(symbols[:limit_tickers]):
        max_retries = 3
        retry_count = 0
        success = False
        
        while retry_count <= max_retries and not success:
            try:
                if retry_count > 0:
                    # Exponential backoff: 1s, 2s, 4s
                    backoff_time = (2 ** retry_count) + random.uniform(0, 1)
                    logger.info(f"Retrying {symbol} (attempt {retry_count + 1}/{max_retries + 1}) after {backoff_time:.1f}s backoff")
                    time.sleep(backoff_time)
                
                logger.info(f"Fetching data for {symbol} ({i+1}/{min(len(symbols), limit_tickers)}) - timeframe: {timeframe}")
                
                # Fetch historical data with specified timeframe
                bars = collector.fetch_recent_ohlcv(symbol, timeframe=timeframe, limit=365)
                
                if len(bars) < 100:  # Skip tickers with insufficient data
                    logger.warning(f"Skipping {symbol}: only {len(bars)} bars available (minimum 100 required)")
                    failed_tickers.append(symbol)
                    success = True  # Don't retry for insufficient data
                    continue
                    
                # Convert to DataFrame
                df = pd.DataFrame(bars)
                df['ticker'] = symbol
                df['Date'] = pd.to_datetime(df['timestamp'])
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High', 
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                
                # Select and reorder columns
                df = df[['Date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
                if len(df) > 0:
                    df = df.sort_values('Date').reset_index(drop=True)
                
                all_data.append(df)
                logger.info(f"Successfully loaded {len(df)} rows for {symbol}")
                success = True
                
                # Rate limiting - be respectful to API
                time.sleep(0.1)
                
            except Exception as e:
                retry_count += 1
                if retry_count <= max_retries:
                    logger.warning(f"Error fetching {symbol} (attempt {retry_count}/{max_retries + 1}): {e}")
                else:
                    logger.error(f"Failed to fetch data for {symbol} after {max_retries + 1} attempts: {e}")
                    failed_tickers.append(symbol)
    
    if not all_data:
        raise ValueError("No market data was successfully fetched for any ticker")
        
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    logger.info(f"Successfully fetched data for {len(all_data)} tickers")
    logger.info(f"Failed tickers ({len(failed_tickers)}): {failed_tickers}")
    logger.info(f"Combined dataset shape: {combined_df.shape}")
    logger.info(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
    
    return combined_df

def load_spy_data(file_path: str = 'data/SPY_1y.csv') -> pd.DataFrame:
    """
    Load SPY historical data from CSV file.
    
    Args:
        file_path (str): Path to SPY data CSV file
        
    Returns:
        pd.DataFrame: Loaded and processed SPY data
    """
    try:
        logger.info(f"Loading SPY data from {file_path}")
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        logger.info(f"Loaded {len(df)} rows of SPY data from {df['Date'].min()} to {df['Date'].max()}")
        return df
    except Exception as e:
        logger.error(f"Error loading SPY data: {e}")
        raise

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices (pd.Series): Price series
        window (int): RSI calculation window
        
    Returns:
        pd.Series: RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_zscore(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate Z-score for price series.
    
    Args:
        prices (pd.Series): Price series
        window (int): Rolling window for Z-score calculation
        
    Returns:
        pd.Series: Z-score values
    """
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    zscore = (prices - rolling_mean) / rolling_std
    return zscore

def calculate_short_term_return(prices: pd.Series, periods: int = 5) -> pd.Series:
    """
    Calculate short-term returns.
    
    Args:
        prices (pd.Series): Price series
        periods (int): Number of periods for return calculation
        
    Returns:
        pd.Series: Short-term returns
    """
    return prices.pct_change(periods)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer technical features from OHLCV data.
    
    Args:
        df (pd.DataFrame): Raw OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    logger.info("Engineering features...")
    
    # Make a copy to avoid modifying original data
    features_df = df.copy()
    
    # Basic price features
    features_df['price_change'] = features_df['Close'].pct_change()
    features_df['high_low_ratio'] = features_df['High'] / features_df['Low']
    features_df['open_close_ratio'] = features_df['Open'] / features_df['Close']
    
    # RSI (14-day)
    features_df['rsi'] = calculate_rsi(features_df['Close'], window=14)
    
    # Z-score (20-day)
    features_df['zscore'] = calculate_zscore(features_df['Close'], window=20)
    
    # Short-term returns
    features_df['return_1d'] = calculate_short_term_return(features_df['Close'], 1)
    features_df['return_5d'] = calculate_short_term_return(features_df['Close'], 5)
    features_df['return_10d'] = calculate_short_term_return(features_df['Close'], 10)
    
    # Moving averages
    features_df['ma_5'] = features_df['Close'].rolling(window=5).mean()
    features_df['ma_10'] = features_df['Close'].rolling(window=10).mean()
    features_df['ma_20'] = features_df['Close'].rolling(window=20).mean()
    
    # Moving average ratios
    features_df['price_ma5_ratio'] = features_df['Close'] / features_df['ma_5']
    features_df['price_ma10_ratio'] = features_df['Close'] / features_df['ma_10']
    features_df['price_ma20_ratio'] = features_df['Close'] / features_df['ma_20']
    features_df['ma5_ma10_ratio'] = features_df['ma_5'] / features_df['ma_10']
    features_df['ma10_ma20_ratio'] = features_df['ma_10'] / features_df['ma_20']
    
    # Volume features
    features_df['volume_change'] = features_df['Volume'].pct_change()
    features_df['volume_ma_20'] = features_df['Volume'].rolling(window=20).mean()
    features_df['volume_ratio'] = features_df['Volume'] / features_df['volume_ma_20']
    
    # Volatility features
    features_df['volatility_5d'] = features_df['return_1d'].rolling(window=5).std()
    features_df['volatility_20d'] = features_df['return_1d'].rolling(window=20).std()
    
    # Bollinger Bands
    bb_window = 20
    bb_std = 2
    features_df['bb_middle'] = features_df['Close'].rolling(window=bb_window).mean()
    bb_rolling_std = features_df['Close'].rolling(window=bb_window).std()
    features_df['bb_upper'] = features_df['bb_middle'] + (bb_rolling_std * bb_std)
    features_df['bb_lower'] = features_df['bb_middle'] - (bb_rolling_std * bb_std)
    features_df['bb_position'] = (features_df['Close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
    
    # MACD-like momentum
    ema_12 = features_df['Close'].ewm(span=12).mean()
    ema_26 = features_df['Close'].ewm(span=26).mean()
    features_df['macd'] = ema_12 - ema_26
    features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
    features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
    
    logger.info(f"Engineered {len([col for col in features_df.columns if col not in df.columns])} new features")
    
    return features_df

def create_target_variable_multi_ticker(df: pd.DataFrame, prediction_horizon: int = 5) -> pd.Series:
    """
    Create binary classification target for multi-ticker data: 1 if price increases in N days, 0 otherwise.
    
    Args:
        df (pd.DataFrame): DataFrame with price data including 'ticker' column
        prediction_horizon (int): Number of days to look ahead
        
    Returns:
        pd.Series: Binary target variable
    """
    logger.info(f"Creating binary target variable with {prediction_horizon}-day horizon for multi-ticker data")
    
    targets = []
    
    # Process each ticker separately to avoid cross-ticker contamination
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy().sort_values('Date')
        
        # Calculate future returns for this ticker
        future_returns = ticker_df['Close'].pct_change(prediction_horizon).shift(-prediction_horizon)
        
        # Binary classification: 1 if positive return, 0 if negative
        ticker_target = (future_returns > 0).astype(int)
        ticker_target.index = ticker_df.index  # Maintain original index
        
        targets.append(ticker_target)
    
    # Combine all targets
    combined_target = pd.concat(targets).sort_index()
    
    logger.info(f"Target distribution: {combined_target.value_counts().to_dict()}")
    logger.info(f"Target balance: {combined_target.mean():.3f} positive, {1-combined_target.mean():.3f} negative")
    
    return combined_target

def create_target_variable(df: pd.DataFrame, prediction_horizon: int = 5) -> pd.Series:
    """
    Create binary classification target: 1 if price increases in N days, 0 otherwise.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        prediction_horizon (int): Number of days to look ahead
        
    Returns:
        pd.Series: Binary target variable
    """
    logger.info(f"Creating binary target variable with {prediction_horizon}-day horizon")
    
    # Check if this is multi-ticker data
    if 'ticker' in df.columns:
        return create_target_variable_multi_ticker(df, prediction_horizon)
    
    # Single ticker logic (original)
    future_returns = df['Close'].pct_change(prediction_horizon).shift(-prediction_horizon)
    target = (future_returns > 0).astype(int)
    
    logger.info(f"Target distribution: {target.value_counts().to_dict()}")
    logger.info(f"Target balance: {target.mean():.3f} positive, {1-target.mean():.3f} negative")
    
    return target

def select_features(df: pd.DataFrame) -> list:
    """
    Select the most relevant features for model training.
    
    Args:
        df (pd.DataFrame): DataFrame with all engineered features
        
    Returns:
        list: List of selected feature column names
    """
    # Select key technical features
    selected_features = [
        'rsi',
        'zscore', 
        'return_1d',
        'return_5d',
        'return_10d',
        'price_ma5_ratio',
        'price_ma10_ratio',
        'price_ma20_ratio',
        'ma5_ma10_ratio',
        'ma10_ma20_ratio',
        'volume_ratio',
        'volatility_5d',
        'volatility_20d',
        'bb_position',
        'macd',
        'macd_signal',
        'macd_histogram',
        'high_low_ratio',
        'open_close_ratio',
        'volume_change'
    ]
    
    # Filter out features that don't exist in the dataframe
    available_features = [f for f in selected_features if f in df.columns]
    
    logger.info(f"Selected {len(available_features)} features for training: {available_features}")
    
    return available_features

def cap_majority_ratio(X, y, max_ratio=2.0):
    """
    Cap extreme majority-minority ratios to prevent training instability.
    
    Args:
        X: Feature matrix
        y: Target labels
        max_ratio: Maximum allowed majority:minority ratio
        
    Returns:
        tuple: (X_capped, y_capped) with capped ratios
    """
    import pandas as pd
    
    # Convert to DataFrame for easier manipulation
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])] if hasattr(X, 'shape') else [f"f{i}" for i in range(len(X[0]))])
    
    df = X.copy()
    df['y'] = y
    
    n0 = (df.y == 0).sum()
    n1 = (df.y == 1).sum()
    
    if min(n0, n1) == 0:  # Handle edge case
        return X, y
        
    current_ratio = max(n0, n1) / min(n0, n1)
    
    if current_ratio > max_ratio:
        maj, mino = (0, 1) if n0 > n1 else (1, 0)
        keep_majority = int(max_ratio * min(n0, n1))
        
        df_major = df[df.y == maj].sample(keep_majority, random_state=42)
        df_mino = df[df.y == mino]
        df_new = pd.concat([df_major, df_mino]).sample(frac=1, random_state=42)
        
        logger.info(f"Capped majority ratio from {current_ratio:.2f} to ~{max_ratio}")
        return df_new.drop('y', axis=1), df_new['y'].values
    
    return X, y

def balance_dataset(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """
    Balance the dataset using a controlled approach to handle class imbalance.
    Instead of perfect 1:1 balancing, aims for ~60:40 to prevent overfitting.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        
    Returns:
        tuple: (X_train_balanced, y_train_balanced) - balanced dataset
    """
    logger.info("Balancing dataset with controlled ratio...")
    
    # Print original distribution
    try:
        original_counts = y_train.value_counts().sort_index()
        logger.info(f"Original class distribution: {original_counts.to_dict()}")
    except:
        logger.info("Could not compute original distribution")
    
    # Cap extreme ratios first
    X_capped, y_capped = cap_majority_ratio(X_train, y_train, max_ratio=2.5)
    
    # Apply modest oversampling instead of perfect balance
    # Target ~60:40 instead of 50:50 to prevent overfitting to minority class
    desired_ratio = 0.6  # 60% majority, 40% minority
    
    class_0 = X_capped[y_capped == 0] if hasattr(X_capped, '__getitem__') else X_train[y_train == 0]
    class_1 = X_capped[y_capped == 1] if hasattr(X_capped, '__getitem__') else X_train[y_train == 1]
    
    num_class_1 = len(class_1)
    num_class_0 = len(class_0)
    
    # If minority class is very small, use RandomOverSampler as fallback
    if min(num_class_0, num_class_1) < 100:
        logger.info("Small minority class detected, using RandomOverSampler")
        oversampler = RandomOverSampler(random_state=42)
        X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train, y_train)
        
        # Convert back to pandas for consistency
        X_train_balanced = pd.DataFrame(X_train_balanced, columns=X_train.columns)
        y_train_balanced = pd.Series(y_train_balanced, name=y_train.name if hasattr(y_train, 'name') else 'target')
    else:
        # Use controlled balancing
        if num_class_0 > num_class_1:
            # Class 0 is majority
            target_class_0 = min(num_class_0, int(num_class_1 / (1 - desired_ratio) * desired_ratio))
            class_0_down = class_0.sample(target_class_0, random_state=42) if hasattr(class_0, 'sample') else class_0[:target_class_0]
            X_train_balanced = pd.concat([class_1, class_0_down])
            y_balanced = [1] * len(class_1) + [0] * len(class_0_down)
        else:
            # Class 1 is majority  
            target_class_1 = min(num_class_1, int(num_class_0 / (1 - desired_ratio) * desired_ratio))
            class_1_down = class_1.sample(target_class_1, random_state=42) if hasattr(class_1, 'sample') else class_1[:target_class_1]
            X_train_balanced = pd.concat([class_0, class_1_down])
            y_balanced = [0] * len(class_0) + [1] * len(class_1_down)
        
        y_train_balanced = pd.Series(y_balanced, name=y_train.name if hasattr(y_train, 'name') else 'target')
    
    # Print new distribution
    try:
        new_counts = y_train_balanced.value_counts().sort_index()
        logger.info(f"Balanced class distribution: {new_counts.to_dict()}")
        logger.info(f"Dataset size changed from {len(X_train)} to {len(X_train_balanced)} samples")
    except:
        logger.info("Could not compute new distribution")
    
    return X_train_balanced, y_train_balanced

def clean_data_for_ml(X: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data for machine learning by handling infinite and extreme values.
    
    Args:
        X: Feature matrix to clean
        
    Returns:
        Cleaned feature matrix
    """
    X_clean = X.copy()
    
    # Replace infinite values with NaN
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    
    # Handle extreme outliers (values beyond 99.9th percentile)
    for col in X_clean.columns:
        if X_clean[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            # Calculate reasonable bounds
            q1 = X_clean[col].quantile(0.01)
            q99 = X_clean[col].quantile(0.99)
            
            # Cap extreme values
            X_clean[col] = X_clean[col].clip(lower=q1, upper=q99)
    
    # Log cleaning summary
    inf_count = np.isinf(X.values).sum() if len(X) > 0 else 0
    nan_count = X.isnull().sum().sum()
    cleaned_nan_count = X_clean.isnull().sum().sum()
    
    logger.info(f"Data cleaning: removed {inf_count} inf values, "
               f"NaN count before: {nan_count}, after: {cleaned_nan_count}")
    
    return X_clean

def prepare_training_data(df: pd.DataFrame, feature_columns: list, target_column: str = 'target') -> tuple:
    """
    Prepare training data by cleaning and splitting with group-based splitting by ticker.
    
    Args:
        df (pd.DataFrame): DataFrame with features and target
        feature_columns (list): List of feature column names
        target_column (str): Name of target column
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) splits
    """
    logger.info("Preparing training data with group-based splitting...")
    
    # Create feature matrix and target vector
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    # Clean data for ML
    X_clean = clean_data_for_ml(X)
    
    # Remove rows with NaN values
    mask = ~(X_clean.isnull().any(axis=1) | y.isnull())
    X_final = X_clean[mask]
    y_final = y[mask]
    df_final = df[mask]  # Keep original df for ticker column
    
    logger.info(f"After cleaning: {len(X_final)} samples with {X_final.shape[1]} features")
    logger.info(f"Features: {list(X_final.columns)}")
    
    # Check data quality
    if len(X_final) < 100:
        logger.warning(f"Very few samples remaining: {len(X_final)}")
    
    # Group-based split by ticker to prevent leakage
    if 'ticker' in df_final.columns:
        from sklearn.model_selection import GroupShuffleSplit
        logger.info("Using GroupShuffleSplit by ticker to prevent cross-ticker leakage")
        
        groups = df_final['ticker']
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X_final, y_final, groups))
        
        X_train, X_test = X_final.iloc[train_idx], X_final.iloc[test_idx]
        y_train, y_test = y_final.iloc[train_idx], y_final.iloc[test_idx]
        
        # Log ticker distribution in train/test
        train_tickers = groups.iloc[train_idx].nunique()
        test_tickers = groups.iloc[test_idx].nunique()
        logger.info(f"Train set has {train_tickers} unique tickers")
        logger.info(f"Test set has {test_tickers} unique tickers")
        
    else:
        # Fallback to stratified split if no ticker column
        logger.info("No ticker column found, using stratified split")
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
        )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Log class balance for debugging
    try:
        train_0 = (y_train == 0).sum() if hasattr(y_train, '__iter__') else 0
        train_1 = (y_train == 1).sum() if hasattr(y_train, '__iter__') else 0
        test_0 = (y_test == 0).sum() if hasattr(y_test, '__iter__') else 0
        test_1 = (y_test == 1).sum() if hasattr(y_test, '__iter__') else 0
        
        logger.info(f"[BATCH CLASS BALANCE] Train -> 0: {train_0}, 1: {train_1}")
        logger.info(f"[BATCH CLASS BALANCE]  Test -> 0: {test_0}, 1: {test_1}")
        
        # Handle potential Series/DataFrame issues for value_counts
        train_dist = y_train.value_counts().sort_index().to_dict() if hasattr(y_train, 'value_counts') else {}
        test_dist = y_test.value_counts().sort_index().to_dict() if hasattr(y_test, 'value_counts') else {}
        logger.info(f"Training target distribution: {train_dist}")
        logger.info(f"Test target distribution: {test_dist}")
    except Exception as e:
        logger.warning(f"Could not compute target distribution: {e}")
    
    return X_train, X_test, y_train, y_test

def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, use_balanced_data: bool = True) -> XGBoostModel:
    """
    Train XGBoost model for binary classification.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data for evaluation
        use_balanced_data (bool): Whether to use balanced training data
        
    Returns:
        XGBoostModel: Trained XGBoost model
    """
    logger.info("Training XGBoost model...")
    
    # Balance dataset if requested
    if use_balanced_data:
        X_train_final, y_train_final = balance_dataset(X_train, y_train)
        logger.info("Using balanced dataset for XGBoost training")
    else:
        X_train_final, y_train_final = X_train, y_train
        logger.info("Using original imbalanced dataset for XGBoost training")
        
        # Alternative: Use scale_pos_weight to handle imbalance
        # Calculate scale_pos_weight: (number of negative samples) / (number of positive samples)
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        logger.info(f"Alternative approach: Could use scale_pos_weight={scale_pos_weight:.4f} in XGBoost params")
    
    # Initialize model
    model = XGBoostModel()
    
    # Train model
    model.train(X_train_final, y_train_final)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred_binary, average=None)
    
    logger.info(f"XGBoost Test Accuracy: {accuracy:.4f}")
    logger.info(f"XGBoost Per-Class Results:")
    logger.info(f"  Class 0 - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1: {f1[0]:.4f}, Support: {support[0]}")
    logger.info(f"  Class 1 - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1: {f1[1]:.4f}, Support: {support[1]}")
    logger.info(f"XGBoost Classification Report:\n{classification_report(y_test, y_pred_binary)}")
    logger.info(f"XGBoost Confusion Matrix:\n{confusion_matrix(y_test, y_pred_binary)}")
    
    # Save model
    save_path = "models/saved/xgb_model.json"
    model.save_model(save_path)
    logger.info(f"XGBoost model saved to {save_path}")
    
    return model

def train_neural_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, use_balanced_data: bool = True) -> NeuralPredictor:
    """
    Train Neural Network model for binary classification.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data for evaluation
        use_balanced_data (bool): Whether to use balanced training data
        
    Returns:
        NeuralPredictor: Trained neural network model
    """
    logger.info("Training Neural Network model...")
    
    try:
        # Balance dataset if requested
        if use_balanced_data:
            X_train_final, y_train_final = balance_dataset(X_train, y_train)
            logger.info("Using balanced dataset for Neural Network training")
        else:
            X_train_final, y_train_final = X_train, y_train
            logger.info("Using original imbalanced dataset for Neural Network training")
        
        # Initialize model
        model = NeuralPredictor(model_type='mlp')
        
        # Train model with more epochs for balanced data
        epochs = 100 if use_balanced_data else 50
        training_results = model.train(X_train_final, y_train_final, epochs=epochs, batch_size=32)
        
        if training_results.get('status') == 'completed':
            logger.info(f"Neural Network training completed successfully")
            logger.info(f"Final validation accuracy: {training_results.get('final_val_accuracy', 'N/A'):.4f}")
            
            # Store training results for visualization
            if hasattr(model, '_last_training_results'):
                model._last_training_results = training_results
            
            # Save model
            save_path = "models/saved/neural_model.pt"
            success = model.save_model(save_path)
            if success:
                logger.info(f"Neural Network model saved to {save_path}")
                
                # Save best threshold if available
                if hasattr(training_results, 'get') and training_results.get('history', {}).get('best_threshold'):
                    import json
                    threshold_info = {
                        'best_threshold': training_results['history']['best_threshold'],
                        'best_threshold_score': training_results['history']['best_threshold_score']
                    }
                    with open('models/saved/nn_threshold.json', 'w') as f:
                        json.dump(threshold_info, f)
                    logger.info(f"Best threshold {threshold_info['best_threshold']:.3f} saved (F1: {threshold_info['best_threshold_score']:.3f})")
            else:
                logger.warning("Failed to save Neural Network model")
        else:
            logger.error(f"Neural Network training failed: {training_results.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Error training Neural Network: {e}")
        # Create a dummy model for consistency
        model = NeuralPredictor(model_type='mlp')
    
    return model
def create_visualizations(xgb_model, neural_model, X_test_full, y_test, top_features, neural_training_results, batch_num=None, device=None):
    """
    Creates and saves model visualizations in the 'reports' directory, unique to each batch.
    Includes: XGBoost confusion matrix, precision/recall/f1 bar charts,
    and neural network training/validation loss and accuracy curves.
    
    Args:
        xgb_model: Trained XGBoost model
        neural_model: Trained neural network model
        X_test_full: Test feature data (the corrected parameter name is used here)
        y_test: Test labels
        neural_training_results: Training results dict from neural model (should include history)
        batch_num: (Optional) int, used to append to filenames for per-batch reporting
    """
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import shutil
    from sklearn.metrics import confusion_matrix, classification_report
    import torch
    logger.info(f"create_visualizations() called with batch_num={batch_num}")
    # FIX: Changed X_test to X_test_full
    logger.info(f"[VISUALIZATION] Features received: {X_test_full.columns.tolist()}")

    # Determine suffix and batch directory
    suffix = f"_batch{batch_num}" if batch_num is not None else ""
    batch_dir = f"reports/batch{batch_num}" if batch_num is not None else "reports"

    # Recreate the batch-specific output directory
    if os.path.exists(batch_dir):
        shutil.rmtree(batch_dir)
    os.makedirs(batch_dir, exist_ok=True)
    
    # --- XGBoost Confusion Matrix ---
    # FIX: Changed X_test to X_test_full
    y_pred_xgb = xgb_model.predict(X_test_full)
    y_pred_xgb_binary = (y_pred_xgb > 0.5).astype(int)
    cm_xgb = confusion_matrix(y_test, y_pred_xgb_binary)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("XGBoost Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{batch_dir}/xgb_confusion_matrix_batch.png")
    plt.close()
    
    #NN confusion matrix - use best threshold
    best_threshold = 0.5
    if neural_training_results and 'history' in neural_training_results:
        best_threshold = neural_training_results['history'].get('best_threshold', 0.5)
        logger.info(f"Using optimized threshold: {best_threshold:.3f}")
    
    neural_model.model.eval()
    with torch.no_grad():
        neural_model.model.eval()
    with torch.no_grad():
        # Add this line to filter the data to the correct features
        X_test_reduced_for_nn = X_test_full[top_features]
        logits = neural_model.model(torch.tensor(X_test_reduced_for_nn.values, dtype=torch.float32).to(device))
        probs = torch.sigmoid(logits)
        y_pred_nn = (probs.cpu().numpy() > best_threshold).astype(int)
        
        # Add probability distribution histogram for debugging
        plt.figure(figsize=(6,4))
        counts, bin_edges, _ = plt.hist(probs.cpu().numpy(), bins=50, alpha=0.7, edgecolor='black')
        plt.title(f"P(class=1) distribution (Batch {batch_num}, threshold={best_threshold:.3f})")
        plt.xlabel("P(class=1)")
        plt.ylabel("Count")
        plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Threshold={best_threshold:.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{batch_dir}/nn_prob_hist_batch.png")
        plt.close()
        
        # Save separate histograms for each class
        probs_np = probs.cpu().numpy()
        y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
        
        # Class 0 probabilities
        class_0_probs = probs_np[y_test_np == 0]
        if len(class_0_probs) > 0:
            plt.figure(figsize=(6,4))
            plt.hist(class_0_probs, bins=30, alpha=0.7, color='red', label='Class 0', edgecolor='black')
            plt.title(f"P(class=1) for Class 0 samples (Batch {batch_num})")
            plt.xlabel("P(class=1)")
            plt.ylabel("Count")
            plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Threshold={best_threshold:.3f}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{batch_dir}/nn_prob_hist_class0_batch.png")
            plt.close()
        
        # Class 1 probabilities
        class_1_probs = probs_np[y_test_np == 1]
        if len(class_1_probs) > 0:
            plt.figure(figsize=(6,4))
            plt.hist(class_1_probs, bins=30, alpha=0.7, color='green', label='Class 1', edgecolor='black')
            plt.title(f"P(class=1) for Class 1 samples (Batch {batch_num})")
            plt.xlabel("P(class=1)")
            plt.ylabel("Count")
            plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Threshold={best_threshold:.3f}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{batch_dir}/nn_prob_hist_class1_batch.png")
            plt.close()
        
        # Save histogram data for drift analysis
        np.savez(f"{batch_dir}/nn_prob_hist.npz", 
                 counts=counts, 
                 bin_edges=bin_edges,
                 class_0_probs=class_0_probs if len(class_0_probs) > 0 else np.array([]),
                 class_1_probs=class_1_probs if len(class_1_probs) > 0 else np.array([]))

    cm_nn = confusion_matrix(y_test, y_pred_nn)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_nn, annot=True, fmt="d", cmap="Greens", cbar=False)
    plt.title(f"Neural Network Confusion Matrix (Batch {batch_num})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{batch_dir}/nn_confusion_matrix_batch.png")
    plt.close()

    # --- XGBoost Precision/Recall/F1 Bar Charts ---
    # FIX: Changed y_pred_xgb to y_pred_xgb_binary to match its definition
    report = classification_report(y_test, y_pred_xgb_binary, output_dict=True)
    metrics = ['precision', 'recall', 'f1-score']
    classes = ['0', '1']

    for metric in metrics:
        values = [report[c][metric] for c in classes if c in report]
        plt.bar(classes, values)
        plt.ylim(0, 1)
        plt.title(f"XGBoost {metric.capitalize()}")
        plt.xlabel("Class")
        plt.ylabel(metric.capitalize())
        plt.savefig(f"{batch_dir}/xgb_{metric}.png")
        plt.close()

    # --- Neural Network Training Curves (if available) ---
    if neural_training_results and 'history' in neural_training_results:
        hist = neural_training_results['history']
        # Debug: print to see structure and length
        print(f"Batch {batch_num}: Found NN training history.")
        print("Keys:", list(hist.keys()))
        print("Lengths:", {k: len(hist[k]) if isinstance(hist[k], (list, tuple)) else 'N/A' for k in hist})
        # Defensive checks for non-empty arrays
        if all(k in hist and isinstance(hist[k], (list, tuple)) and len(hist[k]) > 0 for k in ['train_loss', 'val_loss', 'val_acc']):
            epochs = list(range(1, len(hist['train_loss'])+1))
            train_loss = hist['train_loss']
            val_loss = hist['val_loss']
            val_acc = hist['val_acc']

            fig, ax1 = plt.subplots()
            ax1.plot(epochs, train_loss, label='Train Loss', color='red')
            ax1.plot(epochs, val_loss, label='Val Loss', color='blue')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend(loc='upper left')

            ax2 = ax1.twinx()
            ax2.plot(epochs, val_acc, label='Val Accuracy', color='green')
            ax2.set_ylabel('Validation Accuracy')
            ax2.legend(loc='upper right')

            plt.title("Neural Network Training Metrics")
            plt.tight_layout()
            plt.savefig(f"{batch_dir}/neural_training_curve_batch.png")
            plt.close()
            print(f"Saved NN training curve for batch {batch_num}.")
        else:
            print(f"Batch {batch_num}: Incomplete NN history, skipping plot. Contents: {hist}")
    else:
        print(f"Batch {batch_num}: No neural_training_results or missing 'history' key.")

def load_training_progress(progress_file: str = 'training_progress.json') -> dict:
    """Load training progress to avoid retraining completed batches."""
    try:
        with open(progress_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'completed_batches': [], 'last_updated': None}
    except Exception as e:
        logger.warning(f"Error loading training progress: {e}")
        return {'completed_batches': [], 'last_updated': None}

def save_training_progress(batch_number: int, batch_file: str, progress_file: str = 'training_progress.json') -> None:
    """Save training progress after completing a batch."""
    try:
        progress = load_training_progress(progress_file)
        
        if batch_number not in progress['completed_batches']:
            progress['completed_batches'].append(batch_number)
            progress['completed_batches'].sort()
        
        progress['last_updated'] = datetime.now().isoformat()
        progress['last_batch_file'] = batch_file
        
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        
        logger.info(f"Marked batch {batch_number} as completed in {progress_file}")
        
    except Exception as e:
        logger.warning(f"Error saving training progress: {e}")

def load_batch_symbols(batch_file: str = 'filtered_iex_batches.json', batch_number: int = 1) -> List[str]:
    """
    Load symbols from a specific batch in the filtered IEX batches file.
    
    Args:
        batch_file: Path to the filtered batches JSON file
        batch_number: Which batch to load (1-indexed)
    
    Returns:
        List of symbols for the specified batch
    """
    try:
        with open(batch_file, 'r') as f:
            data = json.load(f)
        
        batch_key = f"batch_{batch_number}"
        
        # Handle both old and new format
        if 'batches' in data:
            batches = data['batches']
        else:
            batches = data
        
        if batch_key not in batches:
            available_batches = list(batches.keys())
            raise ValueError(f"Batch {batch_number} not found. Available batches: {available_batches}")
        
        symbols = batches[batch_key]
        logger.info(f"Loaded batch {batch_number}: {len(symbols)} symbols")
        logger.info(f"Symbols: {symbols[:10]}{'...' if len(symbols) > 10 else ''}")
        
        # Check if batch was already trained
        progress = load_training_progress()
        if batch_number in progress['completed_batches']:
            logger.warning(f"⚠️  Batch {batch_number} was already trained on {progress.get('last_updated', 'unknown date')}")
            logger.warning("Models will be overwritten. Use --force to suppress this warning.")
        
        return symbols
        
    except FileNotFoundError:
        logger.error(f"Batch file {batch_file} not found. Run IEX filtering first:")
        logger.error("python data/collectors/iex_symbol_filter.py")
        raise
    except Exception as e:
        logger.error(f"Error loading batch {batch_number} from {batch_file}: {e}")
        raise

def get_best_batch_model(metadata_dir="models/saved", prefer_f1=True, use_js_score=False, js_scores=None):
    """
    Loads all batch metadata files, ranks by validation accuracy (or F1 if available),
    and returns the best model, scaler, and threshold paths for inference.
    If use_js_score=True and js_scores is provided, uses score = val_accuracy * (1 - js_divergence).
    """
    meta_files = sorted(glob.glob(f"{metadata_dir}/batch_*_nn_metadata.json"))
    best_score = -np.inf
    best_info = None
    batch_summaries = []
    for meta_path in meta_files:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        score = meta.get("final_val_accuracy", 0)
        if prefer_f1 and meta.get("threshold_score") is not None:
            score = meta["threshold_score"]
        batch_num = int(meta_path.split("batch_")[1].split("_")[0])
        js = None
        if use_js_score and js_scores is not None and batch_num in js_scores:
            js = js_scores[batch_num]
            score = score * (1 - js)
        batch_summaries.append({
            "batch": batch_num,
            "score": score,
            "val_acc": meta.get("final_val_accuracy", 0),
            "f1": meta.get("threshold_score"),
            "js": js,
            "model": f"{metadata_dir}/batch_{batch_num}_nn.pt",
            "scaler": f"{metadata_dir}/batch_{batch_num}_scaler.pkl",
            "threshold": f"{metadata_dir}/batch_{batch_num}_nn_threshold.json",
        })
        if score > best_score:
            best_score = score
            best_info = batch_summaries[-1]
    # Print ranked summary
    print("\nBatch Ranking:")
    for s in sorted(batch_summaries, key=lambda x: x['score'], reverse=True):
        print(f"Batch {s['batch']}: score={s['score']:.4f}, val_acc={s['val_acc']:.4f}, f1={s['f1']}, js={s['js']}")
    if best_info:
        print(f"\nBest batch: {best_info['batch']} (score={best_score:.4f})")
        return best_info
    else:
        print("No batch metadata found.")
        return None

def compute_js_divergence_across_batches(hist_dir="reports", return_scores=False):
    """
    Computes Jensen-Shannon divergence between probability histograms of consecutive batches.
    Plots or logs drift values. Returns dict of batch_num: js_score if return_scores=True.
    """
    import os
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import jensenshannon
    npz_files = sorted(glob.glob(f"{hist_dir}/batch*/nn_prob_hist.npz"),
                       key=lambda x: int(x.split('batch')[1].split('/')[0]))
    if len(npz_files) < 2:
        print("Not enough histogram data for JS divergence. Please save .npz histograms during training.")
        return {} if return_scores else None
    js_scores = {}
    batch_nums = []
    js_vals = []
    for i in range(len(npz_files)-1):
        data1 = np.load(npz_files[i])
        data2 = np.load(npz_files[i+1])
        h1, _ = data1['counts'], data1['bin_edges']
        h2, _ = data2['counts'], data2['bin_edges']
        # Normalize
        h1 = h1 / np.sum(h1)
        h2 = h2 / np.sum(h2)
        js = jensenshannon(h1, h2)
        batch_num = i+2  # JS between batch i+1 and i+2
        js_scores[batch_num] = js
        batch_nums.append(batch_num)
        js_vals.append(js)
        print(f"JS divergence between batch {i+1} and {i+2}: {js:.4f}")
    plt.figure(figsize=(7,4))
    plt.plot(batch_nums, js_vals, marker='o')
    plt.xlabel('Batch Number')
    plt.ylabel('JS Divergence')
    plt.title('Jensen-Shannon Divergence between Consecutive Batches')
    plt.tight_layout()
    plt.show()
    if return_scores:
        return js_scores

def main():
    """
    Main training pipeline.
    """
    import argparse
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train ML models on stock data from IEX filtered batches')
    parser.add_argument('--batch', '-b', type=int,
                       help='Train on specific batch from filtered_iex_batches.json (1-11) [REQUIRED unless using --list-batches or --show-progress]')
    parser.add_argument('--batch-file', default='filtered_iex_batches.json',
                       help='Path to filtered batches file (default: filtered_iex_batches.json)')
    parser.add_argument('--limit-tickers', '-l', type=int, default=None,
                       help='Maximum number of tickers to process from batch (default: all in batch)')
    parser.add_argument('--bar-timeframe', '-t', type=str, default='1Day', 
                       choices=['1Min', '5Min', '15Min', '1Hour', '1Day'],
                       help='Bar timeframe for historical data (default: 1Day). Options: 1Min, 5Min, 15Min, 1Hour, 1Day')
    parser.add_argument('--force', action='store_true',
                       help='Force retraining even if batch was already completed')
    parser.add_argument('--show-progress', action='store_true',
                       help='Show training progress and exit')
    parser.add_argument('--list-batches', action='store_true',
                       help='List available batches and their symbol counts')
    parser.add_argument('--tune-xgb', action='store_true', help='Use Optuna tuning for XGBoost')
    parser.add_argument('--tune-xgb-grid', action='store_true', help='Use GridSearchCV for XGBoost')
    parser.add_argument('--tune-nn', action='store_true', help='Run Optuna tuning for neural network')
    parser.add_argument('--analyze-batches', action='store_true', help='Analyze batch evolution, drift, and best model selection')
    parser.add_argument('--no-stacking', action='store_true', help='Disable model stacking (train individual models only)')
    args = parser.parse_args()
    
    # Handle batch listing
    if args.list_batches:
        try:
            from utils.helpers import get_iex_batch_info
            batch_info = get_iex_batch_info(args.batch_file)
            
            logger.info("=" * 60)
            logger.info("AVAILABLE IEX FILTERED BATCHES")
            logger.info("=" * 60)
            logger.info(f"Total symbols: {batch_info['total_symbols']}")
            logger.info(f"Total batches: {batch_info['total_batches']}")
            logger.info("")
            logger.info("Batch details:")
            for batch_name, count in batch_info['symbols_per_batch'].items():
                batch_num = batch_name.replace('batch_', '')
                logger.info(f"  Batch {batch_num}: {count} symbols")
            
            logger.info("")
            logger.info("Usage examples:")
            logger.info("  python train_models.py --batch 1")
            logger.info("  python train_models.py --batch 5 --limit-tickers 25")
            logger.info("=" * 60)
            return
        except Exception as e:
            logger.error(f"Error loading batch info: {e}")
            return
    
    # Handle progress display
    if args.show_progress:
        progress = load_training_progress()
        completed = progress['completed_batches']
        last_updated = progress.get('last_updated', 'Never')
        
        logger.info("=" * 60)
        logger.info("TRAINING PROGRESS REPORT")
        logger.info("=" * 60)
        logger.info(f"Completed batches: {completed}")
        logger.info(f"Total completed: {len(completed)}")
        logger.info(f"Last updated: {last_updated}")
        logger.info("=" * 60)
        return
    
    # Validate that batch is required for training
    if not args.batch:
        logger.error("Error: --batch argument is required for training")
        logger.error("Use one of:")
        logger.error("  python train_models.py --batch 1")
        logger.error("  python train_models.py --list-batches")  
        logger.error("  python train_models.py --show-progress")
        parser.print_help()
        return
    
    # Check if batch was already completed (unless force is used)
    if not args.force:
        progress = load_training_progress()
        if args.batch in progress['completed_batches']:
            logger.warning("=" * 60)
            logger.warning(f"BATCH {args.batch} ALREADY COMPLETED")
            logger.warning("=" * 60)
            logger.warning(f"This batch was trained on: {progress.get('last_updated', 'unknown date')}")
            logger.warning("Use --force to retrain and overwrite existing models")
            logger.warning("Example: python train_models.py --batch {} --force".format(args.batch))
            logger.warning("=" * 60)
            return
    elif args.force:
        logger.info(f"Force flag enabled - retraining batch {args.batch} even if already completed")
    
    logger.info("=" * 60)
    logger.info("Starting Investment Committee Model Training")
    logger.info("=" * 60)

    # Start timing
    start_time = time.time()
    
    try:
        # Create necessary directories
        os.makedirs('models/saved', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Load the specified batch
        logger.info("=" * 60)
        logger.info(f"LOADING IEX FILTERED BATCH {args.batch}")
        logger.info("=" * 60)
        
        symbols = load_batch_symbols(args.batch_file, args.batch)
        
        # Apply ticker limit if specified
        if args.limit_tickers and args.limit_tickers < len(symbols):
            original_count = len(symbols)
            symbols = symbols[:args.limit_tickers]
            logger.info(f"Limited to {args.limit_tickers} tickers (out of {original_count} available)")
        
        logger.info(f"Processing {len(symbols)} symbols from batch {args.batch}")
        logger.info(f"Symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
        
        # Load multi-ticker data using Alpaca API exclusively
        logger.info("=" * 60)
        logger.info(f"FETCHING REAL MARKET DATA FROM ALPACA API (BATCH {args.batch})")
        logger.info("=" * 60)
        
        # Use Alpaca exclusively for IEX filtered symbols
        df = fetch_market_data_multi_ticker(symbols, limit_tickers=len(symbols), alpaca_only=True, timeframe=args.bar_timeframe)
        
        logger.info(f"Raw data shape: {df.shape}")
        
        # Engineer features
        logger.info("=" * 60)
        logger.info("ENGINEERING FEATURES")
        logger.info("=" * 60)
        
        # Process each ticker separately to avoid cross-contamination
        ticker_dataframes = []
        for ticker in df['ticker'].unique():
            ticker_df = df[df['ticker'] == ticker].copy().sort_values('Date')
            ticker_features = engineer_features(ticker_df)
            ticker_features['ticker'] = ticker
            ticker_dataframes.append(ticker_features)
        
        df_features = pd.concat(ticker_dataframes, ignore_index=True)
        logger.info(f"Features engineered. Shape: {df_features.shape}")
        
        # Create target variable
        df_features['target'] = create_target_variable(df_features)
        
        # Select features
        feature_columns = select_features(df_features)
        
        # Prepare training data
        X_train, X_test, y_train, y_test = prepare_training_data(df_features, feature_columns)
        
        # Train XGBoost model with balanced data
        logger.info("=" * 60)
        logger.info("TRAINING MODELS WITH BALANCED DATA")
        logger.info("=" * 60)
        if args.tune_xgb:
            # Balance your dataset as before
            X_train_final, y_train_final = balance_dataset(X_train, y_train)
            print("🔍 Using Optuna for XGBoost hyperparameter tuning...")
            best_model, best_params = XGBoostModel.optuna_search_xgb(X_train_final, y_train_final, n_trials=30)
            xgb_model = XGBoostModel(model_params=best_params)
            xgb_model.model = best_model
            xgb_model.is_trained = True
        elif args.tune_xgb_grid:
            X_train_final, y_train_final = balance_dataset(X_train, y_train)
            print("🔍 Using GridSearchCV for XGBoost hyperparameter tuning...")
            best_model, best_params = XGBoostModel.grid_search_xgb(X_train_final, y_train_final)
            xgb_model = XGBoostModel(model_params=best_params)
            xgb_model.model = best_model
            xgb_model.is_trained = True
        else:
            xgb_model = train_xgboost_model(X_train, y_train, X_test, y_test, use_balanced_data=True)
        
        # Extract top features from XGBoost model for neural network training
        logger.info("🔍 Extracting top features from XGBoost model...")
        try:
            # Get feature importance from XGBoost
            feature_importance = xgb_model.model.feature_importances_
            feature_names = X_train.columns.tolist()
            
            # Create feature importance dataframe
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            # Select top 10-12 features
            top_n_features = min(12, len(feature_names))
            top_features = feature_importance_df.head(top_n_features)['feature'].tolist()
            
            logger.info(f"Selected top {len(top_features)} features for neural network:")
            for i, feature in enumerate(top_features):
                importance = feature_importance_df[feature_importance_df['feature'] == feature]['importance'].iloc[0]
                logger.info(f"  {i+1}. {feature}: {importance:.4f}")
            
            
            # Save full versions for XGBoost prediction later
            X_train_full = X_train.copy()
            X_test_full = X_test.copy()
            # Reduce X_train to top features
            X_train_reduced = X_train[top_features]
            X_test_reduced = X_test[top_features]
            
            logger.info(f"Reduced feature set: {X_train_reduced.shape[1]} features")
            
        except Exception as e:
            logger.warning(f"Could not extract XGBoost features: {e}")
            logger.info("Using all features for neural network training")
            X_train_reduced = X_train
            X_test_reduced = X_test
            top_features = X_train.columns.tolist()

        # Always use Optuna for neural network training
        X_train_final, y_train_final = balance_dataset(X_train_reduced, y_train)
        logger.info("🔍 Using Optuna for Neural Network hyperparameter tuning...")
        input_size = X_train_final.shape[1]
        from models.neural_network import optuna_search_mlp
        
        params_save_path = f"models/saved/batch_{args.batch}_nn_params.json"
        model, scaler, params, history = optuna_search_mlp(
            X_train_final.values, y_train_final.values, input_size,
            n_trials=30,
            params_save_path=params_save_path
        )
        import torch, joblib
        # Save per-batch model, scaler, and threshold
        batch_model_path = f"models/saved/batch_{args.batch}_nn.pt"
        torch.save(model.state_dict(), batch_model_path)
        joblib.dump(scaler, f"models/saved/batch_{args.batch}_scaler.pkl")
        threshold_info = {
            "best_threshold": history["best_threshold"],
            "best_threshold_score": history["best_threshold_score"]
        }
        with open(f"models/saved/batch_{args.batch}_nn_threshold.json", "w") as f:
            json.dump(threshold_info, f)
        logger.info(f"Optuna-tuned NN for batch {args.batch} saved (+ scaler + threshold). Threshold={history['best_threshold']:.3f}")
        
        # Create NeuralPredictor instance with the trained model
        neural_model = NeuralPredictor(model_type='mlp')
        neural_model.model = model
        neural_model.scaler = scaler
        neural_model.is_trained = True
        neural_model.best_threshold = history['best_threshold']
        neural_training_results = {"history": history}
        # Save metadata for this batch
        metadata = {
            "hidden1": params.get("hidden1"),
            "hidden2": params.get("hidden2"),
            "dropout": params.get("dropout"),
            "learning_rate": params.get("lr"),
            "batch_size": params.get("batch_size"),
            "epochs": history.get("epochs", 0),
            "final_train_loss": history.get("final_train_loss"),
            "final_val_loss": history.get("final_val_loss"),
            "final_val_accuracy": history.get("final_val_accuracy"),
            "threshold": history.get("best_threshold"),
            "threshold_score": history.get("best_threshold_score")
        }
        with open(f"models/saved/batch_{args.batch}_nn_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        # 🧠 STEP 2: GENERATE META-FEATURES FROM THE TRAINED NEURAL NETWORK
        logger.info("=" * 60)
        logger.info("GENERATING META-FEATURES FROM NEURAL NETWORK")
        logger.info("=" * 60)

        # IMPORTANT: Use the same scaler that the NN was trained on
        # The `optuna_search_mlp` function returns the scaler
        
        # Scale the original train/test sets (using the reduced features)
        X_train_scaled_for_meta = scaler.transform(X_train_reduced)
        X_test_scaled_for_meta = scaler.transform(X_test_reduced)

        # Convert to tensors
        X_train_tensor_for_meta = torch.tensor(X_train_scaled_for_meta, dtype=torch.float32).to(device)
        X_test_tensor_for_meta = torch.tensor(X_test_scaled_for_meta, dtype=torch.float32).to(device)

        # Use the new method to get the learned features
        nn_meta_features_train = neural_model.extract_features(X_train_tensor_for_meta)
        nn_meta_features_test = neural_model.extract_features(X_test_tensor_for_meta)

        # Convert to DataFrames
        meta_feature_names = [f'nn_feat_{i}' for i in range(nn_meta_features_train.shape[1])]
        df_meta_train = pd.DataFrame(nn_meta_features_train, columns=meta_feature_names, index=X_train_full.index)
        df_meta_test = pd.DataFrame(nn_meta_features_test, columns=meta_feature_names, index=X_test_full.index)

        logger.info(f"Generated {df_meta_train.shape[1]} meta-features for the training and test sets.")
        # 🧠 STEP 3: MODEL STACKING
        if not args.no_stacking:
            logger.info("=" * 60)
            logger.info("IMPLEMENTING MODEL STACKING")
            logger.info("=" * 60)
            
            try:
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import accuracy_score, roc_auc_score
                
                # Get predictions from both models
                logger.info("Getting probability predictions from both models...")
                
                # XGBoost probabilities (class 1)
                xgb_proba = xgb_model.predict_proba(X_test_full)[:, 1]
                logger.info(f"XGBoost probability range: {xgb_proba.min():.4f} to {xgb_proba.max():.4f}")
                
                # Neural network probabilities (class 1)
                # Convert X_test_reduced to tensor format for neural model
                if hasattr(neural_model, 'scaler') and neural_model.scaler is not None:
                    X_test_scaled = neural_model.scaler.transform(X_test_reduced)
                else:
                    X_test_scaled = X_test_reduced.values
                
                nn_proba = neural_model.predict_proba(X_test_scaled)
                logger.info(f"Neural net probability range: {nn_proba.min():.4f} to {nn_proba.max():.4f}")
                
                # Stack predictions as new features
                stacked_features = np.column_stack((xgb_proba, nn_proba))
                logger.info(f"Stacked features shape: {stacked_features.shape}")
                
                # Train meta-model (LogisticRegression)
                logger.info("Training meta-model (LogisticRegression)...")
                meta_model = LogisticRegression(random_state=42, max_iter=1000)
                meta_model.fit(stacked_features, y_test)
                
                # Evaluate stacked model
                final_preds = meta_model.predict(stacked_features)
                final_proba = meta_model.predict_proba(stacked_features)[:, 1]
                plt.hist(final_proba[y_test == 0], bins=50, alpha=0.5, label="Class 0")
                plt.hist(final_proba[y_test == 1], bins=50, alpha=0.5, label="Class 1")
                plt.title("Stacked Meta-Model Probability Histogram")
                plt.legend()
                plt.savefig(f"outputs/stacked_proba_hist_batch{batch_num}.png")
                plt.close()
                
                # Calculate metrics
                stacked_accuracy = accuracy_score(y_test, final_preds)
                stacked_roc_auc = roc_auc_score(y_test, final_proba)
                
                # Individual model metrics for comparison
                xgb_preds = (xgb_proba > 0.5).astype(int)
                xgb_accuracy = accuracy_score(y_test, xgb_preds)
                xgb_roc_auc = roc_auc_score(y_test, xgb_proba)
                
                nn_preds = (nn_proba > 0.5).astype(int)
                nn_accuracy = accuracy_score(y_test, nn_preds)
                nn_roc_auc = roc_auc_score(y_test, nn_proba)
                
                # Log results
                logger.info("=" * 60)
                logger.info("STACKED MODEL RESULTS")
                logger.info("=" * 60)
                logger.info(f"Individual Models:")
                logger.info(f"  XGBoost     - Accuracy: {xgb_accuracy:.4f}, ROC AUC: {xgb_roc_auc:.4f}")
                logger.info(f"  Neural Net  - Accuracy: {nn_accuracy:.4f}, ROC AUC: {nn_roc_auc:.4f}")
                logger.info(f"Stacked Model:")
                logger.info(f"  Combined    - Accuracy: {stacked_accuracy:.4f}, ROC AUC: {stacked_roc_auc:.4f}")
                
                # Check if stacking improved performance
                best_individual_accuracy = max(xgb_accuracy, nn_accuracy)
                best_individual_roc_auc = max(xgb_roc_auc, nn_roc_auc)
                
                if stacked_accuracy > best_individual_accuracy:
                    logger.info(f"✅ Stacking improved accuracy by {stacked_accuracy - best_individual_accuracy:.4f}")
                else:
                    logger.info(f"⚠️  Stacking did not improve accuracy (best individual: {best_individual_accuracy:.4f})")
                
                if stacked_roc_auc > best_individual_roc_auc:
                    logger.info(f"✅ Stacking improved ROC AUC by {stacked_roc_auc - best_individual_roc_auc:.4f}")
                else:
                    logger.info(f"⚠️  Stacking did not improve ROC AUC (best individual: {best_individual_roc_auc:.4f})")
                
                # Save meta-model
                import joblib
                meta_model_path = f"models/saved/batch_{args.batch}_meta_model.pkl"
                joblib.dump(meta_model, meta_model_path)
                logger.info(f"Meta-model saved to {meta_model_path}")
                
                # Save stacking results
                stacking_results = {
                    'xgb_accuracy': xgb_accuracy,
                    'xgb_roc_auc': xgb_roc_auc,
                    'nn_accuracy': nn_accuracy,
                    'nn_roc_auc': nn_roc_auc,
                    'stacked_accuracy': stacked_accuracy,
                    'stacked_roc_auc': stacked_roc_auc,
                    'meta_model_path': meta_model_path,
                    'improved_accuracy': stacked_accuracy > best_individual_accuracy,
                    'improved_roc_auc': stacked_roc_auc > best_individual_roc_auc
                }
                
                stacking_path = f"models/saved/batch_{args.batch}_stacking_results.json"
                with open(stacking_path, 'w') as f:
                    json.dump(stacking_results, f, indent=2)
                logger.info(f"Stacking results saved to {stacking_path}")
                
            except Exception as e:
                logger.error(f"Model stacking failed: {e}")
                logger.warning("Continuing without stacked model...")
        else:
            logger.info("Model stacking disabled with --no-stacking flag")
        
        # Create visualizations
        logger.info("=" * 60)
        logger.info("CREATING VISUALIZATIONS")
        logger.info("=" * 60)
        
        # Create visualizations
        if VISUALIZATION_AVAILABLE:
            os.makedirs('reports', exist_ok=True)
            # Use reduced feature set for neural network visualization
            create_visualizations(xgb_model, neural_model, X_test_full, y_test, top_features, neural_training_results, batch_num=args.batch)
        else:
            logger.warning("Visualization libraries not available. Skipping chart generation.")
        
        # Save training progress for IEX batches
        if args.batch:
            save_training_progress(args.batch, args.batch_file)
        
        logger.info("=" * 60)
        logger.info("MULTI-TICKER MODEL TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("Trained models saved to:")
        logger.info("  - XGBoost: models/saved/xgb_model.json")
        logger.info("  - Neural Network: models/saved/neural_model.pt")
        if not args.no_stacking:
            logger.info("  - Meta-model: models/saved/batch_{}_meta_model.pkl".format(args.batch))
        logger.info("")
        logger.info("Visualizations saved to reports/ directory")
        
        # Calculate training time and log to CSV
        end_time = time.time()
        training_time = end_time - start_time
        
        if args.batch:
            # Get accuracy metrics
            xgb_accuracy = getattr(xgb_model, '_last_test_accuracy', 0.0)
            nn_accuracy = getattr(neural_model, '_last_val_accuracy', 0.0)
            
            # Import and use training logger  
            from utils.training_logger import log_training_summary
            
            # Log to CSV
            log_training_summary(
                batch_number=args.batch,
                symbols_trained=len(symbols),
                xgb_accuracy=xgb_accuracy,
                nn_accuracy=nn_accuracy,
                training_time_seconds=training_time,
                timeframe=args.bar_timeframe
            )
            
            logger.info("")
            logger.info(f"✅ Batch {args.batch} training completed and marked as done!")
            logger.info("Progress saved to training_progress.json")
        
        logger.info("")
        logger.info(f"Total training time: {training_time/60:.1f} minutes")
        logger.info("=" * 60)
        try:
            logger.info(f"Best XGBoost Params: {best_params}")
        except NameError:
            pass
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 