#!/usr/bin/env python3
"""
LightGBM Regressor Demo with Daily Returns
==========================================

This script demonstrates the new regression-based approach for predicting
daily returns using the LightGBM regressor with Huber loss and threshold optimization.

Features:
- Regression targets based on daily returns (normalized by holding period)
- LightGBM with Huber loss for outlier robustness
- Automatic threshold optimization for binary decision conversion
- Comparison with traditional binary classification approach
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_sample_data():
    """Create sample financial data for demonstration."""
    np.random.seed(42)
    
    # Simulate 6 months of daily data
    n_days = 180
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    
    # Simulate price data with some trend and volatility
    base_price = 100
    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns: 0.05% mean, 2% std
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(500000, 2000000, n_days)
    })
    
    # Add some technical indicators
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['rsi'] = 50 + np.random.normal(0, 15, n_days)  # Simplified RSI
    df['volatility'] = df['close'].pct_change().rolling(10).std()
    
    # Set timestamp as index
    df = df.set_index('timestamp')
    
    return df

def demo_regression_approach():
    """Demonstrate the regression-based approach."""
    print("🚀 LightGBM Regressor Demo: Daily Returns Prediction\n")
    
    # Create sample data
    print("📊 Creating sample financial data...")
    df = create_sample_data()
    print(f"   Data shape: {df.shape}")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    
    # Initialize data collector for target creation
    from data_collection_alpaca import AlpacaDataCollector
    collector = AlpacaDataCollector()
    
    # Create regression targets (daily returns)
    print("\n🎯 Creating regression targets...")
    target_3d = collector._create_regression_target(df, 'DEMO', target_horizon=3)
    target_5d = collector._create_regression_target(df, 'DEMO', target_horizon=5)
    
    # Prepare features (remove NaN values)
    feature_cols = ['sma_5', 'sma_20', 'rsi', 'volatility']
    features = df[feature_cols].dropna()
    
    # Align targets with features
    valid_idx = features.index
    target_3d = target_3d.loc[valid_idx].dropna()
    target_5d = target_5d.loc[valid_idx].dropna()
    
    # Further align features with valid targets
    common_idx = features.index.intersection(target_3d.index).intersection(target_5d.index)
    features = features.loc[common_idx]
    target_3d = target_3d.loc[common_idx]
    target_5d = target_5d.loc[common_idx]
    
    print(f"   3-day targets: {len(target_3d)} valid values")
    print(f"   5-day targets: {len(target_5d)} valid values")
    print(f"   Features shape: {features.shape}")
    
    if len(features) < 50:
        print("❌ Insufficient data for training. Need at least 50 samples.")
        return
    
    # Split data (80% train, 20% test)
    split_idx = int(len(features) * 0.8)
    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train_3d, y_test_3d = target_3d.iloc[:split_idx], target_3d.iloc[split_idx:]
    y_train_5d, y_test_5d = target_5d.iloc[:split_idx], target_5d.iloc[split_idx:]
    
    print(f"\n📈 Training/Test split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train LightGBM Regressor for 3-day predictions
    print(f"\n🤖 Training LightGBM Regressor (3-day horizon)...")
    from models.lightgbm_regressor import LightGBMRegressor
    
    model_3d = LightGBMRegressor(
        name="LightGBM_3Day_Returns",
        objective='huber',
        alpha=0.9,
        learning_rate=0.05,
        num_leaves=15
    )
    
    # Train with validation
    model_3d.train(X_train, y_train_3d, X_test, y_test_3d)
    
    # Make predictions
    reg_pred_3d, binary_pred_3d = model_3d.predict(X_test, return_probabilities=True)
    
    # Print results
    print(f"\n📊 3-Day Model Results:")
    print(f"   Model: {model_3d}")
    metrics_3d = model_3d.get_metrics()
    for metric, value in metrics_3d.items():
        print(f"   {metric}: {value:.4f}")
    
    print(f"\n🎯 Prediction Analysis (3-day):")
    print(f"   Regression predictions range: [{reg_pred_3d.min():.4f}, {reg_pred_3d.max():.4f}]")
    print(f"   Binary predictions: {binary_pred_3d.sum()}/{len(binary_pred_3d)} positive ({100*binary_pred_3d.mean():.1f}%)")
    print(f"   Optimal threshold: {model_3d.optimal_threshold_:.4f}")
    
    # Train 5-day model
    print(f"\n🤖 Training LightGBM Regressor (5-day horizon)...")
    model_5d = LightGBMRegressor(
        name="LightGBM_5Day_Returns",
        objective='huber',
        alpha=0.9,
        learning_rate=0.05,
        num_leaves=15
    )
    
    model_5d.train(X_train, y_train_5d, X_test, y_test_5d)
    reg_pred_5d, binary_pred_5d = model_5d.predict(X_test, return_probabilities=True)
    
    print(f"\n📊 5-Day Model Results:")
    print(f"   Model: {model_5d}")
    metrics_5d = model_5d.get_metrics()
    for metric, value in metrics_5d.items():
        print(f"   {metric}: {value:.4f}")
    
    print(f"\n🎯 Prediction Analysis (5-day):")
    print(f"   Regression predictions range: [{reg_pred_5d.min():.4f}, {reg_pred_5d.max():.4f}]")
    print(f"   Binary predictions: {binary_pred_5d.sum()}/{len(binary_pred_5d)} positive ({100*binary_pred_5d.mean():.1f}%)")
    print(f"   Optimal threshold: {model_5d.optimal_threshold_:.4f}")
    
    # Feature importance
    print(f"\n🔍 Feature Importance (3-day model):")
    importance_3d = model_3d.get_feature_importance()
    if importance_3d is not None:
        for feature, importance in importance_3d.head().items():
            print(f"   {feature}: {importance:.3f}")
    
    # Summary
    print(f"\n✅ Demo Summary:")
    print(f"   • Successfully trained LightGBM regressors with Huber loss")
    print(f"   • Predicted daily returns for 3-day and 5-day horizons")
    print(f"   • Automatically found optimal thresholds for binary decisions")
    print(f"   • Regression approach provides continuous predictions with robustness")
    print(f"   • Ready for integration into full training pipeline")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Run full pipeline: python train_models.py --models lightgbm_regressor")
    print(f"   2. Include in ensemble: python train_models.py --models xgboost lightgbm lightgbm_regressor catboost")
    print(f"   3. Use regression targets: Data collection already defaults to use_regression=True")

if __name__ == "__main__":
    demo_regression_approach()
