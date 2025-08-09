#!/usr/bin/env python3
"""
ULTIMATE DATA LEAKAGE TEST
=========================

This script creates a completely clean test by:
1. Removing ALL potentially leaky features
2. Using only features that are guaranteed to be available BEFORE the prediction
3. Testing the model on this truly clean dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, classification_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def remove_all_leaky_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove ALL features that could possibly contain future information.
    """
    # List of obviously leaky features
    definitely_leaky = [
        'pnl_ratio',           # This IS the target calculation!
        'daily_return',        # Also future-looking
        'holding_days',        # Future knowledge
    ]
    
    # Potentially leaky features (anything with target in name)
    potentially_leaky = [col for col in df.columns if 'target' in col.lower() and col != 'target']
    
    # Features that might use future data patterns
    suspicious_features = [
        'gap_follow_through',  # Might use future price action
        'risk_adjusted_return_5d',  # Risk adjustment might be forward-looking
        'risk_adjusted_return_20d',
        'quality_score',       # Might aggregate future-looking metrics
    ]
    
    # Combine all potentially problematic features
    features_to_remove = definitely_leaky + potentially_leaky + suspicious_features
    
    # Keep only clearly backwards-looking features
    clean_df = df.copy()
    
    removed_features = []
    for feature in features_to_remove:
        if feature in clean_df.columns:
            clean_df = clean_df.drop(columns=[feature])
            removed_features.append(feature)
    
    logger.info(f"🚫 Removed {len(removed_features)} potentially leaky features:")
    for feature in removed_features:
        logger.info(f"   - {feature}")
    
    return clean_df

def test_clean_model_performance():
    """Test model performance on truly clean data."""
    
    # Load the dataset
    df = pd.read_csv('data/ultra_clean_batch.csv')
    logger.info(f"📊 Original dataset: {len(df)} samples, {len(df.columns)} columns")
    
    # Remove all leaky features
    clean_df = remove_all_leaky_features(df)
    logger.info(f"📊 Clean dataset: {len(clean_df)} samples, {len(clean_df.columns)} columns")
    
    # Prepare features and target
    feature_cols = [col for col in clean_df.columns if col not in ['ticker', 'target', 'timestamp']]
    
    logger.info(f"🔧 Using {len(feature_cols)} clean features:")
    for i, col in enumerate(feature_cols[:10]):  # Show first 10
        logger.info(f"   {i+1}. {col}")
    if len(feature_cols) > 10:
        logger.info(f"   ... and {len(feature_cols) - 10} more")
    
    X = clean_df[feature_cols]
    y = clean_df['target']
    
    logger.info(f"🎯 Target distribution: {y.value_counts().to_dict()}")
    logger.info(f"🎯 Positive rate: {y.mean():.1%}")
    
    # Clean any remaining NaN values
    X = X.fillna(0)
    
    # Temporal split (not random!) - use earlier data for training, later for testing
    # Sort by timestamp if available
    if 'timestamp' in clean_df.columns:
        clean_df_sorted = clean_df.sort_values('timestamp').reset_index(drop=True)
        split_idx = int(len(clean_df_sorted) * 0.8)
        
        train_indices = clean_df_sorted.index[:split_idx]
        test_indices = clean_df_sorted.index[split_idx:]
        
        X_train = X.loc[train_indices]
        X_test = X.loc[test_indices]
        y_train = y.loc[train_indices]
        y_test = y.loc[test_indices]
        
        logger.info("📅 Using TEMPORAL split (no data leakage)")
        logger.info(f"   Training period: {clean_df_sorted.loc[train_indices, 'timestamp'].min()} to {clean_df_sorted.loc[train_indices, 'timestamp'].max()}")
        logger.info(f"   Test period: {clean_df_sorted.loc[test_indices, 'timestamp'].min()} to {clean_df_sorted.loc[test_indices, 'timestamp'].max()}")
    else:
        # Fallback to random split (less ideal)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info("📅 Using random split (timestamp not available)")
    
    logger.info(f"📊 Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Train model
    logger.info("🤖 Training Random Forest on CLEAN data...")
    rf = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced',
        max_depth=10,  # Limit depth to prevent overfitting
        min_samples_split=20,  # Require more samples to split
        min_samples_leaf=10    # Require more samples in leaves
    )
    rf.fit(X_train, y_train)
    
    # Predict
    y_proba = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)
    
    # Calculate metrics
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    
    # Per-ticker analysis
    logger.info("📊 Per-ticker performance on CLEAN data:")
    test_df = X_test.copy()
    test_df['true_target'] = y_test
    test_df['predicted_target'] = y_pred
    test_df['probability'] = y_proba
    test_df['ticker'] = clean_df.loc[X_test.index, 'ticker']
    
    for ticker in test_df['ticker'].unique():
        ticker_data = test_df[test_df['ticker'] == ticker]
        if len(ticker_data) > 10:  # Only analyze tickers with enough samples
            ticker_precision, ticker_recall, _ = precision_recall_curve(
                ticker_data['true_target'], ticker_data['probability']
            )
            ticker_auc = auc(ticker_recall, ticker_precision)
            
            actual_pos_rate = ticker_data['true_target'].mean()
            pred_pos_rate = ticker_data['predicted_target'].mean()
            
            logger.info(f"   {ticker}: {len(ticker_data)} samples")
            logger.info(f"      Actual: {actual_pos_rate:.1%} positive")
            logger.info(f"      Predicted: {pred_pos_rate:.1%} positive") 
            logger.info(f"      PR-AUC: {ticker_auc:.4f}")
    
    # Overall results
    print(f"""
🎉 CLEAN MODEL TEST RESULTS:

📊 Dataset: {len(clean_df)} samples, {len(feature_cols)} clean features
🎯 Target: {y.mean():.1%} positive rate

🤖 Model Performance (NO DATA LEAKAGE):
   📈 PR-AUC: {pr_auc:.4f}
   
🔍 INTERPRETATION:
""")
    
    if pr_auc > 0.9:
        print("   🚨 STILL SUSPICIOUSLY HIGH - More investigation needed!")
        print("   Possible remaining issues:")
        print("   - Features may still contain subtle future information")
        print("   - Data collection itself may have temporal issues")
        print("   - There may be other forms of leakage not yet identified")
    elif pr_auc > 0.7:
        print("   ⚠️  MODERATELY HIGH - Could be legitimate but should investigate")
        print("   - This performance level could be realistic for some market patterns")
        print("   - But worth double-checking feature engineering")
    elif pr_auc > 0.55:
        print("   ✅ REALISTIC PERFORMANCE - This looks legitimate!")
        print("   - Performance level consistent with authentic market prediction")
        print("   - No obvious signs of data leakage")
    else:
        print("   📉 LOW PERFORMANCE - Model may not be learning well")
        print("   - Could indicate successful leakage removal")
        print("   - Or insufficient signal in the clean features")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"""
🔍 TOP 10 MOST IMPORTANT CLEAN FEATURES:""")
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    return pr_auc, feature_importance

if __name__ == "__main__":
    test_clean_model_performance()
