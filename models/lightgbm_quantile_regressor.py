#!/usr/bin/env python3
"""
LightGBM Quantile Regression Model
==================================

This module implements a LightGBM quantile regression model for predicting
multiple quantiles of daily returns in the Investment Committee project.
It provides uncertainty estimation and risk-aware predictions for better
investment decision making.

Features:
- Multiple quantile predictions (e.g., 0.1, 0.5, 0.9)
- Pinball loss optimization for each quantile
- Uncertainty-based confidence intervals
- Risk-aware threshold selection
- Integration with existing ensemble pipeline

Phase 3 of Advanced Signal Improvements: Quantile Loss Options
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple, Union, List
import joblib
import os

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

from sklearn.model_selection import train_test_split

from .base_model import BaseModel

# Import quantile utilities using absolute import
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.quantile_loss import (
    pinball_loss, quantile_score, calculate_prediction_intervals,
    risk_aware_threshold_selection, evaluate_quantile_predictions,
    quantile_to_binary_predictions, validate_quantile_levels
)

logger = logging.getLogger(__name__)

class LightGBMQuantileRegressor(BaseModel):
    """
    LightGBM quantile regressor for predicting multiple quantiles of daily returns.
    
    This model predicts multiple quantiles simultaneously and includes methods
    for uncertainty estimation and risk-aware decision making.
    """
    
    def __init__(self, name: str = "LightGBMQuantileRegressor",
                 quantile_levels: Optional[List[float]] = None,
                 boosting_type: str = 'gbdt',
                 num_leaves: int = 31,
                 learning_rate: float = 0.1,
                 feature_fraction: float = 0.9,
                 bagging_fraction: float = 0.8,
                 bagging_freq: int = 5,
                 verbose: int = -1,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize LightGBM quantile regressor.
        
        Args:
            name: Model name
            quantile_levels: List of quantile levels to predict (e.g., [0.1, 0.5, 0.9])
            boosting_type: Boosting type for LightGBM
            num_leaves: Number of leaves in trees
            learning_rate: Learning rate
            feature_fraction: Feature sampling fraction
            bagging_fraction: Data sampling fraction
            bagging_freq: Bagging frequency
            verbose: Verbosity level
            random_state: Random state for reproducibility
            **kwargs: Additional parameters
        """
        super().__init__(name)
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required for LightGBMQuantileRegressor")
        
        if quantile_levels is None:
            quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        self.quantile_levels = validate_quantile_levels(quantile_levels)
        self.models = {}  # Dictionary to store one model per quantile
        self.validation_metrics = {}
        self.feature_importance_ = None
        
        # Base parameters for all quantile models
        self.base_params = {
            'boosting_type': boosting_type,
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'verbose': verbose,
            'random_state': random_state,
            **kwargs
        }
        
        self.log(f"Initialized LightGBM quantile regressor with {len(self.quantile_levels)} quantiles: {self.quantile_levels}")
    
    def predict(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        """
        Predict multiple quantiles for input features.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary mapping quantile levels to predictions
        """
        if not self.models:
            raise ValueError("Model not trained. Call train() first.")
        
        # Clean input data
        X_clean = X.copy()
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # Remove rows with NaN values
        mask = ~X_clean.isnull().any(axis=1)
        X_final = X_clean[mask]
        
        if len(X_final) == 0:
            self.log("Warning: All samples contain NaN values after cleaning")
            return {q: np.array([]) for q in self.quantile_levels}
        
        # Predict for each quantile
        quantile_predictions = {}
        for quantile in self.quantile_levels:
            if quantile in self.models:
                pred = self.models[quantile].predict(X_final)
                
                # Handle case where some rows were removed due to NaN
                if len(pred) < len(X):
                    full_pred = np.full(len(X), np.nan)
                    full_pred[mask] = pred
                    quantile_predictions[quantile] = full_pred
                else:
                    quantile_predictions[quantile] = pred
            else:
                self.log(f"Warning: No model trained for quantile {quantile}")
                quantile_predictions[quantile] = np.zeros(len(X))
        
        return quantile_predictions
    
    def predict_single_quantile(self, X: pd.DataFrame, quantile: float) -> np.ndarray:
        """
        Predict a single quantile.
        
        Args:
            X: Input features
            quantile: Quantile level to predict
            
        Returns:
            Predictions for the specified quantile
        """
        quantile_preds = self.predict(X)
        return quantile_preds.get(quantile, np.zeros(len(X)))
    
    def get_prediction_intervals(self, X: pd.DataFrame, confidence_level: float = 0.8) -> Dict[str, np.ndarray]:
        """
        Get prediction intervals from quantile predictions.
        
        Args:
            X: Input features
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with interval bounds and width
        """
        quantile_preds = self.predict(X)
        return calculate_prediction_intervals(quantile_preds, confidence_level)
    
    def predict_binary(self, X: pd.DataFrame, 
                      decision_strategy: str = 'threshold_optimization',
                      risk_tolerance: str = 'moderate',
                      y_true: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Convert quantile predictions to binary predictions.
        
        Args:
            X: Input features
            decision_strategy: Strategy for binary conversion
            risk_tolerance: Risk tolerance level
            y_true: True values for threshold optimization
            
        Returns:
            Tuple of (binary_predictions, conversion_info)
        """
        quantile_preds = self.predict(X)
        return quantile_to_binary_predictions(
            quantile_preds, y_true, decision_strategy, risk_tolerance
        )
    
    def fit(self, X, y, **kwargs):
        """Compatibility method for sklearn-style interface"""
        return self.train(X, y, **kwargs)
    
    def save(self, path: str) -> bool:
        """
        Save all quantile models to disk.
        
        Args:
            path: Base path for saving models
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save each quantile model
            for quantile, model in self.models.items():
                model_path = f"{path}_q{quantile:.2f}.lgb"
                model.save_model(model_path)
            
            # Save metadata
            metadata = {
                'quantile_levels': self.quantile_levels,
                'base_params': self.base_params,
                'validation_metrics': self.validation_metrics,
                'feature_importance': self.feature_importance_
            }
            
            metadata_path = f"{path}_metadata.pkl"
            joblib.dump(metadata, metadata_path)
            
            self.log(f"Model saved to {path}")
            return True
            
        except Exception as e:
            self.log(f"Error saving model: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """
        Load all quantile models from disk.
        
        Args:
            path: Base path for loading models
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load metadata
            metadata_path = f"{path}_metadata.pkl"
            metadata = joblib.load(metadata_path)
            
            self.quantile_levels = metadata['quantile_levels']
            self.base_params = metadata['base_params']
            self.validation_metrics = metadata.get('validation_metrics', {})
            self.feature_importance_ = metadata.get('feature_importance')
            
            # Load each quantile model
            self.models = {}
            for quantile in self.quantile_levels:
                model_path = f"{path}_q{quantile:.2f}.lgb"
                if os.path.exists(model_path):
                    self.models[quantile] = lgb.Booster(model_file=model_path)
                else:
                    self.log(f"Warning: Model file not found for quantile {quantile}: {model_path}")
            
            self.log(f"Model loaded from {path}")
            return True
            
        except Exception as e:
            self.log(f"Error loading model: {e}")
            return False
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              early_stopping_rounds: int = 100,
              positive_weight: float = 10.0,
              use_smote: bool = True) -> None:
        """
        Train quantile regression models for all specified quantiles.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            early_stopping_rounds: Early stopping rounds
            positive_weight: Weight for positive samples
            use_smote: Whether to use SMOTE (will be applied if enabled)
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required for training")
        
        self.log(f"Training quantile regression for {len(self.quantile_levels)} quantiles...")
        
        # Apply SMOTE enhancement if enabled
        X_train_enhanced = X_train.copy()
        y_train_enhanced = y_train.copy()
        
        if use_smote:
            try:
                from ..utils.sampling import apply_combined_enhancement
                X_train_enhanced, y_train_enhanced = apply_combined_enhancement(
                    X_train, y_train, positive_weight=positive_weight
                )
                self.log(f"SMOTE + sample weighting applied: {len(X_train)} → {len(X_train_enhanced)} samples")
            except Exception as e:
                self.log(f"SMOTE enhancement failed, using sample weighting only: {e}")
        
        # Clean training data
        X_train_enhanced = X_train_enhanced.replace([np.inf, -np.inf], np.nan)
        
        # Remove rows with NaN values
        train_mask = ~(X_train_enhanced.isnull().any(axis=1) | y_train_enhanced.isnull())
        X_train_clean = X_train_enhanced[train_mask]
        y_train_clean = y_train_enhanced[train_mask]
        
        if len(X_train_clean) == 0:
            raise ValueError("No valid training samples after cleaning")
        
        self.log(f"Training data after cleaning: {len(X_train_clean)} samples")
        
        # Apply sample weights for positive examples
        if not use_smote:  # Only apply if SMOTE wasn't used
            sample_weights = np.where(y_train_clean > 0, positive_weight, 1.0)
            positive_count = np.sum(y_train_clean > 0)
            self.log(f"Sample weighting: {positive_count}/{len(y_train_clean)} positive samples with {positive_weight}x weight")
        else:
            sample_weights = None
        
        # Train a separate model for each quantile
        self.models = {}
        all_metrics = {}
        
        for quantile in self.quantile_levels:
            self.log(f"Training quantile {quantile:.2f}...")
            
            # Set up quantile-specific parameters
            params = self.base_params.copy()
            params.update({
                'objective': 'quantile',
                'alpha': quantile,  # Quantile parameter for LightGBM
                'metric': 'quantile'
            })
            
            # Get feature names for dataset creation
            feature_names = list(X_train_clean.columns) if hasattr(X_train_clean, 'columns') else None
            
            # Create LightGBM dataset with sample weights
            if sample_weights is not None:
                train_set = lgb.Dataset(
                    X_train_clean, 
                    label=y_train_clean,
                    weight=sample_weights,
                    feature_name=feature_names
                )
            else:
                train_set = lgb.Dataset(
                    X_train_clean, 
                    label=y_train_clean,
                    feature_name=feature_names
                )
            
            # Set up validation
            valid_sets = [train_set]
            if X_val is not None and y_val is not None:
                # Clean validation data
                val_mask = ~(X_val.isna().any(axis=1) | y_val.isna())
                X_val_clean = X_val[val_mask]
                y_val_clean = y_val[val_mask]
                
                if len(X_val_clean) > 0:
                    val_set = lgb.Dataset(
                        X_val_clean, 
                        label=y_val_clean,
                        reference=train_set,
                        feature_name=feature_names
                    )
                    valid_sets.append(val_set)
                    
                    # Train with validation
                    model = lgb.train(
                        params,
                        train_set,
                        valid_sets=valid_sets,
                        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
                        num_boost_round=1000
                    )
                    
                    # Calculate validation metrics
                    val_pred = model.predict(X_val_clean)
                    val_metrics = {
                        f'q_{quantile:.2f}_pinball_loss': pinball_loss(y_val_clean, val_pred, quantile),
                        f'q_{quantile:.2f}_quantile_score': quantile_score(y_val_clean, val_pred, quantile)
                    }
                    all_metrics.update(val_metrics)
                    
                    self.log(f"Quantile {quantile:.2f} validation - Pinball loss: {val_metrics[f'q_{quantile:.2f}_pinball_loss']:.6f}")
                    
                else:
                    # Train without validation
                    model = lgb.train(
                        params,
                        train_set,
                        num_boost_round=1000
                    )
            else:
                # Train without validation
                model = lgb.train(
                    params,
                    train_set,
                    num_boost_round=1000
                )
            
            self.models[quantile] = model
        
        # Calculate combined validation metrics if we have validation data
        if X_val is not None and y_val is not None and len(X_val_clean) > 0:
            # Get all quantile predictions for comprehensive evaluation
            quantile_preds = {}
            for quantile, model in self.models.items():
                quantile_preds[quantile] = model.predict(X_val_clean)
            
            # Evaluate quantile predictions
            quantile_metrics = evaluate_quantile_predictions(y_val_clean, quantile_preds)
            all_metrics.update(quantile_metrics)
            
            self.validation_metrics = all_metrics
        
        # Store feature importance (from median quantile model if available)
        if 0.5 in self.models:
            feature_names = list(X_train_clean.columns) if hasattr(X_train_clean, 'columns') else [f'feature_{i}' for i in range(X_train_clean.shape[1])]
            importance_scores = self.models[0.5].feature_importance(importance_type='gain')
            self.feature_importance_ = dict(zip(feature_names, importance_scores))
        
        self.log(f"✅ Quantile regression training completed for {len(self.models)} quantiles")
