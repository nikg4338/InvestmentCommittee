
"""
Random Forest Regressor Model
=============================

This module implements a Random Forest regressor for predicting daily returns
in the Investment Committee project. While Random Forest doesn't natively support
Huber loss, this implementation includes robust features and threshold optimization
for converting regression predictions to binary decisions.

Features:
- Ensemble regression for robust predictions
- Threshold optimization for binary decisions
- Feature importance analysis
- Out-of-bag scoring
- Model persistence
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple, Union
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_recall_curve, f1_score

from .base_model import BaseModel, clean_data_for_model_prediction

logger = logging.getLogger(__name__)

class RandomForestRegressorModel(BaseModel):
    """
    Random Forest regressor for predicting daily returns.
    
    This model predicts continuous daily returns and includes methods
    for converting predictions to binary decisions via threshold optimization.
    """
    
    def __init__(self, name: str = "RandomForestRegressor",
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str = 'sqrt',
                 bootstrap: bool = True,
                 oob_score: bool = True,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 **kwargs):
        """
        Initialize Random Forest regressor.
        
        Args:
            name: Model name
            n_estimators: Number of trees in the forest
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required at leaf
            max_features: Number of features for best split
            bootstrap: Whether bootstrap samples are used
            oob_score: Whether to compute out-of-bag score
            random_state: Random seed
            n_jobs: Number of parallel jobs
            **kwargs: Additional RandomForest parameters
        """
        super().__init__(name)
        
        # Store hyperparameters
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'bootstrap': bootstrap,
            'oob_score': oob_score,
            'random_state': random_state,
            'n_jobs': n_jobs,
            **kwargs
        }
        
        # Initialize model
        self.model = RandomForestRegressor(**self.params)
        
        # Training tracking
        self.validation_metrics = {}
        self.feature_importance_ = None
        self.optimal_threshold_ = 0.0
        
        self.log(f"Initialized Random Forest Regressor with {n_estimators} trees")

    def fit(self, X, y, **kwargs):
        """
        BaseModel interface for training.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional arguments passed to train method
        """
        # Extract validation data if provided
        X_val = kwargs.get('X_val')
        y_val = kwargs.get('y_val')
        find_optimal_threshold = kwargs.get('find_optimal_threshold', True)
        
        self.train(X, y, X_val, y_val, find_optimal_threshold)
        self.is_trained = True

    def save(self, path: str) -> bool:
        """
        BaseModel interface for saving.
        
        Args:
            path: File path to save the model
            
        Returns:
            True if successful
        """
        try:
            self.save_model(path)
            return True
        except Exception as e:
            self.log(f"❌ Save failed: {str(e)}")
            return False

    def load(self, path: str) -> bool:
        """
        BaseModel interface for loading.
        
        Args:
            path: File path to load the model from
            
        Returns:
            True if successful
        """
        try:
            self.load_model(path)
            self.is_trained = True
            return True
        except Exception as e:
            self.log(f"❌ Load failed: {str(e)}")
            return False

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              find_optimal_threshold: bool = True) -> None:
        """
        Train the Random Forest regressor with optional validation.
        
        Args:
            X_train: Training features
            y_train: Training targets (continuous returns)
            X_val: Validation features
            y_val: Validation targets
            find_optimal_threshold: Whether to find optimal threshold for binary decisions
        """
        try:
            # Clean data for training (remove categorical columns)
            X_train = clean_data_for_model_prediction(X_train)
            if X_val is not None:
                X_val = clean_data_for_model_prediction(X_val)
            
            # Remove NaN values
            train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
            X_train_clean = X_train[train_mask]
            y_train_clean = y_train[train_mask]
            
            self.log(f"Training on {len(X_train_clean)} samples (removed {(~train_mask).sum()} NaN samples)")
            
            # Train the model
            self.model.fit(X_train_clean, y_train_clean)
            
            # Store feature importance
            self.feature_importance_ = pd.Series(
                self.model.feature_importances_,
                index=X_train_clean.columns
            ).sort_values(ascending=False)
            
            # Calculate metrics
            if X_val is not None and y_val is not None:
                # Remove NaN values from validation set
                val_mask = ~(X_val.isna().any(axis=1) | y_val.isna())
                X_val_clean = X_val[val_mask]
                y_val_clean = y_val[val_mask]
                
                self.log(f"Validating on {len(X_val_clean)} samples")
                
                # Calculate validation metrics
                val_pred = self.model.predict(X_val_clean)
                self.validation_metrics = {
                    'mse': mean_squared_error(y_val_clean, val_pred),
                    'mae': mean_absolute_error(y_val_clean, val_pred),
                    'rmse': np.sqrt(mean_squared_error(y_val_clean, val_pred))
                }
                
                # Add OOB score if available
                if hasattr(self.model, 'oob_score_') and self.model.oob_score_:
                    self.validation_metrics['oob_score'] = self.model.oob_score_
                
                # Find optimal threshold for binary decisions
                if find_optimal_threshold:
                    self.optimal_threshold_ = self._find_optimal_threshold(val_pred, y_val_clean)
                    
                    # Calculate F1 score at optimal threshold
                    binary_pred = (val_pred > self.optimal_threshold_).astype(int)
                    binary_true = (y_val_clean > 0).astype(int)
                    f1 = f1_score(binary_true, binary_pred, zero_division=0)
                    self.validation_metrics['f1_score'] = f1
                    
                    self.log(f"Optimal threshold: {self.optimal_threshold_:.4f}, F1: {f1:.3f}")
                
            else:
                # Use training data metrics if no validation data
                train_pred = self.model.predict(X_train_clean)
                self.validation_metrics = {
                    'mse': mean_squared_error(y_train_clean, train_pred),
                    'mae': mean_absolute_error(y_train_clean, train_pred),
                    'rmse': np.sqrt(mean_squared_error(y_train_clean, train_pred))
                }
                
                # Add OOB score if available
                if hasattr(self.model, 'oob_score_') and self.model.oob_score_:
                    self.validation_metrics['oob_score'] = self.model.oob_score_
                
                # Use training data to find threshold if no validation data
                if find_optimal_threshold:
                    self.optimal_threshold_ = self._find_optimal_threshold(train_pred, y_train_clean)
                    self.log(f"Optimal threshold (from training): {self.optimal_threshold_:.4f}")
            
            self.log(f"Training completed. OOB Score: {getattr(self.model, 'oob_score_', 'N/A')}")
            
        except Exception as e:
            self.log(f"❌ Training failed: {str(e)}")
            raise
    
    def _find_optimal_threshold(self, predictions: np.ndarray, true_values: np.ndarray) -> float:
        """
        Find optimal threshold for converting regression predictions to binary decisions.
        
        Args:
            predictions: Regression predictions
            true_values: True target values
            
        Returns:
            Optimal threshold value
        """
        # Convert true values to binary (positive returns = 1)
        binary_true = (true_values > 0).astype(int)
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(binary_true, predictions)
        
        # Find threshold that maximizes F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.0
        
        self.log(f"Threshold optimization: Best F1={f1_scores[best_idx]:.3f} at threshold={optimal_threshold:.4f}")
        
        return optimal_threshold

    def predict(self, X: pd.DataFrame, return_probabilities: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            return_probabilities: If True, return both regression predictions and binary probabilities
            
        Returns:
            Regression predictions or tuple of (regression_pred, binary_pred)
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        try:
            # Clean data for prediction
            X = clean_data_for_model_prediction(X)
            
            # Get regression predictions
            reg_predictions = self.model.predict(X)
            
            if return_probabilities:
                # Convert to binary predictions using optimal threshold
                binary_predictions = (reg_predictions > self.optimal_threshold_).astype(int)
                return reg_predictions, binary_predictions
            else:
                return reg_predictions
                
        except Exception as e:
            self.log(f"❌ Prediction failed: {str(e)}")
            raise

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance scores."""
        return self.feature_importance_

    def get_metrics(self) -> Dict[str, float]:
        """Get validation metrics."""
        return self.validation_metrics.copy()

    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        try:
            model_data = {
                'model': self.model,
                'params': self.params,
                'feature_importance': self.feature_importance_,
                'validation_metrics': self.validation_metrics,
                'optimal_threshold': self.optimal_threshold_,
                'name': self.name
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(model_data, filepath)
            self.log(f"Model saved to {filepath}")
            
        except Exception as e:
            self.log(f"❌ Failed to save model: {str(e)}")
            raise

    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.params = model_data['params']
            self.feature_importance_ = model_data.get('feature_importance')
            self.validation_metrics = model_data.get('validation_metrics', {})
            self.optimal_threshold_ = model_data.get('optimal_threshold', 0.0)
            self.name = model_data.get('name', self.name)
            
            self.log(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.log(f"❌ Failed to load model: {str(e)}")
            raise

    def __str__(self) -> str:
        """String representation of the model."""
        if self.model is None:
            return f"{self.name} (not trained)"
        
        metrics_str = ", ".join([f"{k}={v:.3f}" for k, v in self.validation_metrics.items()])
        return f"{self.name} (trained, threshold={self.optimal_threshold_:.3f}, {metrics_str})"
