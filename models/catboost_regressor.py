
"""
CatBoost Regressor Model
========================

This module implements a CatBoost regressor for predicting daily returns
in the Investment Committee project. It uses Huber loss for robustness
against outliers and includes threshold optimization for converting
regression predictions to binary decisions.

Features:
- Huber loss objective for outlier robustness
- Regression with threshold optimization
- Early stopping capabilities
- Feature importance analysis
- Model persistence
- Categorical feature handling
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple, Union, List
import joblib
import os

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostRegressor = None

from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_recall_curve, f1_score

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class CatBoostRegressorModel(BaseModel):
    """
    CatBoost regressor for predicting daily returns with Huber loss.
    
    This model predicts continuous daily returns and includes methods
    for converting predictions to binary decisions via threshold optimization.
    """
    
    def __init__(self, name: str = "CatBoostRegressor",
                 objective: str = 'Huber',
                 huber_slope: float = 1.0,  # CatBoost uses 'huber_slope' instead of 'huber_alpha'
                 iterations: int = 1000,
                 depth: int = 6,
                 learning_rate: float = 0.03,
                 l2_leaf_reg: float = 3.0,
                 random_seed: int = 42,
                 verbose: bool = False,
                 **kwargs):
        """
        Initialize CatBoost regressor with Huber loss.
        
        Args:
            name: Model name
            objective: CatBoost objective function ('Huber' for Huber loss)
            huber_slope: Huber loss parameter (slope parameter for CatBoost)
            iterations: Number of boosting iterations
            depth: Tree depth
            learning_rate: Learning rate
            l2_leaf_reg: L2 regularization
            random_seed: Random seed
            verbose: Whether to show training progress
            **kwargs: Additional CatBoost parameters
        """
        super().__init__(name)
        
        if not CATBOOST_AVAILABLE:
            self.log("❌ CatBoost not available. Install with: pip install catboost")
            self.model = None
            self.params = {}
            self.validation_metrics = {}
            self.feature_importance_ = None
            self.optimal_threshold_ = 0.0
            return
        
        # Store hyperparameters - CatBoost uses different parameter names
        self.params = {
            'loss_function': objective,
            'iterations': iterations,
            'depth': depth,
            'learning_rate': learning_rate,
            'l2_leaf_reg': l2_leaf_reg,
            'random_seed': random_seed,
            'verbose': verbose,
            'thread_count': -1,  # Use all available cores
            **kwargs
        }
        
        # Initialize model (CatBoost Huber loss doesn't need extra parameters in constructor)
        self.model = CatBoostRegressor(**self.params)
        
        # Training tracking
        self.validation_metrics = {}
        self.feature_importance_ = None
        self.optimal_threshold_ = 0.0
        
        self.log(f"Initialized CatBoost Regressor with Huber loss (delta={huber_slope})")

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
        early_stopping_rounds = kwargs.get('early_stopping_rounds', 100)
        find_optimal_threshold = kwargs.get('find_optimal_threshold', True)
        
        self.train(X, y, X_val, y_val, early_stopping_rounds, find_optimal_threshold)
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
              early_stopping_rounds: int = 100,
              find_optimal_threshold: bool = True,
              cat_features: Optional[List] = None) -> None:
        """
        Train the CatBoost regressor with optional validation and early stopping.
        
        Args:
            X_train: Training features
            y_train: Training targets (continuous returns)
            X_val: Validation features
            y_val: Validation targets
            early_stopping_rounds: Early stopping patience
            find_optimal_threshold: Whether to find optimal threshold for binary decisions
            cat_features: List of categorical feature indices or names
        """
        if not CATBOOST_AVAILABLE or self.model is None:
            raise ValueError("CatBoost not available")
        
        try:
            # Remove NaN values
            train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
            X_train_clean = X_train[train_mask]
            y_train_clean = y_train[train_mask]
            
            self.log(f"Training on {len(X_train_clean)} samples (removed {(~train_mask).sum()} NaN samples)")
            
            if X_val is not None and y_val is not None:
                # Remove NaN values from validation set
                val_mask = ~(X_val.isna().any(axis=1) | y_val.isna())
                X_val_clean = X_val[val_mask]
                y_val_clean = y_val[val_mask]
                
                self.log(f"Validating on {len(X_val_clean)} samples")
                
                # Train with validation
                self.model.fit(
                    X_train_clean, y_train_clean,
                    eval_set=(X_val_clean, y_val_clean),
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False,
                    cat_features=cat_features
                )
                
                # Calculate validation metrics
                val_pred = self.model.predict(X_val_clean)
                self.validation_metrics = {
                    'mse': mean_squared_error(y_val_clean, val_pred),
                    'mae': mean_absolute_error(y_val_clean, val_pred),
                    'rmse': np.sqrt(mean_squared_error(y_val_clean, val_pred))
                }
                
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
                # Train without validation
                self.model.fit(X_train_clean, y_train_clean, cat_features=cat_features)
                
                # Use training data to find threshold if no validation data
                if find_optimal_threshold:
                    train_pred = self.model.predict(X_train_clean)
                    self.optimal_threshold_ = self._find_optimal_threshold(train_pred, y_train_clean)
                    self.log(f"Optimal threshold (from training): {self.optimal_threshold_:.4f}")
            
            # Store feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance_ = pd.Series(
                    self.model.feature_importances_,
                    index=X_train_clean.columns
                ).sort_values(ascending=False)
            
            self.log(f"Training completed. Best iteration: {getattr(self.model, 'best_iteration_', 'N/A')}")
            
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
        if not CATBOOST_AVAILABLE or self.model is None:
            raise ValueError("Model not trained or CatBoost not available")
        
        try:
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
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # CatBoost models can be saved natively
            catboost_path = filepath.replace('.pkl', '.cbm')
            self.model.save_model(catboost_path)
            
            # Save additional metadata
            metadata = {
                'params': self.params,
                'feature_importance': self.feature_importance_,
                'validation_metrics': self.validation_metrics,
                'optimal_threshold': self.optimal_threshold_,
                'name': self.name,
                'catboost_path': catboost_path
            }
            
            joblib.dump(metadata, filepath)
            self.log(f"Model saved to {filepath} (CatBoost model: {catboost_path})")
            
        except Exception as e:
            self.log(f"❌ Failed to save model: {str(e)}")
            raise

    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        try:
            # Load metadata
            metadata = joblib.load(filepath)
            
            self.params = metadata['params']
            self.feature_importance_ = metadata.get('feature_importance')
            self.validation_metrics = metadata.get('validation_metrics', {})
            self.optimal_threshold_ = metadata.get('optimal_threshold', 0.0)
            self.name = metadata.get('name', self.name)
            
            # Load CatBoost model
            catboost_path = metadata.get('catboost_path', filepath.replace('.pkl', '.cbm'))
            if os.path.exists(catboost_path):
                self.model = CatBoostRegressor()
                self.model.load_model(catboost_path)
            else:
                raise FileNotFoundError(f"CatBoost model file not found: {catboost_path}")
            
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
