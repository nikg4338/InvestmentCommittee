
"""
SVM Regressor Model
===================

This module implements an SVM regressor for predicting daily returns
in the Investment Committee project. It uses epsilon-insensitive loss
which provides some robustness to outliers and includes threshold 
optimization for converting regression predictions to binary decisions.

Features:
- Epsilon-insensitive regression for robustness
- Multiple kernel options (RBF, linear, polynomial)
- Threshold optimization for binary decisions
- Feature scaling integration
- Model persistence
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple, Union
import joblib
import os

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_recall_curve, f1_score

from .base_model import BaseModel, clean_data_for_model_prediction

logger = logging.getLogger(__name__)

class SVMRegressorModel(BaseModel):
    """
    SVM regressor for predicting daily returns with epsilon-insensitive loss.
    
    This model predicts continuous daily returns and includes methods
    for converting predictions to binary decisions via threshold optimization.
    """
    
    def __init__(self, name: str = "SVMRegressor",
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 epsilon: float = 0.1,
                 gamma: str = 'scale',
                 degree: int = 3,
                 coef0: float = 0.0,
                 tol: float = 1e-3,
                 max_iter: int = -1,
                 scale_features: bool = True,
                 **kwargs):
        """
        Initialize SVM regressor with epsilon-insensitive loss.
        
        Args:
            name: Model name
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            C: Regularization parameter
            epsilon: Epsilon parameter in epsilon-insensitive loss
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            degree: Degree for polynomial kernel
            coef0: Independent term in kernel function
            tol: Tolerance for stopping criterion
            max_iter: Maximum number of iterations (-1 for no limit)
            scale_features: Whether to scale features using StandardScaler
            **kwargs: Additional SVM parameters
        """
        super().__init__(name)
        
        # Store hyperparameters
        self.params = {
            'kernel': kernel,
            'C': C,
            'epsilon': epsilon,
            'gamma': gamma,
            'degree': degree,
            'coef0': coef0,
            'tol': tol,
            'max_iter': max_iter,
            **kwargs
        }
        
        self.scale_features = scale_features
        
        # Initialize model and scaler
        self.model = SVR(**self.params)
        self.scaler = StandardScaler() if scale_features else None
        
        # Training tracking
        self.validation_metrics = {}
        self.feature_importance_ = None
        self.optimal_threshold_ = 0.0
        
        self.log(f"Initialized SVM Regressor with {kernel} kernel (C={C}, epsilon={epsilon})")

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
        Train the SVM regressor with optional validation.
        
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
            
            # Scale features if enabled
            if self.scale_features and self.scaler is not None:
                X_train_scaled = self.scaler.fit_transform(X_train_clean)
            else:
                X_train_scaled = X_train_clean.values
            
            # Train the model
            self.model.fit(X_train_scaled, y_train_clean)
            
            # Calculate training predictions for metrics
            train_pred = self.model.predict(X_train_scaled)
            
            # Calculate metrics
            if X_val is not None and y_val is not None:
                # Remove NaN values from validation set
                val_mask = ~(X_val.isna().any(axis=1) | y_val.isna())
                X_val_clean = X_val[val_mask]
                y_val_clean = y_val[val_mask]
                
                self.log(f"Validating on {len(X_val_clean)} samples")
                
                # Scale validation features if enabled
                if self.scale_features and self.scaler is not None:
                    X_val_scaled = self.scaler.transform(X_val_clean)
                else:
                    X_val_scaled = X_val_clean.values
                
                # Calculate validation metrics
                val_pred = self.model.predict(X_val_scaled)
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
                # Use training data metrics if no validation data
                self.validation_metrics = {
                    'mse': mean_squared_error(y_train_clean, train_pred),
                    'mae': mean_absolute_error(y_train_clean, train_pred),
                    'rmse': np.sqrt(mean_squared_error(y_train_clean, train_pred))
                }
                
                # Use training data to find threshold if no validation data
                if find_optimal_threshold:
                    self.optimal_threshold_ = self._find_optimal_threshold(train_pred, y_train_clean)
                    self.log(f"Optimal threshold (from training): {self.optimal_threshold_:.4f}")
            
            # SVM doesn't have direct feature importance, but we can estimate it for linear kernels
            if self.params['kernel'] == 'linear' and hasattr(self.model, 'coef_'):
                self.feature_importance_ = pd.Series(
                    np.abs(self.model.coef_[0]) if len(self.model.coef_.shape) > 1 else np.abs(self.model.coef_),
                    index=X_train_clean.columns
                ).sort_values(ascending=False)
                self.log("Feature importance calculated for linear kernel")
            else:
                self.feature_importance_ = None
                self.log("Feature importance not available for non-linear kernels")
            
            self.log(f"Training completed. Support vectors: {len(self.model.support_)}")
            
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
            
            # Scale features if enabled
            if self.scale_features and self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X.values
            
            # Get regression predictions
            reg_predictions = self.model.predict(X_scaled)
            
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
        """Get feature importance scores (only available for linear kernel)."""
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
                'scaler': self.scaler,
                'params': self.params,
                'scale_features': self.scale_features,
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
            self.scaler = model_data.get('scaler')
            self.params = model_data['params']
            self.scale_features = model_data.get('scale_features', True)
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
        support_vectors = getattr(self.model, 'support_', [])
        return f"{self.name} (trained, SVs={len(support_vectors)}, threshold={self.optimal_threshold_:.3f}, {metrics_str})"
