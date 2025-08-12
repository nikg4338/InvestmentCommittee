# filepath: c:\investment-committee\models\lightgbm_model.py

"""
LightGBM Model
==============

This module implements the LightGBM model for binary classification
in the Investment Committee project. It inherits from the BaseModel 
class to ensure a consistent interface with other models.

The LightGBMModel includes:
- Gradient boosting with LightGBM
- Built-in feature importance
- Early stopping capabilities
- Comprehensive metrics computation
- Robust error handling and logging
- Model persistence with joblib
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple, Union
import joblib
import os

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .base_model import BaseModel, clean_data_for_model_prediction

logger = logging.getLogger(__name__)

class LightGBMModel(BaseModel):
    """
    LightGBM classifier for binary classification tasks.
    
    This model uses LightGBM's gradient boosting with built-in categorical
    feature support and early stopping capabilities.
    """
    
    def __init__(self, name: str = "LightGBMModel",
                 objective: str = 'binary',
                 boosting_type: str = 'gbdt',
                 num_leaves: int = 31,
                 learning_rate: float = 0.05,
                 feature_fraction: float = 0.9,
                 bagging_fraction: float = 0.8,
                 bagging_freq: int = 5,
                 verbose: int = -1,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize LightGBM classifier.
        
        Args:
            name: Model name for logging
            objective: LightGBM objective function
            boosting_type: Type of boosting ('gbdt', 'dart', 'goss')
            num_leaves: Number of leaves in one tree
            learning_rate: Learning rate
            feature_fraction: Fraction of features to use
            bagging_fraction: Fraction of data to use for bagging
            bagging_freq: Frequency for bagging
            verbose: Verbosity level
            random_state: Random state for reproducibility
            **kwargs: Additional LightGBM parameters
        """
        super().__init__(name)
        
        if not LIGHTGBM_AVAILABLE:
            self.log("LightGBM is not available. Install it with 'pip install lightgbm'")
            # Create a dummy model that will fail gracefully
            self.model = None
            self.params = {}
            self.validation_metrics = {}
            self.feature_importance_ = None
            return
        
        # Store hyperparameters
        self.params = {
            'objective': objective,
            'boosting_type': boosting_type,
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'verbose': verbose,
            'random_state': random_state,
            'is_unbalance': True,  # Handle class imbalance
            'eval_metric': 'average_precision',  # Updated eval_metric for PR-AUC
            **kwargs
        }
        
        # Initialize model
        self.model = lgb.LGBMClassifier(**self.params)
        
        # Training tracking
        self.validation_metrics = {}
        self.feature_importance_ = None
        
        self.log(f"Initialized LightGBM with {len(self.params)} parameters")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              early_stopping_rounds: int = 100,
              eval_metric: str = 'auc') -> None:
        """
        Train the LightGBM model with optional validation and early stopping.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            early_stopping_rounds: Early stopping rounds
            eval_metric: Evaluation metric for early stopping
        """
        if not LIGHTGBM_AVAILABLE:
            self.log("LightGBM not available, skipping training")
            return
            
        try:
            self.log("Starting LightGBM training...")
            
            # Prepare evaluation set and early stopping for early stopping
            eval_set = None
            callbacks = []
            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val)]
                # Add early stopping with PR-AUC monitoring
                callbacks = [
                    lgb.early_stopping(stopping_rounds=10),
                    lgb.log_evaluation(period=0)  # Silent logging
                ]
            
            # Fit the model with early stopping if validation data is available
            try:
                if eval_set and callbacks:
                    self.model.fit(
                        X_train, y_train, 
                        eval_set=eval_set,
                        eval_metric='average_precision',  # PR-AUC for imbalanced data
                        callbacks=callbacks
                    )
                    self.log("LightGBM training with early stopping (PR-AUC monitoring)")
                elif eval_set:
                    # Fallback without callbacks
                    self.model.fit(X_train, y_train, eval_set=eval_set)
                    self.log("LightGBM training with eval set (no early stopping)")
                else:
                    self.model.fit(X_train, y_train)
                    self.log("LightGBM training without validation")
            except TypeError as e:
                # If that fails, try with just the basic training data
                self.log(f"Warning: Advanced fit parameters not supported: {e}")
                self.model.fit(X_train, y_train)
            
            self.is_trained = True
            self.feature_importance_ = self.model.feature_importances_
            
            # Compute validation metrics if validation data provided
            if X_val is not None and y_val is not None:
                val_proba = self.predict_proba(X_val)
                self.validation_metrics = self.get_metrics(y_val, val_proba[:, 1])
                self.log(f"Validation metrics: {self.validation_metrics}")
            
            self.log("LightGBM training completed successfully")
            
        except Exception as e:
            self.log(f"Error during LightGBM training: {e}")
            self.is_trained = False
            raise

    def fit(self, X, y, X_val=None, y_val=None, **kwargs) -> None:
        """
        Fit method for BaseModel compatibility.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional arguments passed to train
        """
        self.train(X, y, X_val, y_val, **kwargs)

    def predict(self, X, **kwargs):
        """
        Make binary predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Binary predictions (0 or 1)
        """
        if not LIGHTGBM_AVAILABLE:
            self.log("LightGBM not available, returning dummy predictions")
            return np.zeros(len(X))
            
        if not self.is_trained:
            raise Exception("Model must be trained before predictions can be made.")
        
        try:
            # Clean data before prediction
            X_clean = clean_data_for_model_prediction(X)
            return self.model.predict(X_clean)
        except Exception as e:
            self.log(f"Error during prediction: {e}")
            raise

    def predict_proba(self, X, **kwargs):
        """
        Predict class probabilities.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
        """
        if not LIGHTGBM_AVAILABLE:
            self.log("LightGBM not available, returning dummy probabilities")
            n_samples = len(X)
            return np.column_stack([np.full(n_samples, 0.5), np.full(n_samples, 0.5)])
            
        if not self.is_trained:
            raise Exception("Model must be trained before predictions can be made.")
        
        try:
            # Clean data before prediction
            X_clean = clean_data_for_model_prediction(X)
            return self.model.predict_proba(X_clean)
        except Exception as e:
            self.log(f"Error during probability prediction: {e}")
            raise

    def get_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities for positive class
            
        Returns:
            Dictionary containing accuracy, precision, recall, F1, and ROC-AUC scores
        """
        try:
            # Convert probabilities to binary predictions
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0
            }
            
            return metrics
            
        except Exception as e:
            self.log(f"Error computing metrics: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'roc_auc': 0.0
            }

    def save(self, path: str) -> bool:
        """
        Save the trained model to disk.
        
        Args:
            path: File path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save the model
            joblib.dump(self.model, path)
            
            # Save additional metadata
            metadata_path = path.replace('.pkl', '_metadata.pkl')
            metadata = {
                'name': self.name,
                'is_trained': self.is_trained,
                'params': self.params,
                'validation_metrics': self.validation_metrics,
                'feature_importance': self.feature_importance_.tolist() if self.feature_importance_ is not None else None
            }
            joblib.dump(metadata, metadata_path)
            
            self.log(f"Model saved successfully to {path}")
            return True
            
        except Exception as e:
            self.log(f"Failed to save model to {path}: {e}")
            return False

    def load(self, path: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            path: File path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load the model
            self.model = joblib.load(path)
            self.is_trained = True
            
            # Load metadata if available
            metadata_path = path.replace('.pkl', '_metadata.pkl')
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.name = metadata.get('name', self.name)
                self.params = metadata.get('params', self.params)
                self.validation_metrics = metadata.get('validation_metrics', {})
                feature_importance = metadata.get('feature_importance')
                if feature_importance is not None:
                    self.feature_importance_ = np.array(feature_importance)
            
            self.log(f"Model loaded successfully from {path}")
            return True
            
        except Exception as e:
            self.log(f"Failed to load model from {path}: {e}")
            return False

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get comprehensive model metadata.
        
        Returns:
            Dictionary containing model information
        """
        base_metadata = super().get_metadata()
        
        lgb_metadata = {
            "model_type": "LightGBM",
            "params": self.params,
            "validation_metrics": self.validation_metrics,
            "has_feature_importance": self.feature_importance_ is not None,
            "n_features": len(self.feature_importance_) if self.feature_importance_ is not None else None
        }
        
        return {**base_metadata, **lgb_metadata}

    def get_feature_importance(self, importance_type: str = 'gain') -> Optional[np.ndarray]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('gain', 'split')
            
        Returns:
            Feature importance array
        """
        if not self.is_trained:
            self.log("Model must be trained to get feature importance")
            return None
        
        try:
            if importance_type == 'gain':
                return self.model.feature_importances_
            elif importance_type == 'split':
                # Note: LightGBM doesn't directly provide split importance via sklearn interface
                return self.model.feature_importances_
            else:
                self.log(f"Unknown importance type: {importance_type}")
                return None
        except Exception as e:
            self.log(f"Error getting feature importance: {e}")
            return None

# For compatibility with train_models.py imports
LGBMModel = LightGBMModel
