# filepath: c:\investment-committee\models\catboost_model.py

"""
CatBoost Model
==============

This module implements the CatBoost model for binary classification
in the Investment Committee project. It inherits from the BaseModel 
class to ensure a consistent interface with other models.

The CatBoostModel includes:
- Gradient boosting with CatBoost
- Built-in categorical feature handling
- Early stopping capabilities
- Comprehensive metrics computation
- Robust error handling and logging
- Model persistence with native CatBoost methods
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple, Union
import os


try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostClassifier = None

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class CatBoostModel(BaseModel):
    """
    CatBoost classifier for binary classification tasks.
    
    This model uses CatBoost's gradient boosting with built-in categorical
    feature support and GPU acceleration capabilities.
    """
    
    def __init__(self, name: str = "CatBoostModel",
                 iterations: int = 1000,
                 learning_rate: float = 0.05,
                 depth: int = 6,
                 l2_leaf_reg: float = 3.0,
                 bootstrap_type: str = 'Bayesian',
                 random_seed: int = 42,
                 logging_level: str = 'Silent',
                 **kwargs):
        """
        Initialize CatBoost classifier.
        
        Args:
            name: Model name for logging
            iterations: Number of boosting iterations
            learning_rate: Learning rate
            depth: Tree depth
            l2_leaf_reg: L2 regularization coefficient
            bootstrap_type: Bootstrap type ('Bayesian', 'Bernoulli', 'MVS')
            random_seed: Random seed for reproducibility
            logging_level: CatBoost logging level
            **kwargs: Additional CatBoost parameters
        """
        super().__init__(name)
        
        if not CATBOOST_AVAILABLE:
            self.log("CatBoost is not available. Install it with 'pip install catboost'")
            # Create a dummy model that will fail gracefully
            self.model = None
            self.params = {}
            self.validation_metrics = {}
            self.feature_importance_ = None
            self.categorical_features = None
            return
        
        # Store hyperparameters
        self.params = {
            'iterations': iterations,
            'learning_rate': learning_rate,
            'depth': depth,
            'l2_leaf_reg': l2_leaf_reg,
            'bootstrap_type': bootstrap_type,
            'random_seed': random_seed,
            'logging_level': logging_level,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'auto_class_weights': 'Balanced',  # Handle class imbalance - use auto_class_weights instead
            **kwargs
        }
        
        # Initialize model
        self.model = CatBoostClassifier(**self.params)
        
        # Training tracking
        self.validation_metrics = {}
        self.feature_importance_ = None
        self.categorical_features = None
        
        self.log(f"Initialized CatBoost with {len(self.params)} parameters")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              categorical_features: Optional[list] = None,
              early_stopping_rounds: int = 100) -> None:
        """
        Train the CatBoost model with optional validation and early stopping.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            categorical_features: List of categorical feature indices or names
            early_stopping_rounds: Early stopping rounds
        """
        if not CATBOOST_AVAILABLE:
            self.log("CatBoost not available, skipping training")
            return
            
        try:
            self.log("Starting CatBoost training...")
            
            # Store categorical features
            self.categorical_features = categorical_features
            
            # Prepare evaluation set for early stopping
            eval_set = None
            if X_val is not None and y_val is not None:
                eval_set = (X_val, y_val)
            
            # Fit the model
            self.model.fit(
                X_train, y_train,
                cat_features=categorical_features,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds if eval_set else None,
                verbose=False
            )
            
            self.is_trained = True
            self.feature_importance_ = self.model.get_feature_importance()
            
            # Compute validation metrics if validation data provided
            if X_val is not None and y_val is not None:
                val_proba = self.predict_proba(X_val)
                self.validation_metrics = self.get_metrics(y_val, val_proba[:, 1])
                self.log(f"Validation metrics: {self.validation_metrics}")
            
            self.log("CatBoost training completed successfully")
            
        except Exception as e:
            self.log(f"Error during CatBoost training: {e}")
            self.is_trained = False
            raise

    def fit(self, X, y, **kwargs) -> None:
        """
        Fit method for BaseModel compatibility.
        """
        categorical_features = kwargs.get('categorical_features', None)
        self.train(X, y, categorical_features=categorical_features)

    def predict(self, X, **kwargs):
        """
        Make binary predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Binary predictions (0 or 1)
        """
        if not CATBOOST_AVAILABLE:
            self.log("CatBoost not available, returning dummy predictions")
            return np.zeros(len(X))
            
        if not self.is_trained:
            raise Exception("Model must be trained before predictions can be made.")
        
        try:
            return self.model.predict(X)
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
        if not CATBOOST_AVAILABLE:
            self.log("CatBoost not available, returning dummy probabilities")
            n_samples = len(X)
            return np.column_stack([np.full(n_samples, 0.5), np.full(n_samples, 0.5)])
            
        if not self.is_trained:
            raise Exception("Model must be trained before predictions can be made.")
        
        try:
            return self.model.predict_proba(X)
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
        Save the trained model to disk using CatBoost's native save method.
        
        Args:
            path: File path to save the model (should end with .cbm)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Use CatBoost's native save method
            model_path = path if path.endswith('.cbm') else path + '.cbm'
            self.model.save_model(model_path)
            
            # Save additional metadata using joblib
            import joblib
            metadata_path = path.replace('.cbm', '_metadata.pkl').replace('.pkl', '_metadata.pkl')
            metadata = {
                'name': self.name,
                'is_trained': self.is_trained,
                'params': self.params,
                'validation_metrics': self.validation_metrics,
                'categorical_features': self.categorical_features,
                'feature_importance': self.feature_importance_.tolist() if self.feature_importance_ is not None else None
            }
            joblib.dump(metadata, metadata_path)
            
            self.log(f"Model saved successfully to {model_path}")
            return True
            
        except Exception as e:
            self.log(f"Failed to save model to {path}: {e}")
            return False

    def load(self, path: str) -> bool:
        """
        Load a trained model from disk using CatBoost's native load method.
        
        Args:
            path: File path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load the model
            model_path = path if path.endswith('.cbm') else path + '.cbm'
            self.model = CatBoostClassifier()
            self.model.load_model(model_path)
            self.is_trained = True
            
            # Load metadata if available
            import joblib
            metadata_path = path.replace('.cbm', '_metadata.pkl').replace('.pkl', '_metadata.pkl')
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.name = metadata.get('name', self.name)
                self.params = metadata.get('params', self.params)
                self.validation_metrics = metadata.get('validation_metrics', {})
                self.categorical_features = metadata.get('categorical_features')
                feature_importance = metadata.get('feature_importance')
                if feature_importance is not None:
                    self.feature_importance_ = np.array(feature_importance)
            
            self.log(f"Model loaded successfully from {model_path}")
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
        
        catboost_metadata = {
            "model_type": "CatBoost",
            "params": self.params,
            "validation_metrics": self.validation_metrics,
            "categorical_features": self.categorical_features,
            "has_feature_importance": self.feature_importance_ is not None,
            "n_features": len(self.feature_importance_) if self.feature_importance_ is not None else None
        }
        
        return {**base_metadata, **catboost_metadata}

    def get_feature_importance(self, importance_type: str = 'FeatureImportance') -> Optional[np.ndarray]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('FeatureImportance', 'PredictionValuesChange', 'LossFunctionChange')
            
        Returns:
            Feature importance array
        """
        if not self.is_trained:
            self.log("Model must be trained to get feature importance")
            return None
        
        try:
            return self.model.get_feature_importance(type=importance_type)
        except Exception as e:
            self.log(f"Error getting feature importance: {e}")
            return None

# Note: Do not create alias as it causes circular imports