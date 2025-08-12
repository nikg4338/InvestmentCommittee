# filepath: c:\investment-committee\models\random_forest_model.py

"""
Random Forest Model
===================

This module implements the Random Forest model for binary classification
in the Investment Committee project. It inherits from the BaseModel 
class to ensure a consistent interface with other models.

The RandomForestModel includes:
- Ensemble of decision trees
- Built-in feature importance
- Parallel processing capabilities
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .base_model import BaseModel, clean_data_for_model_prediction

logger = logging.getLogger(__name__)

class RandomForestModel(BaseModel):
    """
    Random Forest classifier for binary classification tasks.
    
    This model uses an ensemble of decision trees with built-in feature
    importance and parallel processing capabilities.
    """
    
    def __init__(self, name: str = "RandomForestModel",
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str = 'sqrt',
                 bootstrap: bool = True,
                 n_jobs: int = -1,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize Random Forest classifier.
        
        Args:
            name: Model name for logging
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider at each split
            bootstrap: Whether to use bootstrap samples
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random state for reproducibility
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
            'n_jobs': n_jobs,
            'random_state': random_state,
            'class_weight': 'balanced',  # Handle class imbalance
            **kwargs
        }
        
        # Initialize model
        self.model = RandomForestClassifier(**self.params)
        
        # Training tracking
        self.validation_metrics = {}
        self.feature_importance_ = None
        
        self.log(f"Initialized Random Forest with {n_estimators} trees")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> None:
        """
        Train the Random Forest model with optional validation tracking.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        try:
            self.log("Starting Random Forest training...")
            
            # Fit the model
            self.model.fit(X_train, y_train)
            self.is_trained = True
            self.feature_importance_ = self.model.feature_importances_
            
            # Compute validation metrics if validation data provided
            if X_val is not None and y_val is not None:
                val_proba = self.predict_proba(X_val)
                self.validation_metrics = self.get_metrics(y_val, val_proba[:, 1])
                self.log(f"Validation metrics: {self.validation_metrics}")
            
            self.log("Random Forest training completed successfully")
            
        except Exception as e:
            self.log(f"Error during Random Forest training: {e}")
            self.is_trained = False
            raise

    def fit(self, X, y, **kwargs) -> None:
        """
        Fit method for BaseModel compatibility.
        """
        self.train(X, y)

    def predict(self, X, **kwargs):
        """
        Make binary predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Binary predictions (0 or 1)
        """
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
        
        rf_metadata = {
            "model_type": "RandomForest",
            "params": self.params,
            "validation_metrics": self.validation_metrics,
            "has_feature_importance": self.feature_importance_ is not None,
            "n_features": len(self.feature_importance_) if self.feature_importance_ is not None else None,
            "n_trees": self.params.get('n_estimators', 0)
        }
        
        return {**base_metadata, **rf_metadata}

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores.
        
        Returns:
            Feature importance array (mean decrease in impurity)
        """
        if not self.is_trained:
            self.log("Model must be trained to get feature importance")
            return None
        
        try:
            return self.model.feature_importances_
        except Exception as e:
            self.log(f"Error getting feature importance: {e}")
            return None

    def get_tree_count(self) -> int:
        """
        Get the number of trees in the forest.
        
        Returns:
            Number of estimators/trees
        """
        return self.model.n_estimators if self.model else 0

    def get_oob_score(self) -> Optional[float]:
        """
        Get out-of-bag score if available.
        
        Returns:
            OOB score if bootstrap=True and oob_score=True, None otherwise
        """
        if not self.is_trained:
            return None
        
        try:
            if hasattr(self.model, 'oob_score_'):
                return self.model.oob_score_
            else:
                self.log("OOB score not available (set oob_score=True during initialization)")
                return None
        except Exception as e:
            self.log(f"Error getting OOB score: {e}")
            return None

# For compatibility - avoid naming conflict with sklearn's RandomForestClassifier
# The train_models.py imports RandomForestModel directly, so this alias is not needed