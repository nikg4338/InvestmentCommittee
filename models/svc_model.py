# filepath: c:\investment-committee\models\svc_model.py

"""
Support Vector Classification Model
====================================

This module implements the Support Vector Classification (SVC) model
using the scikit-learn library with preprocessing pipeline. It inherits 
from the BaseModel class to ensure a consistent interface with other 
models in the Investment Committee project.

The SVMClassifier includes:
- StandardScaler preprocessing pipeline
- SVC with probability=True for ensemble stacking
- Comprehensive metrics computation
- Robust error handling and logging
- Model persistence with joblib
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple
import joblib
import os

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class SVMClassifier(BaseModel):
    """
    Support Vector Machine classifier with preprocessing pipeline.
    
    This model uses SVC with probability=True to enable probability predictions
    for ensemble stacking. It includes StandardScaler preprocessing to handle
    feature scaling automatically.
    """
    
    def __init__(self, name: str = "SVMClassifier", 
                 kernel: str = 'rbf', 
                 C: float = 1.0, 
                 gamma: str = 'scale',
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize SVM classifier with preprocessing pipeline.
        
        Args:
            name: Model name for logging
            kernel: SVM kernel ('rbf', 'linear', 'poly', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient ('scale', 'auto', or float)
            random_state: Random state for reproducibility
            **kwargs: Additional SVC parameters
        """
        super().__init__(name)
        
        # Store hyperparameters
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        self.svc_kwargs = kwargs
        
        # Create preprocessing + model pipeline
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(
                kernel=kernel,
                C=C, 
                gamma=gamma,
                random_state=random_state,
                probability=True,  # Required for ensemble stacking
                class_weight='balanced',  # Handle class imbalance
                **kwargs
            ))
        ])
        
        # Training tracking
        self.validation_metrics = {}
        
        self.log(f"Initialized SVM with kernel={kernel}, C={C}, gamma={gamma}")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> None:
        """
        Train the SVM model with optional validation tracking.
        
        Args:
            X_train: Training features
            y_train: Training targets  
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        try:
            self.log("Starting SVM training...")
            
            # Fit the pipeline (scaler + SVM)
            self.pipeline.fit(X_train, y_train)
            self.is_trained = True
            
            # Compute validation metrics if validation data provided
            if X_val is not None and y_val is not None:
                val_proba = self.predict_proba(X_val)
                self.validation_metrics = self.get_metrics(y_val, val_proba[:, 1])
                self.log(f"Validation metrics: {self.validation_metrics}")
            
            self.log("SVM training completed successfully")
            
        except Exception as e:
            self.log(f"Error during SVM training: {e}")
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
            return self.pipeline.predict(X)
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
            return self.pipeline.predict_proba(X)
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
        Save the trained model pipeline to disk.
        
        Args:
            path: File path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save the entire pipeline
            joblib.dump(self.pipeline, path)
            
            # Save additional metadata
            metadata_path = path.replace('.pkl', '_metadata.pkl')
            metadata = {
                'name': self.name,
                'is_trained': self.is_trained,
                'kernel': self.kernel,
                'C': self.C,
                'gamma': self.gamma,
                'random_state': self.random_state,
                'validation_metrics': self.validation_metrics,
                'svc_kwargs': self.svc_kwargs
            }
            joblib.dump(metadata, metadata_path)
            
            self.log(f"Model saved successfully to {path}")
            return True
            
        except Exception as e:
            self.log(f"Failed to save model to {path}: {e}")
            return False

    def load(self, path: str) -> bool:
        """
        Load a trained model pipeline from disk.
        
        Args:
            path: File path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load the pipeline
            self.pipeline = joblib.load(path)
            self.is_trained = True
            
            # Load metadata if available
            metadata_path = path.replace('.pkl', '_metadata.pkl')
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.name = metadata.get('name', self.name)
                self.kernel = metadata.get('kernel', self.kernel)
                self.C = metadata.get('C', self.C)
                self.gamma = metadata.get('gamma', self.gamma)
                self.random_state = metadata.get('random_state', self.random_state)
                self.validation_metrics = metadata.get('validation_metrics', {})
                self.svc_kwargs = metadata.get('svc_kwargs', {})
            
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
        
        svm_metadata = {
            "model_type": "SVM",
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma,
            "random_state": self.random_state,
            "has_scaler": True,
            "probability_enabled": True,
            "validation_metrics": self.validation_metrics,
            "svc_kwargs": self.svc_kwargs
        }
        
        return {**base_metadata, **svm_metadata}

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance for linear kernels.
        
        Returns:
            Feature importance coefficients (only for linear kernel)
        """
        if not self.is_trained:
            self.log("Model must be trained to get feature importance")
            return None
        
        if self.kernel != 'linear':
            self.log("Feature importance only available for linear kernel")
            return None
        
        try:
            return self.pipeline.named_steps['svc'].coef_[0]
        except Exception as e:
            self.log(f"Error getting feature importance: {e}")
            return None

# For compatibility with train_models.py imports
SVCModel = SVMClassifier