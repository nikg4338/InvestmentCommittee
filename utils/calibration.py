
"""
Probability Calibration Utilities
=================================

This module provides probability calibration utilities for ensemble models
to ensure consistent and well-calibrated probability outputs across different
model types (classifiers and regressors).

Features:
- CalibratedClassifierCV wrapper for any model
- Isotonic and Platt calibration methods
- Regression model calibration via sigmoid transformation
- Validation of calibration quality
- Ensemble-ready calibrated predictions
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import warnings

logger = logging.getLogger(__name__)

class ModelCalibrator:
    """
    Calibrate probability outputs from ensemble models for stable predictions.
    
    This calibrator handles both classification and regression models,
    ensuring all models output well-calibrated probabilities in [0,1] range.
    """
    
    def __init__(self, method: str = 'isotonic', cv: int = 3):
        """
        Initialize the calibrator.
        
        Args:
            method: Calibration method ('isotonic' or 'sigmoid')
            cv: Number of cross-validation folds for calibration
        """
        self.method = method
        self.cv = cv
        self.calibrators = {}
        self.is_fitted = False
        
        logger.info(f"🎯 Initialized ModelCalibrator with {method} method, {cv}-fold CV")
    
    def fit_calibrators(self, models: Dict[str, Any], X_cal: pd.DataFrame, 
                       y_cal: pd.Series) -> None:
        """
        Fit calibrators for each model in the ensemble.
        
        Args:
            models: Dictionary of trained models
            X_cal: Calibration features
            y_cal: Calibration targets
        """
        logger.info(f"🎯 Fitting calibrators for {len(models)} models...")
        
        for model_name, model in models.items():
            try:
                # Get uncalibrated predictions
                if hasattr(model, 'predict_proba'):
                    # Classification model
                    uncal_probs = model.predict_proba(X_cal)[:, 1]
                    
                    # Create calibrated version
                    calibrated_model = CalibratedClassifierCV(
                        model, method=self.method, cv=self.cv
                    )
                    calibrated_model.fit(X_cal, y_cal)
                    self.calibrators[model_name] = calibrated_model
                    
                elif hasattr(model, 'predict'):
                    # Regression model - need to calibrate raw predictions
                    raw_preds = model.predict(X_cal)
                    
                    if self.method == 'isotonic':
                        calibrator = IsotonicRegression(out_of_bounds='clip')
                    else:  # sigmoid/platt
                        calibrator = LogisticRegression()
                    
                    # Fit calibrator to map raw predictions to probabilities
                    calibrator.fit(raw_preds.reshape(-1, 1), y_cal)
                    self.calibrators[model_name] = ('regression', calibrator)
                    
                else:
                    logger.warning(f"⚠️ Model {model_name} has no predict method")
                    continue
                
                logger.info(f"✅ Calibrated {model_name}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to calibrate {model_name}: {e}")
                # Store original model as fallback
                self.calibrators[model_name] = model
        
        self.is_fitted = True
        logger.info(f"✅ Calibration complete for {len(self.calibrators)} models")
    
    def get_calibrated_predictions(self, models: Dict[str, Any], 
                                 X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get calibrated probability predictions from all models.
        
        Args:
            models: Dictionary of trained models
            X: Input features
            
        Returns:
            Dictionary of calibrated probability predictions [0,1]
        """
        if not self.is_fitted:
            raise ValueError("Calibrators must be fitted before prediction")
        
        calibrated_preds = {}
        
        for model_name, model in models.items():
            try:
                if model_name in self.calibrators:
                    calibrator = self.calibrators[model_name]
                    
                    if isinstance(calibrator, tuple) and calibrator[0] == 'regression':
                        # Regression model calibration
                        _, reg_calibrator = calibrator
                        raw_preds = model.predict(X)
                        if hasattr(reg_calibrator, 'predict_proba'):
                            # Logistic regression calibrator
                            cal_probs = reg_calibrator.predict_proba(raw_preds.reshape(-1, 1))[:, 1]
                        else:
                            # Isotonic regression calibrator
                            cal_probs = reg_calibrator.predict(raw_preds.reshape(-1, 1))
                        # Ensure probabilities are in [0,1] range
                        cal_probs = np.clip(cal_probs, 0.001, 0.999)
                    elif hasattr(calibrator, 'predict_proba'):
                        # Calibrated classifier
                        cal_probs = calibrator.predict_proba(X)[:, 1]
                    else:
                        # Fallback to original model
                        if hasattr(model, 'predict_proba'):
                            cal_probs = model.predict_proba(X)[:, 1]
                        else:
                            # Convert raw predictions to probabilities (sigmoid)
                            raw_preds = model.predict(X)
                            cal_probs = 1 / (1 + np.exp(-raw_preds))
                    
                    # Ensure probabilities are in [0,1]
                    cal_probs = np.clip(cal_probs, 0.001, 0.999)
                    calibrated_preds[model_name] = cal_probs
                    
                else:
                    logger.warning(f"⚠️ No calibrator for {model_name}, using raw predictions")
                    if hasattr(model, 'predict_proba'):
                        calibrated_preds[model_name] = model.predict_proba(X)[:, 1]
                    else:
                        # Sigmoid transform for regression
                        raw_preds = model.predict(X)
                        calibrated_preds[model_name] = 1 / (1 + np.exp(-raw_preds))
                
            except Exception as e:
                logger.warning(f"⚠️ Error getting calibrated predictions for {model_name}: {e}")
                # Return neutral probabilities as fallback
                calibrated_preds[model_name] = np.full(len(X), 0.5)
        
        return calibrated_preds
    
    def evaluate_calibration(self, models: Dict[str, Any], X_val: pd.DataFrame, 
                           y_val: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluate calibration quality using reliability diagrams and Brier score.
        
        Args:
            models: Dictionary of trained models
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary of calibration metrics for each model
        """
        if not self.is_fitted:
            logger.warning("⚠️ Calibrators not fitted, evaluating raw predictions")
        
        calibration_metrics = {}
        
        # Get calibrated predictions
        cal_preds = self.get_calibrated_predictions(models, X_val)
        
        for model_name, probs in cal_preds.items():
            try:
                # Compute Brier score (lower is better)
                brier_score = brier_score_loss(y_val, probs)
                
                # Compute calibration curve
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_val, probs, n_bins=10
                )
                
                # Expected Calibration Error (ECE)
                bin_boundaries = np.linspace(0, 1, 11)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]
                
                ece = 0
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (probs > bin_lower) & (probs <= bin_upper)
                    prop_in_bin = in_bin.mean()
                    
                    if prop_in_bin > 0:
                        accuracy_in_bin = y_val[in_bin].mean()
                        avg_confidence_in_bin = probs[in_bin].mean()
                        ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                calibration_metrics[model_name] = {
                    'brier_score': brier_score,
                    'ece': ece,
                    'mean_probability': probs.mean(),
                    'std_probability': probs.std()
                }
                
            except Exception as e:
                logger.warning(f"⚠️ Error evaluating calibration for {model_name}: {e}")
                calibration_metrics[model_name] = {
                    'brier_score': 1.0,
                    'ece': 1.0,
                    'mean_probability': 0.5,
                    'std_probability': 0.0
                }
        
        # Log calibration summary
        logger.info("📊 Calibration Quality Summary:")
        for model_name, metrics in calibration_metrics.items():
            logger.info(f"  {model_name}: Brier={metrics['brier_score']:.4f}, "
                       f"ECE={metrics['ece']:.4f}")
        
        return calibration_metrics

def create_calibrated_ensemble_predictions(models: Dict[str, Any], 
                                         X_cal: pd.DataFrame, y_cal: pd.Series,
                                         X_pred: pd.DataFrame,
                                         method: str = 'isotonic',
                                         cv: int = 3) -> Dict[str, np.ndarray]:
    """
    Create calibrated predictions for ensemble voting.
    
    Args:
        models: Dictionary of trained models
        X_cal: Calibration features
        y_cal: Calibration targets
        X_pred: Features for prediction
        method: Calibration method ('isotonic' or 'sigmoid')
        cv: Cross-validation folds
        
    Returns:
        Dictionary of calibrated probability predictions
    """
    calibrator = ModelCalibrator(method=method, cv=cv)
    calibrator.fit_calibrators(models, X_cal, y_cal)
    return calibrator.get_calibrated_predictions(models, X_pred)
