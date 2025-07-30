# XGBoost model module
# Implements XGBoost model for short-term return prediction # xgboost_model.py
"""
XGBoost Model Module
Implements XGBoost model for short-term return prediction.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not available. Using dummy logic.")

logger = logging.getLogger(__name__)

class XGBoostModel:
    """
    XGBoost model for price/return prediction.
    """
    def __init__(self, model_params: Optional[Dict[str, Any]] = None, model_path: Optional[str] = None):
        self.model_params = model_params or {
            "objective": "binary:logistic",  # Changed to binary classification
            "max_depth": 4,
            "learning_rate": 0.07,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "aucpr",  # Updated eval_metric for PR-AUC
            "seed": 42
        }
        self.model = None
        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(**self.model_params)  # Changed to classifier
            if model_path:
                self.load_model(model_path)
        else:
            logger.warning("xgboost not available. Model will return dummy outputs.")

        self.is_trained = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train XGBoost model.
        """
        if not XGBOOST_AVAILABLE or self.model is None:
            logger.warning("xgboost not available. Training skipped.")
            return
        
        # Calculate class imbalance ratio for scale_pos_weight
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        if n_pos > 0:
            scale_pos_weight = n_neg / n_pos
        else:
            scale_pos_weight = 1.0
        
        # Update model with scale_pos_weight
        self.model.set_params(scale_pos_weight=scale_pos_weight)
        
        self.model.fit(X, y)
        self.is_trained = True
        logger.info(f"XGBoost model trained with scale_pos_weight={scale_pos_weight:.2f}")
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train XGBoost model (alias for fit).
        """
        self.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict returns or price using the trained model.
        """
        if not XGBOOST_AVAILABLE or self.model is None:
            logger.warning("xgboost not available. Returning zeros.")
            return np.zeros(len(X))
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using the trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
        """
        if not XGBOOST_AVAILABLE or self.model is None:
            logger.warning("xgboost not available. Returning dummy probabilities.")
            # Return dummy probabilities (50/50 for each class)
            n_samples = len(X)
            return np.column_stack([np.full(n_samples, 0.5), np.full(n_samples, 0.5)])
        
        try:
            probabilities = self.model.predict_proba(X)
            return probabilities
        except Exception as e:
            logger.error(f"Error in predict_proba: {e}")
            # Fallback to dummy probabilities
            n_samples = len(X)
            return np.column_stack([np.full(n_samples, 0.5), np.full(n_samples, 0.5)])

    def predict_signal(self, features: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
        """
        High-level API: Predict signal given features dict.
        Returns: (signal, confidence, metadata)
        """
        if not XGBOOST_AVAILABLE or self.model is None:
            return 'NEUTRAL', 0.5, {"error": "xgboost not available"}

        try:
            X = pd.DataFrame([features])
            pred = self.model.predict(X)[0]
            # Simple interpretation: >0 bullish, <0 bearish, ~0 neutral
            if pred > 0.02:
                signal = 'BULLISH'
            elif pred < -0.02:
                signal = 'BEARISH'
            else:
                signal = 'NEUTRAL'
            confidence = float(np.clip(abs(pred) * 10, 0, 1))  # scale confidence
            metadata = {
                "raw_prediction": float(pred),
                "features": features
            }
            return signal, confidence, metadata
        except Exception as e:
            logger.error(f"XGBoost prediction error: {e}")
            return 'NEUTRAL', 0.5, {"error": str(e)}

    def save_model(self, path: str) -> None:
        """
        Save trained model to disk.
        """
        if not XGBOOST_AVAILABLE or self.model is None:
            logger.warning("xgboost not available. Save skipped.")
            return
        self.model.save_model(path)
        logger.info(f"XGBoost model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load model from disk.
        """
        if not XGBOOST_AVAILABLE or self.model is None:
            logger.warning("xgboost not available. Load skipped.")
            return
        self.model.load_model(path)
        self.is_trained = True
        logger.info(f"XGBoost model loaded from {path}")
    @staticmethod
    def grid_search_xgb(X, y):
        """
        Run GridSearchCV for XGBoost hyperparameter optimization.
        Returns (best_estimator, best_params).
        """
        from sklearn.model_selection import GridSearchCV
        import xgboost as xgb
        param_grid = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.07, 0.1, 0.2],
            "n_estimators": [50, 100, 150],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
        model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss")
        grid = GridSearchCV(model, param_grid, scoring="accuracy", cv=3, verbose=1, n_jobs=-1)
        grid.fit(X, y)
        print("Best GridSearch params:", grid.best_params_)
        return grid.best_estimator_, grid.best_params_

    @staticmethod
    def optuna_search_xgb(X, y, n_trials=25):
        """
        Run Optuna for XGBoost hyperparameter optimization.
        Returns (best_estimator, best_params).
        """
        import optuna
        import xgboost as xgb
        from sklearn.model_selection import cross_val_score

        def objective(trial):
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": trial.suggest_int('max_depth', 3, 8),
                "learning_rate": trial.suggest_float('learning_rate', 0.01, 0.2),
                "n_estimators": trial.suggest_int('n_estimators', 50, 150),
                "subsample": trial.suggest_float('subsample', 0.7, 1.0),
                "colsample_bytree": trial.suggest_float('colsample_bytree', 0.7, 1.0),
                "use_label_encoder": False,
            }
            model = xgb.XGBClassifier(**params)
            scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
            return scores.mean()
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        print("Best Optuna params:", study.best_params)
        best_model = xgb.XGBClassifier(**study.best_params, eval_metric="logloss")
        best_model.fit(X, y)
        return best_model, study.best_params

# Example/test usage
if __name__ == "__main__":
    if XGBOOST_AVAILABLE:
        # Create dummy features and labels
        X = pd.DataFrame({
            "rsi": np.random.uniform(30, 70, 100),
            "momentum": np.random.normal(0, 1, 100),
            "iv_rank": np.random.uniform(0, 1, 100),
            "zscore": np.random.normal(0, 1, 100),
        })
        y = np.random.choice([0, 1], 100)  # Binary classification labels

        model = XGBoostModel()
        model.fit(X, pd.Series(y))
        # Predict single sample
        features = {
            "rsi": 45.0,
            "momentum": 0.5,
            "iv_rank": 0.55,
            "zscore": 1.2,
        }
        signal, confidence, meta = model.predict_signal(features)
        print(f"Signal: {signal}, Confidence: {confidence:.2f}, Meta: {meta}")
    else:
        print("XGBoost not available, skipping example.")
