# Base model class
# Abstract base class for all ML models with common interface and utilities # base_model.py
"""
BaseModel: Abstract base class for all ML models in the Investment Committee.
Defines common interface for fit, predict, save, load, and metadata.
"""

import abc
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def clean_data_for_model_prediction(X: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data for model prediction by handling categorical and non-numeric columns.
    This ensures consistent data preprocessing across all model wrappers.
    
    Args:
        X: Input DataFrame
        
    Returns:
        Cleaned DataFrame with only numeric columns
    """
    if not isinstance(X, pd.DataFrame):
        return X
    
    Xn = X.copy()
    drop_cols = []
    
    # Process each column
    for c in Xn.columns:
        if Xn[c].dtype == 'object':
            # Try to convert object columns to numeric
            coerced = pd.to_numeric(Xn[c], errors='coerce')
            if coerced.notna().sum() == 0:
                # If no values can be converted, drop the column
                drop_cols.append(c)
            else:
                Xn[c] = coerced
        elif Xn[c].dtype.name not in ['int64', 'int32', 'float64', 'float32', 'bool']:
            # Convert other non-standard types to numeric
            Xn[c] = pd.to_numeric(Xn[c], errors='coerce')
    
    # Drop non-convertible columns
    if drop_cols:
        logger.debug(f"Dropping non-numeric columns for model prediction: {drop_cols}")
        Xn = Xn.drop(columns=drop_cols, errors='ignore')
    
    # Handle infinite values
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with median for numeric columns
    if not Xn.empty:
        Xn = Xn.apply(lambda s: s.fillna(s.median()) if s.dtype.kind in 'fc' else s)
    
    return Xn

class BaseModel(abc.ABC):
    """
    Abstract base class for all committee models (XGBoost, MLP, LSTM, etc.).
    Provides standard interface for training, prediction, saving/loading, and metadata.
    Includes sklearn compatibility for calibration support.
    """

    def __init__(self, name: str = "BaseModel"):
        self.name = name
        self.is_trained = False
        self.metadata: Dict[str, Any] = {}

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility (enables calibration)."""
        # Return any parameters stored in self.params if available
        if hasattr(self, 'params'):
            return self.params.copy() if deep else self.params
        # Return basic model metadata
        return self.metadata.copy()
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility (enables calibration)."""
        # Update params if available
        if hasattr(self, 'params'):
            self.params.update(params)
        # Also update metadata
        self.metadata.update(params)
        return self

    @abc.abstractmethod
    def fit(self, X, y, **kwargs) -> None:
        """
        Train the model on data.
        Args:
            X: Features
            y: Targets
        """
        pass

    @abc.abstractmethod
    def predict(self, X, **kwargs):
        """
        Make predictions for input features X.
        Returns:
            Model output (probabilities, class, etc.)
        """
        pass

    @abc.abstractmethod
    def save(self, path: str) -> bool:
        """
        Save model weights/parameters to file.
        Returns:
            True if successful, False otherwise
        """
        pass

    @abc.abstractmethod
    def load(self, path: str) -> bool:
        """
        Load model weights/parameters from file.
        Returns:
            True if successful, False otherwise
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return model metadata (name, trained status, any extra info).
        """
        return {
            "name": self.name,
            "is_trained": self.is_trained,
            **self.metadata,
        }

    def log(self, msg: str):
        """
        Logging utility for model events.
        """
        logger.info(f"[{self.name}] {msg}")

# Example: How to subclass BaseModel

if __name__ == "__main__":
    # Demo: DummyModel implements BaseModel
    class DummyModel(BaseModel):
        def fit(self, X, y, **kwargs):
            self.log("Fitting dummy model (no-op)")
            self.is_trained = True
        def predict(self, X, **kwargs):
            self.log("Predicting (always NEUTRAL)")
            return ["NEUTRAL" for _ in range(len(X))]
        def save(self, path: str) -> bool:
            self.log(f"Saving to {path} (no-op)")
            return True
        def load(self, path: str) -> bool:
            self.log(f"Loading from {path} (no-op)")
            self.is_trained = True
            return True

    model = DummyModel("DummyModel")
    model.fit([[1,2],[3,4]], [0,1])
    print(model.predict([[1,2],[3,4]]))
    print(model.get_metadata())
