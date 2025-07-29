# Base model class
# Abstract base class for all ML models with common interface and utilities # base_model.py
"""
BaseModel: Abstract base class for all ML models in the Investment Committee.
Defines common interface for fit, predict, save, load, and metadata.
"""

import abc
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class BaseModel(abc.ABC):
    """
    Abstract base class for all committee models (XGBoost, MLP, LSTM, etc.).
    Provides standard interface for training, prediction, saving/loading, and metadata.
    """

    def __init__(self, name: str = "BaseModel"):
        self.name = name
        self.is_trained = False
        self.metadata: Dict[str, Any] = {}

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
