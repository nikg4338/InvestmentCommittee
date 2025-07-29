# Tests for ML models
# Unit tests for XGBoost, Neural Network, LLM, and Meta models # test_models.py
"""
Unit tests for XGBoost, Neural Network, LLM, and Meta models.
"""

import numpy as np
import pandas as pd
import pytest

def test_xgboost_fit_and_predict():
    from models.xgboost_model import XGBoostModel, XGBOOST_AVAILABLE
    if not XGBOOST_AVAILABLE:
        pytest.skip("xgboost not installed")
    X = pd.DataFrame({
        "rsi": np.random.uniform(30, 70, 50),
        "momentum": np.random.normal(0, 1, 50),
        "iv_rank": np.random.uniform(0, 1, 50),
        "zscore": np.random.normal(0, 1, 50),
    })
    y = np.random.normal(0, 0.03, 50)
    model = XGBoostModel()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)

def test_nn_predict_signal():
    from models.neural_predictor import NeuralPredictor, PYTORCH_AVAILABLE
    features = {
        "technicals": {
            "rsi": 45.0, "macd_signal": 0.2, "bollinger_position": 0.4,
            "volume_ratio": 1.2, "price_momentum": 0.1, "volatility_rank": 50.0,
            "vix_level": 18.0, "market_trend": 0.3, "price_volatility": 0.02,
            "support_distance": 0.05, "resistance_distance": 0.03, "trend_strength": 0.15
        }
    }
    predictor = NeuralPredictor(model_type="mlp")
    direction, confidence, meta = predictor.predict_nn_signal(features)
    assert direction in ["BULLISH", "BEARISH", "NEUTRAL"]
    assert 0.0 <= confidence <= 1.0

def test_meta_model_import():
    # Just test import; add ensemble test as implemented
    from models.meta_model import MetaModel
    assert MetaModel is not None
