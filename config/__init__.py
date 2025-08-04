"""
Configuration package for Investment Committee
============================================

Centralized configuration management for all training hyperparameters,
thresholds, and settings. Eliminates magic numbers and makes 
experimentation easier.
"""

# Import main configuration classes
from .training_config import (
    TrainingConfig, DataBalancingConfig, CrossValidationConfig,
    CalibrationConfig, MetaModelConfig, ThresholdConfig, 
    VisualizationConfig, EnsembleConfig,
    get_default_config, get_extreme_imbalance_config, get_fast_training_config
) 