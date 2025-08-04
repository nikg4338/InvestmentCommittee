#!/usr/bin/env python3
"""
Training Configuration
=====================

Centralized configuration for all training hyperparameters, thresholds,
and visualization settings. This eliminates magic numbers scattered
throughout the codebase and makes experimentation easier.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class DataBalancingConfig:
    """Configuration for data balancing and sampling"""
    max_ratio: float = 2.5                    # Maximum majority:minority ratio before capping
    desired_ratio: float = 0.6                # Target ratio for controlled balancing (60% majority, 40% minority)
    minority_threshold: int = 100             # Threshold below which to use oversampling
    smote_k_neighbors: int = 5                # k_neighbors for SMOTE (will be adapted for small datasets)
    
@dataclass
class CrossValidationConfig:
    """Configuration for cross-validation and out-of-fold stacking"""
    n_folds: int = 5                          # Number of cross-validation folds
    random_state: int = 42                    # Random state for reproducibility
    shuffle: bool = True                      # Whether to shuffle in StratifiedKFold
    min_minority_samples: int = 2             # Minimum minority samples for robust folding
    
@dataclass
class CalibrationConfig:
    """Configuration for model calibration"""
    method: str = 'isotonic'                  # 'isotonic' or 'sigmoid'
    cv_folds: int = 3                         # Number of CV folds for calibration
    
@dataclass
class MetaModelConfig:
    """Configuration for meta-model training"""
    model_type: str = 'logistic_regression'   # Meta-model type
    max_iter: int = 1000                      # Max iterations for LogisticRegression
    regularization_c: float = 0.1            # Regularization strength
    class_weight: str = 'balanced'            # Class weighting strategy
    solver: str = 'liblinear'                 # Solver for extreme imbalance
    
@dataclass
class ThresholdConfig:
    """Configuration for threshold optimization"""
    threshold_grid_points: int = 101          # Number of threshold points to test (0.0 to 1.0)
    min_positive_rate: float = 0.01          # Minimum positive prediction rate (1%)
    fallback_percentile: float = 1.0         # Percentile for fallback threshold
    emergency_threshold: float = 0.001       # Emergency threshold when all else fails
    
@dataclass
class VisualizationConfig:
    """Configuration for plotting and visualization"""
    chart_figure_width: int = 10             # Width for metric comparison charts
    chart_figure_height: int = 6             # Height for metric comparison charts  
    matrix_figure_width: int = 8             # Width for confusion matrix plots
    matrix_figure_height: int = 6            # Height for confusion matrix plots
    chart_dpi: int = 150                     # DPI for saved charts
    save_plots: bool = True                  # Whether to save plots to disk
    plot_format: str = 'png'                 # Plot file format
    
@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods"""
    voting_strategy: str = 'rank_and_vote'   # 'threshold' or 'rank_and_vote'
    top_percentile: float = 0.01             # Top percentage for rank-and-vote (1%)
    majority_threshold_factor: float = 0.5   # Factor for majority voting (0.5 = half + 1)
    meta_weight: float = 1.5                 # Weight for meta-model in ensemble
    min_consensus: int = 1                   # Minimum samples for consensus
    
@dataclass
class TrainingConfig:
    """Main training configuration combining all sub-configs"""
    # Sub-configurations
    data_balancing: DataBalancingConfig
    cross_validation: CrossValidationConfig
    calibration: CalibrationConfig
    meta_model: MetaModelConfig
    threshold: ThresholdConfig
    visualization: VisualizationConfig
    ensemble: EnsembleConfig
    
    # General settings
    enable_timing: bool = True               # Whether to measure and log timing
    enable_calibration: bool = True          # Whether to use calibrated models
    enable_advanced_sampling: bool = True    # Whether to use SMOTE/SMOTEENN
    random_state: int = 42                   # Global random state
    
    # Data splitting
    test_size: float = 0.2                   # Test set size
    validation_size: float = 0.2             # Validation set size (from training)
    
    # Model selection
    models_to_train: list = None             # List of model names to train (None = all)
    
    def __post_init__(self):
        """Initialize sub-configs if not provided"""
        if self.data_balancing is None:
            self.data_balancing = DataBalancingConfig()
        if self.cross_validation is None:
            self.cross_validation = CrossValidationConfig()
        if self.calibration is None:
            self.calibration = CalibrationConfig()
        if self.meta_model is None:
            self.meta_model = MetaModelConfig()
        if self.threshold is None:
            self.threshold = ThresholdConfig()
        if self.visualization is None:
            self.visualization = VisualizationConfig()
        if self.ensemble is None:
            self.ensemble = EnsembleConfig()
        if self.models_to_train is None:
            self.models_to_train = ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'svm']

def get_default_config() -> TrainingConfig:
    """Get default training configuration"""
    return TrainingConfig(
        data_balancing=DataBalancingConfig(),
        cross_validation=CrossValidationConfig(),
        calibration=CalibrationConfig(),
        meta_model=MetaModelConfig(),
        threshold=ThresholdConfig(),
        visualization=VisualizationConfig(),
        ensemble=EnsembleConfig()
    )

def get_extreme_imbalance_config() -> TrainingConfig:
    """Get configuration optimized for extreme class imbalance (99%+ negative)"""
    config = get_default_config()
    
    # Adjust for extreme imbalance
    config.data_balancing.max_ratio = 10.0           # Allow higher imbalance
    config.cross_validation.n_folds = 3              # Fewer folds for small minority
    config.cross_validation.min_minority_samples = 5 # Boost minority more aggressively
    config.meta_model.regularization_c = 0.01        # Stronger regularization
    config.threshold.min_positive_rate = 0.005       # Lower minimum (0.5%)
    config.ensemble.top_percentile = 0.005           # More selective voting
    
    return config

def get_fast_training_config() -> TrainingConfig:
    """Get configuration for faster training (fewer folds, simpler methods)"""
    config = get_default_config()
    
    # Speed optimizations
    config.cross_validation.n_folds = 3
    config.calibration.cv_folds = 2
    config.enable_calibration = False
    config.enable_advanced_sampling = False
    config.visualization.save_plots = False
    
    return config

# Global default configuration
DEFAULT_CONFIG = get_default_config()

# Export commonly used values for backward compatibility
DEFAULT_MAX_RATIO = DEFAULT_CONFIG.data_balancing.max_ratio
DEFAULT_DESIRED_RATIO = DEFAULT_CONFIG.data_balancing.desired_ratio
DEFAULT_MINORITY_THRESHOLD = DEFAULT_CONFIG.data_balancing.minority_threshold
DEFAULT_N_FOLDS = DEFAULT_CONFIG.cross_validation.n_folds
DEFAULT_RANDOM_STATE = DEFAULT_CONFIG.cross_validation.random_state
DEFAULT_CALIBRATION_METHOD = DEFAULT_CONFIG.calibration.method
DEFAULT_CALIBRATION_CV = DEFAULT_CONFIG.calibration.cv_folds
DEFAULT_SMOTE_K_NEIGHBORS = DEFAULT_CONFIG.data_balancing.smote_k_neighbors
DEFAULT_META_MODEL = DEFAULT_CONFIG.meta_model.model_type
DEFAULT_META_MAX_ITER = DEFAULT_CONFIG.meta_model.max_iter
