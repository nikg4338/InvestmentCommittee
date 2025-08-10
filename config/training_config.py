#!/usr/bin/env python3
"""
Training Configuration
=====================

Centralized configuration for all training hyperparameters, thresholds,
and         if self.ensemble is None:
            self.ensemble = EnsembleConfig()
        if self.models_to_train is None:
            self.models_to_train = ['xgboost', 'lightgbm', 'catboost', 'random_forest']
        
        # Initialize quantile levels if not provided
        if self.quantile_levels is None:
            self.quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]  # Default quantile levels for uncertainty estimationalization settings. This eliminates magic numbers scattered
throughout the codebase and makes experimentation easier.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class DataBalancingConfig:
    """Configuration for data balancing and sampling"""
    max_ratio: float = 2.5                    # Maximum majority:minority ratio before capping
    desired_ratio: float = 0.5                # Target ratio for controlled balancing (50% majority, 50% minority) - Updated for perfect balance
    minority_threshold: int = 100             # Threshold below which to use oversampling
    smote_k_neighbors: int = 5                # k_neighbors for SMOTE (will be adapted for small datasets)
    # Enhanced SMOTE parameters for regression models
    use_smote_for_regressors: bool = True     # Whether to apply SMOTE to regression models
    smote_sampling_strategy: str = 'auto'     # SMOTE sampling strategy ('auto', 'minority', float)
    smote_random_state: int = 42              # Random state for SMOTE reproducibility
    combine_smote_with_weighting: bool = True # Whether to use both SMOTE and sample weighting
    
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
    model_type: str = 'gradient_boosting'     # Meta-model type (updated to gradient boosting)
    meta_learner_type: str = 'gradientboost'  # Enhanced meta-learner type ('lightgbm', 'gradientboost', 'xgboost')
    max_iter: int = 1000                       # Max iterations (legacy parameter)
    regularization_c: float = 0.1             # Regularization strength (legacy parameter)
    class_weight: str = 'balanced'             # Class weighting strategy (legacy parameter)
    solver: str = 'liblinear'                  # Solver for extreme imbalance (legacy parameter)
    # Gradient Boosting specific parameters
    n_estimators: int = 100                    # Number of boosting stages
    learning_rate: float = 0.1                # Shrinkage factor
    max_depth: int = 3                         # Maximum depth of individual trees
    
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
    
    # Advanced features
    enable_optuna: bool = False              # Enable Optuna hyperparameter tuning
    optuna_trials: int = 20                  # Number of Optuna trials per model
    
    # Pipeline improvements
    enable_enhanced_stacking: bool = True         # Use enhanced stacking with improvements
    use_time_series_cv: bool = True               # Use time-series cross-validation for real market data
    enable_feature_selection: bool = True         # Enable SHAP-based feature selection
    advanced_sampling: str = 'smoteenn'           # Advanced sampling strategy ('smoteenn', 'adasyn', 'smotetomek')
    use_xgb_meta_model: bool = True               # Use XGBoost instead of LogisticRegression for meta-model
    stack_raw_features: bool = False              # Stack raw features with meta-features
    enable_rolling_backtest: bool = True          # Enable rolling backtest for drift detection
    enable_drift_detection: bool = True           # Enable data drift detection
    enable_llm_features: bool = True              # Enable LLM-generated macro signals for real market data
    
    # Enhanced meta-model strategies for F₁ optimization
    meta_model_strategy: str = 'optimal_threshold'  # ('optimal_threshold', 'focal_loss', 'dynamic_weights', 'feature_select')
    optuna_optimize_for: str = 'average_precision'  # Metric for Optuna optimization ('average_precision', 'f1_weighted', 'roc_auc')
    
    # Regression-based approach enhancements for better F₁ scores
    enable_regression_targets: bool = False          # Use regression targets instead of restrictive binary thresholds
    regression_threshold_optimization: bool = False  # Enable automatic threshold optimization for regression predictions
    huber_loss_alpha: float = 0.9                   # Huber loss alpha parameter for outlier robustness
    evaluate_regression_metrics: bool = False        # Include regression metrics (MSE, MAE, RMSE) in evaluation
    multi_horizon_targets: bool = False              # Use multiple target horizons (1d, 3d, 5d, 10d) for ensemble diversity
    
    # Phase 3: Quantile Loss Options for uncertainty estimation and risk-aware decision making
    enable_quantile_regression: bool = False         # Enable quantile regression for uncertainty estimation
    quantile_levels: list = None                     # List of quantile levels (e.g., [0.1, 0.5, 0.9])
    quantile_ensemble_method: str = 'median'         # Method for combining quantile predictions ('mean', 'median', 'weighted')
    quantile_decision_strategy: str = 'threshold_optimization'  # Strategy for binary conversion ('threshold_optimization', 'risk_aware', 'median_based')
    risk_tolerance: str = 'moderate'                 # Risk tolerance for risk-aware decisions ('conservative', 'moderate', 'aggressive')
    evaluate_quantile_metrics: bool = False          # Include quantile-specific metrics (pinball loss, coverage, interval width)
    
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
    """Get configuration optimized for extreme class imbalance (99%+ negative) with regression enhancements"""
    config = get_default_config()
    
    # Adjust for extreme imbalance
    config.data_balancing.max_ratio = 10.0           # Allow higher imbalance
    config.cross_validation.n_folds = 3              # Fewer folds for small minority
    config.cross_validation.min_minority_samples = 5 # Boost minority more aggressively
    config.meta_model.regularization_c = 0.01        # Stronger regularization
    config.threshold.min_positive_rate = 0.005       # Lower minimum (0.5%)
    config.ensemble.top_percentile = 0.005           # More selective voting
    config.enable_optuna = True                      # Enable hyperparameter tuning
    config.optuna_trials = 15                        # Moderate number of trials for balance
    
    # Include LightGBM regressor in model ensemble for enhanced predictions
    # Use regression models for better signal capture with continuous targets
    config.models_to_train = [
        # Classification models (keep for comparison)
        'xgboost', 'lightgbm', 'catboost', 'random_forest',
        # Regression models with Huber loss for robustness
        'lightgbm_regressor', 'xgboost_regressor', 'catboost_regressor', 
        'random_forest_regressor', 'svm_regressor',
        # Phase 3: Quantile regression for uncertainty estimation (commented out due to circular import)
        # 'lightgbm_quantile_regressor'
    ]
    
    # Enable advanced pipeline improvements for extreme imbalance
    config.enable_enhanced_stacking = True           # Use enhanced stacking
    config.enable_calibration = True                 # Probability calibration important for imbalance
    config.advanced_sampling = 'adasyn'              # ADASYN works well for extreme imbalance
    config.use_xgb_meta_model = True                 # XGBoost meta-model for better non-linear learning
    config.enable_drift_detection = True             # Monitor for distribution shifts
    
    # Enhanced meta-model strategies for extreme imbalance with regression support
    config.meta_model_strategy = 'optimal_threshold'  # Use optimal threshold with gradient boosting
    config.optuna_optimize_for = 'average_precision'  # Optimize for PR-AUC (better for imbalance)
    config.stack_raw_features = True                 # Stack raw features for more signal
    
    # Regression-specific enhancements for better F₁ scores
    config.enable_regression_targets = True          # Use regression targets by default
    config.regression_threshold_optimization = True  # Enable automatic threshold optimization
    config.huber_loss_alpha = 0.9                   # Robust regression parameter for outliers
    
    # Enhanced evaluation metrics for regression + classification hybrid approach
    config.evaluate_regression_metrics = True        # Include MSE, MAE, RMSE in evaluation
    config.multi_horizon_targets = True             # Use 1d, 3d, 5d, 10d targets for ensemble diversity
    
    # Phase 3: Enable quantile regression for uncertainty estimation and risk-aware decisions
    config.enable_quantile_regression = True         # Enable quantile regression capabilities
    config.quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]  # Multiple quantiles for uncertainty estimation
    config.quantile_ensemble_method = 'median'       # Robust ensemble method for quantile predictions
    config.quantile_decision_strategy = 'risk_aware' # Risk-aware binary decision strategy
    config.risk_tolerance = 'moderate'               # Balanced risk tolerance for investment decisions
    config.evaluate_quantile_metrics = True          # Include quantile-specific evaluation metrics
    
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
