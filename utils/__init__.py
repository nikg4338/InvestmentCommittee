"""
Utilities package for Investment Committee
==========================================

This package provides modular utilities for the investment committee
training pipeline, including:

- data_splitting: Robust train/test splitting with extreme imbalance handling
- sampling: Advanced SMOTE/SMOTEENN sampling techniques  
- stacking: Out-of-fold stacking and meta-model training
- evaluation: Comprehensive model evaluation and metrics
- visualization: Centralized plotting and report generation
"""

# Import key functions for backward compatibility
from .data_splitting import stratified_train_test_split, ensure_minority_samples
from .sampling import prepare_balanced_data
from .stacking import out_of_fold_stacking, train_meta_model
from .evaluation import evaluate_ensemble_performance, find_optimal_threshold
from .visualization import create_training_report 