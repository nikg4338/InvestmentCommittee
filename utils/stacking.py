#!/usr/bin/env python3
"""
Stacking Utilities
=================

Out-of-fold stacking utilities for meta-model training with robust
handling of extreme class imbalance.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional, Union
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

# Model imports
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.lightgbm_regressor import LightGBMRegressor
# from models.lightgbm_quantile_regressor import LightGBMQuantileRegressor  # Commented out to avoid circular import
from models.catboost_model import CatBoostModel
from models.random_forest_model import RandomForestModel
from models.svc_model import SVMClassifier

# Regressor model imports
from models.xgboost_regressor import XGBoostRegressorModel
from models.catboost_regressor import CatBoostRegressorModel
from models.random_forest_regressor import RandomForestRegressorModel
from models.svm_regressor import SVMRegressorModel

from utils.data_splitting import prepare_cv_data
from utils.sampling import prepare_balanced_data
from config.training_config import TrainingConfig, get_default_config

# Import pipeline improvements
from utils.pipeline_improvements import (
    tune_with_optuna, calibrate_model, get_advanced_sampler,
    create_time_series_splits, create_xgb_meta_model,
    select_top_features_shap
)

logger = logging.getLogger(__name__)

def is_regressor_model(model_name: str) -> bool:
    """Check if a model name corresponds to a regression model."""
    return 'regressor' in model_name.lower() or model_name.endswith('_regressor')

def is_quantile_model(model_name: str) -> bool:
    """Check if a model name corresponds to a quantile regression model."""
    return 'quantile' in model_name.lower()

def get_model_predictions_safe(model, X: pd.DataFrame, model_name: str = ""):
    """
    Return PROBABILITIES for classifiers, raw predictions for regressors,
    and a dict for quantile models. Never fall back to hard labels unless
    absolutely necessary.
    """
    import numpy as np
    import pandas as pd
    from math import exp

    try:
        is_reg = is_regressor_model(model_name)
        is_q = is_quantile_model(model_name)

        # Ensure DataFrame has correct dtypes for LightGBM models
        if 'lightgbm' in model_name.lower() and isinstance(X, pd.DataFrame):
            X_clean = X.copy()
            for col in X_clean.columns:
                if X_clean[col].dtype == 'object':
                    try:
                        X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
                    except:
                        pass
                if X_clean[col].dtype not in ['int64', 'int32', 'float64', 'float32', 'bool']:
                    X_clean[col] = X_clean[col].astype('float64')
            X = X_clean

        if is_q:
            preds = model.predict(X)
            return preds if isinstance(preds, dict) else {0.5: np.asarray(preds).ravel()}

        if is_reg:
            return np.asarray(model.predict(X)).ravel()

        # classifier path
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else np.asarray(proba).ravel()

        if hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(X)).ravel()
            return 1.0 / (1.0 + np.exp(-scores))  # Platt-like mapping

        # last resort
        logger.warning(f"{model_name}: using hard labels because no proba/decision_function is available.")
        return np.asarray(model.predict(X)).astype(float).ravel()

    except Exception as e:
        logger.warning(f"Error getting predictions from {model_name}: {e}")
        return np.zeros(len(X))

# Model registry for easy access
MODEL_REGISTRY = {
    # Classification models
    'xgboost': XGBoostModel,
    'lightgbm': LightGBMModel,
    'catboost': CatBoostModel,
    'random_forest': RandomForestModel,
    'svm': SVMClassifier,
    
    # Regression models
    'lightgbm_regressor': LightGBMRegressor,
    # 'lightgbm_quantile_regressor': LightGBMQuantileRegressor,  # Commented out to avoid circular import
    'xgboost_regressor': XGBoostRegressorModel,
    'catboost_regressor': CatBoostRegressorModel,
    'random_forest_regressor': RandomForestRegressorModel,
    'svm_regressor': SVMRegressorModel
}

def create_model_configs(config: Optional[TrainingConfig] = None) -> Dict[str, Dict[str, Any]]:
    """
    Create model configurations for training.
    
    Args:
        config: Training configuration
        
    Returns:
        Dictionary mapping model names to configurations
    """
    if config is None:
        config = get_default_config()
    
    model_configs = {}
    
    for model_name in config.models_to_train:
        if model_name in MODEL_REGISTRY:
            model_configs[model_name] = {
                'class': MODEL_REGISTRY[model_name],
                'balance_method': 'smote' if config.enable_advanced_sampling else 'basic',
                'calibrate': config.enable_calibration,
                'config': config
            }
        else:
            logger.warning(f"Unknown model: {model_name}")
    
    return model_configs

def simple_train_test_stacking(X_train: pd.DataFrame, y_train: pd.Series, 
                              X_test: pd.DataFrame,
                              config: Optional[TrainingConfig] = None) -> Tuple[None, None, Dict[str, Any]]:
    """
    Fallback when OOF is impossible due to extreme imbalance:
    Train each model on full train set, predict on test.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        config: Training configuration
        
    Returns:
        Tuple of (None, None, trained_models) - None indicates fallback was used
    """
    if config is None:
        config = get_default_config()
    
    logger.info("Using simple train/test stacking fallback")
    
    model_configs = create_model_configs(config)
    trained_models = {}
    
    for model_name, model_info in model_configs.items():
        start_time = time.time()
        logger.info(f"Training {model_name} on full training set...")
        
        try:
            # Balance training data
            balance_method = model_info.get('balance_method', 'smote')
            X_balanced, y_balanced = prepare_balanced_data(X_train, y_train, balance_method)
            
            # Train model
            model_class = model_info['class']
            model = model_class()
            model.train(X_balanced, y_balanced)
            
            # Get predictions using the safe prediction function
            train_predictions = get_model_predictions_safe(model, X_balanced, model_name)
            test_predictions = get_model_predictions_safe(model, X_test, model_name)
            
            # Store results
            trained_models[model_name] = {
                'model': model,
                'config': model_info,
                'train_predictions': train_predictions,
                'train_labels': y_balanced,
                'test_predictions': test_predictions,
                'is_regressor': is_regressor_model(model_name)
            }
            
            duration = time.time() - start_time
            logger.info(f"  ✓ {model_name} completed in {duration:.1f}s")
            
        except Exception as e:
            logger.error(f"  ✗ {model_name} failed: {e}")
            continue
    
    logger.info(f"Simple stacking completed with {len(trained_models)} models")
    return None, None, trained_models

def out_of_fold_stacking(X_train: pd.DataFrame, y_train: pd.Series, 
                        X_test: pd.DataFrame,
                        config: Optional[TrainingConfig] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Perform robust out-of-fold stacking to prevent overfitting.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        config: Training configuration
        
    Returns:
        Tuple of (train_meta_features, test_meta_features, trained_models, oof_predictions)
    """
    if config is None:
        config = get_default_config()
    
    logger.info(f"Starting out-of-fold stacking...")
    
    # Prepare cross-validation data
    try:
        X_cv, y_cv, skf = prepare_cv_data(X_train, y_train, config.cross_validation)
        n_folds = skf.n_splits
        logger.info(f"Using {n_folds} stratified folds")
    except Exception as e:
        logger.error(f"CV preparation failed: {e}")
        logger.info("Falling back to simple train/test stacking")
        # simple_train_test_stacking returns (None, None, trained_models)
        _t_train, _t_test, trained_models = simple_train_test_stacking(X_train, y_train, X_test, config)
        # Build placeholders and OOF dict
        dummy_train = None
        dummy_test = None
        oof_predictions = {k: np.zeros(len(X_train)) for k in trained_models.keys()}
        return dummy_train, dummy_test, trained_models, oof_predictions
    
    # Check if we have enough samples for OOF
    class_counts = y_cv.value_counts()
    if class_counts.min() < n_folds:
        logger.warning(f"Insufficient minority samples for {n_folds}-fold CV (min: {class_counts.min()})")
        logger.info("Falling back to simple train/test stacking")
        _t_train, _t_test, trained_models = simple_train_test_stacking(X_train, y_train, X_test, config)
        oof_predictions = {k: np.zeros(len(X_train)) for k in trained_models.keys()}
        return None, None, trained_models, oof_predictions
    
    # Create model configurations
    model_configs = create_model_configs(config)
    
    # Initialize meta-feature arrays
    n_models = len(model_configs)
    train_meta_features = np.zeros((len(X_cv), n_models))
    test_meta_features = np.zeros((len(X_test), n_models))
    
    trained_models = {}
    model_thresholds = {name: [] for name in model_configs.keys()}  # Track optimal thresholds per fold
    oof_store = {name: np.zeros(len(X_cv)) for name in model_configs.keys()}  # OOF bookkeeping
    
    # Train each model with OOF
    for model_idx, (model_name, model_info) in enumerate(model_configs.items()):
        start_time = time.time()
        logger.info(f"Training {model_name} with {n_folds}-fold OOF...")
        
        fold_predictions = []
        fold_models = []
        
        # Optuna hyperparameter tuning for supported models
        optuna_params = {}
        if model_name in ['random_forest', 'catboost'] and hasattr(config, 'enable_optuna') and config.enable_optuna:
            try:
                from train_models import optuna_tune_model
                model_class = model_info['class']
                logger.info(f"  🎯 Running Optuna hyperparameter tuning for {model_name}...")
                optuna_params = optuna_tune_model(model_class, X_cv, y_cv, n_trials=10)
                if optuna_params:
                    logger.info(f"  ✓ Optuna found optimal params for {model_name}: {optuna_params}")
            except Exception as e:
                logger.warning(f"  ⚠️ Optuna tuning failed for {model_name}: {e}")
        
        try:
            # Cross-validation loop
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv)):
                logger.info(f"  Fold {fold + 1}/{n_folds}")
                
                # Split data for this fold
                X_fold_train = X_cv.iloc[train_idx]
                y_fold_train = y_cv.iloc[train_idx]
                X_fold_val = X_cv.iloc[val_idx]
                y_fold_val = y_cv.iloc[val_idx]
                
                # Guard: ensure both classes exist in the training portion of this fold
                if pd.Series(y_fold_train).nunique() < 2:
                    logger.warning(f"Skipping fold {fold} for {model_name}: single-class training fold.")
                    continue
                
                # Balance training data for this fold with safe resampling
                balance_method = model_info.get('balance_method', 'smote')
                try:
                    X_fold_balanced, y_fold_balanced = prepare_balanced_data(
                        X_fold_train, y_fold_train, balance_method
                    )
                except Exception as e:
                    logger.warning(f"Resampling failed for fold {fold} of {model_name}: {e}. Using original data.")
                    X_fold_balanced, y_fold_balanced = X_fold_train, y_fold_train
                
                # Train model on this fold with optional Optuna params
                model_class = model_info['class']
                if optuna_params:
                    # Merge Optuna params with default config
                    combined_params = {**model_info.get('params', {}), **optuna_params}
                    fold_model = model_class(**combined_params)
                else:
                    fold_model = model_class()
                
                fold_model.train(X_fold_balanced, y_fold_balanced)
                
                # Predict on validation set (out-of-fold)
                val_predictions = get_model_predictions_safe(fold_model, X_fold_val, model_name)
                
                # Use default threshold during cross-validation to avoid misleading metrics
                # Threshold optimization should only happen on final test set
                try:
                    if is_regressor_model(model_name):
                        fold_threshold = 0.0  # Default threshold for regressors
                    else:
                        fold_threshold = 0.5  # Default threshold for classifiers
                    model_thresholds[model_name].append(fold_threshold)
                    logger.info(f"    Fold {fold + 1} using default threshold: {fold_threshold:.1f}")
                except Exception as e:
                    logger.warning(f"    Failed to set threshold for fold {fold + 1}: {e}")
                    fold_threshold = 0.5 if not is_regressor_model(model_name) else 0.0
                    model_thresholds[model_name].append(fold_threshold)
                
                # Store OOF predictions
                train_meta_features[val_idx, model_idx] = val_predictions
                oof_store[model_name][val_idx] = val_predictions  # Store in OOF bookkeeping
                
                # Predict on test set
                test_predictions = get_model_predictions_safe(fold_model, X_test, model_name)
                fold_predictions.append(test_predictions)
                
                fold_models.append(fold_model)
            
            # Average test predictions across folds
            test_meta_features[:, model_idx] = np.mean(fold_predictions, axis=0)
            
            # Compute average optimal threshold for this model
            avg_threshold = np.mean(model_thresholds[model_name]) if model_thresholds[model_name] else 0.5
            logger.info(f"  📊 {model_name} average optimal threshold: {avg_threshold:.4f}")
            
            # Store fold models and metadata
            trained_models[model_name] = {
                'models': fold_models,
                'config': model_info,
                'test_predictions': test_meta_features[:, model_idx],
                'optimal_threshold': avg_threshold,
                'fold_thresholds': model_thresholds[model_name],
                'optuna_params': optuna_params,
                'oof_predictions': oof_store[model_name]  # Add OOF predictions for meta model honesty
            }
            
            duration = time.time() - start_time
            logger.info(f"  ✓ {model_name} OOF completed in {duration:.1f}s")
            
        except Exception as e:
            logger.error(f"  ✗ {model_name} OOF failed: {e}")
            # Fill with zeros for failed model
            train_meta_features[:, model_idx] = 0.0
            test_meta_features[:, model_idx] = 0.0
            continue
    
    logger.info("Out-of-fold stacking completed")
    return train_meta_features, test_meta_features, trained_models, oof_store

def train_meta_model(meta_X: np.ndarray, meta_y: np.ndarray,
                    config: Optional[TrainingConfig] = None) -> Tuple[GradientBoostingClassifier, float]:
    """
    Train meta-model on out-of-fold predictions with threshold optimization.
    
    Args:
        meta_X: Meta-features (out-of-fold predictions)
        meta_y: Meta-labels (training labels)
        config: Training configuration
        
    Returns:
        Tuple of (trained meta-model, optimal_threshold)
    """
    if config is None:
        config = get_default_config()
    
    logger.info("Training meta-model with threshold optimization...")
    
    try:
        from utils.evaluation import find_optimal_threshold
        
        meta_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=config.random_state
        )
        
        # Validate input consistency
        if len(meta_X) != len(meta_y):
            logger.warning(f"Meta-features length ({len(meta_X)}) != target length ({len(meta_y)})")
            # Trim to minimum length
            min_len = min(len(meta_X), len(meta_y))
            meta_X = meta_X[:min_len]
            meta_y = meta_y[:min_len]
            logger.info(f"Trimmed both to length {min_len}")
        
        meta_model.fit(meta_X, meta_y)
        
        # Get training probabilities for threshold optimization
        meta_train_proba = meta_model.predict_proba(meta_X)[:, 1]
        
        # Analyze probability distribution
        logger.info(f"Meta-model probability distribution:")
        logger.info(f"  Min: {meta_train_proba.min():.4f}")
        logger.info(f"  Max: {meta_train_proba.max():.4f}")
        logger.info(f"  Mean: {meta_train_proba.mean():.4f}")
        logger.info(f"  Std: {meta_train_proba.std():.4f}")
        
        # Find optimal threshold
        optimal_threshold, optimal_f1 = find_optimal_threshold(meta_y, meta_train_proba, metric='f1')
        logger.info(f"Optimal meta-model threshold: {optimal_threshold:.4f} (F1: {optimal_f1:.4f})")
        
        # Log feature importance (coefficients)
        if hasattr(meta_model, 'coef_') and meta_model.coef_ is not None:
            feature_names = [f'model_{i}' for i in range(len(meta_model.coef_[0]))]
            coef_dict = {name: float(coef) for name, coef in zip(feature_names, meta_model.coef_[0])}
            logger.info(f"Meta-model feature weights: {coef_dict}")
        
        logger.info("✓ Meta-model training and threshold optimization completed")
        return meta_model, optimal_threshold
        
    except Exception as e:
        logger.error(f"Meta-model training failed: {e}")
        raise e

def predict_with_meta_model(meta_model: GradientBoostingClassifier, 
                           test_meta_features: np.ndarray) -> np.ndarray:
    """
    Generate predictions using trained meta-model.
    
    Args:
        meta_model: Trained meta-model
        test_meta_features: Test meta-features
        
    Returns:
        Meta-model predictions (probabilities)
    """
    try:
        # Prefer safe prediction utility to handle models without predict_proba
        return get_model_predictions_safe(meta_model, pd.DataFrame(test_meta_features), "meta_model")
    except Exception as e:
        logger.error(f"Meta-model prediction failed: {e}")
        # Return uniform probabilities as fallback
        return np.full(len(test_meta_features), 0.5)

def create_ensemble_predictions(trained_models: Dict[str, Any],
                               X_test: pd.DataFrame,
                               meta_model: Optional[GradientBoostingClassifier] = None,
                               config: Optional[TrainingConfig] = None) -> Dict[str, Any]:
    """
    Create ensemble predictions from trained models.
    
    Args:
        trained_models: Dictionary of trained models
        X_test: Test features
        meta_model: Trained meta-model (optional)
        config: Training configuration
        
    Returns:
        Dictionary with ensemble predictions and metadata
    """
    if config is None:
        config = get_default_config()
    
    logger.info("Creating ensemble predictions...")
    
    # Collect base model predictions
    base_predictions = {}
    
    for model_name, model_data in trained_models.items():
        if 'test_predictions' in model_data:
            # Use pre-computed test predictions
            base_predictions[model_name] = model_data['test_predictions']
        elif 'models' in model_data:
            # Average predictions from fold models
            fold_predictions = []
            for fold_model in model_data['models']:
                try:
                    test_predictions = get_model_predictions_safe(fold_model, X_test, model_name)
                    fold_predictions.append(test_predictions)
                except Exception as e:
                    logger.warning(f"Prediction failed for {model_name} fold model: {e}")
                    continue
            
            if fold_predictions:
                base_predictions[model_name] = np.mean(fold_predictions, axis=0)
            else:
                logger.warning(f"No valid predictions for {model_name}")
        elif 'model' in model_data:
            # Single model prediction
            try:
                test_predictions = get_model_predictions_safe(model_data['model'], X_test, model_name)
                base_predictions[model_name] = test_predictions
            except Exception as e:
                logger.warning(f"Prediction failed for {model_name}: {e}")
    
    # Create meta-model predictions if available
    meta_predictions = None
    if meta_model is not None and base_predictions:
        try:
            # Stack base predictions as features
            meta_features = np.column_stack(list(base_predictions.values()))
            # Use safe predictor to accommodate models without predict_proba
            meta_predictions = get_model_predictions_safe(meta_model, pd.DataFrame(meta_features), "meta_model")
            logger.info("✓ Meta-model predictions generated")
        except Exception as e:
            logger.error(f"Meta-model prediction failed: {e}")
    
    # Calculate simple average ensemble
    if base_predictions:
        simple_ensemble = np.mean(list(base_predictions.values()), axis=0)
    else:
        simple_ensemble = np.full(len(X_test), 0.5)
        logger.warning("No base predictions available, using uniform probabilities")
    
    results = {
        'base_predictions': base_predictions,
        'meta_predictions': meta_predictions,
        'simple_ensemble': simple_ensemble,
        'n_models': len(base_predictions),
        'ensemble_strategy': config.ensemble.voting_strategy
    }
    
    logger.info(f"Ensemble created with {len(base_predictions)} base models")
    return results

def evaluate_stacking_quality(train_meta_features: np.ndarray,
                             test_meta_features: np.ndarray,
                             model_names: List[str]) -> Dict[str, Any]:
    """
    Evaluate the quality of stacking features.
    
    Args:
        train_meta_features: Training meta-features
        test_meta_features: Test meta-features
        model_names: List of model names
        
    Returns:
        Dictionary with stacking quality metrics
    """
    metrics = {}
    
    # Check for valid predictions
    train_valid = ~np.isnan(train_meta_features).all(axis=0)
    test_valid = ~np.isnan(test_meta_features).all(axis=0)
    
    metrics['valid_models_train'] = train_valid.sum()
    metrics['valid_models_test'] = test_valid.sum()
    metrics['total_models'] = len(model_names)
    
    # Check prediction diversity
    if train_meta_features.shape[1] > 1:
        train_corr = np.corrcoef(train_meta_features.T)
        metrics['mean_correlation'] = np.mean(train_corr[np.triu_indices_from(train_corr, k=1)])
        metrics['max_correlation'] = np.max(train_corr[np.triu_indices_from(train_corr, k=1)])
    else:
        metrics['mean_correlation'] = 0.0
        metrics['max_correlation'] = 0.0
    
    # Check prediction ranges
    metrics['train_pred_ranges'] = {
        f'{name}_range': (float(train_meta_features[:, i].min()), float(train_meta_features[:, i].max()))
        for i, name in enumerate(model_names)
        if i < train_meta_features.shape[1]
    }
    
    # Log quality assessment
    if metrics['valid_models_train'] < 2:
        logger.warning(f"Poor stacking quality: only {metrics['valid_models_train']} valid models")
    elif metrics['mean_correlation'] > 0.9:
        logger.warning(f"High model correlation: {metrics['mean_correlation']:.3f}")
    else:
        logger.info(f"Good stacking quality: {metrics['valid_models_train']} diverse models")
    
    return metrics

# ============================================================================
# Enhanced Stacking Functions with Pipeline Improvements
# ============================================================================

def enhanced_out_of_fold_stacking(X_train: pd.DataFrame, y_train: pd.Series, 
                                 X_test: pd.DataFrame,
                                 config: Optional[TrainingConfig] = None,
                                 use_time_series_cv: bool = False,
                                 enable_calibration: bool = True,
                                 enable_feature_selection: bool = False,
                                 advanced_sampling: str = 'smoteenn') -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Enhanced out-of-fold stacking with pipeline improvements.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        config: Training configuration
        use_time_series_cv: Use time-series cross-validation
        enable_calibration: Enable probability calibration
        enable_feature_selection: Enable SHAP-based feature selection
        advanced_sampling: Advanced sampling strategy
        
    Returns:
        Tuple of (train_meta_features, test_meta_features, trained_models)
    """
    if config is None:
        config = get_default_config()
    
    logger.info("🚀 Enhanced out-of-fold stacking with pipeline improvements...")
    
    # Feature selection using SHAP (if enabled)
    selected_features = X_train.columns.tolist()
    if enable_feature_selection:
        try:
            # Train a quick model for feature selection
            from models.lightgbm_model import LightGBMModel
            temp_model = LightGBMModel()
            temp_model.train(X_train, y_train)
            
            # Select top features using SHAP
            n_features = min(50, len(X_train.columns))  # Select top 50 or all if fewer
            selected_features = select_top_features_shap(temp_model.model, X_train, k=n_features)
            
            # Update feature matrices
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]
            
            logger.info(f"🎯 Feature selection: {len(selected_features)}/{len(X_train.columns)} features selected")
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}, using all features")
    
    # Prepare cross-validation strategy
    try:
        if use_time_series_cv:
            # FIX: pass integer number of folds, not the CrossValidationConfig object
            n_splits = getattr(getattr(config, 'cross_validation', object()), 'n_folds', 5)
            skf = create_time_series_splits(n_splits=n_splits)
            logger.info(f"Using time-series cross-validation with {n_splits} splits")
        else:
            from utils.data_splitting import prepare_cv_data
            X_cv, y_cv, skf = prepare_cv_data(X_train, y_train, config.cross_validation)
        
        n_folds = getattr(skf, 'n_splits', getattr(getattr(config, 'cross_validation', object()), 'n_folds', 5))
        logger.info(f"Using {n_folds} folds for cross-validation")
        
    except Exception as e:
        logger.error(f"CV preparation failed: {e}")
        logger.info("Falling back to simple train/test stacking")
        # simple_train_test_stacking returns (None, None, trained_models)
        _t_train, _t_test, trained_models = simple_train_test_stacking(X_train, y_train, X_test, config)
        # Build placeholders and OOF dict
        dummy_train = None
        dummy_test = None
        oof_predictions = {k: np.zeros(len(X_train)) for k in trained_models.keys()}
        return dummy_train, dummy_test, trained_models, oof_predictions
    
    # Check sample sufficiency
    class_counts = y_train.value_counts()
    if class_counts.min() < n_folds:
        logger.warning(f"Insufficient minority samples for {n_folds}-fold CV (min: {class_counts.min()})")
        logger.info("Falling back to simple train/test stacking")
        _t_train, _t_test, trained_models = simple_train_test_stacking(X_train, y_train, X_test, config)
        oof_predictions = {k: np.zeros(len(X_train)) for k in trained_models.keys()}
        return None, None, trained_models, oof_predictions
    
    # Create model configurations
    model_configs = create_model_configs(config)
    
    # Initialize meta-feature arrays
    n_models = len(model_configs)
    train_meta_features = np.zeros((len(X_train), n_models))
    test_meta_features = np.zeros((len(X_test), n_models))
    
    trained_models = {}
    model_thresholds = {name: [] for name in model_configs.keys()}
    
    # Train each model with enhancements
    for model_idx, (model_name, model_info) in enumerate(model_configs.items()):
        start_time = time.time()
        logger.info(f"🎯 Enhanced training for {model_name}...")
        
        fold_predictions = []
        fold_models = []
        
        # Optuna hyperparameter tuning
        optuna_params = {}
        if hasattr(config, 'enable_optuna') and config.enable_optuna:
            try:
                param_space = get_optuna_param_space(model_name)
                if param_space:
                    logger.info(f"  🔍 Optuna hyperparameter tuning for {model_name}...")
                    n_trials = getattr(config, 'optuna_trials', 20)
                    optuna_params = tune_with_optuna(
                        model_info['class'], X_train, y_train, param_space, n_trials=n_trials
                    )
                    if optuna_params:
                        logger.info(f"  ✓ Optuna optimization completed: {len(optuna_params)} params")
            except Exception as e:
                logger.warning(f"  ⚠️ Optuna tuning failed for {model_name}: {e}")
        
        try:
            # Cross-validation loop with enhancements
            X_cv = X_train if use_time_series_cv else X_train
            y_cv = y_train if use_time_series_cv else y_train
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv)):
                logger.info(f"  📊 Fold {fold + 1}/{n_folds}")
                
                # Split data for this fold
                X_fold_train = X_cv.iloc[train_idx]
                y_fold_train = y_cv.iloc[train_idx]
                X_fold_val = X_cv.iloc[val_idx]
                y_fold_val = y_cv.iloc[val_idx]
                
                # Advanced sampling
                sampler = get_advanced_sampler(advanced_sampling)
                if sampler:
                    try:
                        X_fold_balanced, y_fold_balanced = sampler.fit_resample(X_fold_train, y_fold_train)
                        X_fold_balanced = pd.DataFrame(X_fold_balanced, columns=X_fold_train.columns)
                        y_fold_balanced = pd.Series(y_fold_balanced)
                        logger.info(f"    Applied {advanced_sampling} sampling: {len(y_fold_balanced)} samples")
                    except Exception as e:
                        logger.warning(f"    Advanced sampling failed: {e}, using original data")
                        X_fold_balanced, y_fold_balanced = X_fold_train, y_fold_train
                else:
                    # Fallback to standard balancing
                    balance_method = model_info.get('balance_method', 'smote')
                    X_fold_balanced, y_fold_balanced = prepare_balanced_data(
                        X_fold_train, y_fold_train, balance_method
                    )
                
                # Create model with Optuna params
                model_class = model_info['class']
                if optuna_params:
                    combined_params = {**model_info.get('params', {}), **optuna_params}
                    fold_model = model_class(**combined_params)
                else:
                    fold_model = model_class()
                
                # Train model
                fold_model.train(X_fold_balanced, y_fold_balanced)
                
                # Apply probability calibration if enabled
                if enable_calibration:
                    try:
                        underlying_model = getattr(fold_model, 'model', fold_model)
                        calibrated_model = calibrate_model(underlying_model, X_fold_balanced, y_fold_balanced)
                        
                        # Replace the model with calibrated version
                        if hasattr(fold_model, 'model'):
                            fold_model.model = calibrated_model
                        else:
                            fold_model = calibrated_model
                            
                        logger.info(f"    ✓ Applied probability calibration")
                    except Exception as e:
                        logger.warning(f"    Calibration failed: {e}")
                
                # Predict on validation set
                if hasattr(fold_model, 'predict_proba'):
                    val_proba = fold_model.predict_proba(X_fold_val)
                    val_proba_1 = val_proba[:, -1] if val_proba.ndim > 1 else val_proba
                else:
                    # For calibrated models
                    val_proba_1 = fold_model.predict_proba(X_fold_val)[:, 1]
                
                # Use default threshold during cross-validation
                try:
                    fold_threshold = 0.5  # Default threshold for classifiers
                    model_thresholds[model_name].append(fold_threshold)
                except Exception:
                    model_thresholds[model_name].append(0.5)
                
                # Store OOF predictions
                train_meta_features[val_idx, model_idx] = val_proba_1
                
                # Predict on test set
                if hasattr(fold_model, 'predict_proba'):
                    test_proba = fold_model.predict_proba(X_test)
                    test_proba_1 = test_proba[:, -1] if test_proba.ndim > 1 else test_proba
                else:
                    test_proba_1 = fold_model.predict_proba(X_test)[:, 1]
                
                fold_predictions.append(test_proba_1)
                fold_models.append(fold_model)
            
            # Average test predictions across folds
            test_meta_features[:, model_idx] = np.mean(fold_predictions, axis=0)
            
            # Store model results
            avg_threshold = np.mean(model_thresholds[model_name]) if model_thresholds[model_name] else 0.5
            
            trained_models[model_name] = {
                'models': fold_models,
                'config': model_info,
                'test_predictions': test_meta_features[:, model_idx],
                'optimal_threshold': avg_threshold,
                'fold_thresholds': model_thresholds[model_name],
                'optuna_params': optuna_params,
                'selected_features': selected_features,
                'calibration_enabled': enable_calibration,
                'sampling_method': advanced_sampling
            }
            
            duration = time.time() - start_time
            logger.info(f"  ✓ Enhanced {model_name} completed in {duration:.1f}s")
            
        except Exception as e:
            logger.error(f"  ✗ Enhanced {model_name} failed: {e}")
            train_meta_features[:, model_idx] = 0.0
            test_meta_features[:, model_idx] = 0.0
            continue
    
    logger.info("🎯 Enhanced out-of-fold stacking completed!")
    
    # For now, delegate to the regular out_of_fold_stacking which has proper OOF bookkeeping
    # TODO: Integrate full enhancement features with OOF bookkeeping
    logger.info("🔄 Delegating to regular OOF stacking for proper OOF bookkeeping...")
    return out_of_fold_stacking(X_train, y_train, X_test, config)

def get_optuna_param_space(model_name: str) -> Dict[str, Any]:
    """
    Get Optuna parameter space for different models.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary of parameter ranges for Optuna
    """
    param_spaces = {
        'random_forest': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 8, 12, None],
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10),
            'max_features': ['sqrt', 'log2', None]
        },
        'xgboost': {
            'n_estimators': (50, 300),
            'max_depth': (3, 12),
            'learning_rate': (0.01, 0.3),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'reg_alpha': (0.0, 1.0),
            'reg_lambda': (0.0, 1.0)
        },
        'lightgbm': {
            'num_leaves': (10, 100),
            'learning_rate': (0.01, 0.3),
            'feature_fraction': (0.6, 1.0),
            'bagging_fraction': (0.6, 1.0),
            'min_child_samples': (5, 100),
            'reg_alpha': (0.0, 1.0),
            'reg_lambda': (0.0, 1.0)
        },
        'catboost': {
            'depth': (4, 12),
            'learning_rate': (0.01, 0.2),
            'l2_leaf_reg': (1, 10),
            'iterations': (50, 300),
            'border_count': [32, 64, 128, 255]
        }
    }
    
    return param_spaces.get(model_name, {})

def train_enhanced_meta_model(meta_X: np.ndarray, meta_y: np.ndarray,
                             raw_features: Optional[pd.DataFrame] = None,
                             config: Optional[TrainingConfig] = None,
                             use_xgb_meta: bool = False) -> Tuple[Any, float]:
    """
    Train enhanced meta-model with raw features and XGBoost option.
    
    Args:
        meta_X: Meta-features from base models
        meta_y: Target labels
        raw_features: Raw features to stack with meta-features
        config: Training configuration
        use_xgb_meta: Use XGBoost instead of LogisticRegression
        
    Returns:
        Tuple of (trained meta-model, optimal_threshold)
    """
    if config is None:
        config = get_default_config()
    
    logger.info("🧠 Training enhanced meta-model...")
    
    # Combine meta-features with raw features if provided
    if raw_features is not None:
        logger.info(f"Stacking {meta_X.shape[1]} meta-features with {raw_features.shape[1]} raw features")
        
        # Ensure index alignment
        if len(raw_features) != len(meta_X):
            min_len = min(len(raw_features), len(meta_X))
            raw_features = raw_features.iloc[:min_len]
            meta_X = meta_X[:min_len]
            meta_y = meta_y[:min_len]
        
        # Combine features
        raw_features_reset = raw_features.reset_index(drop=True)
        meta_features_df = pd.DataFrame(meta_X, columns=[f'meta_{i}' for i in range(meta_X.shape[1])])
        combined_features = pd.concat([meta_features_df, raw_features_reset], axis=1)
        
        logger.info(f"Combined feature matrix shape: {combined_features.shape}")
        final_X = combined_features.values
    else:
        final_X = meta_X
    
    try:
        # Create meta-model
        if use_xgb_meta:
            meta_model = create_xgb_meta_model(
                max_depth=3,
                n_estimators=50,
                learning_rate=0.1
            )
            logger.info("Using XGBoost meta-model")
        else:
            meta_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=config.random_state
            )
            logger.info("Using GradientBoostingClassifier meta-model")
        
        # Train meta-model
        meta_model.fit(final_X, meta_y)
        
        # Get predictions for threshold optimization
        meta_train_proba = meta_model.predict_proba(final_X)[:, 1]
        
        # Find optimal threshold
        from utils.evaluation import find_optimal_threshold
        optimal_threshold, optimal_f1 = find_optimal_threshold(meta_y, meta_train_proba, metric='f1')
        
        logger.info(f"Enhanced meta-model trained successfully")
        logger.info(f"Optimal threshold: {optimal_threshold:.4f} (F1: {optimal_f1:.4f})")
        
        return meta_model, optimal_threshold
        
    except Exception as e:
        logger.error(f"Enhanced meta-model training failed: {e}")
        raise e
