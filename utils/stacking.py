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
from typing import Dict, Any, Tuple, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# Model imports
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.catboost_model import CatBoostModel
from models.random_forest_model import RandomForestModel
from models.svc_model import SVMClassifier

from utils.data_splitting import prepare_cv_data
from utils.sampling import prepare_balanced_data
from config.training_config import TrainingConfig, get_default_config

logger = logging.getLogger(__name__)

# Model registry for easy access
MODEL_REGISTRY = {
    'xgboost': XGBoostModel,
    'lightgbm': LightGBMModel,
    'catboost': CatBoostModel,
    'random_forest': RandomForestModel,
    'svm': SVMClassifier
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
            
            # Get predictions
            train_proba = model.predict_proba(X_balanced)
            train_proba_1 = train_proba[:, -1] if train_proba.ndim > 1 else train_proba
            
            test_proba = model.predict_proba(X_test)
            test_proba_1 = test_proba[:, -1] if test_proba.ndim > 1 else test_proba
            
            # Store results
            trained_models[model_name] = {
                'model': model,
                'config': model_info,
                'train_predictions': train_proba_1,
                'train_labels': y_balanced,
                'test_predictions': test_proba_1
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
                        config: Optional[TrainingConfig] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Perform robust out-of-fold stacking to prevent overfitting.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        config: Training configuration
        
    Returns:
        Tuple of (train_meta_features, test_meta_features, trained_models)
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
        return simple_train_test_stacking(X_train, y_train, X_test, config)
    
    # Check if we have enough samples for OOF
    class_counts = y_cv.value_counts()
    if class_counts.min() < n_folds:
        logger.warning(f"Insufficient minority samples for {n_folds}-fold CV (min: {class_counts.min()})")
        logger.info("Falling back to simple train/test stacking")
        return simple_train_test_stacking(X_train, y_train, X_test, config)
    
    # Create model configurations
    model_configs = create_model_configs(config)
    
    # Initialize meta-feature arrays
    n_models = len(model_configs)
    train_meta_features = np.zeros((len(X_cv), n_models))
    test_meta_features = np.zeros((len(X_test), n_models))
    
    trained_models = {}
    
    # Train each model with OOF
    for model_idx, (model_name, model_info) in enumerate(model_configs.items()):
        start_time = time.time()
        logger.info(f"Training {model_name} with {n_folds}-fold OOF...")
        
        fold_predictions = []
        fold_models = []
        
        try:
            # Cross-validation loop
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv)):
                logger.info(f"  Fold {fold + 1}/{n_folds}")
                
                # Split data for this fold
                X_fold_train = X_cv.iloc[train_idx]
                y_fold_train = y_cv.iloc[train_idx]
                X_fold_val = X_cv.iloc[val_idx]
                
                # Balance training data for this fold
                balance_method = model_info.get('balance_method', 'smote')
                X_fold_balanced, y_fold_balanced = prepare_balanced_data(
                    X_fold_train, y_fold_train, balance_method
                )
                
                # Train model on this fold
                model_class = model_info['class']
                fold_model = model_class()
                fold_model.train(X_fold_balanced, y_fold_balanced)
                
                # Predict on validation set (out-of-fold)
                val_proba = fold_model.predict_proba(X_fold_val)
                val_proba_1 = val_proba[:, -1] if val_proba.ndim > 1 else val_proba
                
                # Store OOF predictions
                train_meta_features[val_idx, model_idx] = val_proba_1
                
                # Predict on test set
                test_proba = fold_model.predict_proba(X_test)
                test_proba_1 = test_proba[:, -1] if test_proba.ndim > 1 else test_proba
                fold_predictions.append(test_proba_1)
                
                fold_models.append(fold_model)
            
            # Average test predictions across folds
            test_meta_features[:, model_idx] = np.mean(fold_predictions, axis=0)
            
            # Store fold models
            trained_models[model_name] = {
                'models': fold_models,
                'config': model_info,
                'test_predictions': test_meta_features[:, model_idx]
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
    return train_meta_features, test_meta_features, trained_models

def train_meta_model(meta_X: np.ndarray, meta_y: np.ndarray,
                    config: Optional[TrainingConfig] = None) -> LogisticRegression:
    """
    Train meta-model on out-of-fold predictions.
    
    Args:
        meta_X: Meta-features (out-of-fold predictions)
        meta_y: Meta-labels (training labels)
        config: Training configuration
        
    Returns:
        Trained meta-model
    """
    if config is None:
        config = get_default_config()
    
    logger.info("Training meta-model...")
    
    try:
        meta_model = LogisticRegression(
            random_state=config.random_state,
            C=config.meta_model.regularization_c,
            class_weight=config.meta_model.class_weight,
            solver=config.meta_model.solver,
            max_iter=config.meta_model.max_iter
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
        
        # Log feature importance (coefficients)
        if hasattr(meta_model, 'coef_') and meta_model.coef_ is not None:
            feature_names = [f'model_{i}' for i in range(len(meta_model.coef_[0]))]
            coef_dict = {name: float(coef) for name, coef in zip(feature_names, meta_model.coef_[0])}
            logger.info(f"Meta-model feature weights: {coef_dict}")
        
        logger.info("✓ Meta-model training completed")
        return meta_model
        
    except Exception as e:
        logger.error(f"Meta-model training failed: {e}")
        raise e

def predict_with_meta_model(meta_model: LogisticRegression, 
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
        meta_proba = meta_model.predict_proba(test_meta_features)
        return meta_proba[:, -1] if meta_proba.ndim > 1 else meta_proba
    except Exception as e:
        logger.error(f"Meta-model prediction failed: {e}")
        # Return uniform probabilities as fallback
        return np.full(len(test_meta_features), 0.5)

def create_ensemble_predictions(trained_models: Dict[str, Any],
                               X_test: pd.DataFrame,
                               meta_model: Optional[LogisticRegression] = None,
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
                    test_proba = fold_model.predict_proba(X_test)
                    test_proba_1 = test_proba[:, -1] if test_proba.ndim > 1 else test_proba
                    fold_predictions.append(test_proba_1)
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
                test_proba = model_data['model'].predict_proba(X_test)
                test_proba_1 = test_proba[:, -1] if test_proba.ndim > 1 else test_proba
                base_predictions[model_name] = test_proba_1
            except Exception as e:
                logger.warning(f"Prediction failed for {model_name}: {e}")
    
    # Create meta-model predictions if available
    meta_predictions = None
    if meta_model is not None and base_predictions:
        try:
            # Stack base predictions as features
            meta_features = np.column_stack(list(base_predictions.values()))
            meta_predictions = predict_with_meta_model(meta_model, meta_features)
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
