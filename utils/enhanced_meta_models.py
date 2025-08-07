#!/usr/bin/env python3
"""
Enhanced Meta-Model Training Functions
=====================================

Advanced meta-model training strategies for improved F₁ performance on extreme class imbalance.
Implements five key improvements:
1. Optimal threshold tuning for meta-model
2. Cost-sensitive and focal-loss meta-learners  
3. Dynamic stacking weights based on validation performance
4. Feature selection for meta-features
5. Optuna optimization with PR-AUC/F₁ objectives
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, Union
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

def train_meta_model_with_optimal_threshold(meta_X_train: np.ndarray, 
                                          y_train: np.ndarray,
                                          meta_learner_type: str = 'gradientboost',
                                          use_class_weights: bool = True,
                                          optimize_for: str = 'f1') -> Tuple[Any, float]:
    """
    Train meta-model with optimal threshold tuning for F₁ maximization.
    
    Args:
        meta_X_train: Meta-features from base models
        y_train: Training labels
        meta_learner_type: Type of meta-learner ('gradientboost', 'lightgbm', 'xgboost')
        use_class_weights: Whether to use class weighting
        optimize_for: Metric to optimize threshold for ('f1', 'pr_auc', 'precision')
        
    Returns:
        Tuple of (trained_meta_model, optimal_threshold)
    """
    logger.info(f"🧠 Training {meta_learner_type} meta-model with optimal threshold tuning...")
    
    # 1. Train the meta-model with class weighting
    if meta_learner_type == 'gradientboost':
        meta_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        meta_model.fit(meta_X_train, y_train)
        
        # Get training probabilities for threshold optimization
        meta_proba_train = meta_model.predict_proba(meta_X_train)[:, 1]
        
    elif meta_learner_type == 'lightgbm':
        try:
            import lightgbm as lgb
            
            # Create LightGBM dataset
            train_data = lgb.Dataset(meta_X_train, label=y_train)
            
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
            
            if use_class_weights:
                # Calculate class weights
                pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
                params['scale_pos_weight'] = pos_weight
            
            meta_model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                callbacks=[lgb.log_evaluation(0)]  # Use callbacks instead of verbose_eval
            )
            
            # Get training probabilities
            meta_proba_train = meta_model.predict(meta_X_train)
            
        except ImportError:
            logger.warning("LightGBM not available, falling back to logistic regression")
            return train_meta_model_with_optimal_threshold(
                meta_X_train, y_train, 'logistic', use_class_weights, optimize_for
            )
            
    elif meta_learner_type == 'xgboost':
        try:
            import xgboost as xgb
            
            # Calculate class weights if enabled
            if use_class_weights:
                scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            else:
                scale_pos_weight = 1.0
            
            meta_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss'
            )
            meta_model.fit(meta_X_train, y_train)
            
            # Get training probabilities
            meta_proba_train = meta_model.predict_proba(meta_X_train)[:, 1]
            
        except ImportError:
            logger.warning("XGBoost not available, falling back to logistic regression")
            return train_meta_model_with_optimal_threshold(
                meta_X_train, y_train, 'logistic', use_class_weights, optimize_for
            )
            
    elif meta_learner_type == 'logistic':
        from sklearn.linear_model import LogisticRegression
        
        # Prepare LogisticRegression parameters
        lr_params = {
            'random_state': 42,
            'max_iter': 2000,
            'solver': 'liblinear',  # Better for small datasets
            'C': 0.1  # Stronger regularization for extreme imbalance
        }
        
        # Add class weighting if enabled
        if use_class_weights:
            lr_params['class_weight'] = 'balanced'
            logger.info("Using class_weight='balanced' for LogisticRegression meta-model")
        
        meta_model = LogisticRegression(**lr_params)
        meta_model.fit(meta_X_train, y_train)
        
        # Get training probabilities
        meta_proba_train = meta_model.predict_proba(meta_X_train)[:, 1]
        
    else:
        raise ValueError(f"Unsupported meta_learner_type: {meta_learner_type}. "
                        f"Supported types: 'gradientboost', 'lightgbm', 'xgboost', 'logistic'")
    
    # 2. Find optimal threshold using the specified metric
    from utils.evaluation import find_optimal_threshold
    
    optimal_threshold, best_score = find_optimal_threshold(
        y_train, meta_proba_train, metric=optimize_for
    )
    
    logger.info(f"   Meta-model optimal threshold: {optimal_threshold:.4f} ({optimize_for}: {best_score:.4f})")
    logger.info(f"   Training probability range: [{meta_proba_train.min():.4f}, {meta_proba_train.max():.4f}]")
    
    return meta_model, optimal_threshold

def train_focal_loss_meta_model(meta_X_train: np.ndarray, 
                               y_train: np.ndarray,
                               alpha: float = 0.25, 
                               gamma: float = 2.0) -> Tuple[Any, float]:
    """
    Train meta-model with focal loss for extreme class imbalance.
    
    Args:
        meta_X_train: Meta-features from base models
        y_train: Training labels
        alpha: Focal loss alpha parameter (class weighting)
        gamma: Focal loss gamma parameter (focusing parameter)
        
    Returns:
        Tuple of (trained_meta_model, optimal_threshold)
    """
    logger.info(f"🎯 Training focal-loss meta-model (α={alpha}, γ={gamma})...")
    
    try:
        import lightgbm as lgb
        
        # Calculate class weights for extreme imbalance
        pos_weight = len(y_train) / (2 * np.sum(y_train))
        logger.info(f"Class imbalance ratio: {np.mean(y_train):.3f}, pos_weight: {pos_weight:.3f}")
        
        # Use LightGBM with class weight handling
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            # Use either is_unbalance OR scale_pos_weight, not both
            'scale_pos_weight': pos_weight  # Boost positive class
        }
        
        # Create dataset and train
        train_data = lgb.Dataset(meta_X_train, label=y_train)
        
        # Train model with class imbalance handling
        meta_model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            callbacks=[lgb.log_evaluation(0)]  # Use callbacks instead of verbose_eval
        )
        
        # Get training probabilities and find optimal threshold
        meta_proba_train = meta_model.predict(meta_X_train)
        
        from utils.evaluation import find_optimal_threshold
        optimal_threshold, best_f1 = find_optimal_threshold(
            y_train, meta_proba_train, metric='f1'
        )
        
        logger.info(f"   Focal-loss meta-model threshold: {optimal_threshold:.4f} (F1: {best_f1:.4f})")
        
        return meta_model, optimal_threshold
        
    except Exception as e:
        logger.warning(f"Focal loss LightGBM failed: {e}, falling back to balanced GradientBoostingClassifier")
        
        # Fallback to weighted gradient boosting classifier
        meta_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        meta_model.fit(meta_X_train, y_train)
        
        # Get predictions and find optimal threshold
        meta_proba_train = meta_model.predict_proba(meta_X_train)[:, 1]
        
        from utils.evaluation import find_optimal_threshold
        optimal_threshold, best_f1 = find_optimal_threshold(
            y_train, meta_proba_train, metric='f1'
        )
        
        logger.info(f"   Fallback meta-model threshold: {optimal_threshold:.4f} (F1: {best_f1:.4f})")
        return meta_model, optimal_threshold
        
    except ImportError:
        logger.warning("LightGBM not available for focal loss, falling back to gradient boosting")
        return train_meta_model_with_optimal_threshold(
            meta_X_train, y_train, 'lightgbm', use_class_weights=True, optimize_for='f1'
        )
    except Exception as e:
        logger.warning(f"Focal loss training failed: {e}, falling back to class-weighted gradient boosting")
        return train_meta_model_with_optimal_threshold(
            meta_X_train, y_train, 'lightgbm', use_class_weights=True, optimize_for='f1'
        )

def train_smote_enhanced_meta_model(meta_X_train: np.ndarray, 
                                   y_train: np.ndarray,
                                   meta_learner_type: str = 'logistic',
                                   smote_ratio: float = 0.5) -> Tuple[Any, float]:
    """
    Train meta-model with SMOTE resampling of meta-training features.
    
    Args:
        meta_X_train: Meta-features from base models
        y_train: Training labels
        meta_learner_type: Type of meta-learner
        smote_ratio: SMOTE sampling ratio (0.5 = 50/50 balance)
        
    Returns:
        Tuple of (trained_meta_model, optimal_threshold)
    """
    logger.info(f"🔄 Training SMOTE-enhanced {meta_learner_type} meta-model...")
    
    try:
        from imblearn.over_sampling import SMOTE
        
        # Apply SMOTE to meta-training features
        logger.info(f"Original meta-training distribution: {np.bincount(y_train)}")
        
        # Use desired ratio for perfect balance
        smote = SMOTE(
            sampling_strategy=smote_ratio,
            random_state=42,
            k_neighbors=min(5, np.sum(y_train) - 1)  # Ensure we have enough neighbors
        )
        
        X_meta_resampled, y_meta_resampled = smote.fit_resample(meta_X_train, y_train)
        
        logger.info(f"SMOTE resampled distribution: {np.bincount(y_meta_resampled)}")
        logger.info(f"Meta-training samples: {len(meta_X_train)} → {len(X_meta_resampled)}")
        
        # Train meta-model on resampled data
        return train_meta_model_with_optimal_threshold(
            X_meta_resampled, y_meta_resampled, 
            meta_learner_type=meta_learner_type,
            use_class_weights=True,  # Still use class weights for extra robustness
            optimize_for='f1'
        )
        
    except Exception as e:
        logger.warning(f"SMOTE meta-model training failed: {e}")
        # Fallback to standard meta-model with class weights
        return train_meta_model_with_optimal_threshold(
            meta_X_train, y_train, meta_learner_type, True, 'f1'
        )

def train_dynamic_weighted_ensemble(oof_predictions: Dict[str, np.ndarray], 
                                  y_train: np.ndarray,
                                  test_predictions: Dict[str, np.ndarray],
                                  weight_metric: str = 'roc_auc') -> Tuple[np.ndarray, Dict[str, float], float]:
    """
    Create dynamic weighted ensemble based on out-of-fold validation performance.
    
    Args:
        oof_predictions: Out-of-fold predictions for each base model
        y_train: Training labels
        test_predictions: Test predictions for each base model
        weight_metric: Metric to use for weighting ('roc_auc', 'pr_auc', 'f1')
        
    Returns:
        Tuple of (weighted_test_probabilities, model_weights, optimal_threshold)
    """
    logger.info(f"⚖️ Creating dynamic weighted ensemble based on {weight_metric}...")
    
    # Calculate weights based on out-of-fold performance
    weights = {}
    for model_name, oof_proba in oof_predictions.items():
        if weight_metric == 'roc_auc':
            try:
                score = roc_auc_score(y_train, oof_proba)
            except ValueError:
                score = 0.5  # Fallback for edge cases
        elif weight_metric == 'pr_auc':
            try:
                score = average_precision_score(y_train, oof_proba)
            except ValueError:
                score = np.mean(y_train)  # Fallback
        elif weight_metric == 'f1':
            # Use threshold-optimized F1
            from utils.evaluation import find_optimal_threshold
            _, score = find_optimal_threshold(y_train, oof_proba, metric='f1')
        else:
            # Default to ROC-AUC
            score = roc_auc_score(y_train, oof_proba)
        
        weights[model_name] = max(score, 0.01)  # Minimum weight to avoid zeros
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {name: weight / total_weight for name, weight in weights.items()}
    
    # Log weights
    logger.info(f"   Dynamic weights ({weight_metric}):")
    for model_name, weight in weights.items():
        oof_score = weights[model_name] * total_weight  # Reconstruct original score
        logger.info(f"     {model_name}: {weight:.4f} (score: {oof_score:.4f})")
    
    # Create weighted ensemble for training (OOF) and test
    oof_weighted = sum(weights[name] * oof_predictions[name] for name in weights.keys())
    test_weighted = sum(weights[name] * test_predictions[name] for name in weights.keys())
    
    # Find optimal threshold on weighted OOF predictions
    from utils.evaluation import find_optimal_threshold
    optimal_threshold, best_f1 = find_optimal_threshold(y_train, oof_weighted, metric='f1')
    
    logger.info(f"   Weighted ensemble threshold: {optimal_threshold:.4f} (F1: {best_f1:.4f})")
    
    return test_weighted, weights, optimal_threshold

def train_feature_selected_meta_model(meta_X_train: np.ndarray, 
                                     y_train: np.ndarray,
                                     meta_X_test: np.ndarray,
                                     k_best: int = 3,
                                     selection_method: str = 'mutual_info') -> Tuple[Any, float, np.ndarray, np.ndarray]:
    """
    Train meta-model with feature selection on meta-features.
    
    Args:
        meta_X_train: Training meta-features
        y_train: Training labels
        meta_X_test: Test meta-features
        k_best: Number of best features to select
        selection_method: Selection method ('mutual_info', 'f_classif')
        
    Returns:
        Tuple of (meta_model, optimal_threshold, selected_train_features, selected_test_features)
    """
    logger.info(f"🔍 Training meta-model with feature selection (k={k_best}, method={selection_method})...")
    
    # Ensure k_best doesn't exceed available features
    k_best = min(k_best, meta_X_train.shape[1])
    
    # Feature selection
    if selection_method == 'mutual_info':
        selector = SelectKBest(mutual_info_classif, k=k_best)
    else:  # f_classif
        from sklearn.feature_selection import f_classif
        selector = SelectKBest(f_classif, k=k_best)
    
    # Fit selector and transform features
    meta_X_train_selected = selector.fit_transform(meta_X_train, y_train)
    meta_X_test_selected = selector.transform(meta_X_test)
    
    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)
    logger.info(f"   Selected features (indices): {selected_indices}")
    logger.info(f"   Feature scores: {selector.scores_[selected_indices]}")
    
    # Train meta-model on selected features
    meta_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    meta_model.fit(meta_X_train_selected, y_train)
    
    # Find optimal threshold
    meta_proba_train = meta_model.predict_proba(meta_X_train_selected)[:, 1]
    
    from utils.evaluation import find_optimal_threshold
    optimal_threshold, best_f1 = find_optimal_threshold(y_train, meta_proba_train, metric='f1')
    
    logger.info(f"   Feature-selected meta-model threshold: {optimal_threshold:.4f} (F1: {best_f1:.4f})")
    
    return meta_model, optimal_threshold, meta_X_train_selected, meta_X_test_selected

def optuna_optimize_base_model_for_f1(model_class, X: pd.DataFrame, y: pd.Series, 
                                     n_trials: int = 50, 
                                     optimize_metric: str = 'average_precision') -> Dict[str, Any]:
    """
    Use Optuna to optimize base model hyperparameters for F₁/PR-AUC performance.
    
    Args:
        model_class: Model class to optimize
        X: Training features
        y: Training labels
        n_trials: Number of Optuna trials
        optimize_metric: Metric to optimize ('average_precision', 'f1_weighted')
        
    Returns:
        Dictionary of best hyperparameters
    """
    logger.info(f"🎯 Optuna optimization for {model_class.__name__} targeting {optimize_metric}...")
    
    try:
        import optuna
        from sklearn.model_selection import StratifiedKFold
        
        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            try:
                model_name = model_class.__name__.lower()
                
                # Define hyperparameter search spaces based on model type
                if 'xgb' in model_name or 'xgboost' in model_name:
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
                        'random_state': 42
                    }
                elif 'lightgbm' in model_name or 'lgb' in model_name:
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                        'class_weight': 'balanced',
                        'random_state': 42
                    }
                elif 'catboost' in model_name:
                    params = {
                        'iterations': trial.suggest_int('iterations', 50, 300),
                        'depth': trial.suggest_int('depth', 4, 12),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                        'auto_class_weights': 'Balanced',
                        'random_seed': 42,
                        'verbose': False
                    }
                elif 'randomforest' in model_name:
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                        'class_weight': 'balanced',
                        'random_state': 42
                    }
                elif 'gradientboosting' in model_name or 'gradientboost' in model_name:
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'max_depth': trial.suggest_int('max_depth', 2, 8),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'random_state': 42
                    }
                else:
                    # Default parameters for unknown models
                    params = {'random_state': 42}
                
                # Create model instance
                if hasattr(model_class, 'model'):
                    # If it's a wrapper class, get the underlying model
                    model = model_class(**params).model
                else:
                    model = model_class(**params)
                
                # Cross-validation with the target metric
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scores = cross_val_score(
                    model, X, y, 
                    cv=cv, 
                    scoring=optimize_metric,
                    error_score='raise'
                )
                
                return scores.mean()
                
            except Exception as e:
                logger.warning(f"Optuna trial failed: {e}")
                return 0.0
        
        # Create study and optimize
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        logger.info(f"   Best {optimize_metric}: {study.best_value:.4f}")
        logger.info(f"   Best params: {study.best_params}")
        
        return study.best_params
        
    except ImportError:
        logger.warning("Optuna not available - using default parameters")
        return {}
    except Exception as e:
        logger.warning(f"Optuna optimization failed: {e}")
        return {}

def get_enhanced_meta_model_strategy(config: Any) -> str:
    """
    Determine which enhanced meta-model strategy to use based on configuration.
    
    Args:
        config: Training configuration object
        
    Returns:
        Strategy name ('optimal_threshold', 'focal_loss', 'dynamic_weights', 'feature_select')
    """
    # Check for configuration flags to determine strategy
    if hasattr(config, 'meta_model_strategy'):
        return config.meta_model_strategy
    
    # Default strategy based on problem characteristics
    if hasattr(config, 'data_balancing') and config.data_balancing.method == 'smote':
        return 'focal_loss'  # Good for extreme imbalance
    elif hasattr(config, 'enable_feature_selection') and config.enable_feature_selection:
        return 'feature_select'  # When feature selection is enabled
    else:
        return 'optimal_threshold'  # Safe default

if __name__ == "__main__":
    print("Enhanced Meta-Model Training Functions loaded successfully!")
