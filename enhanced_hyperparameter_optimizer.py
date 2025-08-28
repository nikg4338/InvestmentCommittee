#!/usr/bin/env python3
"""
Enhanced Hyperparameter Optimization System
==========================================

This module provides sophisticated hyperparameter optimization with broader search spaces,
adaptive trial allocation, and model-specific optimization strategies.
"""

import optuna
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
import time
from datetime import datetime
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import average_precision_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)

class EnhancedHyperparameterOptimizer:
    """Advanced hyperparameter optimization with adaptive strategies."""
    
    def __init__(self, use_time_series_cv: bool = True, cv_folds: int = 5):
        """Initialize enhanced hyperparameter optimizer."""
        self.use_time_series_cv = use_time_series_cv
        self.cv_folds = cv_folds
        self.optimization_history = {}
        
        # Model-specific trial allocations
        self.trial_allocations = {
            'simple_models': 15,    # Random Forest, SVM
            'complex_models': 30,   # XGBoost, Neural Network
            'ensemble_models': 50,  # Meta-learners, Stacking
            'production_models': 100 # Final production optimization
        }
        
        # Custom scorer for imbalanced data - fixed for sklearn 1.6.1
        try:
            # First try the response_method parameter (newer sklearn versions) 
            self.pr_auc_scorer = make_scorer(average_precision_score, response_method='predict_proba')
            logger.info("Using PR-AUC scorer with response_method='predict_proba'")
        except TypeError:
            try:
                # Fallback to string scorer which always works
                self.pr_auc_scorer = 'average_precision'
                logger.info("Using string scorer 'average_precision'")
            except Exception:
                # Final fallback - should never happen
                logger.warning("Using basic accuracy scorer as final fallback")
                self.pr_auc_scorer = 'accuracy'
    
    def get_cv_splitter(self, X: pd.DataFrame, y: np.ndarray):
        """Get appropriate cross-validation splitter."""
        if self.use_time_series_cv:
            return TimeSeriesSplit(n_splits=self.cv_folds)
        else:
            return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
    
    def get_trial_count(self, model_type: str, complexity: str = 'balanced') -> int:
        """Get appropriate number of trials based on model complexity."""
        base_trials = {
            'quick': 10,
            'balanced': 20,
            'intensive': 50,
            'production': 100
        }
        
        if model_type in ['xgboost', 'neural_network', 'ensemble']:
            multiplier = 1.5
        elif model_type in ['catboost', 'lightgbm']:
            multiplier = 1.2
        else:
            multiplier = 1.0
        
        return int(base_trials[complexity] * multiplier)
    
    def optimize_xgboost(self, X: pd.DataFrame, y: np.ndarray, 
                        n_trials: int = 30, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Enhanced XGBoost hyperparameter optimization."""
        logger.info(f"🔧 Optimizing XGBoost hyperparameters ({n_trials} trials)...")
        
        def objective(trial):
            # Expanded parameter space for XGBoost
            params = {
                # Tree structure
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
                'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                
                # Learning parameters
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 800),
                
                # Regularization - fix log distribution issue
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 50.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 50.0, log=True),
                
                # Subsampling
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0),
                'colsample_bynode': trial.suggest_float('colsample_bynode', 0.3, 1.0),
                
                # Class imbalance handling
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 20.0),
                
                # Advanced parameters
                'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
                'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
                
                # Performance
                'tree_method': 'hist',
                'random_state': 42,
                'verbosity': 0
            }
            
            # Conditional parameters
            if params['grow_policy'] == 'lossguide':
                params['max_leaves'] = trial.suggest_int('max_leaves', 10, 100)
            
            try:
                from xgboost import XGBClassifier
                model = XGBClassifier(**params)
                
                cv_splitter = self.get_cv_splitter(X, y)
                scores = cross_val_score(model, X, y, cv=cv_splitter, 
                                       scoring=self.pr_auc_scorer, n_jobs=-1)
                return np.mean(scores)
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0
        
        try:
            study = optuna.create_study(direction='maximize', 
                                      sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=n_trials, timeout=timeout, 
                         show_progress_bar=False)
            
            result = {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'n_trials': len(study.trials),
                'optimization_time': sum(t.duration.total_seconds() for t in study.trials if t.duration),
                'study': study
            }
            
            logger.info(f"✅ XGBoost optimization completed - Best PR-AUC: {result['best_score']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"XGBoost optimization failed: {e}")
            return {'best_params': {}, 'best_score': 0.0, 'error': str(e)}
    
    def optimize_neural_network(self, X: pd.DataFrame, y: np.ndarray, 
                               n_trials: int = 30, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Enhanced Neural Network hyperparameter optimization."""
        logger.info(f"🧠 Optimizing Neural Network hyperparameters ({n_trials} trials)...")
        
        def objective(trial):
            # Advanced NN architecture search
            params = {
                # Architecture
                'hidden_layer_sizes': self._suggest_nn_architecture(trial),
                
                # Learning parameters
                'learning_rate_init': trial.suggest_float('learning_rate_init', 0.0001, 0.1, log=True),
                'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True),  # L2 regularization
                
                # Optimization
                'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs', 'sgd']),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256]),
                
                # Regularization
                'early_stopping': True,
                'validation_fraction': 0.15,
                'n_iter_no_change': 20,
                
                # Advanced parameters
                'beta_1': trial.suggest_float('beta_1', 0.8, 0.99),
                'beta_2': trial.suggest_float('beta_2', 0.9, 0.9999),
                'epsilon': trial.suggest_float('epsilon', 1e-9, 1e-6, log=True),
                
                # Training
                'max_iter': 1000,
                'random_state': 42
            }
            
            # Solver-specific parameters
            if params['solver'] == 'sgd':
                params['momentum'] = trial.suggest_float('momentum', 0.5, 0.99)
                params['nesterovs_momentum'] = trial.suggest_categorical('nesterovs_momentum', [True, False])
            
            # Adaptive learning rate
            if params['solver'] in ['sgd', 'adam']:
                params['learning_rate'] = trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive'])
                if params['learning_rate'] == 'invscaling':
                    params['power_t'] = trial.suggest_float('power_t', 0.1, 0.9)
            
            try:
                from sklearn.neural_network import MLPClassifier
                model = MLPClassifier(**params)
                
                cv_splitter = self.get_cv_splitter(X, y)
                scores = cross_val_score(model, X, y, cv=cv_splitter, 
                                       scoring=self.pr_auc_scorer, n_jobs=1)  # NN doesn't parallelize well
                return np.mean(scores)
                
            except Exception as e:
                logger.warning(f"NN trial failed: {e}")
                return 0.0
        
        try:
            study = optuna.create_study(direction='maximize',
                                      sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=n_trials, timeout=timeout,
                         show_progress_bar=False)
            
            result = {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'n_trials': len(study.trials),
                'optimization_time': sum(t.duration.total_seconds() for t in study.trials if t.duration),
                'study': study
            }
            
            logger.info(f"✅ Neural Network optimization completed - Best PR-AUC: {result['best_score']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Neural Network optimization failed: {e}")
            return {'best_params': {}, 'best_score': 0.0, 'error': str(e)}
    
    def _suggest_nn_architecture(self, trial) -> tuple:
        """Suggest neural network architecture."""
        # Dynamic architecture based on data size and complexity
        n_layers = trial.suggest_int('n_layers', 2, 5)
        
        architecture = []
        for i in range(n_layers):
            if i == 0:  # First layer
                size = trial.suggest_int(f'layer_{i}_size', 64, 512)
            else:  # Subsequent layers - typically smaller
                prev_size = architecture[-1]
                size = trial.suggest_int(f'layer_{i}_size', 16, min(prev_size, 256))
            
            architecture.append(size)
        
        return tuple(architecture)
    
    def optimize_lightgbm(self, X: pd.DataFrame, y: np.ndarray,
                         n_trials: int = 25, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Enhanced LightGBM hyperparameter optimization."""
        logger.info(f"💡 Optimizing LightGBM hyperparameters ({n_trials} trials)...")
        
        def objective(trial):
            params = {
                # Core parameters
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                
                # Regularization - fix log distribution issue
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 100.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                
                # Feature sampling
                'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.3, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                
                # Advanced parameters
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
                'path_smooth': trial.suggest_float('path_smooth', 0.0, 100.0),
                
                # Class imbalance
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 20.0),
                
                # Performance
                'verbosity': -1,
                'random_state': 42,
                'force_col_wise': True
            }
            
            try:
                import lightgbm as lgb
                model = lgb.LGBMClassifier(**params)
                
                cv_splitter = self.get_cv_splitter(X, y)
                scores = cross_val_score(model, X, y, cv=cv_splitter,
                                       scoring=self.pr_auc_scorer, n_jobs=-1)
                return np.mean(scores)
                
            except Exception as e:
                logger.warning(f"LightGBM trial failed: {e}")
                return 0.0
        
        try:
            study = optuna.create_study(direction='maximize',
                                      sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=n_trials, timeout=timeout,
                         show_progress_bar=False)
            
            result = {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'n_trials': len(study.trials),
                'optimization_time': sum(t.duration.total_seconds() for t in study.trials if t.duration),
                'study': study
            }
            
            logger.info(f"✅ LightGBM optimization completed - Best PR-AUC: {result['best_score']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"LightGBM optimization failed: {e}")
            return {'best_params': {}, 'best_score': 0.0, 'error': str(e)}
    
    def optimize_catboost(self, X: pd.DataFrame, y: np.ndarray,
                         n_trials: int = 25, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Enhanced CatBoost hyperparameter optimization."""
        logger.info(f"🐱 Optimizing CatBoost hyperparameters ({n_trials} trials)...")
        
        def objective(trial):
            params = {
                # Core parameters
                'iterations': trial.suggest_int('iterations', 100, 1500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                
                # Regularization
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 30.0),
                'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
                
                # Tree structure
                'border_count': trial.suggest_int('border_count', 32, 255),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
                
                # Feature sampling
                'rsm': trial.suggest_float('rsm', 0.1, 1.0),  # Random subspace method
                
                # Class imbalance
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 20.0),
                
                # Performance
                'verbose': False,
                'random_state': 42,
                'thread_count': -1
            }
            
            # Advanced boosting parameters
            boosting_type = trial.suggest_categorical('boosting_type', ['Ordered', 'Plain'])
            params['boosting_type'] = boosting_type
            
            # Only add compatible parameters based on boosting type
            if boosting_type == 'Plain':
                bootstrap_type = trial.suggest_categorical('bootstrap_type', 
                                                         ['Bayesian', 'Bernoulli', 'MVS'])
                params['bootstrap_type'] = bootstrap_type
                
                if bootstrap_type == 'Bayesian':
                    params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.1, 1.0)
                    # Don't use subsample with Bayesian bootstrap
                elif bootstrap_type == 'Bernoulli':
                    params['subsample'] = trial.suggest_float('subsample_bernoulli', 0.1, 1.0)
                # MVS doesn't need additional parameters
            
            # Tree growing strategy - affects max_leaves compatibility
            grow_policy = trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide'])
            params['grow_policy'] = grow_policy
            
            # Only use max_leaves with Lossguide
            if grow_policy == 'Lossguide':
                params['max_leaves'] = trial.suggest_int('max_leaves', 16, 128)
            else:
                # Use depth for other policies
                params['depth'] = trial.suggest_int('depth', 3, 10)
            
            try:
                import catboost as cb
                model = cb.CatBoostClassifier(**params)
                
                cv_splitter = self.get_cv_splitter(X, y)
                scores = cross_val_score(model, X, y, cv=cv_splitter,
                                       scoring=self.pr_auc_scorer, n_jobs=1)  # CatBoost handles parallelism internally
                return np.mean(scores)
                
            except Exception as e:
                logger.warning(f"CatBoost trial failed: {e}")
                return 0.0
        
        try:
            study = optuna.create_study(direction='maximize',
                                      sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=n_trials, timeout=timeout,
                         show_progress_bar=False)
            
            result = {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'n_trials': len(study.trials),
                'optimization_time': sum(t.duration.total_seconds() for t in study.trials if t.duration),
                'study': study
            }
            
            logger.info(f"✅ CatBoost optimization completed - Best PR-AUC: {result['best_score']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"CatBoost optimization failed: {e}")
            return {'best_params': {}, 'best_score': 0.0, 'error': str(e)}
    
    def optimize_random_forest(self, X: pd.DataFrame, y: np.ndarray,
                              n_trials: int = 20, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Enhanced Random Forest hyperparameter optimization."""
        logger.info(f"🌲 Optimizing Random Forest hyperparameters ({n_trials} trials)...")
        
        def objective(trial):
            params = {
                # Core parameters
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                
                # Advanced parameters
                'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.1),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 100),
                'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.1),
                
                # Sampling
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),
                
                # Class imbalance
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
                
                # Performance
                'n_jobs': -1,
                'random_state': 42
            }
            
            # Conditional parameters
            if not params['bootstrap']:
                del params['max_samples']  # Only valid when bootstrap=True
            
            try:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(**params)
                
                cv_splitter = self.get_cv_splitter(X, y)
                scores = cross_val_score(model, X, y, cv=cv_splitter,
                                       scoring=self.pr_auc_scorer, n_jobs=-1)
                return np.mean(scores)
                
            except Exception as e:
                logger.warning(f"Random Forest trial failed: {e}")
                return 0.0
        
        try:
            study = optuna.create_study(direction='maximize',
                                      sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=n_trials, timeout=timeout,
                         show_progress_bar=False)
            
            result = {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'n_trials': len(study.trials),
                'optimization_time': sum(t.duration.total_seconds() for t in study.trials if t.duration),
                'study': study
            }
            
            logger.info(f"✅ Random Forest optimization completed - Best PR-AUC: {result['best_score']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Random Forest optimization failed: {e}")
            return {'best_params': {}, 'best_score': 0.0, 'error': str(e)}
    
    def optimize_svm(self, X: pd.DataFrame, y: np.ndarray,
                    n_trials: int = 20, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Enhanced SVM hyperparameter optimization."""
        logger.info(f"🎯 Optimizing SVM hyperparameters ({n_trials} trials)...")
        
        def objective(trial):
            kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
            
            params = {
                'kernel': kernel,
                'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                'probability': True,  # Required for predict_proba
                'random_state': 42
            }
            
            # Kernel-specific parameters
            if kernel in ['rbf', 'poly', 'sigmoid']:
                params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
                if params['gamma'] not in ['scale', 'auto']:
                    params['gamma'] = trial.suggest_float('gamma_value', 0.001, 1.0, log=True)
            
            if kernel == 'poly':
                params['degree'] = trial.suggest_int('degree', 2, 5)
                params['coef0'] = trial.suggest_float('coef0', 0.0, 10.0)
            
            if kernel == 'sigmoid':
                params['coef0'] = trial.suggest_float('coef0', 0.0, 10.0)
            
            try:
                from sklearn.svm import SVC
                model = SVC(**params)
                
                cv_splitter = self.get_cv_splitter(X, y)
                scores = cross_val_score(model, X, y, cv=cv_splitter,
                                       scoring=self.pr_auc_scorer, n_jobs=1)  # SVM doesn't parallelize well
                return np.mean(scores)
                
            except Exception as e:
                logger.warning(f"SVM trial failed: {e}")
                return 0.0
        
        try:
            study = optuna.create_study(direction='maximize',
                                      sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=n_trials, timeout=timeout,
                         show_progress_bar=False)
            
            result = {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'n_trials': len(study.trials),
                'optimization_time': sum(t.duration.total_seconds() for t in study.trials if t.duration),
                'study': study
            }
            
            logger.info(f"✅ SVM optimization completed - Best PR-AUC: {result['best_score']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"SVM optimization failed: {e}")
            return {'best_params': {}, 'best_score': 0.0, 'error': str(e)}
    
    def adaptive_optimization(self, model_type: str, X: pd.DataFrame, y: np.ndarray,
                             complexity: str = 'balanced', timeout: Optional[int] = None) -> Dict[str, Any]:
        """Adaptive optimization that adjusts trials based on model complexity and data size."""
        # Calculate appropriate number of trials
        data_size_factor = min(2.0, len(X) / 10000)  # More trials for larger datasets
        feature_complexity_factor = min(1.5, len(X.columns) / 100)  # More trials for more features
        
        base_trials = self.get_trial_count(model_type, complexity)
        adaptive_trials = int(base_trials * data_size_factor * feature_complexity_factor)
        
        logger.info(f"🎛️ Adaptive optimization for {model_type}: {adaptive_trials} trials")
        logger.info(f"   Base: {base_trials}, Data factor: {data_size_factor:.2f}, Feature factor: {feature_complexity_factor:.2f}")
        
        # Route to appropriate optimizer
        optimizers = {
            'xgboost': self.optimize_xgboost,
            'neural_network': self.optimize_neural_network,
            'lightgbm': self.optimize_lightgbm,
            'catboost': self.optimize_catboost,
            'random_forest': self.optimize_random_forest,
            'svm': self.optimize_svm
        }
        
        optimizer = optimizers.get(model_type)
        if optimizer:
            return optimizer(X, y, n_trials=adaptive_trials, timeout=timeout)
        else:
            logger.error(f"Unknown model type: {model_type}")
            return {'best_params': {}, 'best_score': 0.0, 'error': f'Unknown model type: {model_type}'}
    
    def save_optimization_history(self, filepath: str):
        """Save optimization history for analysis."""
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(self.optimization_history, f, indent=2, default=str)
            logger.info(f"✅ Optimization history saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save optimization history: {e}")


# Global enhanced optimizer instance
enhanced_optimizer = EnhancedHyperparameterOptimizer()
