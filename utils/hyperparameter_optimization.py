#!/usr/bin/env python3
"""
Production Hyperparameter Optimization
=====================================

Advanced hyperparameter tuning using Optuna for paper trading deployment.
Optimizes model parameters for real-world portfolio construction and risk management.
"""

import logging
import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
import joblib
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Suppress Optuna logging for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ProductionHyperparameterOptimizer:
    """
    Production-focused hyperparameter optimization with trading-specific objectives.
    """
    
    def __init__(self, portfolio_size: int = 20, 
                 risk_tolerance: str = 'moderate',
                 optimization_timeout: int = 3600):
        """
        Initialize optimizer with trading-specific parameters.
        
        Args:
            portfolio_size: Target number of positions
            risk_tolerance: 'conservative', 'moderate', or 'aggressive' 
            optimization_timeout: Maximum optimization time in seconds
        """
        self.portfolio_size = portfolio_size
        self.risk_tolerance = risk_tolerance
        self.optimization_timeout = optimization_timeout
        
        # Risk tolerance settings
        self.risk_settings = {
            'conservative': {
                'precision_weight': 0.7,
                'min_precision': 0.4,
                'max_portfolio_deviation': 0.3,
                'stability_weight': 0.3
            },
            'moderate': {
                'precision_weight': 0.6,
                'min_precision': 0.3,
                'max_portfolio_deviation': 0.5,
                'stability_weight': 0.2
            },
            'aggressive': {
                'precision_weight': 0.5,
                'min_precision': 0.25,
                'max_portfolio_deviation': 0.8,
                'stability_weight': 0.1
            }
        }
        
        self.current_settings = self.risk_settings[risk_tolerance]
        self.optimization_results = {}
        
    def _trading_objective(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """
        Custom objective function optimized for trading performance.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            
        Returns:
            Trading score (higher is better)
        """
        
        # Find optimal threshold for this prediction set
        thresholds = np.linspace(0.1, 0.9, 50)
        best_score = 0.0
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if np.sum(y_pred) == 0:
                continue
                
            try:
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                
                # Portfolio size constraint
                n_positions = np.sum(y_pred)
                size_deviation = abs(n_positions - self.portfolio_size) / self.portfolio_size
                
                # Skip if minimum precision not met
                if precision < self.current_settings['min_precision']:
                    continue
                    
                # Skip if portfolio size deviation too large
                if size_deviation > self.current_settings['max_portfolio_deviation']:
                    continue
                
                # Calculate trading score
                trading_score = (
                    self.current_settings['precision_weight'] * precision +
                    (1 - self.current_settings['precision_weight']) * recall -
                    0.2 * size_deviation  # Penalty for wrong portfolio size
                )
                
                best_score = max(best_score, trading_score)
                
            except (ValueError, ZeroDivisionError):
                continue
                
        return best_score
    
    def _create_xgboost_objective(self, X: pd.DataFrame, y: pd.Series) -> Callable:
        """Create XGBoost optimization objective."""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42,
                'n_jobs': -1
            }
            
            try:
                from xgboost import XGBClassifier
                model = XGBClassifier(**params)
                
                # Use custom trading scorer
                scorer = make_scorer(self._trading_objective, needs_proba=True)
                
                # Cross-validation with stratified folds
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=1)
                
                return np.mean(scores)
                
            except Exception as e:
                logger.warning(f"XGBoost trial failed: {e}")
                return 0.0
                
        return objective
    
    def _create_lightgbm_objective(self, X: pd.DataFrame, y: pd.Series) -> Callable:
        """Create LightGBM optimization objective."""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': -1
            }
            
            try:
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(**params)
                
                scorer = make_scorer(self._trading_objective, needs_proba=True)
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=1)
                
                return np.mean(scores)
                
            except Exception as e:
                logger.warning(f"LightGBM trial failed: {e}")
                return 0.0
                
        return objective
    
    def _create_catboost_objective(self, X: pd.DataFrame, y: pd.Series) -> Callable:
        """Create CatBoost optimization objective."""
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000, step=50),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_strength': trial.suggest_float('random_strength', 0, 10),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'random_state': 42,
                'verbose': False,
                'thread_count': -1
            }
            
            try:
                from catboost import CatBoostClassifier
                model = CatBoostClassifier(**params)
                
                scorer = make_scorer(self._trading_objective, needs_proba=True)
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=1)
                
                return np.mean(scores)
                
            except Exception as e:
                logger.warning(f"CatBoost trial failed: {e}")
                return 0.0
                
        return objective
    
    def _create_randomforest_objective(self, X: pd.DataFrame, y: pd.Series) -> Callable:
        """Create Random Forest optimization objective."""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=25),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.8, 1.0]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': 42,
                'n_jobs': -1
            }
            
            try:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(**params)
                
                scorer = make_scorer(self._trading_objective, needs_proba=True)
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=1)
                
                return np.mean(scores)
                
            except Exception as e:
                logger.warning(f"Random Forest trial failed: {e}")
                return 0.0
                
        return objective
    
    def optimize_model_hyperparameters(self, model_type: str, 
                                     X_train: pd.DataFrame, 
                                     y_train: pd.Series,
                                     n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific model type.
        
        Args:
            model_type: 'xgboost', 'lightgbm', 'catboost', or 'randomforest'
            X_train: Training features
            y_train: Training labels
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary with best parameters and optimization results
        """
        
        logger.info(f"🎯 Optimizing {model_type} hyperparameters...")
        logger.info(f"   Portfolio size: {self.portfolio_size}")
        logger.info(f"   Risk tolerance: {self.risk_tolerance}")
        logger.info(f"   Trials: {n_trials}")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name=f'{model_type}_trading_optimization',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Select appropriate objective function
        objectives = {
            'xgboost': self._create_xgboost_objective,
            'lightgbm': self._create_lightgbm_objective,
            'catboost': self._create_catboost_objective,
            'randomforest': self._create_randomforest_objective
        }
        
        if model_type not in objectives:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        objective_func = objectives[model_type](X_train, y_train)
        
        # Run optimization
        try:
            study.optimize(
                objective_func, 
                n_trials=n_trials, 
                timeout=self.optimization_timeout,
                show_progress_bar=True
            )
            
            results = {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'n_trials': len(study.trials),
                'optimization_time': sum(t.duration.total_seconds() for t in study.trials if t.duration),
                'study': study  # Keep for detailed analysis
            }
            
            logger.info(f"✅ {model_type} optimization complete!")
            logger.info(f"   Best score: {study.best_value:.4f}")
            logger.info(f"   Trials completed: {len(study.trials)}")
            
            self.optimization_results[model_type] = results
            return results
            
        except Exception as e:
            logger.error(f"❌ Optimization failed for {model_type}: {e}")
            return {'error': str(e)}
    
    def optimize_all_models(self, X_train: pd.DataFrame, 
                           y_train: pd.Series,
                           models_to_optimize: List[str] = None,
                           n_trials: int = 100) -> Dict[str, Dict[str, Any]]:
        """
        Optimize hyperparameters for multiple models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            models_to_optimize: List of model types to optimize
            n_trials: Number of trials per model
            
        Returns:
            Dictionary mapping model types to optimization results
        """
        
        if models_to_optimize is None:
            models_to_optimize = ['xgboost', 'lightgbm', 'catboost', 'randomforest']
        
        logger.info("🚀 Starting comprehensive hyperparameter optimization...")
        
        all_results = {}
        
        for model_type in models_to_optimize:
            logger.info(f"\n📊 Optimizing {model_type}...")
            
            try:
                results = self.optimize_model_hyperparameters(
                    model_type, X_train, y_train, n_trials
                )
                all_results[model_type] = results
                
            except Exception as e:
                logger.error(f"❌ Failed to optimize {model_type}: {e}")
                all_results[model_type] = {'error': str(e)}
        
        return all_results
    
    def save_optimization_results(self, output_dir: str = "models/optimized_params") -> None:
        """
        Save optimization results to disk.
        
        Args:
            output_dir: Directory to save results
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        for model_type, results in self.optimization_results.items():
            if 'error' in results:
                continue
                
            # Save parameters as JSON
            params_file = os.path.join(output_dir, f"{model_type}_best_params.json")
            
            save_data = {
                'model_type': model_type,
                'best_params': results['best_params'],
                'best_score': results['best_score'],
                'optimization_metadata': {
                    'portfolio_size': self.portfolio_size,
                    'risk_tolerance': self.risk_tolerance,
                    'n_trials': results['n_trials'],
                    'optimization_time': results['optimization_time'],
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            with open(params_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            # Save Optuna study
            study_file = os.path.join(output_dir, f"{model_type}_study.pkl")
            joblib.dump(results['study'], study_file)
            
            logger.info(f"💾 Saved {model_type} optimization results to {output_dir}")
    
    def create_production_models(self, best_params: Dict[str, Dict[str, Any]],
                               X_train: pd.DataFrame,
                               y_train: pd.Series) -> Dict[str, Any]:
        """
        Create production models with optimized hyperparameters.
        
        Args:
            best_params: Dictionary of optimized parameters for each model
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of trained production models
        """
        
        logger.info("🏭 Creating production models with optimized parameters...")
        
        production_models = {}
        
        for model_type, params in best_params.items():
            if 'error' in params:
                continue
                
            logger.info(f"🔧 Training production {model_type}...")
            
            try:
                if model_type == 'xgboost':
                    from xgboost import XGBClassifier
                    model = XGBClassifier(**params['best_params'])
                    
                elif model_type == 'lightgbm':
                    from lightgbm import LGBMClassifier
                    model = LGBMClassifier(**params['best_params'])
                    
                elif model_type == 'catboost':
                    from catboost import CatBoostClassifier
                    model = CatBoostClassifier(**params['best_params'])
                    
                elif model_type == 'randomforest':
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(**params['best_params'])
                    
                else:
                    logger.warning(f"Unknown model type: {model_type}")
                    continue
                
                # Train the model
                model.fit(X_train, y_train)
                production_models[f"{model_type}_classifier_optimized"] = model
                
                logger.info(f"✅ {model_type} production model ready!")
                
            except Exception as e:
                logger.error(f"❌ Failed to create production {model_type}: {e}")
        
        return production_models
