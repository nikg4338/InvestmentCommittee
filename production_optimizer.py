#!/usr/bin/env python3
"""
Direct Production Model Training and Optimization
================================================

Train models from scratch with optimized hyperparameters and thresholds.
"""

import logging
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple

import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from sklearn.preprocessing import StandardScaler

# ML Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress optuna output
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ProductionModelOptimizer:
    """
    Direct model training and optimization for production deployment.
    """
    
    def __init__(self, portfolio_size: int = 20, min_precision: float = 0.3):
        self.portfolio_size = portfolio_size
        self.min_precision = min_precision
        self.models = {}
        self.thresholds = {}
        self.optimization_results = {}
    
    def load_and_prepare_data(self, data_file: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load and prepare training data with robust handling.
        """
        logger.info("Loading and preparing data...")
        
        # Try different data sources
        data_files = [
            data_file,
            "alpaca_training_data_batches_1.csv",
            "alpaca_training_data.csv"
        ]
        
        data = None
        for file_path in data_files:
            if file_path and os.path.exists(file_path):
                try:
                    data = pd.read_csv(file_path)
                    logger.info(f"Loaded {len(data)} samples from {file_path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
        
        if data is None:
            raise FileNotFoundError("No valid data file found")
        
        # Prepare features and target
        feature_columns = [col for col in data.columns if col not in ['target', 'symbol', 'timestamp', 'ticker']]
        X = data[feature_columns].copy()
        y = data['target'] if 'target' in data.columns else data.iloc[:, -1]
        
        logger.info(f"Features: {len(feature_columns)}, Target distribution: {y.value_counts().to_dict()}")
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Create train/test split with stratification if possible
        if len(y.unique()) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            # Fallback for single class
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Train: {len(X_train)} samples ({y_train.mean():.3f} positive)")
        logger.info(f"Test: {len(X_test)} samples ({y_test.mean():.3f} positive)")
        
        return X_train, X_test, y_train, y_test
    
    def create_trading_objective(self, X: pd.DataFrame, y: pd.Series):
        """
        Create a trading-focused objective function for hyperparameter optimization.
        """
        def objective(trial):
            model_type = trial.suggest_categorical('model_type', ['xgboost', 'lightgbm', 'catboost'])
            
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    'random_state': 42,
                    'eval_metric': 'logloss'
                }
                model = XGBClassifier(**params)
                
            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    'random_state': 42,
                    'verbosity': -1
                }
                model = LGBMClassifier(**params)
                
            elif model_type == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 500),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'random_state': 42,
                    'verbose': False
                }
                model = CatBoostClassifier(**params)
            
            try:
                # Use cross-validation if we have enough positive samples
                if y.sum() >= 10:
                    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
                    return np.mean(scores)
                else:
                    # Simple train/test for limited data
                    split_idx = int(len(X) * 0.8)
                    X_train_cv, X_val_cv = X.iloc[:split_idx], X.iloc[split_idx:]
                    y_train_cv, y_val_cv = y.iloc[:split_idx], y.iloc[split_idx:]
                    
                    model.fit(X_train_cv, y_train_cv)
                    y_pred_proba = model.predict_proba(X_val_cv)[:, 1]
                    
                    # Return AUC if possible, otherwise custom score
                    if len(y_val_cv.unique()) > 1:
                        return roc_auc_score(y_val_cv, y_pred_proba)
                    else:
                        # Custom score for single class
                        return np.std(y_pred_proba)  # Reward diversity in predictions
                        
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0
        
        return objective
    
    def optimize_and_train_models(self, X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 50):
        """
        Optimize hyperparameters and train production models.
        """
        logger.info(f"Starting model optimization with {n_trials} trials...")
        
        # Create optimization study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        objective = self.create_trading_objective(X_train, y_train)
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials, timeout=300)  # 5 minute timeout
        
        logger.info(f"Best trial score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        # Train the best model
        best_params = study.best_params.copy()
        model_type = best_params.pop('model_type')
        
        if model_type == 'xgboost':
            best_model = XGBClassifier(**best_params)
        elif model_type == 'lightgbm':
            best_model = LGBMClassifier(**best_params)
        elif model_type == 'catboost':
            best_model = CatBoostClassifier(**best_params)
        
        best_model.fit(X_train, y_train)
        
        # Train additional models with fixed good parameters
        additional_models = {}
        
        # Simple Random Forest
        try:
            rf_model = RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_split=10,
                random_state=42, n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            additional_models['random_forest'] = rf_model
        except Exception as e:
            logger.warning(f"Random Forest training failed: {e}")
        
        # Simple SVM (if data is not too large)
        if len(X_train) <= 1000:
            try:
                svm_model = SVC(probability=True, random_state=42)
                svm_model.fit(X_train, y_train)
                additional_models['svm'] = svm_model
            except Exception as e:
                logger.warning(f"SVM training failed: {e}")
        
        # Store models
        self.models[f'optimized_{model_type}'] = best_model
        self.models.update(additional_models)
        
        # Store optimization results
        self.optimization_results = {
            'best_model_type': model_type,
            'best_score': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'study': study
        }
        
        logger.info(f"Trained {len(self.models)} models successfully")
        return self.models
    
    def optimize_thresholds(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Optimize prediction thresholds for trading.
        """
        logger.info("Optimizing prediction thresholds...")
        
        for model_name, model in self.models.items():
            logger.info(f"Optimizing threshold for {model_name}...")
            
            try:
                # Get predictions
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                else:
                    y_proba = model.decision_function(X_test)
                    # Convert to [0,1] range
                    y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
                
                best_threshold = 0.5
                best_score = 0.0
                
                # Test different thresholds
                thresholds = np.concatenate([
                    np.linspace(0.1, 0.9, 30),
                    np.percentile(y_proba, [60, 70, 80, 85, 90, 95, 99])
                ])
                
                for threshold in np.unique(thresholds):
                    y_pred = (y_proba >= threshold).astype(int)
                    
                    if np.sum(y_pred) == 0:
                        continue
                    
                    # Calculate metrics if we have positive examples in test
                    if len(y_test.unique()) > 1:
                        precision = precision_score(y_test, y_pred, zero_division=0)
                        recall = recall_score(y_test, y_pred, zero_division=0)
                        f1 = f1_score(y_test, y_pred, zero_division=0)
                        
                        # Portfolio constraints
                        n_positions = np.sum(y_pred)
                        if 5 <= n_positions <= 40 and precision >= self.min_precision:
                            # Score = weighted combination of precision and recall
                            score = 0.7 * precision + 0.3 * recall
                            if score > best_score:
                                best_score = score
                                best_threshold = threshold
                    else:
                        # For single class test data, use portfolio size constraint
                        n_positions = np.sum(y_pred)
                        if 10 <= n_positions <= 30:  # Reasonable portfolio size
                            # Use prediction diversity as score
                            score = len(np.unique(y_proba)) / len(y_proba)
                            if score > best_score:
                                best_score = score
                                best_threshold = threshold
                
                self.thresholds[model_name] = {
                    'threshold': best_threshold,
                    'score': best_score,
                    'portfolio_size': np.sum((y_proba >= best_threshold).astype(int))
                }
                
                logger.info(f"  {model_name}: threshold={best_threshold:.4f}, "
                           f"portfolio_size={self.thresholds[model_name]['portfolio_size']}")
                
            except Exception as e:
                logger.error(f"Threshold optimization failed for {model_name}: {e}")
                self.thresholds[model_name] = {'threshold': 0.5, 'error': str(e)}
    
    def save_production_models(self, output_dir: str = "models/production"):
        """
        Save optimized models and configuration for production use.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_file = os.path.join(output_dir, f"{model_name}.pkl")
            joblib.dump(model, model_file)
            logger.info(f"Saved {model_name} to {model_file}")
        
        # Save configuration
        config = {
            'creation_timestamp': datetime.now().isoformat(),
            'optimization_results': {
                'best_model_type': self.optimization_results.get('best_model_type'),
                'best_score': self.optimization_results.get('best_score'),
                'best_params': self.optimization_results.get('best_params'),
                'n_trials': self.optimization_results.get('n_trials')
            },
            'model_thresholds': self.thresholds,
            'portfolio_settings': {
                'target_size': self.portfolio_size,
                'min_precision': self.min_precision
            },
            'model_files': {name: f"{name}.pkl" for name in self.models.keys()}
        }
        
        config_file = os.path.join(output_dir, "production_config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Production configuration saved to {config_file}")
        return config
    
    def run_complete_optimization(self, data_file: str = None, n_trials: int = 50):
        """
        Run the complete optimization pipeline.
        """
        logger.info("Starting complete production optimization pipeline...")
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_and_prepare_data(data_file)
        
        # Optimize and train models
        self.optimize_and_train_models(X_train, y_train, n_trials)
        
        # Optimize thresholds
        self.optimize_thresholds(X_test, y_test)
        
        # Save everything
        config = self.save_production_models()
        
        # Print summary
        self.print_summary()
        
        return config
    
    def print_summary(self):
        """
        Print optimization summary.
        """
        print("\n" + "="*70)
        print("PRODUCTION MODEL OPTIMIZATION SUMMARY")
        print("="*70)
        
        print(f"Models Trained: {len(self.models)}")
        print(f"Best Model: {self.optimization_results.get('best_model_type', 'N/A')}")
        print(f"Best Score: {self.optimization_results.get('best_score', 0):.4f}")
        print(f"Optimization Trials: {self.optimization_results.get('n_trials', 0)}")
        
        print("\nModel Thresholds:")
        for model_name, threshold_info in self.thresholds.items():
            if 'error' in threshold_info:
                print(f"  {model_name}: ERROR - {threshold_info['error']}")
            else:
                print(f"  {model_name}: threshold={threshold_info['threshold']:.4f}, "
                      f"portfolio_size={threshold_info['portfolio_size']}")
        
        print("\nNext Steps:")
        print("  1. Review model performance metrics")
        print("  2. Test models with real-time data")
        print("  3. Deploy to paper trading environment")
        print("  4. Monitor performance and retrain as needed")
        print("="*70)


def main():
    """
    Main execution function.
    """
    # Configuration
    PORTFOLIO_SIZE = 20
    MIN_PRECISION = 0.25
    N_TRIALS = 30  # Reduced for faster execution
    
    logger.info("Starting production model optimization...")
    
    try:
        # Initialize optimizer
        optimizer = ProductionModelOptimizer(
            portfolio_size=PORTFOLIO_SIZE,
            min_precision=MIN_PRECISION
        )
        
        # Run complete optimization
        config = optimizer.run_complete_optimization(
            data_file="alpaca_training_data_batches_1.csv",
            n_trials=N_TRIALS
        )
        
        logger.info("Production optimization completed successfully!")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise


if __name__ == "__main__":
    main()
