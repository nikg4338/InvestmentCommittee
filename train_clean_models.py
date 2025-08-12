#!/usr/bin/env python3
"""
Clean Model Training with Optuna Optimization
Train models on data without leakage for realistic performance.
Enhanced with hyperparameter optimization.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import joblib
import json
from datetime import datetime
import os
import optuna
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

class CleanModelTrainer:
    """Enhanced model trainer with Optuna optimization for clean data."""
    
    def __init__(self, data_path: str = "alpaca_training_data_clean.csv"):
        self.data_path = data_path
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        self.results = {}
        
    def load_and_prepare_data(self) -> bool:
        """Load and prepare clean training data."""
        try:
            logger.info(f"Loading clean training data from {self.data_path}")
            
            if not os.path.exists(self.data_path):
                logger.error(f"Clean training data file not found: {self.data_path}")
                return False
            
            # Load data
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(df)} samples with {len(df.columns)} total columns")
            
            # Prepare features and target
            feature_cols = [col for col in df.columns if col not in ['target', 'ticker', 'timestamp']]
            
            X = df[feature_cols]
            y = df['target']
            
            # Handle missing values
            X = X.fillna(X.median())
            
            # Store feature names
            self.feature_names = feature_cols
            
            # Time-aware split (80/20 split)
            split_idx = int(len(X) * 0.8)
            
            self.X_train = X.iloc[:split_idx]
            self.y_train = y.iloc[:split_idx]
            self.X_test = X.iloc[split_idx:]
            self.y_test = y.iloc[split_idx:]
            
            logger.info(f"Training set: {len(self.X_train)} samples")
            logger.info(f"Test set: {len(self.X_test)} samples")
            logger.info(f"Features: {len(self.feature_names)}")
            logger.info(f"Target distribution - Train: {self.y_train.value_counts().to_dict()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def optimize_catboost(self, trial) -> float:
        """Optuna objective for CatBoost optimization."""
        try:
            from catboost import CatBoostClassifier
            
            params = {
                'iterations': trial.suggest_int('iterations', 200, 800),
                'depth': trial.suggest_int('depth', 4, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 128),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'random_strength': trial.suggest_float('random_strength', 0, 1),
                'verbose': False,
                'random_seed': 42,
                'thread_count': -1
            }
            
            model = CatBoostClassifier(**params)
            
            # Use cross-validation for robust evaluation
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train, 
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                scoring='roc_auc', n_jobs=-1
            )
            
            return cv_scores.mean()
            
        except Exception as e:
            logger.warning(f"CatBoost trial failed: {e}")
            return 0.5
    
    def optimize_random_forest(self, trial) -> float:
        """Optuna objective for Random Forest optimization."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': 42,
                'n_jobs': -1
            }
            
            model = RandomForestClassifier(**params)
            
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                scoring='roc_auc', n_jobs=-1
            )
            
            return cv_scores.mean()
            
        except Exception as e:
            logger.warning(f"Random Forest trial failed: {e}")
            return 0.5
    
    def optimize_xgboost(self, trial) -> float:
        """Optuna objective for XGBoost optimization."""
        try:
            import xgboost as xgb
            
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'logloss'
            }
            
            model = xgb.XGBClassifier(**params)
            
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                scoring='roc_auc', n_jobs=-1
            )
            
            return cv_scores.mean()
            
        except Exception as e:
            logger.warning(f"XGBoost trial failed: {e}")
            return 0.5
    
    def optimize_lightgbm(self, trial) -> float:
        """Optuna objective for LightGBM optimization."""
        try:
            import lightgbm as lgb
            
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            
            model = lgb.LGBMClassifier(**params)
            
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                scoring='roc_auc', n_jobs=-1
            )
            
            return cv_scores.mean()
            
        except Exception as e:
            logger.warning(f"LightGBM trial failed: {e}")
            return 0.5
    
    def train_optimized_model(self, model_name: str, n_trials: int = 50) -> dict:
        """Train a model with Optuna optimization."""
        
        logger.info(f"Starting optimization for {model_name} with {n_trials} trials...")
        
        # Create study
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        
        # Get optimization function
        if model_name == 'catboost':
            objective = self.optimize_catboost
        elif model_name == 'random_forest':
            objective = self.optimize_random_forest
        elif model_name == 'xgboost':
            objective = self.optimize_xgboost
        elif model_name == 'lightgbm':
            objective = self.optimize_lightgbm
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best {model_name} CV score: {study.best_value:.4f}")
        logger.info(f"Best {model_name} params: {study.best_params}")
        
        # Train final model with best parameters
        final_model = self._train_final_model(model_name, study.best_params)
        
        # Evaluate on test set
        test_metrics = self._evaluate_model(final_model, model_name)
        
        return {
            'model': final_model,
            'best_params': study.best_params,
            'best_cv_score': study.best_value,
            'test_metrics': test_metrics,
            'study': study
        }
    
    def _train_final_model(self, model_name: str, best_params: dict):
        """Train final model with best parameters."""
        
        if model_name == 'catboost':
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(**best_params, verbose=False, random_seed=42)
            
        elif model_name == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
            
        elif model_name == 'xgboost':
            import xgboost as xgb
            model = xgb.XGBClassifier(**best_params, random_state=42, n_jobs=-1)
            
        elif model_name == 'lightgbm':
            import lightgbm as lgb
            model = lgb.LGBMClassifier(**best_params, random_state=42, n_jobs=-1, verbose=-1)
        
        # Train the model
        model.fit(self.X_train, self.y_train)
        return model
    
    def _evaluate_model(self, model, model_name: str) -> dict:
        """Evaluate model on test set."""
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='weighted'),
            'recall': recall_score(self.y_test, y_pred, average='weighted'),
            'f1': f1_score(self.y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
        }
        
        logger.info(f"{model_name} Test Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def save_model(self, model, model_name: str, best_params: dict, test_metrics: dict):
        """Save optimized model and metadata."""
        
        # Create models directory
        os.makedirs('models/clean', exist_ok=True)
        
        # Save model
        model_path = f'models/clean/{model_name}_clean_optimized.pkl'
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'training_date': datetime.now().isoformat(),
            'best_params': best_params,
            'test_metrics': test_metrics,
            'feature_names': self.feature_names,
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'data_source': 'clean_data_no_leakage'
        }
        
        metadata_path = f'models/clean/{model_name}_clean_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {model_name} model to {model_path}")
        logger.info(f"Saved {model_name} metadata to {metadata_path}")
    
    def run_complete_training(self, n_trials: int = 50) -> dict:
        """Run complete training pipeline for all models."""
        
        print("="*80)
        print("CLEAN MODEL TRAINING WITH OPTUNA OPTIMIZATION")
        print("="*80)
        
        if not self.load_and_prepare_data():
            logger.error("Failed to load data")
            return {}
        
        models_to_train = ['catboost', 'random_forest', 'xgboost', 'lightgbm']
        results = {}
        
        for i, model_name in enumerate(models_to_train, 1):
            print(f"\nTRAINING MODEL {i}/{len(models_to_train)}: {model_name.upper()}")
            print(f"   Optimization trials: {n_trials}")
            
            try:
                start_time = datetime.now()
                result = self.train_optimized_model(model_name, n_trials)
                training_time = datetime.now() - start_time
                
                # Save the model
                self.save_model(
                    result['model'], 
                    model_name, 
                    result['best_params'], 
                    result['test_metrics']
                )
                
                result['training_time'] = training_time.total_seconds()
                results[model_name] = result
                
                print(f"   ✅ {model_name} training completed in {training_time}")
                print(f"   📈 Best CV Score: {result['best_cv_score']:.4f}")
                print(f"   🎯 Test ROC-AUC: {result['test_metrics']['roc_auc']:.4f}")
                print(f"   🎯 Test Accuracy: {result['test_metrics']['accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        # Save overall results
        self._save_training_summary(results, n_trials)
        
        return results
    
    def _save_training_summary(self, results: dict, n_trials: int):
        """Save training summary."""
        
        summary = {
            'training_date': datetime.now().isoformat(),
            'n_trials_per_model': n_trials,
            'models_trained': len([k for k, v in results.items() if 'error' not in v]),
            'training_data_size': len(self.X_train) + len(self.X_test),
            'feature_count': len(self.feature_names),
            'data_source': 'clean_data_no_leakage',
            'results': {}
        }
        
        for model_name, result in results.items():
            if 'error' not in result:
                summary['results'][model_name] = {
                    'best_cv_score': result['best_cv_score'],
                    'test_metrics': result['test_metrics'],
                    'training_time_seconds': result['training_time']
                }
            else:
                summary['results'][model_name] = {'error': result['error']}
        
        with open('models/clean/clean_training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nTRAINING SUMMARY SAVED")
        print(f"   Models successfully trained: {summary['models_trained']}")
        print(f"   Training data size: {summary['training_data_size']:,}")
        print(f"   Features used: {summary['feature_count']}")

def train_clean_models():
    """Main function to train clean models with user input for trials."""
    
    print("CLEAN MODEL TRAINING WITH OPTUNA OPTIMIZATION")
    print("This will train models on clean data (no leakage) with hyperparameter optimization")
    
    # Ask user for number of trials
    try:
        n_trials = int(input("Enter number of Optuna trials per model (25-75 recommended): "))
        if n_trials < 10:
            n_trials = 25
        elif n_trials > 100:
            n_trials = 75
    except ValueError:
        n_trials = 50
        print(f"Using default: {n_trials} trials per model")
    
    print(f"\nStarting training with {n_trials} trials per model...")
    print("This will take 15-30 minutes depending on your hardware.")
    
    trainer = CleanModelTrainer()
    results = trainer.run_complete_training(n_trials=n_trials)
    
    print("\nTRAINING COMPLETE!")
    print("All clean optimized models saved to models/clean/")
    print("These models are ready for production paper trading!")

if __name__ == "__main__":
    train_clean_models()
    print("All clean optimized models saved to models/clean/")
    print("These models are ready for production paper trading!")

if __name__ == "__main__":
    train_clean_models()
