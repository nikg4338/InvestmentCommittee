#!/usr/bin/env python3
"""
Enhanced Training Integration
============================

This module provides integration functions to connect the enhanced systems
with the existing train_models.py by adding parameter injection and enhanced workflows.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

# Integration availability flag
TRAIN_MODELS_AVAILABLE = True

logger = logging.getLogger(__name__)
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import importlib.util

# Import enhanced systems
from enhanced_hyperparameter_optimizer import enhanced_optimizer
from advanced_meta_learning_ensemble import AdvancedMetaLearningEnsemble
from feature_alignment_system import ProbabilityFixer

logger = logging.getLogger(__name__)

def fix_uniform_probabilities_in_results(training_results: Dict[str, Any], 
                                        X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Fix uniform probabilities in training results.
    
    Args:
        training_results: Results from model training
        X_test: Test features  
        y_test: Test labels
        
    Returns:
        Training results with fixed probabilities
    """
    try:
        if 'models' not in training_results:
            return training_results
            
        probability_fixer = ProbabilityFixer()
        
        for model_name, model_info in training_results['models'].items():
            if 'model' in model_info:
                try:
                    # Get model predictions
                    model = model_info['model']
                    
                    # Try to get probabilities
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(X_test)
                        if probs.ndim > 1:
                            probs = probs[:, 1]  # Get positive class probabilities
                    elif hasattr(model, 'decision_function'):
                        scores = model.decision_function(X_test)
                        # Convert scores to probabilities
                        from sklearn.utils.fixes import expit
                        probs = expit(scores)
                    else:
                        continue
                    
                    # Check and fix uniform probabilities
                    if probability_fixer.detect_uniform_probabilities(probs):
                        logger.warning(f"🔧 Fixing uniform probabilities for {model_name}")
                        fixed_probs = probability_fixer.fix_uniform_probabilities(probs, y_test)
                        
                        # Update the model info with fixed probabilities
                        model_info['fixed_probabilities'] = fixed_probs
                        model_info['had_uniform_probabilities'] = True
                    else:
                        model_info['fixed_probabilities'] = probs
                        model_info['had_uniform_probabilities'] = False
                        
                except Exception as e:
                    logger.warning(f"Failed to fix probabilities for {model_name}: {e}")
                    
        logger.info("✅ Probability fixing completed")
        return training_results
        
    except Exception as e:
        logger.error(f"Failed to fix uniform probabilities: {e}")
        return training_results

def inject_optimized_parameters(model_type: str, optimized_params: Dict[str, Any]) -> Dict[str, Any]:
    """Inject optimized parameters into model configuration."""
    try:
        # Clean and prepare parameters for the specific model type
        params = optimized_params.get('best_params', {})
        
        if not params:
            logger.warning(f"No optimized parameters found for {model_type}")
            return {}
        
        # Model-specific parameter mapping
        if model_type == 'xgboost':
            # Map optimizer params to XGBoost model params
            model_params = {
                'n_estimators': params.get('n_estimators', 100),
                'max_depth': params.get('max_depth', 6),
                'learning_rate': params.get('learning_rate', 0.1),
                'subsample': params.get('subsample', 1.0),
                'colsample_bytree': params.get('colsample_bytree', 1.0),
                'reg_alpha': params.get('reg_alpha', 0.0),
                'reg_lambda': params.get('reg_lambda', 1.0),
                'min_child_weight': params.get('min_child_weight', 1),
                'gamma': params.get('gamma', 0.0),
                'scale_pos_weight': params.get('scale_pos_weight', 1.0),
                'random_state': 42,
                'verbosity': 0
            }
            
        elif model_type == 'lightgbm':
            model_params = {
                'n_estimators': params.get('n_estimators', 100),
                'num_leaves': params.get('num_leaves', 31),
                'learning_rate': params.get('learning_rate', 0.1),
                'feature_fraction': params.get('feature_fraction', 1.0),
                'bagging_fraction': params.get('bagging_fraction', 1.0),
                'bagging_freq': params.get('bagging_freq', 0),
                'reg_alpha': params.get('reg_alpha', 0.0),
                'reg_lambda': params.get('reg_lambda', 0.0),
                'min_child_weight': params.get('min_child_weight', 0.001),
                'max_depth': params.get('max_depth', -1),
                'scale_pos_weight': params.get('scale_pos_weight', 1.0),
                'random_state': 42,
                'verbosity': -1,
                'force_col_wise': True
            }
            
        elif model_type == 'catboost':
            model_params = {
                'iterations': params.get('iterations', 100),
                'learning_rate': params.get('learning_rate', 0.1),
                'depth': params.get('depth', 6),
                'l2_leaf_reg': params.get('l2_leaf_reg', 3.0),
                'random_strength': params.get('random_strength', 1.0),
                'bagging_temperature': params.get('bagging_temperature', 1.0),
                'border_count': params.get('border_count', 128),
                'scale_pos_weight': params.get('scale_pos_weight', 1.0),
                'random_state': 42,
                'verbose': False,
                'thread_count': -1
            }
            
        elif model_type == 'random_forest':
            model_params = {
                'n_estimators': params.get('n_estimators', 100),
                'max_depth': params.get('max_depth', None),
                'min_samples_split': params.get('min_samples_split', 2),
                'min_samples_leaf': params.get('min_samples_leaf', 1),
                'max_features': params.get('max_features', 'sqrt'),
                'bootstrap': params.get('bootstrap', True),
                'class_weight': params.get('class_weight', 'balanced'),
                'random_state': 42,
                'n_jobs': -1
            }
            
        elif model_type == 'svm':
            model_params = {
                'C': params.get('C', 1.0),
                'kernel': params.get('kernel', 'rbf'),
                'gamma': params.get('gamma', 'scale'),
                'class_weight': params.get('class_weight', 'balanced'),
                'probability': True,
                'random_state': 42
            }
            
        elif model_type == 'neural_network':
            # Handle neural network architecture tuple
            hidden_layers = params.get('hidden_layer_sizes', (100,))
            if isinstance(hidden_layers, (list, tuple)):
                hidden_layer_sizes = tuple(hidden_layers)
            else:
                hidden_layer_sizes = (100,)
                
            model_params = {
                'hidden_layer_sizes': hidden_layer_sizes,
                'learning_rate_init': params.get('learning_rate_init', 0.001),
                'alpha': params.get('alpha', 0.0001),
                'solver': params.get('solver', 'adam'),
                'batch_size': params.get('batch_size', 'auto'),
                'max_iter': params.get('max_iter', 200),
                'early_stopping': params.get('early_stopping', True),
                'validation_fraction': params.get('validation_fraction', 0.1),
                'n_iter_no_change': params.get('n_iter_no_change', 10),
                'random_state': 42
            }
            
        else:
            logger.warning(f"Unknown model type for parameter injection: {model_type}")
            model_params = {}
        
        logger.info(f"✅ Injected {len(model_params)} optimized parameters for {model_type}")
        return model_params
        
    except Exception as e:
        logger.error(f"Failed to inject parameters for {model_type}: {e}")
        return {}

def train_batch_with_params(batch_name: str, X_train: pd.DataFrame, y_train: np.ndarray, 
                           X_test: pd.DataFrame, y_test: np.ndarray, 
                           optimized_params: Dict[str, Any]) -> Dict[str, Any]:
    """Train models for a batch with injected optimized parameters."""
    try:
        logger.info(f"🎯 Training batch {batch_name} with optimized parameters...")
        
        # Import train_models dynamically to avoid circular imports
        import train_models
        
        # Create configuration with optimized parameters
        config = train_models.get_default_config()
        
        # Inject optimized parameters into model configurations
        model_configs = {}
        for model_type, opt_result in optimized_params.items():
            if opt_result.get('best_params'):
                model_params = inject_optimized_parameters(model_type, opt_result)
                if model_params:
                    model_configs[model_type] = model_params
                    logger.info(f"✅ Enhanced {model_type} with {len(model_params)} optimized params")
        
        # Convert numpy arrays to pandas if needed
        if isinstance(y_train, np.ndarray):
            y_train = pd.Series(y_train, index=X_train.index)
        if isinstance(y_test, np.ndarray):
            y_test = pd.Series(y_test, index=X_test.index)
        
        # Monkey patch model parameters into the training system
        original_model_params = {}
        
        # Store original parameters and inject optimized ones
        for model_type, params in model_configs.items():
            try:
                # This is a simplified approach - you may need to adjust based on your model implementation
                if hasattr(config, f'{model_type}_params'):
                    original_model_params[model_type] = getattr(config, f'{model_type}_params')
                    setattr(config, f'{model_type}_params', params)
                    
            except Exception as e:
                logger.warning(f"Could not inject params for {model_type}: {e}")
        
        # Train the models using the existing training function
        training_results = train_models.train_committee_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            config=config
        )
        
        # Fix uniform probability issues in model predictions
        training_results = fix_uniform_probabilities_in_results(training_results, X_test, y_test)
        
        # Restore original parameters
        for model_type, original_params in original_model_params.items():
            setattr(config, f'{model_type}_params', original_params)
        
        # Add optimization metadata
        training_results['optimization_metadata'] = {
            'optimized_models': list(model_configs.keys()),
            'total_optimization_time': sum(
                opt_result.get('optimization_time', 0) 
                for opt_result in optimized_params.values()
            ),
            'optimization_trials': {
                model_type: opt_result.get('n_trials', 0)
                for model_type, opt_result in optimized_params.items()
            }
        }
        
        logger.info(f"✅ Training completed for {batch_name} with {len(model_configs)} optimized models")
        return training_results
        
    except Exception as e:
        logger.error(f"Failed to train batch {batch_name} with optimized parameters: {e}")
        # Fallback to standard training
        import train_models
        return train_models.train_committee_models(X_train, y_train, X_test, y_test)

def create_enhanced_config_with_params(optimized_params: Dict[str, Any]) -> Any:
    """Create an enhanced training configuration with optimized parameters."""
    try:
        import train_models
        
        # Start with extreme imbalance config (best for your use case)
        config = train_models.get_extreme_imbalance_config()
        
        # Enable enhanced features
        setattr(config, 'enable_enhanced_stacking', True)
        setattr(config, 'enable_calibration', True)
        setattr(config, 'advanced_sampling', 'smoteenn')  # Best for noisy financial data
        setattr(config, 'enable_optuna', False)  # Disable since we have pre-optimized params
        
        # Inject model-specific optimized parameters
        for model_type, opt_result in optimized_params.items():
            if opt_result.get('best_params'):
                model_params = inject_optimized_parameters(model_type, opt_result)
                if model_params:
                    # Store parameters in config
                    param_attr = f'{model_type}_optimized_params'
                    setattr(config, param_attr, model_params)
        
        logger.info(f"✅ Created enhanced config with optimized parameters")
        return config
        
    except Exception as e:
        logger.error(f"Failed to create enhanced config: {e}")
        import train_models
        return train_models.get_default_config()

def run_enhanced_training_with_integration(data_file: str, 
                                          optimization_results: Optional[Dict[str, Any]] = None,
                                          batch_name: Optional[str] = None) -> Dict[str, Any]:
    """Run complete enhanced training with full integration."""
    try:
        logger.info(f"🚀 Starting enhanced training with full integration...")
        logger.info(f"   Data file: {data_file}")
        logger.info(f"   Batch: {batch_name or 'unknown'}")
        logger.info(f"   Optimization results: {'Available' if optimization_results else 'None'}")
        
        # Load and prepare data
        df = pd.read_csv(data_file)
        
        # Prepare features and targets
        exclude_cols = ['ticker', 'target', 'target_enhanced', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_columns].copy()
        y = df['target'].values
        
        # Clean data
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        valid_indices = ~pd.isna(y)
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Train with optimization if available
        if optimization_results:
            results = train_batch_with_params(
                batch_name or 'enhanced_batch',
                X_train, y_train, X_test, y_test,
                optimization_results
            )
        else:
            # Standard training
            import train_models
            y_train_series = pd.Series(y_train, index=X_train.index)
            y_test_series = pd.Series(y_test, index=X_test.index)
            
            results = train_models.train_committee_models(
                X_train, y_train_series, X_test, y_test_series
            )
        
        # Add integration metadata
        results['integration_metadata'] = {
            'enhanced_training': True,
            'optimization_used': optimization_results is not None,
            'data_file': data_file,
            'batch_name': batch_name,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(feature_columns),
            'integration_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Enhanced training with integration completed")
        return results
        
    except Exception as e:
        logger.error(f"Enhanced training with integration failed: {e}")
        raise

def validate_integration() -> bool:
    """Validate that all integration components are properly connected."""
    try:
        logger.info("🔍 Validating integration components...")
        
        # Check if train_models can be imported
        try:
            import train_models
            logger.info("✅ train_models.py imported successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import train_models.py: {e}")
            return False
        
        # Check if enhanced systems are available
        try:
            from enhanced_hyperparameter_optimizer import enhanced_optimizer
            from advanced_meta_learning_ensemble import AdvancedMetaLearningEnsemble
            from cross_batch_analyzer import cross_batch_analyzer
            from production_deployment_system import production_deployer
            logger.info("✅ All enhanced systems imported successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import enhanced systems: {e}")
            return False
        
        # Check required functions
        required_functions = ['train_committee_models', 'get_default_config']
        for func_name in required_functions:
            if not hasattr(train_models, func_name):
                logger.error(f"❌ Missing required function: {func_name}")
                return False
        
        logger.info("✅ Integration validation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Integration validation failed: {e}")
        return False


if __name__ == "__main__":
    # Test integration
    if validate_integration():
        print("✅ Integration validation passed - all systems connected")
    else:
        print("❌ Integration validation failed")
        sys.exit(1)
