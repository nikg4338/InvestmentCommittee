#!/usr/bin/env python3
"""
Pipeline Improvements Module
===========================

Advanced ML pipeline improvements including Optuna optimization,
probability calibration, dynamic weighting, and more.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)

def tune_with_optuna(model_cls, X: pd.DataFrame, y: pd.Series, 
                    param_space: Dict[str, Any], n_trials: int = 50, 
                    cv: int = 3, scoring: str = 'f1') -> Dict[str, Any]:
    """
    Run Optuna to tune hyperparameters for any sklearn-like estimator.
    
    Args:
        model_cls: Model class to instantiate
        X: Training features
        y: Training labels
        param_space: Dictionary of parameter ranges
        n_trials: Number of optimization trials
        cv: Cross-validation folds
        scoring: Scoring metric
        
    Returns:
        Dictionary of best parameters
    """
    try:
        import optuna
        
        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            try:
                # Build parameter dict for this trial
                params = {}
                for k, v in param_space.items():
                    if isinstance(v, list):
                        params[k] = trial.suggest_categorical(k, v)
                    elif isinstance(v, tuple) and len(v) == 2:
                        if isinstance(v[0], int) and isinstance(v[1], int):
                            params[k] = trial.suggest_int(k, v[0], v[1])
                        else:
                            params[k] = trial.suggest_float(k, v[0], v[1])
                    else:
                        params[k] = v  # Fixed parameter
                
                # Create and evaluate model
                # Handle different model constructor patterns
                if model_cls.__name__ in ['XGBoostModel']:
                    model = model_cls(model_params=params)
                else:
                    model = model_cls(**params)
                
                # Get underlying sklearn estimator if wrapped
                if hasattr(model, 'model'):
                    estimator = model.model
                elif hasattr(model, 'pipeline'):
                    estimator = model.pipeline
                else:
                    estimator = model
                
                scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=1)
                return scores.mean()
                
            except Exception as e:
                logger.warning(f"Optuna trial failed: {e}")
                return 0.0
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        logger.info(f"Optuna optimization completed: best score = {study.best_value:.4f}")
        return study.best_params
        
    except ImportError:
        logger.warning("Optuna not available - returning empty params")
        return {}
    except Exception as e:
        logger.warning(f"Optuna optimization failed: {e}")
        return {}

def calibrate_model(estimator, X_train: pd.DataFrame, y_train: pd.Series, 
                   method: str = 'isotonic', cv: int = 3):
    """
    Return a calibrated version of the estimator using CalibratedClassifierCV.
    
    Args:
        estimator: Fitted sklearn estimator
        X_train: Training features
        y_train: Training labels
        method: Calibration method ('isotonic' or 'sigmoid')
        cv: Cross-validation folds
        
    Returns:
        Calibrated classifier
    """
    try:
        calibrator = CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
        calibrator.fit(X_train, y_train)
        logger.info(f"Model calibrated using {method} method with {cv}-fold CV")
        return calibrator
    except Exception as e:
        logger.warning(f"Calibration failed: {e}, returning original estimator")
        return estimator

def get_advanced_sampler(name: str = 'smoteenn', random_state: int = 42):
    """
    Get advanced sampling strategy for imbalanced data.
    
    Args:
        name: Sampler name ('smoteenn', 'adasyn', 'smotetomek')
        random_state: Random state for reproducibility
        
    Returns:
        Configured sampler or None if unavailable
    """
    try:
        if name == 'smoteenn':
            from imblearn.combine import SMOTEENN
            return SMOTEENN(random_state=random_state)
        elif name == 'adasyn':
            from imblearn.over_sampling import ADASYN
            return ADASYN(random_state=random_state)
        elif name == 'smotetomek':
            from imblearn.combine import SMOTETomek
            return SMOTETomek(random_state=random_state)
        else:
            logger.warning(f"Unknown sampler: {name}")
            return None
    except ImportError:
        logger.warning(f"Sampler {name} not available - imbalanced-learn not installed")
        return None
    except Exception as e:
        logger.warning(f"Failed to create sampler {name}: {e}")
        return None

def compute_dynamic_weights(metrics_dict: Dict[str, Dict[str, float]], 
                          metric: str = 'roc_auc') -> Dict[str, float]:
    """
    Compute dynamic ensemble weights based on model performance.
    
    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        metric: Metric to use for weighting
        
    Returns:
        Dictionary of normalized weights
    """
    try:
        scores = np.array([v.get(metric, 0.5) for v in metrics_dict.values()])
        
        # Ensure all scores are positive
        scores = np.clip(scores, 0.01, None)  # Minimum weight
        
        if scores.sum() == 0:
            # Equal weights if all scores are zero
            equal_weight = 1.0 / len(scores)
            return {m: equal_weight for m in metrics_dict.keys()}
        
        # Normalize to sum to 1
        normalized_scores = scores / scores.sum()
        
        weights = dict(zip(metrics_dict.keys(), normalized_scores))
        
        logger.info(f"Dynamic weights computed using {metric}:")
        for model, weight in weights.items():
            score = metrics_dict[model].get(metric, 0.0)
            logger.info(f"  {model}: {weight:.4f} (score: {score:.4f})")
        
        return weights
        
    except Exception as e:
        logger.warning(f"Failed to compute dynamic weights: {e}")
        # Return equal weights as fallback
        equal_weight = 1.0 / len(metrics_dict)
        return {m: equal_weight for m in metrics_dict.keys()}

def select_top_features_shap(model, X_train: pd.DataFrame, k: int = 20) -> List[str]:
    """
    Select top-k features using SHAP feature importance.
    
    Args:
        model: Fitted model (preferably tree-based)
        X_train: Training features
        k: Number of top features to select
        
    Returns:
        List of selected feature names
    """
    try:
        import shap
        
        # If k >= number of features, return all features
        if k >= len(X_train.columns):
            logger.info(f"Requested features ({k}) >= available features ({len(X_train.columns)}) - returning all")
            return X_train.columns.tolist()
        
        # Create explainer based on model type
        if hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model)
        
        # Compute SHAP values (sample if dataset is large)
        sample_size = min(1000, len(X_train))
        X_sample = X_train.sample(n=sample_size, random_state=42)
        
        shap_values = explainer.shap_values(X_sample)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Binary classification - use positive class
            shap_vals = np.abs(shap_values[1])
        else:
            shap_vals = np.abs(shap_values)
        
        # Compute mean absolute SHAP values
        mean_importance = np.mean(shap_vals, axis=0)
        
        # Get top-k features
        top_indices = np.argsort(mean_importance)[::-1][:k]
        selected_features = X_train.columns[top_indices].tolist()
        
        logger.info(f"Selected top {len(selected_features)} features using SHAP:")
        for i, feature in enumerate(selected_features[:10]):  # Show top 10
            importance = mean_importance[top_indices[i]]
            logger.info(f"  {i+1}. {feature}: {importance:.4f}")
        
        return selected_features
        
    except ImportError:
        logger.warning("SHAP not available - returning all features")
        return X_train.columns.tolist()
    except Exception as e:
        logger.warning(f"SHAP feature selection failed: {e} - returning all features")
        return X_train.columns.tolist()

def create_time_series_splits(n_splits: int = 5) -> TimeSeriesSplit:
    """
    Create TimeSeriesSplit for time-aware cross-validation.
    
    Args:
        n_splits: Number of splits
        
    Returns:
        TimeSeriesSplit object
    """
    return TimeSeriesSplit(n_splits=n_splits)

def create_xgb_meta_model(**kwargs) -> Any:
    """
    Create XGBoost meta-model as alternative to LogisticRegression.
    
    Args:
        **kwargs: Additional XGBoost parameters
        
    Returns:
        Configured XGBoost classifier
    """
    try:
        from xgboost import XGBClassifier
        
        default_params = {
            'objective': 'binary:logistic',
            'max_depth': 3,
            'n_estimators': 50,
            'learning_rate': 0.1,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': 42
        }
        default_params.update(kwargs)
        
        return XGBClassifier(**default_params)
        
    except ImportError:
        logger.warning("XGBoost not available for meta-model")
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight='balanced',  # ← FIXED: Added balanced class weights
            solver='liblinear'  # Better for balanced weights
        )

def rolling_backtest(model, X: pd.DataFrame, y: pd.Series, 
                    window: int = 200, step: int = 50) -> pd.DataFrame:
    """
    Perform rolling-window backtest for drift detection.
    
    Args:
        model: Sklearn-like model
        X: Feature matrix
        y: Target vector
        window: Training window size
        step: Test step size
        
    Returns:
        DataFrame with backtest results
    """
    results = []
    
    for start in range(0, len(X) - window - step, step):
        try:
            train_slice = slice(start, start + window)
            test_slice = slice(start + window, start + window + step)
            
            X_train = X.iloc[train_slice]
            y_train = y.iloc[train_slice]
            X_test = X.iloc[test_slice]
            y_test = y.iloc[test_slice]
            
            # Clone and fit model
            m = clone(model)
            m.fit(X_train, y_train)
            
            # Make predictions
            y_pred = m.predict(X_test)
            y_proba = m.predict_proba(X_test)[:, 1] if hasattr(m, 'predict_proba') else None
            
            # Compute metrics
            metrics = {
                'window_start': start,
                'window_end': start + window,
                'test_start': start + window,
                'test_end': start + window + step,
                'n_train': len(y_train),
                'n_test': len(y_test),
                'train_pos_rate': y_train.mean(),
                'test_pos_rate': y_test.mean(),
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
            
            if y_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
            
            results.append(metrics)
            
        except Exception as e:
            logger.warning(f"Rolling backtest failed at window {start}: {e}")
            continue
    
    df_results = pd.DataFrame(results)
    
    if not df_results.empty:
        logger.info(f"Rolling backtest completed: {len(df_results)} windows")
        logger.info(f"Mean accuracy: {df_results['accuracy'].mean():.4f} ± {df_results['accuracy'].std():.4f}")
        logger.info(f"Mean F1: {df_results['f1'].mean():.4f} ± {df_results['f1'].std():.4f}")
    
    return df_results

def add_macro_llm_signal(df: pd.DataFrame, llm_analyzer, 
                        feature_columns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add LLM-generated macro signal to the dataset.
    
    Args:
        df: Input DataFrame
        llm_analyzer: LLM analyzer instance
        feature_columns: Current feature columns list
        
    Returns:
        Updated DataFrame and feature columns
    """
    try:
        # Generate macro signal using LLM
        if hasattr(llm_analyzer, 'analyze_macro_conditions'):
            macro_analysis = llm_analyzer.analyze_macro_conditions()
            macro_signal = getattr(macro_analysis, 'confidence', 0.5)
        else:
            # Fallback: use a simple sentiment score
            macro_signal = 0.5  # Neutral signal
        
        # Add signal to all rows
        df_updated = df.copy()
        df_updated['macro_llm_signal'] = macro_signal
        
        # Update feature columns
        updated_features = feature_columns + ['macro_llm_signal']
        
        logger.info(f"Added LLM macro signal: {macro_signal:.4f}")
        return df_updated, updated_features
        
    except Exception as e:
        logger.warning(f"Failed to add LLM macro signal: {e}")
        return df, feature_columns

def detect_data_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame, 
                     threshold: float = 0.3) -> Dict[str, Any]:
    """
    Detect data drift between reference and current datasets.
    
    Args:
        reference_data: Reference dataset (historical)
        current_data: Current dataset (recent)
        threshold: Drift threshold
        
    Returns:
        Dictionary with drift analysis results
    """
    try:
        from scipy import stats
        
        drift_results = {
            'drift_detected': False,
            'drifted_features': [],
            'drift_scores': {},
            'recommendation': 'continue'
        }
        
        common_features = list(set(reference_data.columns) & set(current_data.columns))
        
        for feature in common_features:
            if reference_data[feature].dtype in ['float64', 'int64']:
                # Use Kolmogorov-Smirnov test for continuous features
                try:
                    _, p_value = stats.ks_2samp(reference_data[feature], current_data[feature])
                    drift_score = 1 - p_value  # Higher score = more drift
                    
                    drift_results['drift_scores'][feature] = drift_score
                    
                    if drift_score > threshold:
                        drift_results['drifted_features'].append(feature)
                        drift_results['drift_detected'] = True
                        
                except Exception:
                    continue
        
        if drift_results['drift_detected']:
            drift_results['recommendation'] = 'retrain_model'
            logger.warning(f"Data drift detected in {len(drift_results['drifted_features'])} features")
        else:
            logger.info("No significant data drift detected")
        
        return drift_results
        
    except ImportError:
        logger.warning("SciPy not available for drift detection")
        return {'drift_detected': False, 'recommendation': 'continue'}
    except Exception as e:
        logger.warning(f"Drift detection failed: {e}")
        return {'drift_detected': False, 'recommendation': 'continue'}
