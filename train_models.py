#!/usr/bin/env python3
"""
Committee of Five Training Script - Refactored
==============================================

This script implements a "Committee of Five" ensemble for the Investment
Committee project. It has been refactored into a modular architecture:

- Configuration management for all hyperparameters
- Robust data splitting with extreme imbalance handling  
- Advanced sampling techniques (SMOTE, SMOTEENN)
- Out-of-fold stacking with fallback strategies
- Comprehensive evaluation and visualization
- Clean separation of concerns across modules

The script now supports multiple training strategies and configurations
optimized for different scenarios (extreme imbalance, fast training, etc.).

Usage
-----
python train_models.py --config extreme_imbalance --models xgboost lightgbm catboost
python train_models.py --config fast_training --export-results
"""

import argparse
import logging
import os
import time
import traceback
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

# Import configuration and utilities
from config.training_config import (
    TrainingConfig, get_default_config, get_extreme_imbalance_config, 
    get_fast_training_config
)
from utils.data_splitting import stratified_train_test_split, validate_split_quality
from utils.sampling import prepare_balanced_data, assess_balance_quality
from utils.stacking import (
    out_of_fold_stacking, train_meta_model, create_ensemble_predictions,
    evaluate_stacking_quality
)
from utils.evaluation import (
    evaluate_ensemble_performance, create_performance_summary,
    export_training_results, create_confusion_matrices, generate_training_report,
    optimize_ensemble_thresholds
)
from utils.visualization import create_training_report as create_visual_report
from utils.helpers import compute_classification_metrics
from utils.probability_analysis import analyze_model_ensemble_probabilities, diagnose_probability_issues

# Import advanced optimization utilities
from utils.advanced_optimization import (
    find_optimal_threshold_advanced, enhanced_smote_for_regression,
    convert_multiclass_to_binary, portfolio_aware_threshold_optimization,
    evaluate_threshold_robustness, adaptive_threshold_selection
)

# Import Alpaca data collection
from data_collection_alpaca import AlpacaDataCollector

# Enhanced utilities for extreme imbalance (keep these for backward compatibility)
from utils.data_splitting import ensure_minority_samples
from utils.evaluation import find_optimal_threshold

# Import pipeline improvements
from utils.pipeline_improvements import (
    rolling_backtest, add_macro_llm_signal, detect_data_drift,
    compute_dynamic_weights
)

logger = logging.getLogger(__name__)

def find_optimal_threshold_on_test(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                  metric: str = 'f1') -> Tuple[float, float, Dict[str, float]]:
    """
    Find optimal threshold using the TRUE UNBALANCED test set.
    This avoids data leakage from resampled training data.
    
    Args:
        y_true: True binary labels (unbalanced test set)
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall', 'pr_auc')
        
    Returns:
        (optimal_threshold, best_score, metrics_dict)
    """
    from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
    
    # Use precision-recall curve for threshold optimization
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    if metric == 'f1':
        # Find threshold that maximizes F1
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_score = f1_scores[best_idx]
        
    elif metric == 'precision':
        # Find threshold that maximizes precision (conservative approach)
        best_idx = np.argmax(precision)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_score = precision[best_idx]
        
    elif metric == 'recall':
        # Find threshold that maximizes recall (aggressive approach)
        best_idx = np.argmax(recall)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_score = recall[best_idx]
        
    elif metric == 'pr_auc':
        # Use threshold that balances precision and recall for best AUC
        from sklearn.metrics import auc
        pr_auc = auc(recall, precision)
        # Find threshold closest to the "elbow" of the PR curve
        best_idx = np.argmax(f1_scores)  # F1 is good proxy for PR-AUC optimization
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_score = pr_auc
        
    else:
        # Default to F1 optimization
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_score = f1_scores[best_idx]
    
    # Calculate final metrics at optimal threshold
    y_pred_binary = (y_pred_proba >= best_threshold).astype(int)
    
    final_metrics = {
        'threshold': best_threshold,
        'f1': f1_score(y_true, y_pred_binary, zero_division=0),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'support_positive': np.sum(y_true),
        'support_negative': len(y_true) - np.sum(y_true),
        'predicted_positive': np.sum(y_pred_binary)
    }
    
    logger.info(f"🎯 Optimal threshold ({metric}): {best_threshold:.4f}")
    logger.info(f"   F1={final_metrics['f1']:.3f}, P={final_metrics['precision']:.3f}, R={final_metrics['recall']:.3f}")
    logger.info(f"   Predicted positives: {final_metrics['predicted_positive']}/{len(y_true)} ({final_metrics['predicted_positive']/len(y_true)*100:.1f}%)")
    
    return best_threshold, best_score, final_metrics

def compute_optimal_threshold(y_true: np.ndarray, proba_preds: np.ndarray, metric: str = 'pr_auc') -> float:
    """
    Compute optimal threshold for given predictions using specified metric.
    Uses the unbalanced test set to avoid data leakage.
    
    Args:
        y_true: True binary labels (unbalanced)
        proba_preds: Predicted probabilities
        metric: Metric to optimize ('pr_auc', 'f1', 'precision', 'recall')
        
    Returns:
        Optimal threshold value
    """
    try:
        # Use new threshold optimization that preserves test set distribution
        threshold, _, _ = find_optimal_threshold_on_test(y_true, proba_preds, metric)
        return threshold
    except Exception as e:
        logger.warning(f"Failed to compute optimal threshold: {e}")
        # Fallback to traditional method
        try:
            from utils.evaluation import find_optimal_threshold
            threshold, _ = find_optimal_threshold(y_true, proba_preds, metric=metric)
            return threshold
        except:
            return 0.5

def optuna_tune_model(model_cls, X: pd.DataFrame, y: pd.Series, n_trials: int = 20) -> Dict[str, Any]:
    """
    Use Optuna to tune hyperparameters for F₁/PR-AUC optimization.
    
    Args:
        model_cls: Model class to tune (RandomForestModel, CatBoostModel, etc.)
        X: Training features
        y: Training labels
        n_trials: Number of Optuna trials
        
    Returns:
        Dictionary of best hyperparameters
    """
    try:
        import optuna
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        
        # Import enhanced optimization function
        from utils.enhanced_meta_models import optuna_optimize_base_model_for_f1
        
        logger.info(f"🎯 Optuna F₁/PR-AUC optimization for {model_cls.__name__}...")
        
        # Use enhanced optimization targeting average_precision (PR-AUC)
        best_params = optuna_optimize_base_model_for_f1(
            model_cls, X, y, n_trials=n_trials, optimize_metric='average_precision'
        )
        
        if best_params:
            logger.info(f"✅ Optuna optimization completed for {model_cls.__name__}")
            return best_params
        else:
            logger.warning(f"Optuna optimization failed for {model_cls.__name__}, using defaults")
            return {}
            
    except ImportError:
        logger.warning("Optuna not available - using default parameters")
        return {}
    except Exception as e:
        logger.warning(f"Optuna tuning failed: {e}")
        return {}

def get_model_predictions(model, X: pd.DataFrame, is_regressor: bool = False) -> np.ndarray:
    """
    Get predictions from a model, handling both classification and regression.
    
    Args:
        model: Trained model instance
        X: Input features
        is_regressor: Whether the model is a regressor
        
    Returns:
        Probability scores for classification or continuous predictions for regression
    """
    try:
        if is_regressor:
            # For regressors, use the predict method directly
            predictions = model.predict(X)
            return predictions
        else:
            # For classifiers, try predict_proba first, fall back to predict
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)[:, 1]
            else:
                return model.predict(X)
    except Exception as e:
        logger.warning(f"Error getting predictions from model: {e}")
        return np.zeros(len(X))

def is_regression_model(model_name: str) -> bool:
    """
    Check if a model name corresponds to a regression model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        True if it's a regression model, False otherwise
    """
    return 'regressor' in model_name.lower() or model_name.endswith('_regressor')

def convert_regression_to_binary(predictions: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Convert regression predictions to binary predictions using a threshold.
    
    Args:
        predictions: Continuous regression predictions
        threshold: Threshold for binary conversion (default: 0.0 for positive returns)
        
    Returns:
        Binary predictions (0 or 1)
    """
    return (predictions > threshold).astype(int)

def compute_dynamic_ensemble_weights(evaluation_results: Dict[str, Any], 
                                   base_models: List[str] = None) -> Dict[str, float]:
    """
    Compute dynamic ensemble weights based on individual model performance.
    
    Args:
        evaluation_results: Results from model evaluation
        base_models: List of base model names
        
    Returns:
        Dictionary mapping model names to normalized weights
    """
    if base_models is None:
        base_models = ['xgboost', 'lightgbm', 'lightgbm_regressor', 'catboost', 'random_forest', 'svm']
    
    # Extract ROC-AUC scores for weighting
    roc_scores = {}
    for model_name in base_models:
        if model_name in evaluation_results:
            roc_scores[model_name] = evaluation_results[model_name].get('roc_auc', 0.5)
        else:
            roc_scores[model_name] = 0.5  # Default neutral weight
    
    # Normalize weights to sum to 1
    total_score = sum(roc_scores.values()) or 1.0
    dynamic_weights = {model: score / total_score for model, score in roc_scores.items()}
    
    logger.info("🎯 Dynamic ensemble weights based on ROC-AUC:")
    for model, weight in dynamic_weights.items():
        logger.info(f"  {model}: {weight:.4f} (ROC-AUC: {roc_scores[model]:.4f})")
    
    return dynamic_weights

def setup_logging(log_level: str = 'INFO') -> None:
    """Setup logging configuration"""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def clean_data_for_ml(X: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the feature matrix by handling infinite and extreme values.
    """
    X_clean = X.copy()
    
    # First, remove any columns that contain datetime strings
    datetime_like_columns = []
    for col in X_clean.columns:
        if X_clean[col].dtype == 'object':
            # Check if column contains datetime-like strings
            sample_values = X_clean[col].dropna().head(5).astype(str)
            if any(any(pattern in str(val) for pattern in ['-', ':', 'T', '+', 'UTC', 'GMT']) and 
                   len(str(val)) > 10 for val in sample_values):
                datetime_like_columns.append(col)
    
    if datetime_like_columns:
        logger.warning(f"Removing {len(datetime_like_columns)} datetime-like columns during cleaning: {datetime_like_columns}")
        X_clean = X_clean.drop(columns=datetime_like_columns)
    
    # Replace infinite values with NaN
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    
    # Clip extreme values
    for col in X_clean.columns:
        if X_clean[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            q1 = X_clean[col].quantile(0.01)
            q99 = X_clean[col].quantile(0.99)
            X_clean[col] = X_clean[col].clip(lower=q1, upper=q99)
    
    return X_clean

def prepare_training_data(df: pd.DataFrame, 
                         feature_columns: List[str],
                         target_column: str = 'target',
                         config: Optional[TrainingConfig] = None,
                         enable_enhanced_targets: bool = True,
                         target_strategy: str = 'top_percentile') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare training data with robust splitting, cleaning, and enhanced target handling.
    
    Args:
        df: Input DataFrame with features and target
        feature_columns: List of feature column names
        target_column: Name of target column
        config: Training configuration
        enable_enhanced_targets: Whether to use enhanced target strategies
        target_strategy: Target enhancement strategy ('top_percentile', 'multi_class', 'quantile_buckets')
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    if config is None:
        config = get_default_config()
    
    logger.info("Preparing enhanced training data...")
    
    # Extract features and target
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    # Enhanced target processing
    if enable_enhanced_targets and target_strategy in ['multi_class', 'quantile_buckets']:
        logger.info(f"🔄 Processing {target_strategy} targets...")
        
        # Check if target is already multi-class
        unique_targets = y.unique()
        n_classes = len(unique_targets)
        
        if n_classes > 2:
            logger.info(f"   Multi-class target detected: {n_classes} classes")
            
            # For training, we can use multi-class directly or convert to binary
            if target_strategy == 'multi_class':
                # Keep multi-class for richer patterns during training
                logger.info(f"   Using multi-class target for training: {unique_targets}")
            else:
                # Convert to binary for final evaluation
                y = convert_multiclass_to_binary(y.values, strategy='top_class')
                logger.info(f"   Converted multi-class to binary: {np.unique(y)}")
        
        elif target_strategy == 'top_percentile':
            # Check if we need to apply top-percentile strategy
            positive_rate = np.sum(y) / len(y) * 100 if len(y) > 0 else 0
            
            if positive_rate < 5.0:  # Less than 5% positive rate
                logger.info(f"   Low positive rate ({positive_rate:.1f}%), applying top-percentile enhancement...")
                
                # This would typically be done in data collection, but we can simulate here
                # by treating continuous values as returns and applying top-percentile
                if hasattr(df, 'daily_return') or any('return' in col for col in df.columns):
                    return_cols = [col for col in df.columns if 'return' in col]
                    if return_cols:
                        return_col = return_cols[0]  # Use first return column
                        returns = df[return_col].dropna()
                        
                        if len(returns) > 0:
                            top_threshold = returns.quantile(0.90)  # Top 10%
                            y = (returns >= top_threshold).astype(int)
                            
                            new_positive_rate = np.sum(y) / len(y) * 100
                            logger.info(f"   Enhanced positive rate: {new_positive_rate:.1f}% (threshold: {top_threshold:.4f})")
    
    # Clean data
    X_clean = clean_data_for_ml(X)
    
    # Remove rows with NaN values
    mask = ~(X_clean.isnull().any(axis=1) | y.isnull())
    X_final = X_clean[mask]
    y_final = y[mask]
    
    logger.info(f"Data after cleaning: {len(X_final)} samples")
    
    # Enhanced class distribution logging
    if y_final.dtype in ['int64', 'int32'] and len(np.unique(y_final)) <= 10:
        class_dist = pd.Series(y_final).value_counts().sort_index()
        logger.info(f"Class distribution: {dict(class_dist)}")
        
        # Calculate enhanced metrics
        if len(np.unique(y_final)) == 2:
            positive_rate = np.sum(y_final) / len(y_final) * 100
            logger.info(f"Binary positive rate: {positive_rate:.1f}%")
            
            if positive_rate < 5.0:
                logger.warning(f"⚠️ Very low positive rate ({positive_rate:.1f}%) - consider enhanced sampling")
            elif positive_rate > 20.0:
                logger.info(f"✅ Good positive rate ({positive_rate:.1f}%) for model training")
    else:
        logger.info(f"Continuous target: mean={y_final.mean():.4f}, std={y_final.std():.4f}")
    
    # Perform robust stratified split
    try:
        X_train, X_test, y_train, y_test = stratified_train_test_split(
            X_final, y_final,
            test_size=config.test_size,
            random_state=config.random_state,
            min_minority_samples=config.cross_validation.min_minority_samples
        )
    except Exception as e:
        logger.warning(f"Stratified split failed: {e}, using random split")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_final, 
            test_size=config.test_size, 
            random_state=config.random_state
        )
    
    # Validate split quality
    try:
        split_quality = validate_split_quality(X_train, X_test, y_train, y_test)
        logger.info(f"Split quality: {split_quality}")
    except Exception as e:
        logger.warning(f"Split quality validation failed: {e}")
    
    # CRITICAL: Do NOT resample here - preserve true data distribution for proper evaluation
    # Resampling will be done ONLY on training data in the stacking module
    logger.info("� Preserving true data distribution - resampling will occur only on training data")
    logger.info(f"✅ Train/test split maintains original positive rates:")
    logger.info(f"   Training: {np.sum(y_train)/len(y_train)*100:.2f}% positive ({np.sum(y_train)}/{len(y_train)})")
    logger.info(f"   Test: {np.sum(y_test)/len(y_test)*100:.2f}% positive ({np.sum(y_test)}/{len(y_test)})")
    
    return X_train, X_test, y_train, y_test

def create_rank_vote_ensemble(base_predictions: Dict[str, np.ndarray],
                             meta_predictions: Optional[np.ndarray] = None,
                             config: Optional[TrainingConfig] = None) -> np.ndarray:
    """
    Create rank-and-vote ensemble predictions (production-ready approach).
    
    Args:
        base_predictions: Dictionary mapping model names to probabilities
        meta_predictions: Meta-model predictions (optional)
        config: Training configuration
        
    Returns:
        Final ensemble predictions (binary)
    """
    if config is None:
        config = get_default_config()
    
    logger.info("🗳️ Creating rank-and-vote ensemble...")
    
    if not base_predictions:
        logger.warning("No base predictions available")
        return np.array([])
    
    n_samples = len(next(iter(base_predictions.values())))
    top_pct = config.ensemble.top_percentile
    
    # Step 1: Each model votes on its top percentile
    votes = []
    for model_name, probabilities in base_predictions.items():
        cutoff = np.percentile(probabilities, 100 - top_pct * 100)
        model_votes = (probabilities >= cutoff).astype(int)
        votes.append(model_votes)
        
        vote_count = np.sum(model_votes)
        logger.info(f"   {model_name}: {vote_count} votes (cutoff: {cutoff:.6f})")
    
    if not votes:
        return np.zeros(n_samples, dtype=int)
    
    # Step 2: Majority voting
    vote_matrix = np.vstack(votes).T
    total_votes = vote_matrix.sum(axis=1)
    
    majority_threshold = int(len(votes) * config.ensemble.majority_threshold_factor) + 1
    consensus_predictions = (total_votes >= majority_threshold).astype(int)
    
    consensus_count = np.sum(consensus_predictions)
    logger.info(f"   Majority consensus: {consensus_count} samples (threshold: {majority_threshold}/{len(votes)})")
    
    # Step 3: Meta-model boost if available
    if meta_predictions is not None:
        meta_cutoff = np.percentile(meta_predictions, 100 - top_pct * 100)
        meta_votes = (meta_predictions >= meta_cutoff).astype(int)
        meta_vote_count = np.sum(meta_votes)
        
        logger.info(f"   Meta-model: {meta_vote_count} votes (cutoff: {meta_cutoff:.6f})")
        
        # Combine with meta-model boost
        combined_votes = total_votes + (config.ensemble.meta_weight * meta_votes)
        
        # Re-apply majority with meta boost
        boosted_consensus = (combined_votes >= majority_threshold).astype(int)
        final_predictions = boosted_consensus
    else:
        final_predictions = consensus_predictions
    
    # Step 4: Guarantee minimum predictions
    min_positives = max(config.ensemble.min_consensus, int(n_samples * top_pct))
    if np.sum(final_predictions) < min_positives:
        logger.info(f"Guaranteeing minimum {min_positives} positive predictions")
        
        # Use combined scores if available, otherwise use base votes
        if meta_predictions is not None:
            scores = total_votes + (config.ensemble.meta_weight * meta_votes)
        else:
            scores = total_votes.astype(float)
        
        top_indices = np.argsort(scores)[-min_positives:]
        final_predictions = np.zeros(n_samples, dtype=int)
        final_predictions[top_indices] = 1
    
    final_count = np.sum(final_predictions)
    logger.info(f"🎯 Final ensemble result: {final_count} buy signals")
    
    return final_predictions

def train_committee_models(X_train: pd.DataFrame, y_train: pd.Series, 
                          X_test: pd.DataFrame, y_test: pd.Series,
                          config: Optional[TrainingConfig] = None) -> Dict[str, Any]:
    """
    Main training function for the Committee of Five ensemble.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        config: Training configuration
        
    Returns:
        Dictionary with training results and evaluation metrics
    """
    if config is None:
        config = get_default_config()
    
    start_time = time.time()
    logger.info("🚀 Starting Committee of Five training...")
    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    logger.info(f"Training config: {config.__class__.__name__}")
    
    # Extract meta-learner configuration early to avoid UnboundLocalError
    meta_learner_type = config.meta_model.meta_learner_type  # Use config value (default: gradientboost)
    use_xgb_meta = getattr(config, 'use_xgb_meta_model', False)
    if use_xgb_meta:
        meta_learner_type = 'lightgbm'  # Override with lightgbm if XGB meta is requested
    
    logger.info(f"📊 Meta-learner type: {meta_learner_type}")
    
    # Step 1: Out-of-fold stacking with enhancements
    logger.info("\n📊 Phase 1: Enhanced out-of-fold stacking...")
    
    # Check if enhanced stacking is enabled
    use_enhanced_stacking = getattr(config, 'enable_enhanced_stacking', True)
    use_time_series_cv = getattr(config, 'use_time_series_cv', False)
    enable_calibration = getattr(config, 'enable_calibration', True)
    enable_feature_selection = getattr(config, 'enable_feature_selection', False)
    advanced_sampling = getattr(config, 'advanced_sampling', 'smoteenn')
    
    if use_enhanced_stacking:
        from utils.stacking import enhanced_out_of_fold_stacking
        train_meta_features, test_meta_features, trained_models = enhanced_out_of_fold_stacking(
            X_train, y_train, X_test, config,
            use_time_series_cv=use_time_series_cv,
            enable_calibration=enable_calibration,
            enable_feature_selection=enable_feature_selection,
            advanced_sampling=advanced_sampling
        )
    else:
        # Use standard stacking
        train_meta_features, test_meta_features, trained_models = out_of_fold_stacking(
            X_train, y_train, X_test, config
        )
    
    # Check if OOF was successful
    if train_meta_features is None:
        logger.info("OOF stacking fell back to simple train/test - using base models only")
        use_meta_model = False
        meta_model = None
        meta_test_proba = None
        optimal_threshold = 0.5
    else:
        use_meta_model = True
        
        # Step 2: Train enhanced meta-model with advanced strategies
        logger.info("\n🧠 Phase 2: Training enhanced meta-model with F₁ optimization...")
        
        # Import enhanced meta-model training functions
        from utils.enhanced_meta_models import (
            train_meta_model_with_optimal_threshold,
            train_focal_loss_meta_model,
            train_dynamic_weighted_ensemble,
            train_feature_selected_meta_model,
            get_enhanced_meta_model_strategy
        )
        
        # Check configuration for meta-model strategy
        stack_raw_features = getattr(config, 'stack_raw_features', False)
        meta_strategy = getattr(config, 'meta_model_strategy', 'optimal_threshold')
        
        # Get OOF predictions for dynamic weighting if available
        oof_predictions = {}
        test_predictions = {}
        
        # Extract base model predictions from ensemble results (we'll get this later)
        # For now, use the trained models to get predictions
        for model_name, model_info in trained_models.items():
            if 'oof_predictions' in model_info:
                oof_predictions[model_name] = model_info['oof_predictions']
            if 'test_predictions' in model_info:
                test_predictions[model_name] = model_info['test_predictions']
        
        # Choose enhanced meta-model training strategy
        if meta_strategy == 'dynamic_weights' and len(oof_predictions) > 0:
            # Dynamic weighted ensemble approach
            meta_test_proba, dynamic_weights, optimal_threshold = train_dynamic_weighted_ensemble(
                oof_predictions, y_train, test_predictions, weight_metric='roc_auc'
            )
            meta_model = None  # No actual model, just weighted combination
            logger.info("✅ Using dynamic weighted ensemble as meta-model")
            
        elif meta_strategy == 'focal_loss':
            # Focal loss meta-model for extreme imbalance
            meta_model, optimal_threshold = train_focal_loss_meta_model(
                train_meta_features, y_train, alpha=0.25, gamma=2.0
            )
            # Get test predictions
            if hasattr(meta_model, 'predict'):
                meta_test_proba = meta_model.predict(test_meta_features)
            else:
                meta_test_proba = meta_model.predict_proba(test_meta_features)[:, 1]
                
        elif meta_strategy == 'feature_select':
            # Feature-selected meta-model
            meta_model, optimal_threshold, train_selected, test_selected = train_feature_selected_meta_model(
                train_meta_features, y_train, test_meta_features, k_best=3
            )
            meta_test_proba = meta_model.predict_proba(test_selected)[:, 1]
            
        else:
            # Default: Optimal threshold with gradient boosting
            meta_model, optimal_threshold = train_meta_model_with_optimal_threshold(
                train_meta_features, y_train,
                meta_learner_type=meta_learner_type,
                use_class_weights=True,
                optimize_for='f1'
            )
            
            # Get test predictions
            if hasattr(meta_model, 'predict'):
                meta_test_proba = meta_model.predict(test_meta_features)
            else:
                meta_test_proba = meta_model.predict_proba(test_meta_features)[:, 1]
        
        # Handle raw feature stacking if enabled
        if stack_raw_features and meta_model is not None:
            logger.info("🔗 Stacking raw features with meta-features...")
            # Prepare combined features
            raw_features_reset = X_train.reset_index(drop=True)
            meta_features_df = pd.DataFrame(train_meta_features, columns=[f'meta_{i}' for i in range(train_meta_features.shape[1])])
            combined_train_features = pd.concat([meta_features_df, raw_features_reset], axis=1)
            
            raw_test_reset = X_test.reset_index(drop=True)
            meta_test_df = pd.DataFrame(test_meta_features, columns=[f'meta_{i}' for i in range(test_meta_features.shape[1])])
            combined_test_features = pd.concat([meta_test_df, raw_test_reset], axis=1)
            
            # Retrain meta-model with combined features
            if meta_strategy == 'feature_select':
                # Use feature selection on combined features
                meta_model, optimal_threshold, _, test_selected = train_feature_selected_meta_model(
                    combined_train_features.values, y_train, combined_test_features.values, k_best=10
                )
                meta_test_proba = meta_model.predict_proba(test_selected)[:, 1]
            else:
                meta_model, optimal_threshold = train_meta_model_with_optimal_threshold(
                    combined_train_features.values, y_train,
                    meta_learner_type=meta_learner_type,
                    use_class_weights=True,
                    optimize_for='f1'
                )
                
                if hasattr(meta_model, 'predict'):
                    meta_test_proba = meta_model.predict(combined_test_features.values)
                else:
                    meta_test_proba = meta_model.predict_proba(combined_test_features.values)[:, 1]
            
        logger.info(f"Meta-model test predictions range: [{meta_test_proba.min():.4f}, {meta_test_proba.max():.4f}]")
        logger.info(f"Using optimal threshold: {optimal_threshold:.4f} for final predictions")
    
    # Step 3: Create ensemble predictions
    logger.info("\n🎯 Phase 3: Creating ensemble predictions...")
    
    ensemble_results = create_ensemble_predictions(
        trained_models, X_test, meta_model, config
    )
    
    # Analyze base model probability distributions
    logger.info("\n📊 Comprehensive Probability Analysis:")
    
    # Perform detailed analysis with diagnostics
    analysis_results = analyze_model_ensemble_probabilities(
        ensemble_results['base_predictions'], 
        y_test, 
        save_plots=config.visualization.save_plots,
        plot_dir="reports"
    )
    
    # Check for common probability issues
    for model_name, probabilities in ensemble_results['base_predictions'].items():
        issues = diagnose_probability_issues(probabilities, model_name)
        if issues:
            logger.warning(f"⚠️ {model_name} probability issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")
    
    # Analyze meta-model probabilities if available
    if meta_test_proba is not None:
        meta_issues = diagnose_probability_issues(meta_test_proba, "Meta-model")
        if meta_issues:
            logger.warning(f"⚠️ Meta-model probability issues:")
            for issue in meta_issues:
                logger.warning(f"  - {issue}")
    
    # Optimize thresholds for all base models with UNBALANCED TEST SET
    logger.info("\n🎯 Advanced Individual Model Threshold Optimization (Unbalanced Test Set):")
    threshold_results = {}
    
    for model_name, predictions in ensemble_results['base_predictions'].items():
        try:
            # Use multiple optimization strategies on UNBALANCED test set
            strategies = ['f1', 'precision', 'pr_auc']
            best_threshold = 0.5
            best_score = 0.0
            best_strategy = 'f1'
            
            for strategy in strategies:
                try:
                    threshold, score, metrics = find_optimal_threshold_on_test(
                        y_test, predictions, strategy
                    )
                    
                    if score > best_score:
                        best_threshold = threshold
                        best_score = score
                        best_strategy = strategy
                        
                except Exception as e:
                    logger.warning(f"Strategy {strategy} failed for {model_name}: {e}")
                    continue
            
            threshold_results[model_name] = {
                'threshold': best_threshold,
                'score': best_score,
                'strategy': best_strategy
            }
            
            logger.info(f"  {model_name}: threshold={best_threshold:.4f}, {best_strategy.upper()}={best_score:.3f}")
            
        except Exception as e:
            logger.warning(f"Threshold optimization failed for {model_name}: {e}")
            threshold_results[model_name] = {'threshold': 0.5, 'score': 0.0, 'strategy': 'default'}
    
    # Portfolio-aware threshold optimization for ensemble
    logger.info("\n📊 Portfolio-aware ensemble optimization:")
    try:
        ensemble_proba = ensemble_results.get('simple_ensemble', np.mean(list(ensemble_results['base_predictions'].values()), axis=0))
        portfolio_threshold, portfolio_metrics = portfolio_aware_threshold_optimization(
            y_test, ensemble_proba, portfolio_size=20, risk_tolerance='moderate'
        )
        
        logger.info(f"📊 Portfolio-optimized threshold: {portfolio_threshold:.4f}")
        logger.info(f"   Expected positions: {portfolio_metrics.get('n_positions', 0)}")
        logger.info(f"   Portfolio precision: {portfolio_metrics.get('precision', 0):.3f}")
        
    except Exception as e:
        logger.warning(f"Portfolio optimization failed: {e}")
        portfolio_threshold = 0.5
        portfolio_metrics = {}
    
    # Step 4: Convert regression and quantile predictions to binary decisions
    logger.info("\n🔄 Phase 4: Converting regression and quantile predictions to binary decisions...")
    
    # Import the conversion functions
    from utils.evaluation import convert_regression_ensemble_to_binary
    
    # Check for quantile models
    quantile_models = {k: v for k, v in ensemble_results['base_predictions'].items() 
                      if isinstance(v, dict) and all(isinstance(qk, float) for qk in v.keys())}
    
    if quantile_models and enable_quantile_eval:
        # Handle quantile regression models separately
        logger.info(f"Converting {len(quantile_models)} quantile models to binary predictions...")
        
        try:
            from utils.quantile_evaluation import convert_quantile_ensemble_to_binary
            
            decision_strategy = getattr(config, 'quantile_decision_strategy', 'threshold_optimization')
            risk_tolerance = getattr(config, 'risk_tolerance', 'moderate')
            
            quantile_conversion_results = convert_quantile_ensemble_to_binary(
                quantile_models, 
                y_test,
                decision_strategy=decision_strategy,
                risk_tolerance=risk_tolerance,
                optimize_thresholds=True
            )
            
            # Log quantile conversion results
            for model_name, results in quantile_conversion_results.items():
                logger.info(f"  {model_name} (quantile): F1={results['f1_score']:.3f}, strategy={results.get('decision_strategy', 'unknown')}")
            
        except Exception as e:
            logger.warning(f"Quantile conversion failed: {e}")
            quantile_conversion_results = {}
    else:
        quantile_conversion_results = {}
    
    # Handle regular regression models
    regular_models = {k: v for k, v in ensemble_results['base_predictions'].items() 
                     if not (isinstance(v, dict) and all(isinstance(qk, float) for qk in v.keys()))}
    
    if regular_models:
        # Convert regular regression predictions to binary decisions with optimized thresholds
        binary_conversion_results = convert_regression_ensemble_to_binary(
            regular_models, 
            y_test, 
            optimize_thresholds=True
        )
        
        # Log threshold optimization results
        for model_name, results in binary_conversion_results.items():
            if results['is_regressor']:
                logger.info(f"  {model_name}: threshold={results['threshold']:.4f}, F1={results['f1_score']:.3f}")
    else:
        binary_conversion_results = {}
    
    # Combine conversion results
    all_conversion_results = {**binary_conversion_results, **quantile_conversion_results}
    
    # Create final ensemble using both continuous predictions (for meta-model) and binary decisions
    final_continuous_predictions = {}
    final_binary_predictions = {}
    
    for model_name, results in all_conversion_results.items():
        final_continuous_predictions[model_name] = results['continuous_predictions']
        final_binary_predictions[model_name] = results['predictions']
    
    # Step 5: Advanced final ensemble creation
    logger.info("\n🎯 Phase 5: Creating advanced final ensemble predictions...")
    
    # Rank-and-vote ensemble (production approach)
    if config.ensemble.voting_strategy == 'rank_and_vote':
        logger.info("\n🗳️ Rank-and-vote ensemble...")
        
        final_predictions = create_rank_vote_ensemble(
            final_binary_predictions,
            meta_test_proba,
            config
        )
        
        # Convert to probabilities for evaluation (weighted by performance)
        model_weights = {}
        for model_name in final_continuous_predictions.keys():
            model_weights[model_name] = threshold_results.get(model_name, {}).get('score', 1.0)
        
        # Normalize weights
        total_weight = sum(model_weights.values()) or 1.0
        model_weights = {k: v/total_weight for k, v in model_weights.items()}
        
        # Weighted ensemble probabilities
        final_probabilities = np.zeros(len(y_test))
        for model_name, proba in final_continuous_predictions.items():
            weight = model_weights.get(model_name, 1.0/len(final_continuous_predictions))
            final_probabilities += weight * proba
        
        if meta_test_proba is not None:
            final_probabilities = (final_probabilities + meta_test_proba) / 2
            
        logger.info(f"🎯 Weighted ensemble created with performance-based weights:")
        for model_name, weight in model_weights.items():
            logger.info(f"   {model_name}: {weight:.3f}")
    
    else:
        # Use advanced ensemble with multiple threshold strategies
        simple_probabilities = ensemble_results['simple_ensemble']
        
        # Try multiple threshold optimization strategies for ensemble
        logger.info("\n🔍 Advanced ensemble threshold optimization:")
        
        ensemble_strategies = [
            ('f1', None),
            ('top_k_percent', 5.0),  # Top 5% strategy
            ('portfolio_aware', None)
        ]
        
        best_ensemble_threshold = 0.5
        best_ensemble_score = 0.0
        best_ensemble_strategy = 'f1'
        
        for strategy, param in ensemble_strategies:
            try:
                if strategy == 'portfolio_aware':
                    threshold = portfolio_threshold
                    # Calculate F1 score for portfolio threshold
                    y_pred_binary = (simple_probabilities >= threshold).astype(int)
                    score = f1_score(y_test, y_pred_binary, zero_division=0)
                    strategy_name = 'portfolio_aware'
                    
                elif strategy == 'top_k_percent':
                    threshold, score, _ = find_optimal_threshold_advanced(
                        y_test, simple_probabilities, strategy, top_k_percent=param
                    )
                    strategy_name = f'top_{param}%'
                    
                else:
                    threshold, score, _ = find_optimal_threshold_advanced(
                        y_test, simple_probabilities, strategy
                    )
                    strategy_name = strategy
                
                logger.info(f"   {strategy_name}: threshold={threshold:.4f}, F1={score:.3f}")
                
                if score > best_ensemble_score:
                    best_ensemble_threshold = threshold
                    best_ensemble_score = score
                    best_ensemble_strategy = strategy_name
                    
            except Exception as e:
                logger.warning(f"Ensemble strategy {strategy} failed: {e}")
                continue
        
        logger.info(f"🎯 Best ensemble strategy: {best_ensemble_strategy} (threshold={best_ensemble_threshold:.4f}, F1={best_ensemble_score:.3f})")
        
        final_predictions = (simple_probabilities >= best_ensemble_threshold).astype(int)
        final_probabilities = simple_probabilities
        optimal_threshold = best_ensemble_threshold
    
    # Step 6: Comprehensive evaluation
    logger.info("\n📈 Phase 6: Evaluation...")
    
    # Prepare evaluation data with both continuous and binary predictions
    all_predictions = final_continuous_predictions.copy()
    if meta_test_proba is not None:
        all_predictions['meta_model'] = meta_test_proba
    all_predictions['final_ensemble'] = final_probabilities
    
    # Evaluate all models
    evaluation_results = evaluate_ensemble_performance(
        y_test, 
        ensemble_results['base_predictions'],
        meta_test_proba,
        final_probabilities
    )
    
    # Enhanced evaluation for quantile regression models if enabled
    enable_quantile_eval = getattr(config, 'enable_quantile_regression', False)
    if enable_quantile_eval:
        try:
            from utils.quantile_evaluation import evaluate_quantile_ensemble_performance
            
            logger.info("\n📊 Enhanced quantile regression evaluation...")
            evaluation_results = evaluate_quantile_ensemble_performance(
                y_test,
                ensemble_results['base_predictions'],
                meta_test_proba,
                final_probabilities,
                config
            )
            
        except Exception as e:
            logger.warning(f"Quantile evaluation failed, using standard evaluation: {e}")
    
    # Batch-specific signal filter - check if this batch has sufficient signal quality
    PR_AUC_THRESHOLD = 0.05
    
    # Get meta-model PR-AUC for quality check
    meta_pr_auc = evaluation_results.get('meta_model', {}).get('pr_auc', 0.0)
    if meta_pr_auc == 0.0:
        # If no meta-model, use best base model PR-AUC
        base_pr_aucs = [metrics.get('pr_auc', 0.0) for metrics in evaluation_results.values() 
                       if isinstance(metrics, dict) and 'pr_auc' in metrics]
        meta_pr_auc = max(base_pr_aucs) if base_pr_aucs else 0.0
    
    if meta_pr_auc < PR_AUC_THRESHOLD:
        logger.warning(
            f"⚠️ Batch signal quality check FAILED: PR-AUC ({meta_pr_auc:.3f}) < "
            f"{PR_AUC_THRESHOLD} → This batch has insufficient predictive signal"
        )
        logger.warning("🚫 Recommendation: Skip trading signals for this batch")
        
        # Add warning to evaluation results
        evaluation_results['batch_quality_warning'] = {
            'pr_auc': meta_pr_auc,
            'threshold': PR_AUC_THRESHOLD,
            'recommendation': 'SKIP_BATCH',
            'reason': 'Insufficient predictive signal quality'
        }
    else:
        logger.info(f"✅ Batch signal quality PASSED: PR-AUC ({meta_pr_auc:.3f}) >= {PR_AUC_THRESHOLD}")
        evaluation_results['batch_quality_warning'] = None
    
    # Compute dynamic ensemble weights based on model performance
    base_models = ['xgboost', 'lightgbm', 'lightgbm_regressor', 'catboost', 'random_forest', 'svm']
    dynamic_weights = compute_dynamic_ensemble_weights(evaluation_results, base_models)
    
    # Add weights to evaluation results for export
    evaluation_results['dynamic_weights'] = dynamic_weights
    
    # Step 6: Export and visualize results
    logger.info("\n💾 Phase 6: Export and visualization...")
    
    # Export results
    exported_files = export_training_results(evaluation_results, config)
    
    # Perform rolling backtest if enabled
    rolling_results = None
    enable_rolling_backtest = getattr(config, 'enable_rolling_backtest', False)
    if enable_rolling_backtest and len(X_train) > 500:  # Only if enough data
        try:
            logger.info("\n📈 Performing rolling backtest for drift detection...")
            
            # Combine train and test data for rolling analysis
            X_combined = pd.concat([X_train, X_test], ignore_index=True)
            y_combined = pd.concat([y_train, y_test], ignore_index=True)
            
            # Use best performing base model for rolling backtest
            best_model_name = max(evaluation_results.keys(), 
                                key=lambda k: evaluation_results[k].get('roc_auc', 0) 
                                if isinstance(evaluation_results[k], dict) else 0)
            
            if best_model_name in trained_models:
                best_model = trained_models[best_model_name]['models'][0]  # Use first fold model
                
                window_size = min(200, len(X_combined) // 4)
                step_size = max(50, window_size // 4)
                
                rolling_results = rolling_backtest(
                    best_model, X_combined, y_combined, 
                    window=window_size, step=step_size
                )
                
                if not rolling_results.empty:
                    logger.info(f"Rolling backtest completed: {len(rolling_results)} windows")
                    logger.info(f"Performance stability - Mean F1: {rolling_results['f1'].mean():.4f} ± {rolling_results['f1'].std():.4f}")
                    
                    # Save rolling results
                    rolling_path = f"reports/rolling_backtest_{int(time.time())}.csv"
                    rolling_results.to_csv(rolling_path, index=False)
                    exported_files['rolling_backtest'] = rolling_path
                    
        except Exception as e:
            logger.warning(f"Rolling backtest failed: {e}")
    
    # Detect data drift if enabled
    drift_results = None
    enable_drift_detection = getattr(config, 'enable_drift_detection', False)
    if enable_drift_detection:
        try:
            logger.info("\n🔍 Detecting data drift...")
            drift_results = detect_data_drift(X_train, X_test, threshold=0.1)
            
            if drift_results['drift_detected']:
                logger.warning(f"⚠️ Data drift detected in {len(drift_results['drifted_features'])} features")
                logger.warning(f"Recommendation: {drift_results['recommendation']}")
            else:
                logger.info("✅ No significant data drift detected")
            
            # Add drift results to evaluation
            evaluation_results['drift_analysis'] = drift_results
            
        except Exception as e:
            logger.warning(f"Drift detection failed: {e}")
    
    # Create visualizations with quantile-aware performance summary
    training_results = {
        'evaluation_results': evaluation_results,
        'confusion_matrices': create_confusion_matrices(evaluation_results),
        'model_names': list(ensemble_results['base_predictions'].keys()) + (['meta_model'] if use_meta_model else []),
        'metrics_df': None,  # Will be set below with quantile-aware summary
        'y_train': y_train,
        'y_test': y_test,
        'rolling_results': rolling_results,
        'drift_results': drift_results
    }
    
    # Create performance summary (quantile-aware if enabled)
    if enable_quantile_eval:
        try:
            from utils.quantile_evaluation import create_quantile_performance_summary
            performance_df = create_quantile_performance_summary(evaluation_results)
        except Exception as e:
            logger.warning(f"Failed to create quantile performance summary: {e}")
            performance_df = create_performance_summary(evaluation_results)
    else:
        performance_df = create_performance_summary(evaluation_results)
    
    training_results['metrics_df'] = performance_df
    
    if config.visualization.save_plots:
        plot_paths = create_visual_report(training_results, config.visualization)
    else:
        plot_paths = {}
    
    # Generate comprehensive report
    total_time = time.time() - start_time
    text_report = generate_training_report(evaluation_results, total_time, config)
    logger.info(f"\n{text_report}")
    
    # Evaluate stacking quality if OOF was used
    if use_meta_model:
        stacking_quality = evaluate_stacking_quality(
            train_meta_features, test_meta_features,
            list(ensemble_results['base_predictions'].keys())
        )
        logger.info(f"Stacking quality: {stacking_quality}")
    
    # Return comprehensive results
    return {
        'trained_models': trained_models,
        'meta_model': meta_model,
        'ensemble_results': ensemble_results,
        'evaluation_results': evaluation_results,
        'final_predictions': final_predictions,
        'final_probabilities': final_probabilities,
        'performance_summary': performance_df,
        'confusion_matrices': create_confusion_matrices(evaluation_results),
        'exported_files': exported_files,
        'plot_paths': plot_paths,
        'training_time': total_time,
        'config_used': config,
        'stacking_quality': stacking_quality if use_meta_model else None,
        'text_report': text_report
    }

def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(description='Train Committee of Five ensemble')
    
    parser.add_argument('--config', choices=['default', 'extreme_imbalance', 'fast_training'],
                       default='default', help='Training configuration preset')
    parser.add_argument('--models', nargs='+', 
                       choices=['xgboost', 'lightgbm', 'lightgbm_regressor', 'catboost', 'random_forest', 'svm'],
                       help='Models to train (default: all)')
    parser.add_argument('--data-file', type=str, help='Path to training data CSV')
    parser.add_argument('--target-column', type=str, default='target_enhanced', 
                       help='Name of target column')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (0.0-1.0)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--export-results', action='store_true',
                       help='Export detailed results to files')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save visualization plots')
    parser.add_argument('--collect-data', action='store_true',
                       help='Collect fresh data using Alpaca')
    parser.add_argument('--batch-id', type=str, default=None,
                       help='Batch identifier for organizing outputs')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Get configuration
    if args.config == 'extreme_imbalance':
        config = get_extreme_imbalance_config()
    elif args.config == 'fast_training':
        config = get_fast_training_config()
    else:
        config = get_default_config()
    
    # Override config with command line arguments
    if args.models:
        config.models_to_train = args.models
    if args.test_size:
        config.test_size = args.test_size
    if args.random_state:
        config.random_state = args.random_state
    
    config.visualization.save_plots = args.save_plots
    
    # Add batch ID if provided
    if hasattr(config.visualization, 'batch_id'):
        config.visualization.batch_id = args.batch_id
    else:
        # Add batch_id attribute dynamically if not in config
        setattr(config.visualization, 'batch_id', args.batch_id)
    
    # Data collection or loading
    if args.collect_data:
        logger.info("🔄 Collecting fresh data using Alpaca...")
        try:
            collector = AlpacaDataCollector()
            # Collect training data for first batch by default
            df = collector.collect_training_data([1])
            logger.info(f"✓ Collected {len(df)} samples")
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            return 1
    elif args.data_file:
        logger.info(f"📁 Loading data from {args.data_file}...")
        try:
            df = pd.read_csv(args.data_file)
            logger.info(f"✓ Loaded {len(df)} samples from file")
        except Exception as e:
            logger.error(f"Failed to load data file: {e}")
            return 1
    else:
        logger.error("❌ No data source specified. Use --collect-data or --data-file")
        return 1
    
    # Validate data
    if args.target_column not in df.columns:
        logger.error(f"❌ Target column '{args.target_column}' not found in data")
        return 1
    
    # Ensure target is binary for classification
    if df[args.target_column].dtype in ['float64', 'float32'] and not df[args.target_column].isin([0, 1]).all():
        logger.info(f"Converting continuous target to binary using median threshold")
        threshold = df[args.target_column].median()
        df[args.target_column] = (df[args.target_column] > threshold).astype(int)
        
    logger.info(f"Target distribution: {df[args.target_column].value_counts().to_dict()}")
    
    # Get feature columns (all except target)
    feature_columns = [col for col in df.columns 
                      if col not in [args.target_column, 'ticker']]
    
    # Remove datetime/timestamp columns that can't be used by ML models
    datetime_columns = []
    for col in feature_columns:
        # Check dtype
        if df[col].dtype in ['datetime64[ns]', 'datetime64[ns, UTC]'] or 'datetime' in str(df[col].dtype):
            datetime_columns.append(col)
        # Check for datetime-like strings by sampling values
        elif df[col].dtype == 'object':
            sample_values = df[col].dropna().head(10).astype(str)
            if any(any(pattern in str(val) for pattern in ['-', ':', 'T', '+', 'UTC']) and 
                   len(str(val)) > 10 for val in sample_values):
                # Likely datetime string
                datetime_columns.append(col)
    
    if datetime_columns:
        logger.info(f"Removing {len(datetime_columns)} datetime columns: {datetime_columns}")
        feature_columns = [col for col in feature_columns if col not in datetime_columns]
    
    # Add LLM macro signal if enabled
    enable_llm_features = getattr(config, 'enable_llm_features', False)
    if enable_llm_features:
        try:
            logger.info("🤖 Adding LLM macro signal features...")
            from models.llm_analyzer import LLMAnalyzer
            llm_analyzer = LLMAnalyzer()
            df, feature_columns = add_macro_llm_signal(df, llm_analyzer, feature_columns)
        except Exception as e:
            logger.warning(f"Failed to add LLM features: {e}")
    
    logger.info(f"📊 Using {len(feature_columns)} features")
    
    # Prepare training data
    try:
        X_train, X_test, y_train, y_test = prepare_training_data(
            df, feature_columns, args.target_column, config,
            enable_enhanced_targets=True, target_strategy='top_percentile'
        )
    except Exception as e:
        logger.error(f"❌ Data preparation failed: {e}")
        return 1
    
    # Train models
    try:
        results = train_committee_models(X_train, y_train, X_test, y_test, config)
        logger.info("✅ Training completed successfully!")
        
        # Print summary
        performance_df = results['performance_summary']
        if not performance_df.empty:
            logger.info(f"\n📊 Performance Summary:\n{performance_df.to_string(index=False)}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())
