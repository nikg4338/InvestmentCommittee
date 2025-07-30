#!/usr/bin/env python3
"""
Committee of Five Training Script
=================================

This script implements a "Committee of Five" ensemble for the Investment
Committee project.  It trains five different base models—XGBoost,
LightGBM, CatBoost, Random Forest, and Support Vector Classification—on
the same training data, using the same feature set and train/test
splits.  Each model outputs probabilities on the validation/test set.
Those probabilities are stacked as new features and used to train a
logistic regression meta‑model, which produces the final ensemble
predictions.  Metrics (accuracy, F1 and PR‑AUC) are computed for each
base model and for the stacked ensemble.  Confusion matrices and
bar charts of the metrics are saved to disk for diagnostics.  A
training summary is logged to CSV via the ``utils.training_logger``
module.

To keep the file self‑contained, light utility functions for
balancing datasets and preparing train/test splits are included.
This script does **not** rely on the previous neural network or
neural_predictor components—those references have been removed.

Usage
-----
This file can be used as a module within a larger training pipeline.
To run it as a script, you must first assemble a DataFrame with
engineered features and a target column named ``target``.  Then
invoke ``main()`` with appropriate arguments or adapt the call to
your application.
"""

import logging
import os
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    VISUALIZATION_AVAILABLE = False

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# Oversampling is optional; handle missing imblearn gracefully
try:
    from imblearn.over_sampling import RandomOverSampler
    IMBLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    IMBLEARN_AVAILABLE = False

from utils.helpers import compute_classification_metrics, compute_classification_metrics_with_threshold, find_optimal_threshold
from utils.training_logger import log_training_summary

# Import Alpaca data collection
from data_collection_alpaca import AlpacaDataCollector

# Ensure log directory exists before configuring logging
os.makedirs('logs', exist_ok=True)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Advanced ML imports for class imbalance (after logger is defined)
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN
    ADVANCED_SAMPLING_AVAILABLE = True
except ImportError:
    ADVANCED_SAMPLING_AVAILABLE = False
    logger.warning("Advanced sampling (SMOTE) not available. Install with: pip install imbalanced-learn")

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    logger.warning("Calibration not available in this sklearn version")

# Import model classes
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.catboost_model import CatBoostModel
from models.random_forest_model import RandomForestModel
from models.svc_model import SVMClassifier


def cap_majority_ratio(X: pd.DataFrame, y: pd.Series, max_ratio: float = 2.5) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Cap extreme majority–minority ratios to prevent training instability.

    Args:
        X: Feature matrix (DataFrame)
        y: Target labels (Series)
        max_ratio: Maximum allowed majority:minority ratio

    Returns:
        A tuple ``(X_capped, y_capped)`` with capped ratios.  If the
        existing ratio is below ``max_ratio``, the original data is returned.
    """
    # Convert to DataFrame for easier manipulation if necessary
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df = X.copy()
    df['y'] = y.values if hasattr(y, 'values') else np.array(y)

    n0 = (df.y == 0).sum()
    n1 = (df.y == 1).sum()
    if min(n0, n1) == 0:
        return X, y
    current_ratio = max(n0, n1) / min(n0, n1)
    if current_ratio <= max_ratio:
        return X, y
    # Identify majority and minority classes
    maj, mino = (0, 1) if n0 > n1 else (1, 0)
    keep_majority = int(max_ratio * min(n0, n1))
    df_major = df[df.y == maj].sample(keep_majority, random_state=42)
    df_mino = df[df.y == mino]
    df_new = pd.concat([df_major, df_mino]).sample(frac=1, random_state=42)
    return df_new.drop('y', axis=1), df_new['y'].values


def prepare_balanced_data(X_train: pd.DataFrame, y_train: pd.Series, method: str = 'smote') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Advanced data balancing using multiple techniques for extreme class imbalance.
    
    Args:
        X_train: Training features
        y_train: Training labels  
        method: Balancing method ('smote', 'smoteenn', 'combined', 'basic')
        
    Returns:
        Balanced training data
    """
    logger.info(f"Preparing balanced data using method: {method}")
    
    # Log original distribution
    original_counts = y_train.value_counts().sort_index()
    logger.info(f"Original class distribution: {original_counts.to_dict()}")
    
    if not ADVANCED_SAMPLING_AVAILABLE and method in ['smote', 'smoteenn', 'combined']:
        logger.warning(f"Advanced sampling not available, falling back to basic balancing")
        method = 'basic'
    
    try:
        if method == 'smote':
            # SMOTE with adaptive k-neighbors for small datasets
            minority_count = min(original_counts)
            k_neighbors = min(5, minority_count - 1) if minority_count > 1 else 1
            
            if minority_count < 2:
                logger.warning("Not enough minority samples for SMOTE, using basic oversampling")
                return balance_dataset(X_train, y_train)
            
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
            
            # Convert back to pandas
            X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
            y_balanced = pd.Series(y_balanced, name=y_train.name or 'target')
            
        elif method == 'smoteenn':
            # SMOTE + Edited Nearest Neighbours for combined over/under sampling
            minority_count = min(original_counts)
            k_neighbors = min(5, minority_count - 1) if minority_count > 1 else 1
            
            if minority_count < 2:
                logger.warning("Not enough minority samples for SMOTEENN, using basic oversampling")
                return balance_dataset(X_train, y_train)
            
            smoteenn = SMOTEENN(random_state=42, smote=SMOTE(k_neighbors=k_neighbors))
            X_balanced, y_balanced = smoteenn.fit_resample(X_train, y_train)
            
            # Convert back to pandas
            X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
            y_balanced = pd.Series(y_balanced, name=y_train.name or 'target')
            
        elif method == 'combined':
            # Custom combined approach: Under-sample majority + SMOTE minority
            majority_class = original_counts.idxmax()
            minority_class = original_counts.idxmin()
            majority_count = original_counts[majority_class]
            minority_count = original_counts[minority_class]
            
            if minority_count < 2:
                logger.warning("Not enough minority samples for combined approach, using basic")
                return balance_dataset(X_train, y_train)
            
            # Step 1: Under-sample majority to 3:1 ratio
            target_majority = min(majority_count, minority_count * 3)
            
            majority_indices = y_train[y_train == majority_class].index
            minority_indices = y_train[y_train == minority_class].index
            
            # Random under-sampling of majority
            sampled_majority_indices = np.random.choice(
                majority_indices, size=target_majority, replace=False
            )
            
            # Combine with all minority samples
            combined_indices = np.concatenate([sampled_majority_indices, minority_indices])
            X_intermediate = X_train.loc[combined_indices]
            y_intermediate = y_train.loc[combined_indices]
            
            # Step 2: Apply SMOTE to balance
            k_neighbors = min(5, minority_count - 1)
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_balanced, y_balanced = smote.fit_resample(X_intermediate, y_intermediate)
            
            # Convert back to pandas
            X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
            y_balanced = pd.Series(y_balanced, name=y_train.name or 'target')
            
        else:  # method == 'basic'
            X_balanced, y_balanced = balance_dataset(X_train, y_train)
        
        # Log new distribution
        new_counts = y_balanced.value_counts().sort_index()
        logger.info(f"Balanced class distribution: {new_counts.to_dict()}")
        logger.info(f"Dataset size: {len(X_train)} → {len(X_balanced)} samples")
        
        return X_balanced, y_balanced
        
    except Exception as e:
        logger.error(f"Error in {method} balancing: {e}")
        logger.info("Falling back to basic balancing")
        return balance_dataset(X_train, y_train)


def create_calibrated_model(base_estimator, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 3):
    """
    Create and train a calibrated version of a model using CalibratedClassifierCV.
    
    Args:
        base_estimator: The base estimator to calibrate (sklearn-compatible)
        X_train: Training features
        y_train: Training labels
        cv_folds: Number of CV folds (will be adapted for small datasets)
        
    Returns:
        Trained calibrated model or trained base model if calibration fails
    """
    if not CALIBRATION_AVAILABLE:
        logger.warning(f"Calibration not available, using base model")
        base_estimator.fit(X_train, y_train)
        return base_estimator
    
    try:
        # Adaptive CV folds for small datasets
        adapted_cv = min(cv_folds, 3)  # Use max 3 folds for small datasets
        
        # Use isotonic regression for better performance with limited data
        calibrated_model = CalibratedClassifierCV(
            base_estimator, 
            method='isotonic',  # Better for small datasets than 'sigmoid'
            cv=adapted_cv
        )
        
        # Fit the calibrated model
        calibrated_model.fit(X_train, y_train)
        
        logger.info(f"Created and trained calibrated model with {adapted_cv} CV folds")
        return calibrated_model
        
    except Exception as e:
        logger.warning(f"Failed to calibrate model: {e}")
        # Fall back to base estimator
        base_estimator.fit(X_train, y_train)
        return base_estimator


def balance_dataset(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Balance the dataset with a controlled approach to handle class imbalance.

    The function first caps extreme class ratios, then either applies
    modest oversampling to achieve roughly a 60:40 majority/minority
    split or falls back to a simple ``RandomOverSampler`` when the
    minority class is extremely small.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        ``(X_train_balanced, y_train_balanced)`` – the balanced dataset
    """
    # Log the original distribution
    try:
        counts = y_train.value_counts().to_dict()
        logger.info(f"Original class distribution: {counts}")
    except Exception:
        logger.info("Could not compute original class distribution")
    X_capped, y_capped = cap_majority_ratio(X_train, y_train, max_ratio=2.5)
    # Determine class counts
    class_0 = X_capped[y_capped == 0] if hasattr(X_capped, '__getitem__') else X_train[y_train == 0]
    class_1 = X_capped[y_capped == 1] if hasattr(X_capped, '__getitem__') else X_train[y_train == 1]
    num_class_0 = len(class_0)
    num_class_1 = len(class_1)
    desired_ratio = 0.6  # 60% majority, 40% minority
    # If the minority class is tiny, oversample using RandomOverSampler
    if min(num_class_0, num_class_1) < 100:
        logger.info("Small minority class detected, using oversampling")
        if IMBLEARN_AVAILABLE:
            ros = RandomOverSampler(random_state=42)
            X_bal, y_bal = ros.fit_resample(X_train, y_train)
            return pd.DataFrame(X_bal, columns=X_train.columns), pd.Series(y_bal, name=y_train.name)
        else:
            # Fallback oversampling: duplicate minority class samples to approximate balance
            minority_class = 0 if num_class_0 < num_class_1 else 1
            X_min = X_train[y_train == minority_class]
            y_min = y_train[y_train == minority_class]
            n_samples = abs(num_class_0 - num_class_1)
            idx = np.random.choice(len(X_min), size=n_samples, replace=True)
            X_oversampled = pd.concat([X_train, X_min.iloc[idx]]).reset_index(drop=True)
            y_oversampled = pd.concat([y_train, y_min.iloc[idx]]).reset_index(drop=True)
            return X_oversampled, y_oversampled
    # Controlled balancing
    if num_class_0 > num_class_1:
        # Class 0 is majority
        target_class_0 = min(num_class_0, int(num_class_1 / (1 - desired_ratio) * desired_ratio))
        class_0_down = class_0.sample(target_class_0, random_state=42) if hasattr(class_0, 'sample') else class_0[:target_class_0]
        X_bal = pd.concat([class_1, class_0_down])
        y_bal = [1] * len(class_1) + [0] * len(class_0_down)
    else:
        # Class 1 is majority
        target_class_1 = min(num_class_1, int(num_class_0 / (1 - desired_ratio) * desired_ratio))
        class_1_down = class_1.sample(target_class_1, random_state=42) if hasattr(class_1, 'sample') else class_1[:target_class_1]
        X_bal = pd.concat([class_0, class_1_down])
        y_bal = [0] * len(class_0) + [1] * len(class_1_down)
    y_balanced = pd.Series(y_bal, name=y_train.name if hasattr(y_train, 'name') else 'target')
    # Shuffle the balanced dataset
    X_balanced = X_bal.copy()
    X_balanced['y'] = y_balanced.values
    X_balanced = X_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    y_balanced = X_balanced.pop('y')
    return X_balanced, y_balanced
    """
    Balance the dataset with a controlled approach to handle class imbalance.

    The function first caps extreme class ratios, then either applies
    modest oversampling to achieve roughly a 60:40 majority/minority
    split or falls back to a simple ``RandomOverSampler`` when the
    minority class is extremely small.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        ``(X_train_balanced, y_train_balanced)`` – the balanced dataset
    """
    # Log the original distribution
    try:
        counts = y_train.value_counts().to_dict()
        logger.info(f"Original class distribution: {counts}")
    except Exception:
        logger.info("Could not compute original class distribution")
    X_capped, y_capped = cap_majority_ratio(X_train, y_train, max_ratio=2.5)
    # Determine class counts
    class_0 = X_capped[y_capped == 0] if hasattr(X_capped, '__getitem__') else X_train[y_train == 0]
    class_1 = X_capped[y_capped == 1] if hasattr(X_capped, '__getitem__') else X_train[y_train == 1]
    num_class_0 = len(class_0)
    num_class_1 = len(class_1)
    desired_ratio = 0.6  # 60% majority, 40% minority
    # If the minority class is tiny, oversample using RandomOverSampler
    if min(num_class_0, num_class_1) < 100:
        logger.info("Small minority class detected, using oversampling")
        if IMBLEARN_AVAILABLE:
            ros = RandomOverSampler(random_state=42)
            X_bal, y_bal = ros.fit_resample(X_train, y_train)
            return pd.DataFrame(X_bal, columns=X_train.columns), pd.Series(y_bal, name=y_train.name)
        else:
            # Fallback oversampling: duplicate minority class samples to approximate balance
            minority_class = 0 if num_class_0 < num_class_1 else 1
            X_min = X_train[y_train == minority_class]
            y_min = y_train[y_train == minority_class]
            n_samples = abs(num_class_0 - num_class_1)
            idx = np.random.choice(len(X_min), size=n_samples, replace=True)
            X_oversampled = pd.concat([X_train, X_min.iloc[idx]]).reset_index(drop=True)
            y_oversampled = pd.concat([y_train, y_min.iloc[idx]]).reset_index(drop=True)
            return X_oversampled, y_oversampled
    # Controlled balancing
    if num_class_0 > num_class_1:
        # Class 0 is majority
        target_class_0 = min(num_class_0, int(num_class_1 / (1 - desired_ratio) * desired_ratio))
        class_0_down = class_0.sample(target_class_0, random_state=42) if hasattr(class_0, 'sample') else class_0[:target_class_0]
        X_bal = pd.concat([class_1, class_0_down])
        y_bal = [1] * len(class_1) + [0] * len(class_0_down)
    else:
        # Class 1 is majority
        target_class_1 = min(num_class_1, int(num_class_0 / (1 - desired_ratio) * desired_ratio))
        class_1_down = class_1.sample(target_class_1, random_state=42) if hasattr(class_1, 'sample') else class_1[:target_class_1]
        X_bal = pd.concat([class_0, class_1_down])
        y_bal = [0] * len(class_0) + [1] * len(class_1_down)
    y_balanced = pd.Series(y_bal, name=y_train.name if hasattr(y_train, 'name') else 'target')
    # Shuffle the balanced dataset
    X_balanced = X_bal.copy()
    X_balanced['y'] = y_balanced.values
    X_balanced = X_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    y_balanced = X_balanced.pop('y')
    return X_balanced, y_balanced


def out_of_fold_stacking(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, 
                        models_config: Dict[str, Any], n_folds: int = 3) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Perform out-of-fold stacking to prevent overfitting in meta-model training.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        models_config: Dictionary with model configurations
        n_folds: Number of cross-validation folds
        
    Returns:
        Tuple of (train_meta_features, test_meta_features, trained_models)
    """
    logger.info(f"Starting out-of-fold stacking with {n_folds} folds")
    
    # Adaptive folds for small datasets
    actual_folds = min(n_folds, len(y_train) // 10, 5)  # At least 10 samples per fold
    if len(np.unique(y_train)) > 1:
        # Ensure each fold has both classes
        min_class_count = min(y_train.value_counts())
        actual_folds = min(actual_folds, min_class_count)
    
    actual_folds = max(2, actual_folds)  # At least 2 folds
    logger.info(f"Using {actual_folds} folds for stacking")
    
    # Initialize meta-feature arrays
    n_models = len(models_config)
    train_meta_features = np.zeros((len(X_train), n_models))
    test_meta_features = np.zeros((len(X_test), n_models))
    
    # Cross-validation setup
    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    trained_models = {}
    
    for model_idx, (model_name, model_info) in enumerate(models_config.items()):
        logger.info(f"Training {model_name} with out-of-fold stacking...")
        
        fold_predictions = []
        fold_models = []
        
        # Cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            logger.info(f"  Fold {fold + 1}/{actual_folds}")
            
            # Split data
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            
            # Balance training data for this fold
            balance_method = model_info.get('balance_method', 'smote')
            X_fold_balanced, y_fold_balanced = prepare_balanced_data(X_fold_train, y_fold_train, balance_method)
            
            # Train model
            model_class = model_info['class']
            fold_model = model_class()
            fold_model.train(X_fold_balanced, y_fold_balanced)
            
            # Apply calibration if specified
            if model_info.get('calibrate', False):
                fold_model = create_calibrated_model(fold_model, f"{model_name}_fold_{fold}")
                # Re-fit calibrated model
                if hasattr(fold_model, 'fit'):  # CalibratedClassifierCV
                    fold_model.fit(X_fold_balanced, y_fold_balanced)
            
            # Predict on validation set
            val_proba = fold_model.predict_proba(X_fold_val)
            val_proba_1 = val_proba[:, -1] if val_proba.ndim > 1 else val_proba
            
            # Store predictions for this fold
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
            'config': model_info
        }
        
        logger.info(f"  Completed {model_name} stacking")
    
    logger.info("Out-of-fold stacking completed")
    return train_meta_features, test_meta_features, trained_models


def clean_data_for_ml(X: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the feature matrix by handling infinite and extreme values.

    Infinite values are replaced with NaN.  Numeric columns are clipped
    to the 1st and 99th percentile to mitigate extreme outliers.  The
    function returns a copy of the cleaned feature matrix.
    """
    X_clean = X.copy()
    # Replace infinite values with NaN
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    # Clip extreme values
    for col in X_clean.columns:
        if X_clean[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            q1 = X_clean[col].quantile(0.01)
            q99 = X_clean[col].quantile(0.99)
            X_clean[col] = X_clean[col].clip(lower=q1, upper=q99)
    return X_clean


def prepare_training_data(df: pd.DataFrame, feature_columns: List[str], target_column: str = 'target') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare training and test splits using group‑based splitting by ticker.

    Rows with NaN values in features or target are dropped.  If a
    ``ticker`` column is present, a ``GroupShuffleSplit`` is used to
    prevent leakage across tickers.  Otherwise, a stratified train/test
    split is performed.

    Args:
        df: DataFrame containing features and target
        feature_columns: List of feature column names to use for training
        target_column: Name of the target column (default ``'target'``)

    Returns:
        ``(X_train, X_test, y_train, y_test)`` – the split feature and label sets
    """
    logger.info("Preparing training data with group‑based splitting…")
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    X_clean = clean_data_for_ml(X)
    mask = ~(X_clean.isnull().any(axis=1) | y.isnull())
    X_final = X_clean[mask]
    y_final = y[mask]
    df_final = df[mask]
    
    # Check class distribution
    class_counts = y_final.value_counts()
    logger.info(f"Total class distribution: {class_counts.to_dict()}")
    
    # If we have very few positive examples, use stratified split to ensure both classes in test
    if class_counts.min() < 10:
        logger.info("Few minority samples detected, using stratified split to ensure both classes in test set")
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
        )
    # Group split by ticker if available and we have enough samples per class
    elif 'ticker' in df_final.columns:
        try:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(gss.split(X_final, y_final, groups=df_final['ticker']))
            X_train, X_test = X_final.iloc[train_idx], X_final.iloc[test_idx]
            y_train, y_test = y_final.iloc[train_idx], y_final.iloc[test_idx]
            
            # Check if test set has both classes
            test_class_counts = y_test.value_counts()
            if len(test_class_counts) < 2:
                logger.warning("Group split resulted in test set with only one class, falling back to stratified split")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
                )
        except Exception as e:
            logger.warning(f"Group split failed: {e}, using stratified split")
            X_train, X_test, y_train, y_test = train_test_split(
                X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
            )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
        )
    
    return X_train, X_test, y_train, y_test


def train_committee_models_advanced(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """
    Train Committee of Five with advanced techniques for extreme class imbalance.
    
    Uses sklearn-compatible estimators directly to enable calibration.
    Implements 5 advanced ML techniques:
    1. Threshold optimization per model using F1 metric  
    2. Probability calibration with CalibratedClassifierCV and isotonic regression
    3. SMOTE oversampling with adaptive k-neighbors for small datasets
    4. Out-of-fold stacking with StratifiedKFold to prevent overfitting
    5. Combined under/over-sampling with custom majority class reduction + SMOTE
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Tuple of (models dict, metrics dict) where models contains trained model info
        and metrics contains performance metrics for each model
    """
    logger.info("🚀 Starting Advanced Committee of Five training...")
    logger.info("📊 Implementing: SMOTE, Calibration, Threshold Optimization, Out-of-fold Stacking")
    
    # Import required sklearn-compatible estimators
    try:
        import xgboost as xgb
        import lightgbm as lgb
        from catboost import CatBoostClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
    except ImportError as e:
        logger.error(f"❌ Missing required library: {e}")
        raise
    
    models = {}
    
    # Model configurations with sklearn-compatible estimators
    model_configs = {
        'xgboost': {
            'estimator': xgb.XGBClassifier(
                objective='binary:logistic',
                max_depth=4,
                learning_rate=0.07,
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            ),
            'balance_method': 'smote'
        },
        'lightgbm': {
            'estimator': lgb.LGBMClassifier(
                objective='binary',
                max_depth=4,
                learning_rate=0.07,
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ),
            'balance_method': 'smote'
        },
        'catboost': {
            'estimator': CatBoostClassifier(
                iterations=100,
                depth=4,
                learning_rate=0.07,
                random_seed=42,
                verbose=False,
                allow_writing_files=False
            ),
            'balance_method': 'smoteenn'
        },
        'random_forest': {
            'estimator': RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                random_state=42
            ),
            'balance_method': 'smote'
        },
        'svm': {
            'estimator': SVC(
                probability=True,
                random_state=42,
                kernel='rbf'
            ),
            'balance_method': 'combined'
        }
    }
    
    # Use StratifiedKFold for proper cross-validation with class imbalance
    from sklearn.model_selection import StratifiedKFold
    n_folds = min(3, min(np.bincount(y_train)))  # Adaptive based on minority class
    logger.info(f"Starting out-of-fold stacking with {n_folds} folds")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    logger.info(f"Using {n_folds} folds for stacking")
    
    # Train base models with out-of-fold predictions
    for name, config in model_configs.items():
        logger.info(f"Training {name} with out-of-fold stacking...")
        
        fold_predictions = []
        test_predictions = []
        trained_fold_models = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            logger.info(f"  Fold {fold_idx + 1}/{n_folds}")
            
            # Split fold data
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            # Apply balancing technique
            X_fold_balanced, y_fold_balanced = prepare_balanced_data(
                X_fold_train, y_fold_train, method=config['balance_method']
            )
            
            # Create and train calibrated model
            fold_model = create_calibrated_model(
                config['estimator'], X_fold_balanced, y_fold_balanced
            )
            
            # Get validation predictions
            val_proba = fold_model.predict_proba(X_fold_val)[:, 1]
            fold_predictions.extend(list(zip(val_idx, val_proba)))
            
            # Get test predictions
            test_proba = fold_model.predict_proba(X_test)[:, 1]
            test_predictions.append(test_proba)
            trained_fold_models.append(fold_model)
        
        # Average test predictions across folds
        avg_test_proba = np.mean(test_predictions, axis=0)
        
        # Find optimal threshold using validation predictions
        val_indices, val_probas = zip(*fold_predictions)
        val_y_true = y_train.iloc[list(val_indices)]
        val_y_proba = np.array(val_probas)
        
        optimal_threshold = find_optimal_threshold(val_y_true, val_y_proba, metric='f1')
        
        # Store model information
        models[name] = {
            'trained_models': trained_fold_models,
            'optimal_threshold': optimal_threshold,
            'test_probabilities': avg_test_proba,
            'balance_method': config['balance_method']
        }
        
        logger.info(f"✅ {name} training complete - optimal threshold: {optimal_threshold:.3f}")
    
    # Create meta-features from base model predictions for stacking
    logger.info("🔗 Creating stacked ensemble...")
    
    # Properly reconstruct meta-features - one column per base model
    meta_features_train = []
    for model_name in model_configs.keys():
        model_meta_features = np.zeros(len(X_train))
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            fold_model = models[model_name]['trained_models'][fold_idx]
            val_proba = fold_model.predict_proba(X_train.iloc[val_idx])[:, 1]
            model_meta_features[val_idx] = val_proba
        meta_features_train.append(model_meta_features)
    
    meta_X_train = np.column_stack(meta_features_train)
    
    # Create meta-features for test set
    meta_X_test = np.column_stack([
        models[model_name]['test_probabilities'] 
        for model_name in model_configs.keys()
    ])
    
    # Train meta-model (logistic regression)
    meta_model = LogisticRegression(random_state=42, max_iter=1000)
    meta_model.fit(meta_X_train, y_train)
    
    # Get meta-model predictions and find optimal threshold
    meta_train_proba = meta_model.predict_proba(meta_X_train)[:, 1]
    meta_optimal_threshold = find_optimal_threshold(y_train, meta_train_proba, metric='f1')
    
    meta_test_proba = meta_model.predict_proba(meta_X_test)[:, 1]
    
    # Store stacked model
    models['stacked'] = {
        'meta_model': meta_model,
        'test_meta_features': meta_X_test,
        'optimal_threshold': meta_optimal_threshold,
        'test_probabilities': meta_test_proba
    }
    
    logger.info(f"✅ Stacked ensemble complete - optimal threshold: {meta_optimal_threshold:.3f}")
    
    # Compute metrics for all models
    metrics = {}
    for name, model_info in models.items():
        logger.info(f"📊 Computing metrics for {name}...")
        
        test_proba = model_info['test_probabilities']
        optimal_threshold = model_info['optimal_threshold']
        
        model_metrics = compute_classification_metrics_with_threshold(
            y_test, test_proba, optimal_threshold
        )
        metrics[name] = model_metrics
        
        logger.info(f"✅ {name} - F1: {model_metrics['f1']:.3f}, "
                   f"ROC-AUC: {model_metrics['roc_auc']:.3f}, "
                   f"Accuracy: {model_metrics['accuracy']:.3f}")
    
    # Log summary
    f1_scores = [metrics[m]['f1'] for m in metrics.keys()]
    avg_f1 = np.mean(f1_scores)
    best_f1 = max(f1_scores)
    best_model = max(metrics.keys(), key=lambda k: metrics[k]['f1'])
    
    logger.info(f"🎯 Advanced Training Summary:")
    logger.info(f"   Average F1 Score: {avg_f1:.3f}")
    logger.info(f"   Best F1 Score: {best_f1:.3f} ({best_model})")
    logger.info(f"   Stacked F1 Score: {metrics['stacked']['f1']:.3f}")
    
    return models, metrics


def create_visualizations_advanced(models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, 
                                  metrics: Dict[str, Dict[str, float]], batch_num: Any = None) -> None:
    """
    Generate and save confusion matrices and metric bar charts for advanced models.

    This version handles the new model structure with optimal thresholds and
    out-of-fold trained models.

    Args:
        models: Mapping of model names to model configurations
        X_test: Test features
        y_test: Test labels
        metrics: Nested dictionary of metrics for each model
        batch_num: Optional batch identifier used to name the output directory
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization libraries not available; skipping plots.")
        return
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        # Determine output directory
        if batch_num is not None:
            if isinstance(batch_num, str) and '_' in batch_num:
                batch_parts = batch_num.split('_')
                if len(batch_parts) > 1:
                    batch_dir = f"reports/batch multiple"
                    suffix = f"_batch_multiple"
                else:
                    batch_dir = f"reports/batch {batch_num}"
                    suffix = f"_batch_{batch_num}"
            else:
                batch_dir = f"reports/batch {batch_num}"
                suffix = f"_batch_{batch_num}"
        else:
            batch_dir = "reports"
            suffix = ""
        
        logger.info(f"📊 Creating advanced visualizations in directory: {batch_dir}")
        
        # Recreate directory
        if os.path.exists(batch_dir):
            import shutil
            shutil.rmtree(batch_dir)
            logger.info(f"🗑️  Cleaned existing directory: {batch_dir}")
        
        os.makedirs(batch_dir, exist_ok=True)
        logger.info(f"📁 Created visualization directory: {batch_dir}")
        
        # Generate confusion matrices
        confusion_matrix_count = 0
        for name, model_info in models.items():
            try:
                logger.info(f"📈 Generating confusion matrix for {name}...")
                
                if name == 'stacked':
                    # Handle stacked model predictions
                    test_meta_features = model_info['test_meta_features']
                    meta_model = model_info['meta_model']
                    optimal_threshold = model_info['optimal_threshold']
                    
                    pred_proba = meta_model.predict_proba(test_meta_features)[:, 1]
                    y_pred = (pred_proba >= optimal_threshold).astype(int)
                else:
                    # Handle individual model predictions with optimal threshold
                    optimal_threshold = model_info['optimal_threshold']
                    trained_models = model_info['trained_models']
                    
                    # Average predictions across folds
                    fold_predictions = []
                    for fold_model in trained_models:
                        fold_proba = fold_model.predict_proba(X_test)
                        fold_proba_1 = fold_proba[:, -1] if fold_proba.ndim > 1 else fold_proba
                        fold_predictions.append(fold_proba_1)
                    
                    avg_proba = np.mean(fold_predictions, axis=0)
                    y_pred = (avg_proba >= optimal_threshold).astype(int)
                
                # Create confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                plt.title(f"{name.replace('_', ' ').title()} Confusion Matrix\n(Threshold: {optimal_threshold:.3f})")
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.tight_layout()
                
                # Save the plot
                plot_path = f"{batch_dir}/{name}_confusion_matrix{suffix}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                confusion_matrix_count += 1
                logger.info(f"✅ Saved confusion matrix: {plot_path}")
                
            except Exception as e:
                logger.error(f"❌ Error generating confusion matrix for {name}: {e}")
                continue
        
        logger.info(f"📊 Generated {confusion_matrix_count} confusion matrices")
        
        # Bar charts for metrics (including threshold info)
        metric_names = ['accuracy', 'f1', 'roc_auc']
        metric_chart_count = 0
        
        for metric_name in metric_names:
            try:
                logger.info(f"📊 Generating {metric_name} comparison chart...")
                
                plt.figure(figsize=(8, 5))
                labels = list(metrics.keys())
                values = [metrics[m][metric_name] for m in labels]
                thresholds = [models[m]['optimal_threshold'] for m in labels]
                
                # Create bar plot with threshold annotations
                bars = sns.barplot(x=labels, y=values, palette="viridis")
                plt.ylim(0, 1)
                plt.title(f"Advanced Committee - {metric_name.upper()} Comparison")
                plt.ylabel(metric_name.upper())
                plt.xlabel('Model')
                
                # Add value and threshold labels on bars
                for idx, (val, thresh) in enumerate(zip(values, thresholds)):
                    plt.text(idx, val + 0.02, f"{val:.3f}\n(T:{thresh:.3f})", 
                            ha='center', va='bottom', fontsize=8)
                
                # Rotate x-axis labels if they're long
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save the plot
                plot_path = f"{batch_dir}/{metric_name}_comparison_advanced{suffix}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                metric_chart_count += 1
                logger.info(f"✅ Saved metric chart: {plot_path}")
                
            except Exception as e:
                logger.error(f"❌ Error generating {metric_name} chart: {e}")
                continue
        
        # Create threshold optimization plot
        try:
            logger.info("📊 Generating threshold optimization summary...")
            
            plt.figure(figsize=(10, 6))
            model_names = list(models.keys())
            thresholds = [models[m]['optimal_threshold'] for m in model_names]
            f1_scores = [metrics[m]['f1'] for m in model_names]
            
            # Create subplot for thresholds and F1 scores
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Threshold plot
            bars1 = ax1.bar(model_names, thresholds, color='skyblue')
            ax1.set_title('Optimal Thresholds by Model')
            ax1.set_ylabel('Threshold')
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis='x', rotation=45)
            
            for bar, thresh in zip(bars1, thresholds):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{thresh:.3f}', ha='center', va='bottom')
            
            # F1 score plot
            bars2 = ax2.bar(model_names, f1_scores, color='lightgreen')
            ax2.set_title('F1 Scores with Optimal Thresholds')
            ax2.set_ylabel('F1 Score')
            ax2.set_ylim(0, max(f1_scores) * 1.2 if max(f1_scores) > 0 else 1)
            ax2.tick_params(axis='x', rotation=45)
            
            for bar, f1 in zip(bars2, f1_scores):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{f1:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plot_path = f"{batch_dir}/threshold_optimization_summary{suffix}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Saved threshold optimization summary: {plot_path}")
            metric_chart_count += 1
            
        except Exception as e:
            logger.error(f"❌ Error generating threshold plot: {e}")
        
        logger.info(f"📊 Generated {metric_chart_count} metric comparison charts")
        logger.info(f"🎉 Advanced visualization creation complete! Total images: {confusion_matrix_count + metric_chart_count}")
        logger.info(f"📂 All visualizations saved to: {batch_dir}")
        
    except Exception as e:
        logger.error(f"❌ Critical error in advanced visualization creation: {e}")
        return
    """
    Generate and save confusion matrices and metric bar charts for each model.

    Each model's confusion matrix is saved into the designated batch
    directory.  In addition, three bar charts (accuracy, F1, ROC-AUC) are
    produced across all models for easy comparison.

    Args:
        models: Mapping of model names to model instances (including the meta‑model)
        X_test: Test features
        y_test: Test labels
        metrics: Nested dictionary of metrics for each model
        batch_num: Optional batch identifier used to name the output directory
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization libraries not available; skipping plots.")
        return
    
    try:
        import matplotlib.pyplot as plt  # reimport within function to appease linters
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        # Determine output directory
        if batch_num is not None:
            # Create user-friendly batch directory names
            if isinstance(batch_num, str) and '_' in batch_num:
                # Handle multiple batches like "1_2" -> "batch multiple"
                batch_parts = batch_num.split('_')
                if len(batch_parts) > 1:
                    batch_dir = f"reports/batch multiple"
                    suffix = f"_batch_multiple"
                else:
                    batch_dir = f"reports/batch {batch_num}"
                    suffix = f"_batch_{batch_num}"
            else:
                # Handle single batch like "1" -> "batch 1"
                batch_dir = f"reports/batch {batch_num}"
                suffix = f"_batch_{batch_num}"
        else:
            batch_dir = "reports"
            suffix = ""
        
        logger.info(f"📊 Creating visualizations in directory: {batch_dir}")
        
        # Recreate directory
        if os.path.exists(batch_dir):
            import shutil
            shutil.rmtree(batch_dir)
            logger.info(f"🗑️  Cleaned existing directory: {batch_dir}")
        
        os.makedirs(batch_dir, exist_ok=True)
        logger.info(f"📁 Created visualization directory: {batch_dir}")
        
        # Compute confusion matrices and plot
        confusion_matrix_count = 0
        for name, model in models.items():
            try:
                logger.info(f"📈 Generating confusion matrix for {name}...")
                
                if name == 'stacked':
                    # Recompute stacked probabilities using base models
                    # Create stacked features again from base models
                    base_names = ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'svc']
                    stacked = []
                    for b in base_names:
                        if b in models:
                            proba = models[b].predict_proba(X_test)
                            stacked.append(proba[:, -1] if proba.ndim > 1 else proba)
                        else:
                            logger.warning(f"Model {b} not found for stacking")
                    
                    if stacked:
                        stacked_arr = np.column_stack(stacked)
                        pred_proba = model.predict_proba(stacked_arr)
                        y_pred = (pred_proba[:, -1] >= 0.001).astype(int)  # Extremely low threshold
                    else:
                        logger.error(f"No base models available for stacking {name}")
                        continue
                else:
                    pred_proba = model.predict_proba(X_test)
                    y_pred = (pred_proba[:, -1] >= 0.001).astype(int)  # Extremely low threshold
                
                # Create confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                plt.title(f"{name.replace('_', ' ').title()} Confusion Matrix")
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.tight_layout()
                
                # Save the plot
                plot_path = f"{batch_dir}/{name}_confusion_matrix{suffix}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                confusion_matrix_count += 1
                logger.info(f"✅ Saved confusion matrix: {plot_path}")
                
            except Exception as e:
                logger.error(f"❌ Error generating confusion matrix for {name}: {e}")
                continue
        
        logger.info(f"📊 Generated {confusion_matrix_count} confusion matrices")
        
        # Bar charts for metrics
        metric_names = ['accuracy', 'f1', 'roc_auc']
        metric_chart_count = 0
        
        for metric_name in metric_names:
            try:
                logger.info(f"📊 Generating {metric_name} comparison chart...")
                
                plt.figure(figsize=(6, 4))
                labels = list(metrics.keys())
                values = [metrics[m][metric_name] for m in labels]
                
                # Create bar plot
                bars = sns.barplot(x=labels, y=values, palette="viridis")
                plt.ylim(0, 1)
                plt.title(f"Model {metric_name.upper()} Comparison")
                plt.ylabel(metric_name.upper())
                plt.xlabel('Model')
                
                # Add value labels on bars
                for idx, val in enumerate(values):
                    plt.text(idx, val + 0.02, f"{val:.3f}", ha='center', va='bottom')
                
                # Rotate x-axis labels if they're long
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save the plot
                plot_path = f"{batch_dir}/{metric_name}_comparison{suffix}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                metric_chart_count += 1
                logger.info(f"✅ Saved metric chart: {plot_path}")
                
            except Exception as e:
                logger.error(f"❌ Error generating {metric_name} chart: {e}")
                continue
        
        logger.info(f"📊 Generated {metric_chart_count} metric comparison charts")
        logger.info(f"🎉 Visualization creation complete! Total images: {confusion_matrix_count + metric_chart_count}")
        logger.info(f"📂 All visualizations saved to: {batch_dir}")
        
    except Exception as e:
        logger.error(f"❌ Critical error in visualization creation: {e}")
        return


def collect_alpaca_training_data(batch_numbers: List[int], max_symbols_per_batch: int = 10, use_cached: bool = True) -> pd.DataFrame:
    """
    Collect training data from Alpaca API for specified batches.
    
    Args:
        batch_numbers: List of batch numbers to process
        max_symbols_per_batch: Maximum symbols to process per batch
        use_cached: Whether to use cached data if available
        
    Returns:
        DataFrame with engineered features and targets ready for training
    """
    cache_file = f"alpaca_training_data_batches_{'_'.join(map(str, batch_numbers))}.csv"
    
    # Check for cached data
    if use_cached and os.path.exists(cache_file):
        logger.info(f"Loading cached training data from {cache_file}")
        try:
            df = pd.read_csv(cache_file)
            logger.info(f"✅ Loaded {len(df)} samples from cache")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}, collecting fresh data")
    
    # Collect fresh data from Alpaca
    logger.info(f"Collecting fresh training data from Alpaca API...")
    try:
        collector = AlpacaDataCollector()
        df = collector.collect_training_data(
            batch_numbers=batch_numbers,
            max_symbols_per_batch=max_symbols_per_batch
        )
        
        if len(df) > 0:
            # Cache the data
            df.to_csv(cache_file, index=False)
            logger.info(f"💾 Training data cached to {cache_file}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to collect Alpaca data: {e}")
        raise


def main(batch_numbers: List[int] = None, max_symbols_per_batch: int = 10, use_alpaca: bool = True) -> None:
    """
    Main function for Committee of Five training with Alpaca data integration.
    
    Args:
        batch_numbers: List of batch numbers to process (default: [1, 2])
        max_symbols_per_batch: Maximum symbols per batch (default: 10)
        use_alpaca: Whether to use Alpaca API for data collection (default: True)
    """
    if batch_numbers is None:
        batch_numbers = [1, 2]  # Default to first 2 batches for testing
    
    logger.info(f"🚀 Starting Committee of Five training for batches: {batch_numbers}")
    
    try:
        if use_alpaca:
            # Collect data from Alpaca API
            df = collect_alpaca_training_data(
                batch_numbers=batch_numbers,
                max_symbols_per_batch=max_symbols_per_batch,
                use_cached=True
            )
        else:
            # Fallback to CSV file method
            data_path = os.environ.get('COMMITTEE_DATA_PATH', 'data.csv')
            if not os.path.exists(data_path):
                logger.error(f"Data file '{data_path}' not found. Set COMMITTEE_DATA_PATH to override.")
                return
            df = pd.read_csv(data_path)
        
        if len(df) == 0:
            logger.error("No training data available")
            return
        
        # Assume a 'target' column exists and drop rows without it
        if 'target' not in df.columns:
            logger.error("The training data must contain a 'target' column for labels.")
            return
        
        # Use all columns except the target and ticker as features
        feature_columns = [col for col in df.columns if col not in ['target', 'ticker']]
        logger.info(f"📋 Using {len(feature_columns)} features for training")
        
        # Prepare train/test split
        X_train, X_test, y_train, y_test = prepare_training_data(df, feature_columns, target_column='target')
        
        logger.info(f"📊 Training set: {len(X_train)} samples")
        logger.info(f"📊 Test set: {len(X_test)} samples")
        
        # Train committee models with advanced techniques
        models, metrics = train_committee_models_advanced(X_train, y_train, X_test, y_test)
        
        # Generate visualizations
        if VISUALIZATION_AVAILABLE:
            batch_suffix = '_'.join(map(str, batch_numbers)) if batch_numbers else 'default'
            create_visualizations_advanced(models, X_test, y_test, metrics, batch_num=batch_suffix)
        
        # Log training summary
        log_training_summary(
            batch_number=batch_numbers[0] if batch_numbers else 0,
            symbols_trained=df['ticker'].nunique() if 'ticker' in df.columns else len(df),
            xgb_accuracy=metrics.get('xgboost', {}).get('accuracy', 0.0),
            nn_accuracy=metrics.get('stacked', {}).get('accuracy', 0.0),  # Use meta-model as "NN"
            training_time_seconds=0.0,  # not measured in this simplified example
            timeframe='Alpaca_API_Data' if use_alpaca else 'CSV_Data'
        )
        
        logger.info("🎉 Committee of Five training completed successfully!")
        
        # Print final metrics summary
        logger.info("\n📊 Final Model Performance Summary:")
        for model_name, model_metrics in metrics.items():
            accuracy = model_metrics.get('accuracy', 0)
            f1 = model_metrics.get('f1', 0)
            logger.info(f"   {model_name}: Accuracy={accuracy:.3f}, F1={f1:.3f}")
            
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Committee of Five Training with Alpaca Integration')
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 2], 
                       help='Batch numbers to process (default: 1 2)')
    parser.add_argument('--max-symbols', type=int, default=10,
                       help='Maximum symbols per batch (default: 10)')
    parser.add_argument('--no-alpaca', action='store_true',
                       help='Use CSV file instead of Alpaca API')
    parser.add_argument('--force-fresh', action='store_true',
                       help='Force fresh data collection (ignore cache)')
    
    args = parser.parse_args()
    
    # Override cache setting if force-fresh is specified
    if args.force_fresh:
        # Clear any existing cache files
        import glob
        cache_files = glob.glob("alpaca_training_data_batches_*.csv")
        for cache_file in cache_files:
            try:
                os.remove(cache_file)
                logger.info(f"🗑️  Removed cache file: {cache_file}")
            except Exception as e:
                logger.warning(f"Could not remove cache file {cache_file}: {e}")
    
    main(
        batch_numbers=args.batches,
        max_symbols_per_batch=args.max_symbols,
        use_alpaca=not args.no_alpaca
    )