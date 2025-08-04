#!/usr/bin/env python3
"""
Data Splitting Utilities
========================

Robust data splitting utilities that handle extreme class imbalance
and ensure both classes are present in all splits.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedKFold

from config.training_config import CrossValidationConfig, get_default_config

logger = logging.getLogger(__name__)

def ensure_minority_samples(X: pd.DataFrame, y: pd.Series, 
                          min_samples: int = 2,
                          noise_factor: float = 0.01) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Ensure minority class has at least min_samples by duplicating with noise.
    
    This prevents stratified splitting from failing due to insufficient 
    minority samples and adds diversity through gaussian noise.
    
    Args:
        X: Feature matrix
        y: Target labels
        min_samples: Minimum samples required for minority class
        noise_factor: Standard deviation factor for gaussian noise (default 1%)
        
    Returns:
        Enhanced dataset with sufficient minority samples
    """
    counts = y.value_counts()
    if len(counts) < 2:
        logger.warning(f"Only one class found: {counts.index.tolist()}")
        return X.copy(), y.copy()
    
    minority_class = counts.idxmin()
    minority_count = counts.min()
    
    if minority_count >= min_samples:
        logger.info(f"Minority class {minority_class} has sufficient samples: {minority_count}")
        return X.copy(), y.copy()
    
    needed_samples = min_samples - minority_count
    logger.info(f"Boosting minority class {minority_class} from {minority_count} to {min_samples} samples")
    
    # Get minority samples
    minority_mask = y == minority_class
    minority_X = X[minority_mask].copy()
    minority_y = y[minority_mask].copy()
    
    if len(minority_X) == 0:
        logger.error("No minority samples found for boosting")
        return X.copy(), y.copy()
    
    # Create replicas with small noise for diversity
    replicas_list = []
    replicas_y_list = []
    
    for i in range(needed_samples):
        # Sample with replacement if needed
        sample_idx = i % len(minority_X)
        replica = minority_X.iloc[[sample_idx]].copy()
        
        # Add gaussian noise to numeric columns only
        for col in replica.columns:
            if replica[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                if len(minority_X[col].dropna()) > 0:
                    col_std = minority_X[col].std()
                    if col_std > 0:
                        noise_std = col_std * noise_factor
                        noise = np.random.normal(0, noise_std, size=len(replica))
                        replica[col] = replica[col] + noise
        
        replicas_list.append(replica)
        replicas_y_list.append(pd.Series([minority_class], index=replica.index, name=y.name))
    
    # Combine replicas
    if replicas_list:
        replicas_X = pd.concat(replicas_list, ignore_index=True)
        replicas_y = pd.concat(replicas_y_list, ignore_index=True)
        
        # Combine with original data
        X_enhanced = pd.concat([X, replicas_X], ignore_index=True)
        y_enhanced = pd.concat([y, replicas_y], ignore_index=True)
        
        logger.info(f"Enhanced dataset: {y_enhanced.value_counts().to_dict()}")
        return X_enhanced, y_enhanced
    else:
        return X.copy(), y.copy()

def stratified_train_test_split(X: pd.DataFrame, y: pd.Series,
                               test_size: float = 0.2,
                               random_state: int = 42,
                               min_minority_samples: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Robust stratified train-test split that handles extreme class imbalance.
    
    Args:
        X: Feature matrix
        y: Target labels
        test_size: Proportion of dataset for test set
        random_state: Random state for reproducibility
        min_minority_samples: Minimum minority samples needed for stratification
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info(f"Performing stratified train-test split (test_size={test_size})")
    
    # Check class distribution
    counts = y.value_counts()
    logger.info(f"Original class distribution: {counts.to_dict()}")
    
    if len(counts) < 2:
        logger.error(f"Cannot perform stratified split with only one class: {counts.index.tolist()}")
        # Fall back to random split
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Calculate required samples for stratified split
    minority_count = counts.min()
    test_samples_needed = max(1, int(minority_count * test_size))
    train_samples_needed = minority_count - test_samples_needed
    
    # Ensure we have enough samples for both train and test
    required_total = max(min_minority_samples, test_samples_needed + train_samples_needed + 1)
    
    if minority_count < required_total:
        logger.info(f"Boosting minority samples for stratified split")
        X_boosted, y_boosted = ensure_minority_samples(X, y, min_samples=required_total)
    else:
        X_boosted, y_boosted = X.copy(), y.copy()
    
    # Perform stratified split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_boosted, y_boosted,
            test_size=test_size,
            stratify=y_boosted,
            random_state=random_state
        )
        
        # Verify both classes are present
        train_counts = y_train.value_counts()
        test_counts = y_test.value_counts()
        
        logger.info(f"Train class distribution: {train_counts.to_dict()}")
        logger.info(f"Test class distribution: {test_counts.to_dict()}")
        
        # Check if stratification was successful
        if len(train_counts) < 2 or len(test_counts) < 2:
            logger.warning("Stratification failed - missing classes in splits")
            raise ValueError("Missing classes after stratification")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Stratified split failed: {e}")
        logger.info("Falling back to custom balanced split")
        return _fallback_balanced_split(X_boosted, y_boosted, test_size, random_state)

def _fallback_balanced_split(X: pd.DataFrame, y: pd.Series,
                           test_size: float,
                           random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Fallback split that manually ensures both classes in train and test.
    """
    np.random.seed(random_state)
    
    # Split each class separately
    unique_classes = y.unique()
    train_indices = []
    test_indices = []
    
    for class_label in unique_classes:
        class_indices = y[y == class_label].index.tolist()
        n_test = max(1, int(len(class_indices) * test_size))
        
        # Randomly select test indices for this class
        test_class_indices = np.random.choice(class_indices, size=n_test, replace=False)
        train_class_indices = [idx for idx in class_indices if idx not in test_class_indices]
        
        test_indices.extend(test_class_indices)
        train_indices.extend(train_class_indices)
    
    # Create splits
    X_train = X.loc[train_indices].reset_index(drop=True)
    X_test = X.loc[test_indices].reset_index(drop=True)
    y_train = y.loc[train_indices].reset_index(drop=True)
    y_test = y.loc[test_indices].reset_index(drop=True)
    
    logger.info(f"Fallback split - Train: {y_train.value_counts().to_dict()}")
    logger.info(f"Fallback split - Test: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test

def robust_stratified_kfold(X: pd.DataFrame, y: pd.Series,
                           n_splits: int = 5,
                           random_state: int = 42,
                           min_minority_samples: int = None) -> StratifiedKFold:
    """
    Create a robust StratifiedKFold that handles extreme class imbalance.
    
    Args:
        X: Feature matrix (used for minority boosting if needed)
        y: Target labels
        n_splits: Number of folds desired
        random_state: Random state for reproducibility
        min_minority_samples: Minimum minority samples (defaults to n_splits)
        
    Returns:
        Configured StratifiedKFold object
    """
    if min_minority_samples is None:
        min_minority_samples = n_splits
    
    # Check if we need to boost minority samples
    counts = y.value_counts()
    if len(counts) < 2:
        raise ValueError(f"Cannot create StratifiedKFold with only one class: {counts.index.tolist()}")
    
    minority_count = counts.min()
    
    # Adjust n_splits if necessary
    actual_splits = min(n_splits, minority_count)
    if actual_splits < n_splits:
        logger.warning(f"Reducing n_splits from {n_splits} to {actual_splits} due to minority class size")
    
    # Ensure minimum of 2 splits
    actual_splits = max(2, actual_splits)
    
    logger.info(f"Creating StratifiedKFold with {actual_splits} splits")
    return StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)

def group_aware_split(X: pd.DataFrame, y: pd.Series,
                     groups: pd.Series,
                     test_size: float = 0.2,
                     random_state: int = 42,
                     fallback_to_stratified: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Group-aware splitting (e.g., by ticker symbol) with fallback to stratified.
    
    Args:
        X: Feature matrix
        y: Target labels
        groups: Group labels (e.g., ticker symbols)
        test_size: Proportion for test set
        random_state: Random state
        fallback_to_stratified: Whether to fallback if group split fails
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info(f"Attempting group-aware split with {len(groups.unique())} groups")
    
    try:
        # Use GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))
        
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
        
        # Check if both classes are present
        train_counts = y_train.value_counts()
        test_counts = y_test.value_counts()
        
        if len(train_counts) < 2 or len(test_counts) < 2:
            raise ValueError("Group split resulted in missing classes")
        
        logger.info(f"Group split successful - Train: {train_counts.to_dict()}")
        logger.info(f"Group split successful - Test: {test_counts.to_dict()}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.warning(f"Group-aware split failed: {e}")
        
        if fallback_to_stratified:
            logger.info("Falling back to stratified split")
            return stratified_train_test_split(X, y, test_size, random_state)
        else:
            raise e

def prepare_cv_data(X: pd.DataFrame, y: pd.Series,
                   config: Optional[CrossValidationConfig] = None) -> Tuple[pd.DataFrame, pd.Series, StratifiedKFold]:
    """
    Prepare data for cross-validation with minority boosting if needed.
    
    Args:
        X: Feature matrix
        y: Target labels
        config: Cross-validation configuration
        
    Returns:
        Enhanced X, enhanced y, configured StratifiedKFold
    """
    if config is None:
        config = get_default_config().cross_validation
    
    logger.info(f"Preparing CV data with {config.n_folds} folds")
    
    # Ensure sufficient minority samples
    X_cv, y_cv = ensure_minority_samples(
        X, y, 
        min_samples=config.min_minority_samples
    )
    
    # Create robust StratifiedKFold
    skf = robust_stratified_kfold(
        X_cv, y_cv,
        n_splits=config.n_folds,
        random_state=config.random_state
    )
    
    return X_cv, y_cv, skf

def validate_split_quality(X_train: pd.DataFrame, X_test: pd.DataFrame,
                          y_train: pd.Series, y_test: pd.Series) -> Dict[str, float]:
    """
    Validate the quality of a train-test split.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        
    Returns:
        Dictionary with split quality metrics
    """
    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()
    
    # Calculate imbalance ratios
    train_ratio = train_counts.max() / train_counts.min() if train_counts.min() > 0 else float('inf')
    test_ratio = test_counts.max() / test_counts.min() if test_counts.min() > 0 else float('inf')
    
    # Calculate proportion differences
    train_props = train_counts / len(y_train)
    test_props = test_counts / len(y_test)
    
    # Align indices for comparison
    all_classes = sorted(set(train_props.index) | set(test_props.index))
    train_props = train_props.reindex(all_classes, fill_value=0.0)
    test_props = test_props.reindex(all_classes, fill_value=0.0)
    
    prop_differences = abs(train_props - test_props)
    
    quality_metrics = {
        'train_imbalance_ratio': train_ratio,
        'test_imbalance_ratio': test_ratio,
        'max_proportion_difference': prop_differences.max(),
        'mean_proportion_difference': prop_differences.mean(),
        'train_size': len(y_train),
        'test_size': len(y_test),
        'train_minority_count': train_counts.min(),
        'test_minority_count': test_counts.min(),
        'classes_in_train': len(train_counts),
        'classes_in_test': len(test_counts)
    }
    
    # Log quality assessment
    if quality_metrics['classes_in_train'] < 2 or quality_metrics['classes_in_test'] < 2:
        logger.error("POOR SPLIT: Missing classes in train or test set")
    elif quality_metrics['max_proportion_difference'] > 0.1:
        logger.warning(f"MODERATE SPLIT: Large proportion difference ({quality_metrics['max_proportion_difference']:.3f})")
    else:
        logger.info("GOOD SPLIT: Balanced class proportions maintained")
    
    return quality_metrics
