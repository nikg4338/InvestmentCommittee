#!/usr/bin/env python3
"""
Sampling Utilities
=================

Advanced sampling techniques for handling extreme class imbalance,
including SMOTE, SMOTEENN, and custom balancing strategies.
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union

# Oversampling is optional; handle missing imblearn gracefully
try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

from config.training_config import DataBalancingConfig, get_default_config

logger = logging.getLogger(__name__)

def cap_majority_ratio(X: pd.DataFrame, y: pd.Series, 
                      max_ratio: float = None) -> Tuple[pd.DataFrame, pd.Series]:
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
    if max_ratio is None:
        max_ratio = get_default_config().data_balancing.max_ratio
    
    # Check class distribution
    counts = y.value_counts()
    if len(counts) < 2:
        logger.warning(f"Only one class found: {counts.index.tolist()}")
        return X.copy(), y.copy()
    
    n0 = counts.get(0, 0)
    n1 = counts.get(1, 0)
    
    if min(n0, n1) == 0:
        logger.warning("One class has zero samples")
        return X.copy(), y.copy()
    
    current_ratio = max(n0, n1) / min(n0, n1)
    logger.info(f"Current class ratio: {current_ratio:.2f}")
    
    if current_ratio <= max_ratio:
        logger.info(f"Ratio within limit ({max_ratio}), no capping needed")
        return X.copy(), y.copy()
    
    # Identify majority and minority classes
    maj_class = 0 if n0 > n1 else 1
    min_class = 1 - maj_class
    
    # Calculate how many majority samples to keep
    minority_count = min(n0, n1)
    keep_majority = int(max_ratio * minority_count)
    
    logger.info(f"Capping majority class {maj_class} from {max(n0, n1)} to {keep_majority} samples")
    
    # Create temporary DataFrame for easier manipulation
    df = X.copy()
    df['target'] = y.values
    
    # Sample majority class
    df_majority = df[df.target == maj_class].sample(keep_majority, random_state=42)
    df_minority = df[df.target == min_class]
    
    # Combine and shuffle
    df_balanced = pd.concat([df_majority, df_minority]).sample(frac=1, random_state=42)
    
    # Split back into X and y
    X_capped = df_balanced.drop('target', axis=1)
    y_capped = df_balanced['target']
    
    new_counts = y_capped.value_counts()
    logger.info(f"After capping: {new_counts.to_dict()}")
    
    return X_capped.reset_index(drop=True), y_capped.reset_index(drop=True)

def basic_oversample(X: pd.DataFrame, y: pd.Series,
                    random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Basic oversampling using RandomOverSampler or manual duplication.
    
    Args:
        X: Feature matrix
        y: Target labels
        random_state: Random state for reproducibility
        
    Returns:
        Oversampled dataset
    """
    logger.info("Applying basic oversampling")
    
    if IMBLEARN_AVAILABLE:
        try:
            ros = RandomOverSampler(random_state=random_state)
            X_resampled, y_resampled = ros.fit_resample(X, y)
            
            # Convert back to pandas
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            y_resampled = pd.Series(y_resampled, name=y.name or 'target')
            
            logger.info(f"RandomOverSampler: {len(X)} → {len(X_resampled)} samples")
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.warning(f"RandomOverSampler failed: {e}. Using manual oversampling.")
    
    # Manual oversampling fallback
    counts = y.value_counts()
    if len(counts) < 2:
        return X.copy(), y.copy()
    
    majority_count = counts.max()
    minority_class = counts.idxmin()
    minority_count = counts.min()
    
    # Get minority samples
    minority_mask = y == minority_class
    X_minority = X[minority_mask]
    y_minority = y[minority_mask]
    
    # Calculate how many duplicates needed
    needed_duplicates = majority_count - minority_count
    
    if needed_duplicates <= 0:
        return X.copy(), y.copy()
    
    # Create duplicates with replacement
    duplicate_indices = np.random.choice(len(X_minority), size=needed_duplicates, replace=True)
    X_duplicates = X_minority.iloc[duplicate_indices].reset_index(drop=True)
    y_duplicates = pd.Series([minority_class] * needed_duplicates, name=y.name)
    
    # Combine original and duplicates
    X_balanced = pd.concat([X, X_duplicates], ignore_index=True)
    y_balanced = pd.concat([y, y_duplicates], ignore_index=True)
    
    logger.info(f"Manual oversampling: {len(X)} → {len(X_balanced)} samples")
    return X_balanced, y_balanced

def smote_oversample(X: pd.DataFrame, y: pd.Series,
                    k_neighbors: int = None,
                    random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    SMOTE oversampling with adaptive k_neighbors for small datasets.
    
    Args:
        X: Feature matrix
        y: Target labels
        k_neighbors: Number of neighbors (auto-adapted if None)
        random_state: Random state for reproducibility
        
    Returns:
        SMOTE oversampled dataset
    """
    if not IMBLEARN_AVAILABLE:
        logger.warning("SMOTE not available, falling back to basic oversampling")
        return basic_oversample(X, y, random_state)
    
    logger.info("Applying SMOTE oversampling")
    
    # Check if we have enough samples
    counts = y.value_counts()
    if len(counts) < 2:
        logger.warning("Only one class found, cannot apply SMOTE")
        return X.copy(), y.copy()
    
    min_count = counts.min()
    if min_count < 2:
        logger.warning(f"Insufficient minority samples for SMOTE (need ≥2, got {min_count})")
        return basic_oversample(X, y, random_state)
    
    # Adaptive k_neighbors
    if k_neighbors is None:
        k_neighbors = get_default_config().data_balancing.smote_k_neighbors
    
    k_neighbors = min(k_neighbors, min_count - 1)
    if k_neighbors < 1:
        k_neighbors = 1
    
    try:
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Convert back to pandas
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name or 'target')
        
        new_counts = y_resampled.value_counts()
        logger.info(f"SMOTE oversampling: {counts.to_dict()} → {new_counts.to_dict()}")
        
        return X_resampled, y_resampled
        
    except Exception as e:
        logger.error(f"SMOTE failed: {e}")
        logger.info("Falling back to basic oversampling")
        return basic_oversample(X, y, random_state)

def smoteenn_resample(X: pd.DataFrame, y: pd.Series,
                     k_neighbors: int = None,
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    SMOTEENN combined over/under sampling.
    
    Args:
        X: Feature matrix
        y: Target labels
        k_neighbors: Number of neighbors for SMOTE (auto-adapted if None)
        random_state: Random state for reproducibility
        
    Returns:
        SMOTEENN resampled dataset
    """
    if not IMBLEARN_AVAILABLE:
        logger.warning("SMOTEENN not available, falling back to SMOTE")
        return smote_oversample(X, y, k_neighbors, random_state)
    
    logger.info("Applying SMOTEENN resampling")
    
    # Check if we have enough samples
    counts = y.value_counts()
    if len(counts) < 2:
        logger.warning("Only one class found, cannot apply SMOTEENN")
        return X.copy(), y.copy()
    
    min_count = counts.min()
    if min_count < 2:
        logger.warning(f"Insufficient minority samples for SMOTEENN (need ≥2, got {min_count})")
        return smote_oversample(X, y, k_neighbors, random_state)
    
    # Adaptive k_neighbors
    if k_neighbors is None:
        k_neighbors = get_default_config().data_balancing.smote_k_neighbors
    
    k_neighbors = min(k_neighbors, min_count - 1)
    if k_neighbors < 1:
        k_neighbors = 1
    
    try:
        smoteenn = SMOTEENN(
            random_state=random_state,
            smote=SMOTE(k_neighbors=k_neighbors)
        )
        X_resampled, y_resampled = smoteenn.fit_resample(X, y)
        
        # Convert back to pandas
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name or 'target')
        
        new_counts = y_resampled.value_counts()
        logger.info(f"SMOTEENN resampling: {counts.to_dict()} → {new_counts.to_dict()}")
        
        return X_resampled, y_resampled
        
    except Exception as e:
        logger.error(f"SMOTEENN failed: {e}")
        logger.info("Falling back to SMOTE")
        return smote_oversample(X, y, k_neighbors, random_state)

def controlled_balance(X: pd.DataFrame, y: pd.Series,
                      desired_ratio: float = None,
                      minority_threshold: int = None,
                      random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Controlled balancing to achieve a target majority:minority ratio.
    
    Args:
        X: Feature matrix
        y: Target labels
        desired_ratio: Target ratio (0.6 = 60% majority, 40% minority)
        minority_threshold: Threshold below which to use oversampling
        random_state: Random state for reproducibility
        
    Returns:
        Balanced dataset
    """
    config = get_default_config().data_balancing
    if desired_ratio is None:
        desired_ratio = config.desired_ratio
    if minority_threshold is None:
        minority_threshold = config.minority_threshold
    
    logger.info(f"Applying controlled balancing (target ratio: {desired_ratio})")
    
    # Check class distribution
    counts = y.value_counts()
    if len(counts) < 2:
        logger.warning("Only one class found, cannot balance")
        return X.copy(), y.copy()
    
    class_0_count = counts.get(0, 0)
    class_1_count = counts.get(1, 0)
    
    if min(class_0_count, class_1_count) == 0:
        logger.warning("One class has zero samples")
        return X.copy(), y.copy()
    
    # If minority class is very small, use oversampling
    min_count = min(class_0_count, class_1_count)
    if min_count < minority_threshold:
        logger.info(f"Small minority class ({min_count} < {minority_threshold}), using oversampling")
        return basic_oversample(X, y, random_state)
    
    # Determine which class is majority
    if class_0_count > class_1_count:
        majority_class, minority_class = 0, 1
        majority_count, minority_count = class_0_count, class_1_count
    else:
        majority_class, minority_class = 1, 0
        majority_count, minority_count = class_1_count, class_0_count
    
    # Calculate target counts based on desired ratio
    # desired_ratio = majority / (majority + minority)
    # So: majority = desired_ratio * total, minority = (1 - desired_ratio) * total
    # We keep minority count fixed and adjust majority
    total_target = minority_count / (1 - desired_ratio)
    majority_target = int(total_target * desired_ratio)
    
    # Don't increase majority count, only decrease
    majority_target = min(majority_target, majority_count)
    
    logger.info(f"Balancing: majority {majority_class} from {majority_count} to {majority_target}")
    
    # Create temporary DataFrame
    df = X.copy()
    df['target'] = y.values
    
    # Sample majority class down
    df_majority = df[df.target == majority_class]
    df_minority = df[df.target == minority_class]
    
    if len(df_majority) > majority_target:
        df_majority = df_majority.sample(majority_target, random_state=random_state)
    
    # Combine and shuffle
    df_balanced = pd.concat([df_majority, df_minority]).sample(frac=1, random_state=random_state)
    
    # Split back into X and y
    X_balanced = df_balanced.drop('target', axis=1)
    y_balanced = df_balanced['target']
    
    new_counts = y_balanced.value_counts()
    logger.info(f"Controlled balancing result: {new_counts.to_dict()}")
    
    return X_balanced.reset_index(drop=True), y_balanced.reset_index(drop=True)

def prepare_balanced_data(X: pd.DataFrame, y: pd.Series, 
                         method: str = 'smote',
                         config: Optional[DataBalancingConfig] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Main function for data balancing using various techniques.
    
    Args:
        X: Feature matrix
        y: Target labels
        method: Balancing method ('basic', 'smote', 'smoteenn', 'controlled', 'cap_only')
        config: Data balancing configuration
        
    Returns:
        Balanced dataset
    """
    if config is None:
        config = get_default_config().data_balancing
    
    logger.info(f"Preparing balanced data using method: {method}")
    
    # Check if we have multiple classes
    counts = y.value_counts()
    if len(counts) < 2:
        logger.warning(f"Only one class found: {counts.index.tolist()}. Returning original data.")
        return X.copy(), y.copy()
    
    logger.info(f"Original class distribution: {counts.to_dict()}")
    
    # Step 1: Cap extreme ratios if needed
    X_capped, y_capped = cap_majority_ratio(X, y, config.max_ratio)
    
    # Step 2: Apply selected balancing method
    if method == 'cap_only':
        return X_capped, y_capped
    elif method == 'basic':
        return basic_oversample(X_capped, y_capped)
    elif method == 'smote':
        return smote_oversample(X_capped, y_capped, config.smote_k_neighbors)
    elif method == 'smoteenn':
        return smoteenn_resample(X_capped, y_capped, config.smote_k_neighbors)
    elif method == 'controlled':
        return controlled_balance(X_capped, y_capped, config.desired_ratio, config.minority_threshold)
    else:
        logger.warning(f"Unknown method '{method}', falling back to 'smote'")
        return smote_oversample(X_capped, y_capped, config.smote_k_neighbors)

def assess_balance_quality(y_original: pd.Series, y_balanced: pd.Series) -> dict:
    """
    Assess the quality of data balancing.
    
    Args:
        y_original: Original labels
        y_balanced: Balanced labels
        
    Returns:
        Dictionary with balance quality metrics
    """
    orig_counts = y_original.value_counts()
    bal_counts = y_balanced.value_counts()
    
    # Calculate imbalance ratios
    orig_ratio = orig_counts.max() / orig_counts.min() if orig_counts.min() > 0 else float('inf')
    bal_ratio = bal_counts.max() / bal_counts.min() if bal_counts.min() > 0 else float('inf')
    
    # Calculate size change
    size_change = len(y_balanced) / len(y_original)
    
    metrics = {
        'original_imbalance_ratio': orig_ratio,
        'balanced_imbalance_ratio': bal_ratio,
        'ratio_improvement': orig_ratio / bal_ratio if bal_ratio > 0 else float('inf'),
        'size_change_factor': size_change,
        'original_size': len(y_original),
        'balanced_size': len(y_balanced),
        'original_distribution': orig_counts.to_dict(),
        'balanced_distribution': bal_counts.to_dict()
    }
    
    logger.info(f"Balance quality - Ratio: {orig_ratio:.2f} → {bal_ratio:.2f}, Size: {size_change:.2f}x")
    
    return metrics
