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

def _ensure_numeric_dataframe(X: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the feature matrix contains only numeric columns suitable for
    resampling libraries (e.g., SMOTE). Non-numeric/object columns are
    dropped; convertible strings are coerced to numeric.

    Returns a new DataFrame with numeric dtypes only.
    """
    if not isinstance(X, pd.DataFrame):
        return X

    X_num = X.copy()

    # Attempt to coerce object columns to numeric; drop those that remain non-numeric
    non_numeric_to_drop = []
    for col in X_num.columns:
        if X_num[col].dtype == 'object':
            coerced = pd.to_numeric(X_num[col], errors='coerce')
            # If coercion results in all NaNs, drop the column (e.g., ticker strings)
            if coerced.notna().sum() == 0:
                non_numeric_to_drop.append(col)
            else:
                X_num[col] = coerced
        elif X_num[col].dtype.name not in ['int64', 'int32', 'float64', 'float32', 'bool']:
            # Coerce any exotic dtypes to float64
            X_num[col] = pd.to_numeric(X_num[col], errors='coerce')

    if non_numeric_to_drop:
        logger.warning(f"Dropping non-numeric columns before resampling: {non_numeric_to_drop}")
        X_num = X_num.drop(columns=non_numeric_to_drop, errors='ignore')

    # Replace inf with NaN, then fill NaNs with column medians (SMOTE cannot handle NaNs)
    X_num = X_num.replace([np.inf, -np.inf], np.nan)
    if not X_num.empty:
        X_num = X_num.apply(lambda s: s.fillna(s.median()) if s.dtype.kind in 'fc' else s)

    return X_num

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
    df = _ensure_numeric_dataframe(X)
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
    
    # Ensure numeric safety for resampling
    X = _ensure_numeric_dataframe(X)

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
    # Ensure numeric safety for resampling
    X = _ensure_numeric_dataframe(X)

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
    # Ensure numeric safety for resampling
    X = _ensure_numeric_dataframe(X)

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
    df = _ensure_numeric_dataframe(X)
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

    # Safety: ensure numeric-only features before any balancing
    X = _ensure_numeric_dataframe(X)
    
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

def apply_smote_for_regression(X: pd.DataFrame, y: pd.Series, 
                              threshold: float = 0.0,
                              k_neighbors: int = 5,
                              sampling_strategy: str = 'auto',
                              random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to regression data by treating positive returns as minority class.
    
    This function converts the regression problem to a temporary classification
    problem, applies SMOTE to generate synthetic positive examples, then converts
    back to regression targets while preserving the continuous nature.
    
    Args:
        X: Feature matrix
        y: Continuous target values (returns)
        threshold: Threshold for defining positive class (default: 0.0)
        k_neighbors: Number of nearest neighbors for SMOTE
        sampling_strategy: SMOTE sampling strategy
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_resampled, y_resampled) with synthetic positive examples
    """
    if not IMBLEARN_AVAILABLE:
        logger.warning("⚠️ imblearn not available. Returning original data.")
        return X.copy(), y.copy()
    
    try:
        # Convert to binary classification for SMOTE
        y_binary = (y > threshold).astype(int)
        
        # Check if we have both classes
        unique_classes = np.unique(y_binary)
        if len(unique_classes) < 2:
            logger.warning(f"Only one class found for SMOTE (threshold={threshold}). Returning original data.")
            return X.copy(), y.copy()
        
        # Get class distribution
        pos_count = np.sum(y_binary == 1)
        neg_count = np.sum(y_binary == 0)
        
        logger.info(f"📊 Pre-SMOTE distribution: {neg_count} negative, {pos_count} positive")
        
        # Adjust k_neighbors for small datasets
        min_class_size = min(pos_count, neg_count)
        actual_k_neighbors = min(k_neighbors, min_class_size - 1)
        
        if actual_k_neighbors < 1:
            logger.warning("⚠️ Not enough samples for SMOTE. Returning original data.")
            return X.copy(), y.copy()
        
        # Determine appropriate sampling strategy
        # For SMOTE, use a dictionary to specify exactly how many samples we want
        if pos_count < neg_count:  # Positive is minority
            minority_class = 1
            majority_count = neg_count
            minority_count = pos_count
        else:  # Negative is minority 
            minority_class = 0
            majority_count = pos_count
            minority_count = neg_count
            
        # SMOTE should only over-sample, never under-sample
        # Target count should be greater than current minority count
        target_count = max(minority_count, int(majority_count * 0.8))
        
        # Ensure we're not trying to under-sample with SMOTE
        if target_count <= minority_count:
            # If target is not larger than current minority, just balance equally
            target_count = majority_count
            
        # Use dictionary strategy for precise control
        sampling_strategy_dict = {minority_class: target_count}
        
        logger.info(f"📊 SMOTE strategy: upsample class {minority_class} to {target_count} samples")
        
        # Apply SMOTE with robust error handling
        try:
            smote = SMOTE(
                k_neighbors=actual_k_neighbors,
                sampling_strategy=sampling_strategy_dict,
                random_state=random_state
            )
            X_resampled, y_binary_resampled = smote.fit_resample(X, y_binary)
            logger.info(f"✅ SMOTE successful: {len(X)} → {len(X_resampled)} samples")
            
        except ValueError as ve:
            logger.warning(f"⚠️ SMOTE ValueError (likely k_neighbors too high): {ve}")
            # Try with fewer neighbors
            if actual_k_neighbors > 1:
                try:
                    smote = SMOTE(
                        k_neighbors=1,
                        sampling_strategy=sampling_strategy_dict,
                        random_state=random_state
                    )
                    X_resampled, y_binary_resampled = smote.fit_resample(X, y_binary)
                    logger.info(f"✅ SMOTE successful with k_neighbors=1: {len(X)} → {len(X_resampled)} samples")
                except Exception as ve2:
                    logger.warning(f"⚠️ SMOTE failed even with k_neighbors=1: {ve2}")
                    raise ve2
            else:
                raise ve
                
        except Exception as smote_error:
            logger.warning(f"⚠️ SMOTE failed with unexpected error: {smote_error}")
            logger.info("🔄 Falling back to simple oversampling...")
            
            # Fall back to simple random oversampling
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(sampling_strategy=sampling_strategy_dict, random_state=random_state)
            try:
                X_resampled, y_binary_resampled = ros.fit_resample(X, y_binary)
                logger.info("✅ Random oversampling successful")
            except Exception as ros_error:
                logger.warning(f"⚠️ Random oversampling also failed: {ros_error}. Returning original data.")
                return X.copy(), y.copy()
        
        # Now reconstruct continuous targets for the resampled data
        # SMOTE reorders and duplicates samples, so we need to handle this carefully
        
        # Create a mapping from the original samples to their continuous values
        original_y_mapping = dict(zip(range(len(y)), y.values))
        
        # Initialize resampled continuous targets 
        y_resampled = np.zeros(len(y_binary_resampled))
        
        # For each resampled sample, assign appropriate continuous value
        n_original = len(y)
        
        for i in range(len(y_binary_resampled)):
            if i < n_original:
                # This might be an original sample (but SMOTE may have reordered)
                # Use the binary class to determine the continuous value strategy
                if y_binary_resampled[i] == 1:  # Positive class
                    # Find a positive value from original data
                    positive_values = y[y > threshold]
                    if len(positive_values) > 0:
                        y_resampled[i] = np.random.choice(positive_values)
                    else:
                        y_resampled[i] = threshold + 0.001
                else:  # Negative class
                    # Find a negative value from original data  
                    negative_values = y[y <= threshold]
                    if len(negative_values) > 0:
                        y_resampled[i] = np.random.choice(negative_values)
                    else:
                        y_resampled[i] = threshold - 0.001
            else:
                # This is definitely a synthetic sample
                if y_binary_resampled[i] == 1:  # Synthetic positive
                    positive_values = y[y > threshold]
                    if len(positive_values) > 0:
                        base_value = np.random.choice(positive_values)
                        noise = np.random.normal(0, positive_values.std() * 0.1)
                        y_resampled[i] = max(threshold + 0.001, base_value + noise)
                    else:
                        y_resampled[i] = threshold + 0.001
                else:  # Synthetic negative
                    negative_values = y[y <= threshold]
                    if len(negative_values) > 0:
                        base_value = np.random.choice(negative_values)
                        noise = np.random.normal(0, negative_values.std() * 0.1)
                        y_resampled[i] = min(threshold - 0.001, base_value + noise)
                    else:
                        y_resampled[i] = threshold - 0.001
        # Convert back to pandas
        X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled_series = pd.Series(y_resampled, name=y.name)
        
        # Log results
        final_pos_count = np.sum(y_resampled > threshold)
        final_neg_count = np.sum(y_resampled <= threshold)
        n_synthetic = len(y_resampled) - len(y)
        
        logger.info(f"📈 Post-SMOTE distribution: {final_neg_count} negative, {final_pos_count} positive")
        logger.info(f"✨ Generated {n_synthetic} synthetic samples")
        logger.info(f"📊 Total samples: {len(y)} → {len(y_resampled)} ({len(y_resampled)/len(y):.2f}x)")
        
        return X_resampled_df, y_resampled_series
        
    except Exception as e:
        logger.error(f"❌ SMOTE failed: {e}")
        return X.copy(), y.copy()

def apply_combined_enhancement(X: pd.DataFrame, y: pd.Series,
                             use_smote: bool = True,
                             positive_weight: float = 10.0,
                             threshold: float = 0.0,
                             config: Optional[DataBalancingConfig] = None) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Apply combined SMOTE + sample weighting enhancement for regression models.
    
    Args:
        X: Feature matrix
        y: Continuous target values
        use_smote: Whether to apply SMOTE upsampling
        positive_weight: Weight multiplier for positive samples
        threshold: Threshold for defining positive class
        config: Data balancing configuration
        
    Returns:
        Tuple of (X_enhanced, y_enhanced, sample_weights)
    """
    if config is None:
        config = get_default_config().data_balancing
    
    X_result, y_result = X.copy(), y.copy()
    
    # Step 1: Apply SMOTE if enabled
    if use_smote and config.use_smote_for_regressors:
        logger.info("🔄 Applying SMOTE for positive example upsampling...")
        X_result, y_result = apply_smote_for_regression(
            X_result, y_result,
            threshold=threshold,
            k_neighbors=config.smote_k_neighbors,
            sampling_strategy=config.smote_sampling_strategy,
            random_state=config.smote_random_state
        )
    
    # Step 2: Create sample weights
    if config.combine_smote_with_weighting:
        sample_weights = np.where(y_result > threshold, positive_weight, 1.0)
        pos_count = np.sum(y_result > threshold)
        total_count = len(y_result)
        logger.info(f"⚖️ Sample weighting: {pos_count}/{total_count} positive samples with {positive_weight}x weight")
    else:
        sample_weights = np.ones(len(y_result))
        logger.info("⚖️ No sample weighting applied")
    
    return X_result, y_result, sample_weights
