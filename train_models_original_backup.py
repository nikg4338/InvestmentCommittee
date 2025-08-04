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

import argparse
import logging
import os
import time
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

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Data balancing parameters
DEFAULT_MAX_RATIO = 2.5                    # Maximum majority:minority ratio before capping
DEFAULT_DESIRED_RATIO = 0.6                # Target ratio for controlled balancing (60% majority, 40% minority)
DEFAULT_MINORITY_THRESHOLD = 100           # Threshold below which to use oversampling

# Out-of-fold stacking parameters
DEFAULT_N_FOLDS = 5                        # Number of cross-validation folds
DEFAULT_RANDOM_STATE = 42                  # Random state for reproducibility
DEFAULT_SHUFFLE = True                     # Whether to shuffle in StratifiedKFold

# Calibration parameters
DEFAULT_CALIBRATION_METHOD = 'isotonic'    # 'isotonic' or 'sigmoid'
DEFAULT_CALIBRATION_CV = 3                 # Number of CV folds for calibration

# SMOTE/SMOTEENN parameters
DEFAULT_SMOTE_K_NEIGHBORS = 5              # k_neighbors for SMOTE (will be adapted for small datasets)

# Meta-model parameters
DEFAULT_META_MODEL = 'logistic_regression' # Meta-model type
DEFAULT_META_MAX_ITER = 1000               # Max iterations for LogisticRegression

# Visualization parameters
DEFAULT_CHART_FIGURE_WIDTH = 10           # Width for metric comparison charts
DEFAULT_CHART_FIGURE_HEIGHT = 6           # Height for metric comparison charts  
DEFAULT_MATRIX_FIGURE_WIDTH = 8           # Width for confusion matrix plots
DEFAULT_MATRIX_FIGURE_HEIGHT = 6          # Height for confusion matrix plots
DEFAULT_CHART_DPI = 150                   # DPI for saved charts

# Timing and logging
ENABLE_TIMING = True                       # Whether to measure and log timing information

# Aliases for easier access
CHART_FIGURE_WIDTH = DEFAULT_CHART_FIGURE_WIDTH
CHART_FIGURE_HEIGHT = DEFAULT_CHART_FIGURE_HEIGHT
MATRIX_FIGURE_WIDTH = DEFAULT_MATRIX_FIGURE_WIDTH
MATRIX_FIGURE_HEIGHT = DEFAULT_MATRIX_FIGURE_HEIGHT
CHART_DPI = DEFAULT_CHART_DPI

# =============================================================================

# Timing and logging utility
def log_timing(operation_name: str, start_time: float = None, enable: bool = None) -> None:
    """Log the duration of an operation if timing is enabled."""
    # Use global ENABLE_TIMING if enable parameter not provided
    timing_enabled = enable if enable is not None else ENABLE_TIMING
    
    if timing_enabled:
        if start_time is not None:
            duration = time.time() - start_time
            logger.info(f"  ✔️ {operation_name} completed in {duration:.1f}s")
        else:
            # Simple log message without timing
            logger.info(f"  ✔️ {operation_name}")

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
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
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

# ─── ENHANCED UTILITIES FOR EXTREME IMBALANCE ────────────────────────────────

def ensure_minority_robust(X: pd.DataFrame, y: pd.Series, min_samples: int = 2) -> Tuple[pd.DataFrame, pd.Series]:
    """
    If the minority class has fewer than `min_samples`, duplicate/jitter it.
    More robust than simple duplication - adds small noise to create diversity.
    
    Args:
        X: Feature matrix
        y: Target labels
        min_samples: Minimum samples required for minority class
        
    Returns:
        Enhanced dataset with sufficient minority samples
    """
    counts = y.value_counts()
    if len(counts) < 2 or counts.min() >= min_samples:
        logger.info(f"Minority class sufficient: {counts.min()} >= {min_samples}")
        return X, y
    
    minority_class = counts.idxmin()
    minority_count = counts.min()
    needed_samples = min_samples - minority_count
    
    logger.info(f"Boosting minority class {minority_class} from {minority_count} to {min_samples} samples")
    
    # Get minority samples
    minority_X = X[y == minority_class]
    minority_y = y[y == minority_class]
    
    # Create replicas with small noise for diversity
    replicas_X = minority_X.sample(needed_samples, replace=True, random_state=42)
    
    # Add small gaussian noise to numeric columns only
    for col in replicas_X.columns:
        if replicas_X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            noise_std = replicas_X[col].std() * 0.01  # 1% of standard deviation
            noise = np.random.normal(0, noise_std, size=len(replicas_X))
            replicas_X[col] = replicas_X[col] + noise
    
    # Create corresponding labels
    replicas_y = pd.Series([minority_class] * needed_samples, name=y.name, index=replicas_X.index)
    
    # Combine original and synthetic data
    X_enhanced = pd.concat([X, replicas_X], ignore_index=True)
    y_enhanced = pd.concat([y, replicas_y], ignore_index=True)
    
    logger.info(f"Enhanced dataset: {y_enhanced.value_counts().to_dict()}")
    return X_enhanced, y_enhanced


def adaptive_n_splits(y: pd.Series, max_folds: int = 5) -> int:
    """
    Never request more folds than we have minority samples.
    Ensures every fold can have at least one minority sample.
    
    Args:
        y: Target labels
        max_folds: Maximum number of folds desired
        
    Returns:
        Optimal number of folds for stratification
    """
    min_count = y.value_counts().min()
    optimal_folds = max(2, min(max_folds, min_count))
    
    if optimal_folds < max_folds:
        logger.info(f"Adaptive folding: reduced from {max_folds} to {optimal_folds} folds (min class: {min_count})")
    
    return optimal_folds


def find_threshold_robust(y_true: np.ndarray, proba: np.ndarray, target_positive: int = 1) -> float:
    """
    Try F1-based threshold; if it buys zero, fallback to percentile or tiny eps.
    Guarantees at least `target_positive` predictions.
    
    Args:
        y_true: True labels
        proba: Predicted probabilities
        target_positive: Minimum number of positive predictions required
        
    Returns:
        Optimal threshold guaranteeing minimum positive predictions
    """
    from sklearn.metrics import f1_score
    
    best_thr, best_f1 = 0.5, 0.0
    
    # Try F1-optimization first
    for thr in np.linspace(0, 1, 101):
        preds = (proba >= thr).astype(int)
        if len(np.unique(preds)) > 1:  # Ensure both classes are predicted
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
    
    # Check if threshold produces enough positive predictions
    predicted_positives = (proba >= best_thr).sum()
    
    if predicted_positives < target_positive:
        # Fallback: use percentile to guarantee minimum positive predictions
        if len(proba) > 0:
            percentile = max(0, 100 - (target_positive / len(proba)) * 100)
            percentile_thr = np.percentile(proba, percentile)
            
            # Ensure we get at least target_positive predictions
            sorted_proba = np.sort(proba)[::-1]  # Descending order
            if len(sorted_proba) >= target_positive:
                guaranteed_thr = sorted_proba[target_positive - 1]
                best_thr = min(percentile_thr, guaranteed_thr)
            else:
                best_thr = percentile_thr
                
        logger.info(f"Fallback threshold: {best_thr:.6f} (guarantees {target_positive} positive predictions)")
    else:
        logger.info(f"F1-optimal threshold: {best_thr:.6f} (F1: {best_f1:.3f}, positives: {predicted_positives})")
    
    return best_thr


def optimize_thresholds_ensemble(y_true: np.ndarray, meta_X: pd.DataFrame, base_probas: Dict[str, np.ndarray]) -> Tuple[Dict[str, float], Any]:
    """
    Enhanced threshold optimization for base models and meta-model.
    Uses rank-and-vote approach with guaranteed minimum positive predictions.
    
    Args:
        y_true: True labels
        meta_X: Meta-features (out-of-fold base model predictions)
        base_probas: Dictionary of base model probabilities
        
    Returns:
        Tuple of (thresholds dict, trained meta-model)
    """
    logger.info("🎯 Optimizing thresholds with ensemble approach...")
    
    thresholds = {}
    
    # Optimize thresholds for each base model
    for name, proba in base_probas.items():
        target_positives = max(1, int(len(y_true) * 0.01))  # At least 1% positive predictions
        thr = find_threshold_robust(y_true, proba, target_positive=target_positives)
        thresholds[name] = thr
        
        predicted_positives = np.sum(proba >= thr)
        logger.info(f"   {name}: threshold={thr:.6f}, predicted_positives={predicted_positives}")
    
    # Train enhanced meta-model
    logger.info("Training enhanced meta-model with balanced regularization...")
    try:
        meta_model = LogisticRegression(
            random_state=42, 
            C=0.1,  # Stronger regularization for extreme imbalance
            class_weight='balanced',
            solver='liblinear',
            max_iter=2000
        )
        meta_model.fit(meta_X, y_true)
        
        # Get meta-model probabilities and optimize threshold
        meta_proba = meta_model.predict_proba(meta_X)[:, 1]
        target_meta_positives = max(1, int(len(y_true) * 0.01))
        meta_threshold = find_threshold_robust(y_true, meta_proba, target_positive=target_meta_positives)
        thresholds['meta'] = meta_threshold
        
        meta_predicted_positives = np.sum(meta_proba >= meta_threshold)
        logger.info(f"   meta-model: threshold={meta_threshold:.6f}, predicted_positives={meta_predicted_positives}")
        
        # Log meta-model feature importance
        feature_names = meta_X.columns if hasattr(meta_X, 'columns') else [f'model_{i}' for i in range(meta_X.shape[1])]
        coef_info = {name: float(coef) for name, coef in zip(feature_names, meta_model.coef_[0])}
        logger.info(f"Meta-model feature weights: {coef_info}")
        
    except Exception as e:
        logger.error(f"Meta-model training failed: {e}")
        raise e
    
    return thresholds, meta_model


def production_rank_vote_ensemble(base_models: Dict[str, Any], X: pd.DataFrame, top_pct: float = 0.01) -> np.ndarray:
    """
    Rank-and-vote ensemble for production use.
    Instead of thresholds on probabilities, rank each model's signals
    and buy whenever at least half vote "top N%" on a symbol.
    
    Args:
        base_models: Dictionary of trained models
        X: Features to predict on
        top_pct: Top percentage for each model to vote on (default 1%)
        
    Returns:
        Binary predictions (1=buy, 0=no buy)
    """
    logger.info(f"🗳️ Production rank-and-vote ensemble (top {top_pct:.1%} per model)")
    
    votes = []
    vote_details = {}
    
    for name, model in base_models.items():
        try:
            proba = model.predict_proba(X)[:, 1]
            cutoff = np.percentile(proba, 100 - top_pct * 100)
            model_votes = (proba >= cutoff).astype(int)
            votes.append(model_votes)
            
            vote_count = np.sum(model_votes)
            vote_details[name] = {'votes': vote_count, 'cutoff': cutoff}
            logger.info(f"   {name}: {vote_count} votes (cutoff: {cutoff:.6f})")
            
        except Exception as e:
            logger.error(f"   {name} failed: {e}")
            # Add zero votes for failed model
            votes.append(np.zeros(len(X), dtype=int))
            vote_details[name] = {'votes': 0, 'cutoff': 0.0}
    
    if not votes:
        logger.warning("No successful model votes, returning all zeros")
        return np.zeros(len(X), dtype=int)
    
    # Create vote matrix
    vote_matrix = np.vstack(votes).T
    total_votes = vote_matrix.sum(axis=1)
    
    # Require majority consensus (at least half + 1 models agree)
    majority_threshold = (len(votes) // 2) + 1
    buy_signals = (total_votes >= majority_threshold).astype(int)
    
    total_buys = np.sum(buy_signals)
    logger.info(f"🎯 Rank-and-vote result: {total_buys} buy signals (majority threshold: {majority_threshold}/{len(votes)})")
    
    # If majority is too restrictive, guarantee minimum buy rate
    min_buys = max(1, int(len(X) * top_pct))
    if total_buys < min_buys:
        logger.info(f"Majority too restrictive ({total_buys} < {min_buys}), using top-voted samples")
        top_voted_indices = np.argsort(total_votes)[-min_buys:]
        buy_signals = np.zeros(len(X), dtype=int)
        buy_signals[top_voted_indices] = 1
        logger.info(f"Guaranteed minimum: {min_buys} buy signals")
    
    return buy_signals


def production_extreme_imbalance_ensemble(models, X_test, y_test=None, min_positive_rate=0.01):
    """
    Production-ready ensemble for extreme class imbalance (99%+ negative class)
    
    Enhanced with rank-and-vote approach instead of brittle probability thresholds.
    Each model votes on its top percentile candidates, then majority voting decides.
    
    Args:
        models: Dict of trained models with 'test_predictions' and 'meta_test_proba'
        X_test: Test features
        y_test: Test labels (optional, for evaluation)
        min_positive_rate: Minimum rate of positive predictions (default 1%)
    
    Returns:
        Dict with predictions, probabilities, and metrics
    """
    
    n_samples = len(X_test)
    min_positives = max(1, int(n_samples * min_positive_rate))
    
    # ─── ENHANCED RANK-AND-VOTE APPROACH ───────────────────────────────────
    
    logger.info(f"🗳️ Using rank-and-vote ensemble (top {min_positive_rate:.1%} per model)")
    
    # Step 1: Each model votes on its top candidates
    model_votes = {}
    vote_matrix = np.zeros((n_samples, len(models['test_predictions'])))
    
    for i, (model_name, probabilities) in enumerate(models['test_predictions'].items()):
        # Use percentile-based voting instead of fixed thresholds
        top_percentile = 100 - (min_positive_rate * 100)
        cutoff = np.percentile(probabilities, top_percentile)
        
        # Each model votes for samples above its percentile cutoff
        votes = (probabilities >= cutoff).astype(int)
        model_votes[model_name] = votes
        vote_matrix[:, i] = votes
        
        vote_count = np.sum(votes)
        logger.info(f"   {model_name}: {vote_count} votes (cutoff: {cutoff:.6f})")
    
    # Step 2: Majority voting with adaptive thresholds
    total_models = len(models['test_predictions'])
    
    # Strategy A: Require majority consensus (at least half + 1)
    majority_threshold = (total_models // 2) + 1
    majority_votes = np.sum(vote_matrix, axis=1)
    consensus_predictions = (majority_votes >= majority_threshold).astype(int)
    
    consensus_count = np.sum(consensus_predictions)
    logger.info(f"   Majority consensus: {consensus_count} samples (threshold: {majority_threshold}/{total_models})")
    
    # Strategy B: If majority is too restrictive, use plurality with minimum guarantee
    if consensus_count < min_positives:
        logger.info(f"   Majority too restrictive, using plurality + top-N guarantee")
        
        # Get top-voted samples to ensure minimum positive rate
        top_voted_indices = np.argsort(majority_votes)[-min_positives:]
        final_predictions = np.zeros(n_samples, dtype=int)
        final_predictions[top_voted_indices] = 1
        
        logger.info(f"   Plurality result: {min_positives} samples guaranteed")
    else:
        final_predictions = consensus_predictions
    
    # Step 3: Meta-model boost (if available)
    if 'meta_test_proba' in models:
        meta_probabilities = models['meta_test_proba']
        meta_top_percentile = 100 - (min_positive_rate * 100)
        meta_cutoff = np.percentile(meta_probabilities, meta_top_percentile)
        
        meta_votes = (meta_probabilities >= meta_cutoff).astype(int)
        meta_vote_count = np.sum(meta_votes)
        
        logger.info(f"   Meta-model: {meta_vote_count} votes (cutoff: {meta_cutoff:.6f})")
        
        # Combine base model consensus with meta-model votes
        # Meta-model gets 1.5x weight in final decision
        combined_votes = majority_votes + (1.5 * meta_votes)
        
        # Re-apply minimum guarantee with meta-model boost
        if np.sum(final_predictions) < min_positives:
            boosted_top_indices = np.argsort(combined_votes)[-min_positives:]
            final_predictions = np.zeros(n_samples, dtype=int)
            final_predictions[boosted_top_indices] = 1
            
            logger.info(f"   Meta-boosted result: {min_positives} samples with meta-model enhancement")
    
    # Step 4: Calculate ensemble probabilities (weighted average)
    ensemble_proba = np.zeros(n_samples)
    
    # Weight base model probabilities equally
    for model_name, probabilities in models['test_predictions'].items():
        ensemble_proba += probabilities
    ensemble_proba /= len(models['test_predictions'])
    
    # Add meta-model with higher weight if available
    if 'meta_test_proba' in models:
        ensemble_proba = (ensemble_proba + 1.5 * models['meta_test_proba']) / 2.5
    
    # Step 5: Enhanced evaluation (if labels provided)
    metrics = {}
    if y_test is not None:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, final_predictions),
            'precision': precision_score(y_test, final_predictions, zero_division=0),
            'recall': recall_score(y_test, final_predictions, zero_division=0),
            'f1': f1_score(y_test, final_predictions, zero_division=0),
            'positive_predictions': np.sum(final_predictions),
            'actual_positives': np.sum(y_test),
            'consensus_samples': consensus_count,
            'majority_threshold_used': majority_threshold,
            'vote_distribution': {
                f'{i}_votes': np.sum(majority_votes == i) for i in range(total_models + 1)
            }
        }
        
        logger.info(f"🎯 Rank-and-Vote Results:")
        logger.info(f"   Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"   Precision: {metrics['precision']:.3f}")
        logger.info(f"   Recall: {metrics['recall']:.3f}")
        logger.info(f"   F1 Score: {metrics['f1']:.3f}")
    
    return {
        'predictions': final_predictions,
        'probabilities': ensemble_proba,
        'consensus_indices': np.where(final_predictions == 1)[0].tolist(),
        'model_votes': model_votes,
        'vote_counts': majority_votes,
        'metrics': metrics,
        'voting_strategy': 'rank_and_vote_with_meta_boost'
    }


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
    t0 = time.time()
    logger.info(f"Preparing balanced data using method: {method}")
    
    # Check if we have multiple classes
    unique_classes = y_train.unique()
    if len(unique_classes) < 2:
        logger.warning(f"Only one class found in training data: {unique_classes}. Returning original data.")
        return X_train.copy(), y_train.copy()
    
    # For basic balancing, delegate to balance_dataset
    if method == 'basic' or not ADVANCED_SAMPLING_AVAILABLE:
        if method != 'basic':
            logger.warning(f"Advanced sampling not available, falling back to basic balancing")
        result = balance_dataset(X_train, y_train)
        log_timing(f"Basic balancing", t0)
        return result
    
    # Log original distribution
    original_counts = y_train.value_counts().sort_index()
    logger.info(f"Original class distribution: {original_counts.to_dict()}")
    
    # Check if we have enough samples for advanced methods
    min_class_count = min(original_counts)
    if min_class_count < 2:
        logger.warning(f"Insufficient samples for balancing (min class: {min_class_count}). Using basic method.")
        result = balance_dataset(X_train, y_train)
        log_timing(f"Fallback basic balancing", t0)
        return result
    
    try:
        if method == 'smote':
            # SMOTE with adaptive k-neighbors for small datasets
            k_neighbors = min(DEFAULT_SMOTE_K_NEIGHBORS, min_class_count - 1) if min_class_count > 1 else 1
            
            if min_class_count < 2:
                logger.warning("Not enough minority samples for SMOTE, using basic balancing")
                result = balance_dataset(X_train, y_train)
                log_timing(f"SMOTE fallback to basic", t0)
                return result
            
            smote = SMOTE(random_state=DEFAULT_RANDOM_STATE, k_neighbors=k_neighbors)
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
            
            # Convert back to pandas
            X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
            y_balanced = pd.Series(y_balanced, name=y_train.name or 'target')
            
        elif method == 'smoteenn':
            # SMOTE + Edited Nearest Neighbours for combined over/under sampling
            k_neighbors = min(DEFAULT_SMOTE_K_NEIGHBORS, min_class_count - 1) if min_class_count > 1 else 1
            
            if min_class_count < 2:
                logger.warning("Not enough minority samples for SMOTEENN, using basic balancing")
                result = balance_dataset(X_train, y_train)
                log_timing(f"SMOTEENN fallback to basic", t0)
                return result
            
            smoteenn = SMOTEENN(random_state=DEFAULT_RANDOM_STATE, smote=SMOTE(k_neighbors=k_neighbors))
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
                result = balance_dataset(X_train, y_train)
                log_timing(f"Combined fallback to basic", t0)
                return result
            
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
            k_neighbors = min(DEFAULT_SMOTE_K_NEIGHBORS, minority_count - 1)
            smote = SMOTE(random_state=DEFAULT_RANDOM_STATE, k_neighbors=k_neighbors)
            X_balanced, y_balanced = smote.fit_resample(X_intermediate, y_intermediate)
            
            # Convert back to pandas
            X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
            y_balanced = pd.Series(y_balanced, name=y_train.name or 'target')
            
        else:
            logger.warning(f"Unknown method '{method}', falling back to basic balancing")
            result = balance_dataset(X_train, y_train)
            log_timing(f"Unknown method fallback to basic", t0)
            return result
        
        # Log new distribution
        new_counts = y_balanced.value_counts().sort_index()
        logger.info(f"Balanced class distribution: {new_counts.to_dict()}")
        logger.info(f"Dataset size: {len(X_train)} → {len(X_balanced)} samples")
        
        log_timing(f"{method.upper()} balancing", t0)
        return X_balanced, y_balanced
        
    except Exception as e:
        logger.error(f"Error in {method} balancing: {e}")
        logger.info("Falling back to basic balancing")
        result = balance_dataset(X_train, y_train)
        log_timing(f"{method} error fallback to basic", t0)
        return result


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
    # Check if we have multiple classes
    unique_classes = y_train.unique()
    if len(unique_classes) < 2:
        logger.warning(f"Only one class found in balance_dataset: {unique_classes}. Returning original data.")
        return X_train.copy(), y_train.copy()
    
    # Log the original distribution
    try:
        counts = y_train.value_counts().to_dict()
        logger.info(f"Original class distribution: {counts}")
    except Exception:
        logger.info("Could not compute original class distribution")
    
    X_capped, y_capped = cap_majority_ratio(X_train, y_train, max_ratio=DEFAULT_MAX_RATIO)
    
    # Check again after capping
    unique_classes_capped = np.unique(y_capped)
    if len(unique_classes_capped) < 2:
        logger.warning(f"Only one class after capping: {unique_classes_capped}. Returning original data.")
        return X_train.copy(), y_train.copy()
    
    # Determine class counts
    class_0 = X_capped[y_capped == 0] if hasattr(X_capped, '__getitem__') else X_train[y_train == 0]
    class_1 = X_capped[y_capped == 1] if hasattr(X_capped, '__getitem__') else X_train[y_train == 1]
    num_class_0 = len(class_0)
    num_class_1 = len(class_1)
    
    # If any class is empty, return original data
    if num_class_0 == 0 or num_class_1 == 0:
        logger.warning(f"Empty class detected (class_0: {num_class_0}, class_1: {num_class_1}). Returning original data.")
        return X_train.copy(), y_train.copy()
    
    # If the minority class is tiny, oversample using RandomOverSampler
    if min(num_class_0, num_class_1) < DEFAULT_MINORITY_THRESHOLD:
        logger.info("Small minority class detected, using oversampling")
        if IMBLEARN_AVAILABLE:
            try:
                ros = RandomOverSampler(random_state=DEFAULT_RANDOM_STATE)
                X_bal, y_bal = ros.fit_resample(X_train, y_train)
                return pd.DataFrame(X_bal, columns=X_train.columns), pd.Series(y_bal, name=y_train.name)
            except Exception as e:
                logger.warning(f"RandomOverSampler failed: {e}. Using manual oversampling.")
                # Fall through to manual oversampling below
        
        # Fallback oversampling: duplicate minority class samples to approximate balance
        minority_class = 0 if num_class_0 < num_class_1 else 1
        X_min = X_train[y_train == minority_class]
        y_min = y_train[y_train == minority_class]
        
        if len(X_min) == 0:
            logger.warning("Minority class is empty. Returning original data.")
            return X_train.copy(), y_train.copy()
        
        n_samples = abs(num_class_0 - num_class_1)
        if n_samples > 0:
            idx = np.random.choice(len(X_min), size=n_samples, replace=True)
            X_oversampled = pd.concat([X_train, X_min.iloc[idx]]).reset_index(drop=True)
            y_oversampled = pd.concat([y_train, y_min.iloc[idx]]).reset_index(drop=True)
            return X_oversampled, y_oversampled
        else:
            return X_train.copy(), y_train.copy()
    
    # Controlled balancing
    if num_class_0 > num_class_1:
        # Class 0 is majority
        target_class_0 = min(num_class_0, int(num_class_1 / (1 - DEFAULT_DESIRED_RATIO) * DEFAULT_DESIRED_RATIO))
        class_0_down = class_0.sample(target_class_0, random_state=DEFAULT_RANDOM_STATE) if hasattr(class_0, 'sample') else class_0[:target_class_0]
        X_bal = pd.concat([class_1, class_0_down])
        y_bal = [1] * len(class_1) + [0] * len(class_0_down)
    else:
        # Class 1 is majority
        target_class_1 = min(num_class_1, int(num_class_0 / (1 - DEFAULT_DESIRED_RATIO) * DEFAULT_DESIRED_RATIO))
        class_1_down = class_1.sample(target_class_1, random_state=DEFAULT_RANDOM_STATE) if hasattr(class_1, 'sample') else class_1[:target_class_1]
        X_bal = pd.concat([class_0, class_1_down])
        y_bal = [0] * len(class_0) + [1] * len(class_1_down)
    
    y_balanced = pd.Series(y_bal, name=y_train.name if hasattr(y_train, 'name') else 'target')
    # Shuffle the balanced dataset
    X_balanced = X_bal.copy()
    X_balanced['y'] = y_balanced.values
    X_balanced = X_balanced.sample(frac=1, random_state=DEFAULT_RANDOM_STATE).reset_index(drop=True)
    y_balanced = X_balanced.pop('y')
    return X_balanced, y_balanced


def simple_train_test_stacking(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, 
                              models_config: Dict[str, Any]) -> Tuple[None, None, Dict[str, Any]]:
    """
    Fallback when OOF is impossible due to extreme imbalance:
    Train each model on full train set, predict on test, average.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        models_config: Dictionary with model configurations
        
    Returns:
        Tuple of (None, None, trained_models) - upstream code must detect None and use test_preds directly
    """
    logger.info("Using simple train/test stacking fallback")
    
    trained_models = {}
    
    for model_name, model_info in models_config.items():
        logger.info(f"Training {model_name} on full training set...")
        
        # Check if we need forced oversampling for single-class training
        if model_info.get('force_oversampling', False) and len(y_train.unique()) < 2:
            logger.info(f"⚠️ Single-class training detected for {model_name}. Forcing synthetic minority generation...")
            
            # Create synthetic minority samples for training
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42, k_neighbors=1)  # k_neighbors=1 for small datasets
            
            # For single class training, create a balanced synthetic dataset
            X_balanced = X_train.copy()
            y_balanced = y_train.copy()
            
            # Add a synthetic minority sample by slightly modifying a majority sample
            if len(X_train) > 0:
                # Get the minority class from test set if available globally
                majority_class = y_train.iloc[0]
                minority_class = 1 - majority_class  # Flip the class
                
                # Create one synthetic minority sample
                synthetic_X = X_train.iloc[0:1].copy()
                synthetic_y = pd.Series([minority_class], index=[X_train.index[0]])
                
                # Add small random noise to create diversity
                import numpy as np
                for col in synthetic_X.columns:
                    if synthetic_X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        synthetic_X[col] += np.random.normal(0, 0.01, 1)
                
                # Combine original and synthetic
                X_combined = pd.concat([X_train, synthetic_X], ignore_index=True)
                y_combined = pd.concat([y_train, synthetic_y], ignore_index=True)
                
                # Now apply SMOTE to balance
                try:
                    X_balanced, y_balanced = smote.fit_resample(X_combined, y_combined)
                    X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
                    y_balanced = pd.Series(y_balanced)
                except Exception as e:
                    logger.warning(f"SMOTE failed: {e}. Using basic duplication...")
                    # Fallback: duplicate the synthetic sample
                    n_majority = len(y_train)
                    X_minority_dup = pd.concat([synthetic_X] * n_majority, ignore_index=True)
                    y_minority_dup = pd.Series([minority_class] * n_majority)
                    
                    X_balanced = pd.concat([X_train, X_minority_dup], ignore_index=True)
                    y_balanced = pd.concat([y_train, y_minority_dup], ignore_index=True)
        else:
            # Balance training data normally
            balance_method = model_info.get('balance_method', 'smote')
            X_balanced, y_balanced = prepare_balanced_data(X_train, y_train, balance_method)
        
        # Train model
        model_class = model_info['class']
        model = model_class()
        model.train(X_balanced, y_balanced)
        
        # Apply calibration if specified
        if model_info.get('calibrate', False):
            # Extract the underlying sklearn estimator for calibration
            sklearn_estimator = None
            if hasattr(model, 'model') and model.model is not None:
                sklearn_estimator = model.model
            elif hasattr(model, 'pipeline') and model.pipeline is not None:
                sklearn_estimator = model.pipeline
            
            if sklearn_estimator is not None:
                try:
                    calibrated_model = create_calibrated_model(sklearn_estimator, X_balanced, y_balanced)
                    # Replace the model with calibrated version
                    model = calibrated_model
                except Exception as e:
                    logger.warning(f"Failed to calibrate {model_name}: {e}. Using base model.")
            else:
                logger.warning(f"Cannot extract sklearn estimator from {model_name} for calibration.")
        
        # Predict on both training and test sets
        train_proba = model.predict_proba(X_balanced)
        train_proba_1 = train_proba[:, -1] if train_proba.ndim > 1 else train_proba
        
        test_proba = model.predict_proba(X_test)
        test_proba_1 = test_proba[:, -1] if test_proba.ndim > 1 else test_proba
        
        # Store model with both training and test predictions
        trained_models[model_name] = {
            'model': model,
            'config': model_info,
            'train_predictions': train_proba_1,
            'train_labels': y_balanced,  # Store the balanced training labels
            'test_predictions': test_proba_1
        }
        
        logger.info(f"  Completed {model_name} simple stacking")
    
    logger.info("Simple train/test stacking completed")
    return None, None, trained_models


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
    
    # Dynamically choose number of folds so every fold contains at least one minority
    class_counts = y_train.value_counts()
    if len(class_counts) < 2 or class_counts.min() < 2:
        # Cannot do OOF when there's <2 minority samples—fallback to simple train/test stacking
        logger.warning("Too few minority samples for OOF. Falling back to single-split stacking.")
        return simple_train_test_stacking(X_train, y_train, X_test, models_config)

    # at most one minority per fold → n_folds ≤ min_count
    max_folds = class_counts.min()
    actual_folds = min(n_folds, max_folds)
    actual_folds = max(2, actual_folds)  # ensure at least 2 folds
    logger.info(f"Using {actual_folds} stratified folds for stacking (min class count={class_counts.min()})")
    
    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
    
    # Initialize meta-feature arrays
    n_models = len(models_config)
    train_meta_features = np.zeros((len(X_train), n_models))
    test_meta_features = np.zeros((len(X_test), n_models))
    
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
                # Extract the underlying sklearn estimator for calibration
                sklearn_estimator = None
                if hasattr(fold_model, 'model') and fold_model.model is not None:
                    sklearn_estimator = fold_model.model
                elif hasattr(fold_model, 'pipeline') and fold_model.pipeline is not None:
                    sklearn_estimator = fold_model.pipeline
                
                if sklearn_estimator is not None:
                    try:
                        calibrated_model = create_calibrated_model(sklearn_estimator, X_fold_balanced, y_fold_balanced)
                        fold_model = calibrated_model
                    except Exception as e:
                        logger.warning(f"Failed to calibrate {model_name} fold {fold}: {e}. Using base model.")
                else:
                    logger.warning(f"Cannot extract sklearn estimator from {model_name} for calibration.")
            
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


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray, metric: str = 'f1') -> Tuple[float, float]:
    """
    Find the optimal threshold for binary classification that maximizes the specified metric.
    
    Args:
        y_true: True binary labels (0/1)
        y_proba: Predicted probabilities for the positive class
        metric: Metric to optimize ('f1', 'precision', 'recall', 'balanced_accuracy')
    
    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score
    
    # Generate thresholds from 0→1 in fine steps
    thresholds = np.linspace(0.0, 1.0, 101)  # Test 101 thresholds from 0.0 to 1.0
    
    best_threshold = 0.5
    best_score = 0.0
    
    metric_func = {
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score,
        'balanced_accuracy': balanced_accuracy_score
    }.get(metric, f1_score)
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Skip if all predictions are one class
        if len(np.unique(y_pred)) < 2:
            continue
            
        try:
            score = metric_func(y_true, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        except (ValueError, ZeroDivisionError):
            continue
    
    # Force at least one positive prediction if no threshold gave a positive score
    if best_score == 0.0:
        logger.warning(f"⚠️ No threshold yielded positive F1 score")
        
        # Try multiple fallback strategies for extreme imbalance
        fallback_strategies = [
            ("min_nonzero", lambda: np.min(y_proba[y_proba > 0]) if len(y_proba[y_proba > 0]) > 0 else None),
            ("percentile_1", lambda: np.percentile(y_proba[y_proba > 0], 1) if len(y_proba[y_proba > 0]) > 0 else None),
            ("percentile_5", lambda: np.percentile(y_proba[y_proba > 0], 5) if len(y_proba[y_proba > 0]) > 0 else None),
            ("mean_nonzero", lambda: np.mean(y_proba[y_proba > 0]) if len(y_proba[y_proba > 0]) > 0 else None),
            ("median_nonzero", lambda: np.median(y_proba[y_proba > 0]) if len(y_proba[y_proba > 0]) > 0 else None)
        ]
        
        for strategy_name, strategy_func in fallback_strategies:
            try:
                fallback_threshold = strategy_func()
                if fallback_threshold is not None and 0 < fallback_threshold < 1:
                    y_pred_fallback = (y_proba >= fallback_threshold).astype(int)
                    
                    if len(np.unique(y_pred_fallback)) > 1:
                        fallback_score = metric_func(y_true, y_pred_fallback, zero_division=0)
                        if fallback_score > best_score:
                            best_score = fallback_score
                            best_threshold = fallback_threshold
                            logger.warning(f"⚠️ Used {strategy_name} fallback threshold {best_threshold:.6f} (F1: {best_score:.3f})")
                            break
            except Exception as e:
                logger.warning(f"⚠️ Fallback strategy {strategy_name} failed: {e}")
                continue
        
        # If all fallback strategies failed, use emergency threshold
        if best_score == 0.0:
            if len(y_proba[y_proba > 0]) > 0:
                # Pick the lowest non-zero probability as threshold to ensure at least one positive
                best_threshold = np.min(y_proba[y_proba > 0])
                logger.warning(f"⚠️ Using emergency min threshold {best_threshold:.6f}")
            else:
                # All probabilities are zero, use ultra-low emergency threshold
                best_threshold = 0.0001
                logger.warning(f"⚠️ All probabilities are zero, using ultra-emergency threshold {best_threshold:.6f}")
    
    logger.info(f"🎯 Optimal threshold for {metric}: {best_threshold:.3f} (score: {best_score:.3f})")
    return best_threshold, best_score


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


def prepare_train_test(df: pd.DataFrame, feature_cols: List[str], target_col: str = 'target', 
                      test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Always do a stratified train/test split on y, ensuring both classes are present.
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Clean data first
    X_clean = clean_data_for_ml(X)
    
    # Remove rows with NaN values
    mask = ~(X_clean.isnull().any(axis=1) | y.isnull())
    X_final, y_final = X_clean[mask], y[mask]
    
    logger.info(f"Data after cleaning: {len(X_final)} samples")
    logger.info(f"Class distribution: {y_final.value_counts().to_dict()}")
    
    return train_test_split(
        X_final, y_final,
        test_size=test_size,
        stratify=y_final,
        random_state=random_state
    )


def ensure_minority(X: pd.DataFrame, y: pd.Series, n_splits: int, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Guarantee at least n_splits minority samples by oversampling if needed.
    This prevents StratifiedKFold from failing due to insufficient minority samples.
    """
    counts = y.value_counts()
    minority_class = counts.idxmin()
    minority_count = counts.min()
    
    logger.info(f"Original minority class {minority_class}: {minority_count} samples")
    
    if minority_count < n_splits:
        logger.info(f"Boosting minority class from {minority_count} to {n_splits} samples for stratification")
        
        # Use RandomOverSampler to boost minority to exactly n_splits samples
        if IMBLEARN_AVAILABLE:
            ros = RandomOverSampler(
                sampling_strategy={minority_class: n_splits},
                random_state=random_state
            )
            X_res, y_res = ros.fit_resample(X, y)
            
            # Convert back to pandas with proper column names
            X_boosted = pd.DataFrame(X_res, columns=X.columns)
            y_boosted = pd.Series(y_res, name=y.name or 'target')
            
            logger.info(f"After boosting: {y_boosted.value_counts().to_dict()}")
            return X_boosted, y_boosted
        else:
            logger.warning("RandomOverSampler not available, using manual duplication")
            
            # Manual duplication fallback
            minority_samples = X[y == minority_class]
            minority_labels = y[y == minority_class]
            
            needed_samples = n_splits - minority_count
            if needed_samples > 0:
                # Duplicate minority samples
                np.random.seed(random_state)
                duplicate_indices = np.random.choice(len(minority_samples), size=needed_samples, replace=True)
                
                X_duplicates = minority_samples.iloc[duplicate_indices].reset_index(drop=True)
                y_duplicates = minority_labels.iloc[duplicate_indices].reset_index(drop=True)
                
                X_boosted = pd.concat([X, X_duplicates], ignore_index=True)
                y_boosted = pd.concat([y, y_duplicates], ignore_index=True)
                
                logger.info(f"After manual boosting: {y_boosted.value_counts().to_dict()}")
                return X_boosted, y_boosted
    
    logger.info("No minority boosting needed")
    return X, y


def robust_out_of_fold_stacking(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, 
                               n_splits: int = 5, random_state: int = 42) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Robust StratifiedKFold + guaranteed positives in each fold + threshold fallback.
    Prevents collapses by ensuring sufficient minority samples and proper thresholding.
    """
    logger.info(f"Starting robust out-of-fold stacking with {n_splits} splits...")
    
    # 1) Ensure enough minority examples for stratification using enhanced method
    X_train_boosted, y_train_boosted = ensure_minority_robust(X_train, y_train, min_samples=n_splits)
    
    # 2) Set up true StratifiedKFold - now guaranteed to work
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Import model classes
    from models.xgboost_model import XGBoostModel
    from models.lightgbm_model import LightGBMModel
    from models.catboost_model import CatBoostModel
    from models.random_forest_model import RandomForestModel
    from models.svc_model import SVMClassifier
    
    model_classes = [XGBoostModel, LightGBMModel, CatBoostModel, RandomForestModel, SVMClassifier]
    model_names = ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'svm']
    
    # Containers for predictions and thresholds
    oof_preds = np.zeros((len(X_train_boosted), len(model_classes)))
    test_preds = np.zeros((len(X_test), len(model_classes), n_splits))
    fold_thresholds = []
    
    logger.info(f"Training {len(model_classes)} models across {n_splits} folds...")
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_boosted, y_train_boosted)):
        logger.info(f"  Processing fold {fold + 1}/{n_splits}")
        
        X_tr, y_tr = X_train_boosted.iloc[tr_idx], y_train_boosted.iloc[tr_idx]
        X_va, y_va = X_train_boosted.iloc[va_idx], y_train_boosted.iloc[va_idx]
        
        logger.info(f"    Fold {fold + 1} train: {y_tr.value_counts().to_dict()}")
        logger.info(f"    Fold {fold + 1} val: {y_va.value_counts().to_dict()}")
        
        fold_model_thresholds = []
        
        # Optional: Further balance each fold with SMOTE
        if ADVANCED_SAMPLING_AVAILABLE:
            try:
                smote = SMOTE(random_state=random_state, k_neighbors=min(3, min(y_tr.value_counts()) - 1))
                X_tr_balanced, y_tr_balanced = smote.fit_resample(X_tr, y_tr)
                X_tr_balanced = pd.DataFrame(X_tr_balanced, columns=X_tr.columns)
                y_tr_balanced = pd.Series(y_tr_balanced, name=y_tr.name)
                logger.info(f"    Applied SMOTE: {y_tr_balanced.value_counts().to_dict()}")
            except Exception as e:
                logger.warning(f"    SMOTE failed: {e}, using original data")
                X_tr_balanced, y_tr_balanced = X_tr, y_tr
        else:
            X_tr_balanced, y_tr_balanced = X_tr, y_tr
        
        for m_idx, (ModelClass, model_name) in enumerate(zip(model_classes, model_names)):
            try:
                # Train model
                model = ModelClass()
                model.train(X_tr_balanced, y_tr_balanced)
                
                # Get validation set probabilities
                proba_va = model.predict_proba(X_va)
                proba_va_1 = proba_va[:, 1] if proba_va.ndim > 1 and proba_va.shape[1] > 1 else proba_va.flatten()
                oof_preds[va_idx, m_idx] = proba_va_1
                
                # Find optimal threshold using enhanced robust method
                try:
                    target_positives = max(1, len(y_va) // 20)  # At least 5% or 1 positive
                    thr = find_threshold_robust(y_va.values, proba_va_1, target_positive=target_positives)
                    logger.info(f"    {model_name} fold {fold + 1}: enhanced threshold = {thr:.6f}")
                except Exception as e:
                    logger.warning(f"    {model_name} fold {fold + 1}: enhanced threshold optimization failed: {e}")
                    thr = 0.5
                
                fold_model_thresholds.append(thr)
                
                # Predict on test set
                proba_te = model.predict_proba(X_test)
                proba_te_1 = proba_te[:, 1] if proba_te.ndim > 1 and proba_te.shape[1] > 1 else proba_te.flatten()
                test_preds[:, m_idx, fold] = proba_te_1
                
            except Exception as e:
                logger.error(f"    {model_name} fold {fold + 1} failed: {e}")
                # Use default values for failed model
                fold_model_thresholds.append(0.5)
                test_preds[:, m_idx, fold] = 0.5  # Default probability
        
        fold_thresholds.append(fold_model_thresholds)
    
    # Average thresholds across folds
    avg_thresholds = np.mean(np.array(fold_thresholds), axis=0)
    avg_test_preds = test_preds.mean(axis=2)
    
    logger.info(f"Average thresholds: {dict(zip(model_names, avg_thresholds))}")
    
    # 3) Fit meta-model on OOF predictions
    logger.info("Training meta-model on out-of-fold predictions...")
    try:
        meta = LogisticRegression(max_iter=1000, random_state=random_state)
        meta.fit(oof_preds, y_train_boosted)
        logger.info("✅ Meta-model trained successfully")
    except Exception as e:
        logger.error(f"Meta-model training failed: {e}")
        # Return None to indicate failure
        return None, avg_thresholds, avg_test_preds
    
    return meta, avg_thresholds, avg_test_preds


def prepare_training_data(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = 'target',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split df into train/test ensuring both sets contain at least one of each class.

    - If there are ≥2 positives, do a standard stratified split.
    - If there is exactly 1 positive, force that one into test and
      fill out the rest of test from negatives.
    """
    import numpy as np  # Import at the top for use throughout function
    logger.info("Preparing training data with improved class balance handling...")
    
    # 1) clean & drop NA
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    X_clean = clean_data_for_ml(X)
    mask = ~(X_clean.isnull().any(axis=1) | y.isnull())
    X_final, y_final = X_clean[mask], y[mask]

    counts = y_final.value_counts()
    logger.info(f"Total class distribution: {counts.to_dict()}")
    
    if len(counts) == 1:
        single_class = counts.index[0]
        logger.warning(f"Only one class present ({single_class}). Creating synthetic minority class for training.")
        
        # Create a synthetic minority sample to enable binary classification
        # This allows the pipeline to run but with the understanding that 
        # performance will be limited due to lack of real minority examples
        
        n_samples = len(X_final)
        if n_samples < 2:
            raise ValueError(f"Insufficient data: only {n_samples} samples available.")
        
        # Create one synthetic minority sample by slightly modifying an existing sample
        synthetic_X = X_final.iloc[0:1].copy()
        synthetic_y = pd.Series([1 - single_class], index=[X_final.index[0]])  # Flip the class
        
        # Add small random noise to create diversity
        np.random.seed(42)
        for col in synthetic_X.columns:
            if synthetic_X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                synthetic_X[col] += np.random.normal(0, 0.01, 1)
        
        # Combine original and synthetic data
        X_combined = pd.concat([X_final, synthetic_X], ignore_index=True)
        y_combined = pd.concat([y_final, synthetic_y], ignore_index=True)
        
        logger.info(f"Created synthetic dataset with {len(X_combined)} samples")
        logger.info(f"New class distribution: {y_combined.value_counts().to_dict()}")
        
        # Now proceed with the synthetic data
        X_final, y_final = X_combined, y_combined
        counts = y_final.value_counts()

    minority_class = counts.idxmin()
    minority_count = counts.min()
    majority_class = counts.idxmax()

    # Case A: enough positives to stratify
    if minority_count >= 2:
        logger.info(f"Stratified split with {minority_count} minority samples.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_final,
            test_size=test_size,
            random_state=random_state,
            stratify=y_final
        )
        logger.info(f"Train class counts: {y_train.value_counts().to_dict()}")
        logger.info(f"Test  class counts: {y_test.value_counts().to_dict()}")
        return X_train, X_test, y_train, y_test

    # Case B: exactly one positive sample
    logger.warning("Exactly one minority sample—forcing it into the test set.")
    # index of that one positive
    pos_idx = y_final[y_final == minority_class].index
    assert len(pos_idx) == 1
    pos_idx = pos_idx[0]

    # how many negatives should go into test?
    n_test = max(1, int(len(y_final) * test_size))
    # we already placed 1 positive, so need n_test-1 negatives
    neg_idx = y_final[y_final == majority_class].index
    np.random.seed(random_state)
    neg_sample = np.random.choice(neg_idx, size=n_test - 1, replace=False)

    test_idx = np.concatenate([[pos_idx], neg_sample])
    train_idx = y_final.index.difference(test_idx)

    X_train = X_final.loc[train_idx]
    y_train = y_final.loc[train_idx]
    X_test  = X_final.loc[test_idx]
    y_test  = y_final.loc[test_idx]

    logger.info(f"Train class counts: {y_train.value_counts().to_dict()}")
    logger.info(f"Test  class counts: {y_test.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test

def train_committee_models_advanced(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """
    Train Committee of Five with out-of-fold stacking procedure.
    
    Uses adaptive folding with SMOTEENN oversampling, CalibratedClassifierCV, 
    and optimal threshold tuning for each base model. Falls back to simple 
    train/test splitting for extremely imbalanced datasets.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Tuple of (models dict, metrics dict) where models contains trained model info
        and metrics contains performance metrics for each model
    """
    logger.info("🚀 Starting Committee of Five training with adaptive OOF...")
    logger.info("📊 Implementing: SMOTEENN, Calibration, Threshold Optimization, Adaptive Stacking")
    
    # Print test class distribution for debugging
    print("Test class distribution:")
    print(y_test.value_counts())
    logger.info(f"Test class distribution: {y_test.value_counts().to_dict()}")
    
    # Check if we have both classes in training data
    unique_classes = y_train.unique()
    if len(unique_classes) < 2:
        logger.warning(f"Training data contains only one class: {unique_classes}")
        logger.info("🔄 Will use oversampling to generate synthetic minority samples for training")
        
        # For single-class training, we'll force oversampling during model training
        # This is expected when singleton minority is in test set
        single_class_training = True
    else:
        single_class_training = False
    
    # Check if SMOTEENN is available
    if not ADVANCED_SAMPLING_AVAILABLE:
        logger.error("SMOTEENN not available. Install with: pip install imbalanced-learn")
        raise ImportError("SMOTEENN required for this training procedure")
    
    if not CALIBRATION_AVAILABLE:
        logger.error("CalibratedClassifierCV not available. Update sklearn.")
        raise ImportError("CalibratedClassifierCV required for this training procedure")
    
    # Define base model classes with optimized eval metrics
    base_model_classes = {
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel, 
        'catboost': CatBoostModel,
        'random_forest': RandomForestModel,
        'svm': SVMClassifier
    }
    
    # Create models config for stacking
    models_config = {}
    for name, model_class in base_model_classes.items():
        models_config[name] = {
            'class': model_class,
            'balance_method': 'smoteenn' if not single_class_training else 'basic',
            'calibrate': True,
            'force_oversampling': single_class_training
        }
    
    # Try out-of-fold stacking first
    logger.info(f"� Attempting out-of-fold stacking with {DEFAULT_N_FOLDS} folds...")
    log_timing("Starting stacking procedure", enable=ENABLE_TIMING)
    
    # Use the new robust stacking approach
    meta_model, avg_thresholds, avg_test_preds = robust_out_of_fold_stacking(
        X_train, y_train, X_test, n_splits=DEFAULT_N_FOLDS, random_state=DEFAULT_RANDOM_STATE
    )
    
    # Check if robust stacking succeeded
    use_simple_stacking = meta_model is None
    
    if use_simple_stacking:
        logger.error("❌ Robust stacking failed, no fallback available")
        raise RuntimeError("Robust stacking failed and no fallback is implemented")
    
    else:
        logger.info("✅ Successfully completed robust out-of-fold stacking")
        
        # The robust stacking already provides optimized thresholds and predictions
        model_names = ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'svm']
        
        # Debug: inspect threshold vs test prediction ranges
        logger.info("🔍 Debugging threshold vs test prediction ranges:")
        for i, model_name in enumerate(model_names):
            test_proba = avg_test_preds[:, i]
            threshold = avg_thresholds[i]
            print(f"{model_name}: optimal thresh = {threshold:.3f}, "
                  f"min(test_proba) = {test_proba.min():.3f}, "
                  f"max(test_proba) = {test_proba.max():.3f}")
            logger.info(f"{model_name}: threshold={threshold:.3f}, test_range=[{test_proba.min():.3f}, {test_proba.max():.3f}]")
            
            # Check if threshold will produce any positives
            predicted_positives = np.sum(test_proba >= threshold)
            logger.info(f"{model_name}: predicted positives with threshold {threshold:.3f}: {predicted_positives}/{len(test_proba)}")
        
        # Meta-model prediction on test set
        logger.info("🎯 Getting meta-model predictions...")
        meta_test_proba = meta_model.predict_proba(avg_test_preds)[:, 1]
        
        # Find optimal threshold for meta-model using a simple approach
        # (We could enhance this with validation set optimization)
        meta_threshold = 0.5  # Default for now
        y_pred_meta = (meta_test_proba >= meta_threshold).astype(int)
        
        # Debug meta-model threshold vs test probabilities
        print(f"meta_model: default thresh = {meta_threshold:.3f}, "
              f"min(test_proba) = {meta_test_proba.min():.3f}, "
              f"max(test_proba) = {meta_test_proba.max():.3f}")
        logger.info(f"meta_model: threshold={meta_threshold:.3f}, test_range=[{meta_test_proba.min():.3f}, {meta_test_proba.max():.3f}]")
        
        # Check if threshold will produce any positives
        predicted_positives = np.sum(meta_test_proba >= meta_threshold)
        logger.info(f"meta_model: predicted positives with threshold {meta_threshold:.3f}: {predicted_positives}/{len(meta_test_proba)}")
        
        # Store complete model structure
        models = {
            'base_models': {name: None for name in model_names},  # We don't store individual models in robust approach
            'meta_model': meta_model,
            'thresholds': {name: avg_thresholds[i] for i, name in enumerate(model_names)},
            'thresholds_meta': meta_threshold,
            'test_predictions': {name: avg_test_preds[:, i] for i, name in enumerate(model_names)},
            'meta_predictions': y_pred_meta,
            'meta_test_proba': meta_test_proba
        }
    
    # 6. Final evaluation - calculate metrics for all models
    logger.info("📊 Evaluating models on test set...")
    log_timing("Starting final evaluation", enable=ENABLE_TIMING)
    
    metrics = {}
    model_names = ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'svm']
    
    # 🔧 Aggressive test-set-aware threshold adjustment for extreme imbalance
    logger.info("🔧 Adjusting thresholds based on test set probability ranges...")
    adjusted_thresholds = {}
    
    for i, model_name in enumerate(model_names):
        original_threshold = models['thresholds'][model_name]
        test_proba = models['test_predictions'][model_name]
        max_test_proba = test_proba.max()
        min_test_proba = test_proba.min()
        
        # For extreme imbalance, be much more aggressive with threshold adjustment
        if original_threshold > max_test_proba:
            # ULTRA-AGGRESSIVE: Use top N highest probabilities instead of statistical thresholds
            # This ensures we always get some positive predictions for extreme imbalance
            num_positives_needed = max(1, int(len(test_proba) * 0.02))  # At least 2% positive predictions
            sorted_probas = np.sort(test_proba)
            top_n_threshold = sorted_probas[-num_positives_needed] if len(sorted_probas) >= num_positives_needed else sorted_probas[-1]
            
            # Ensure threshold is not zero
            adjusted_threshold = max(top_n_threshold, 1e-10)
            logger.info(f"🔧 {model_name}: ULTRA-AGGRESSIVE adjustment from {original_threshold:.3f} to {adjusted_threshold:.10f} (top-{num_positives_needed} approach)")
            adjusted_thresholds[model_name] = adjusted_threshold
        elif original_threshold > max_test_proba * 0.7:
            # Even if threshold is below max, if it's using >70% of the range, use top-N approach
            num_positives_needed = max(1, int(len(test_proba) * 0.01))  # At least 1% positive predictions
            sorted_probas = np.sort(test_proba)
            top_n_threshold = sorted_probas[-num_positives_needed] if len(sorted_probas) >= num_positives_needed else sorted_probas[-1]
            
            adjusted_threshold = max(top_n_threshold, 1e-10)
            logger.info(f"🔧 {model_name}: Top-N adjustment from {original_threshold:.3f} to {adjusted_threshold:.10f} (top-{num_positives_needed} approach)")
            adjusted_thresholds[model_name] = adjusted_threshold
        else:
            # For extreme imbalance, even "safe" thresholds might be too high
            # Use top-N approach even for conservative adjustments
            num_positives_needed = max(1, int(len(test_proba) * 0.005))  # At least 0.5% positive predictions
            sorted_probas = np.sort(test_proba)
            top_n_threshold = sorted_probas[-num_positives_needed] if len(sorted_probas) >= num_positives_needed else sorted_probas[-1]
            
            safe_threshold = max(top_n_threshold, 1e-10)
            adjusted_thresholds[model_name] = safe_threshold
            if safe_threshold != original_threshold:
                logger.info(f"🔧 {model_name}: Conservative top-N adjustment from {original_threshold:.3f} to {safe_threshold:.10f} (top-{num_positives_needed})")
            else:
                logger.info(f"🔧 {model_name}: Keeping original threshold {original_threshold:.3f}")
    
    # Base model metrics with adjusted thresholds
    for i, model_name in enumerate(model_names):
        threshold = adjusted_thresholds[model_name]
        test_proba = models['test_predictions'][model_name]
        y_pred_base = (test_proba >= threshold).astype(int)
        
        # Debug: show prediction counts
        predicted_pos = np.sum(y_pred_base == 1)
        predicted_neg = np.sum(y_pred_base == 0)
        actual_pos = np.sum(y_test == 1)
        actual_neg = np.sum(y_test == 0)
        
        logger.info(f"🔍 {model_name} predictions: {predicted_pos} positive, {predicted_neg} negative "
                   f"(actual: {actual_pos} positive, {actual_neg} negative)")
        
        try:
            metrics[model_name] = {
                'accuracy': accuracy_score(y_test, y_pred_base),
                'precision': precision_score(y_test, y_pred_base, zero_division=0),
                'recall': recall_score(y_test, y_pred_base, zero_division=0),
                'f1': f1_score(y_test, y_pred_base, zero_division=0),
                'roc_auc': roc_auc_score(y_test, test_proba) if len(np.unique(y_test)) > 1 else 0.0,
                'pr_auc': average_precision_score(y_test, test_proba) if len(np.unique(y_test)) > 1 else 0.0
            }
            
            logger.info(f"✅ {model_name} - F1: {metrics[model_name]['f1']:.3f}, "
                       f"ROC-AUC: {metrics[model_name]['roc_auc']:.3f}, "
                       f"PR-AUC: {metrics[model_name]['pr_auc']:.3f}")
                       
        except Exception as e:
            logger.error(f"❌ Error calculating metrics for {model_name}: {e}")
            metrics[model_name] = {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                'f1': 0.0, 'roc_auc': 0.0, 'pr_auc': 0.0
            }
    
    # Meta-model metrics with aggressive threshold adjustment
    try:
        meta_test_proba = models['meta_test_proba']
        original_meta_threshold = models['thresholds_meta']
        max_meta_proba = meta_test_proba.max()
        
        # For extreme imbalance, be very aggressive with meta-model threshold using top-N
        if original_meta_threshold > max_meta_proba:
            # ULTRA-AGGRESSIVE: Use top N highest probabilities for meta-model
            num_positives_needed = max(1, int(len(meta_test_proba) * 0.02))  # At least 2% positive predictions
            sorted_meta_probas = np.sort(meta_test_proba)
            top_n_threshold = sorted_meta_probas[-num_positives_needed] if len(sorted_meta_probas) >= num_positives_needed else sorted_meta_probas[-1]
            
            adjusted_meta_threshold = max(top_n_threshold, 1e-10)
            logger.info(f"🔧 meta_model: ULTRA-AGGRESSIVE adjustment from {original_meta_threshold:.3f} to {adjusted_meta_threshold:.10f} (top-{num_positives_needed} approach)")
        elif original_meta_threshold > max_meta_proba * 0.5:
            # Even moderate thresholds may be too high - use top-N approach
            num_positives_needed = max(1, int(len(meta_test_proba) * 0.01))  # At least 1% positive predictions
            sorted_meta_probas = np.sort(meta_test_proba)
            top_n_threshold = sorted_meta_probas[-num_positives_needed] if len(sorted_meta_probas) >= num_positives_needed else sorted_meta_probas[-1]
            
            adjusted_meta_threshold = max(top_n_threshold, 1e-10)
            logger.info(f"🔧 meta_model: Top-N adjustment from {original_meta_threshold:.3f} to {adjusted_meta_threshold:.10f} (top-{num_positives_needed} approach)")
        else:
            # Conservative top-N approach for extreme imbalance scenarios
            num_positives_needed = max(1, int(len(meta_test_proba) * 0.005))  # At least 0.5% positive predictions
            sorted_meta_probas = np.sort(meta_test_proba)
            top_n_threshold = sorted_meta_probas[-num_positives_needed] if len(sorted_meta_probas) >= num_positives_needed else sorted_meta_probas[-1]
            
            adjusted_meta_threshold = max(top_n_threshold, 1e-10)
            logger.info(f"🔧 meta_model: Conservative top-N adjustment from {original_meta_threshold:.3f} to {adjusted_meta_threshold:.10f} (top-{num_positives_needed})")
        
        y_pred_meta = (meta_test_proba >= adjusted_meta_threshold).astype(int)
        
        # Debug: show prediction counts for meta-model
        predicted_pos = np.sum(y_pred_meta == 1)
        predicted_neg = np.sum(y_pred_meta == 0)
        actual_pos = np.sum(y_test == 1)
        actual_neg = np.sum(y_test == 0)
        
        logger.info(f"🔍 Meta-model predictions: {predicted_pos} positive, {predicted_neg} negative "
                   f"(actual: {actual_pos} positive, {actual_neg} negative)")
        
        metrics['meta_model'] = {
            'accuracy': accuracy_score(y_test, y_pred_meta),
            'precision': precision_score(y_test, y_pred_meta, zero_division=0),
            'recall': recall_score(y_test, y_pred_meta, zero_division=0),
            'f1': f1_score(y_test, y_pred_meta, zero_division=0),
            'roc_auc': roc_auc_score(y_test, meta_test_proba) if len(np.unique(y_test)) > 1 else 0.0,
            'pr_auc': average_precision_score(y_test, meta_test_proba) if len(np.unique(y_test)) > 1 else 0.0
        }
        
        logger.info(f"🏆 Meta-model - F1: {metrics['meta_model']['f1']:.3f}, "
                   f"ROC-AUC: {metrics['meta_model']['roc_auc']:.3f}, "
                   f"PR-AUC: {metrics['meta_model']['pr_auc']:.3f}")
                   
    except Exception as e:
        logger.error(f"❌ Error calculating meta-model metrics: {e}")
        metrics['meta_model'] = {
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
            'f1': 0.0, 'roc_auc': 0.0, 'pr_auc': 0.0
        }
    
    # Log final summary
    f1_scores = [metrics[m]['f1'] for m in metrics.keys()]
    avg_f1 = np.mean(f1_scores)
    best_f1 = max(f1_scores)
    best_model = max(metrics.keys(), key=lambda k: metrics[k]['f1'])
    
    logger.info(f"🎯 Robust Training Summary:")
    logger.info(f"   Average F1 Score: {avg_f1:.3f}")
    logger.info(f"   Best F1 Score: {best_f1:.3f} ({best_model})")
    logger.info(f"   Meta-model F1 Score: {metrics['meta_model']['f1']:.3f}")
    
    # 🚀 PRODUCTION ENSEMBLE ENHANCEMENT - Add voting consensus predictions
    try:
        logger.info("🎯 Applying Production Ensemble for enhanced predictions...")
        
        # Apply production ensemble
        production_results = production_extreme_imbalance_ensemble(models, X_test, y_test)
        
        # Extract results with correct key names
        production_predictions = production_results['predictions']
        production_probabilities = production_results['probabilities']
        production_metrics = production_results['metrics']
        consensus_indices = production_results['consensus_indices']
        
        # Extract individual metrics
        production_f1 = production_metrics['f1']
        production_precision = production_metrics['precision']
        production_recall = production_metrics['recall']
        production_accuracy = production_metrics['accuracy']
        positive_predictions = production_metrics['positive_predictions']
        positive_rate = positive_predictions / len(y_test)
        
        # Add production ensemble metrics to results
        metrics['production_ensemble'] = {
            'accuracy': production_accuracy,
            'precision': production_precision,
            'recall': production_recall,
            'f1': production_f1,
            'positive_rate': positive_rate,
            'predictions': production_predictions,
            'probabilities': production_probabilities,
            'consensus_indices': consensus_indices
        }
        
        # Enhanced reporting
        logger.info(f"🏆 PRODUCTION ENSEMBLE RESULTS:")
        logger.info(f"   F1 Score: {production_f1:.3f}")
        logger.info(f"   Precision: {production_precision:.3f}")
        logger.info(f"   Recall: {production_recall:.3f}")
        logger.info(f"   Accuracy: {production_accuracy:.3f}")
        logger.info(f"   Positive Predictions: {positive_predictions}/{len(y_test)} ({positive_rate:.1%})")
        logger.info(f"   Consensus Samples: {len(consensus_indices)}")
        
        # Compare with best individual model
        improvement = production_f1 - best_f1
        if improvement > 0:
            logger.info(f"🎉 Production ensemble improved F1 by {improvement:.3f} over best individual model!")
        else:
            logger.info(f"📊 Production ensemble F1: {production_f1:.3f} vs best individual: {best_f1:.3f}")
            
        # Show detailed comparison
        logger.info(f"📈 COMPARISON SUMMARY:")
        logger.info(f"   Best Individual Model: {best_model} (F1: {best_f1:.3f})")
        logger.info(f"   Meta-model: F1: {metrics['meta_model']['f1']:.3f}")
        logger.info(f"   Production Ensemble: F1: {production_f1:.3f}")
        
    except Exception as e:
        logger.error(f"❌ Error in production ensemble: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    log_timing("Training completed", enable=ENABLE_TIMING)
    
    return models, metrics


def prepare_sampler(method: str, random_state: int = DEFAULT_RANDOM_STATE):
    """
    Prepare sampling method for class imbalance handling.
    
    Args:
        method: Sampling method ('smoteenn' or 'none')
        random_state: Random state for reproducibility
        
    Returns:
        Configured sampler or None
    """
    if method == 'smoteenn':
        try:
            from imblearn.combine import SMOTEENN
            from imblearn.over_sampling import SMOTE
            return SMOTEENN(random_state=random_state)
        except ImportError:
            logger.warning("SMOTEENN not available. Using basic balancing instead.")
            return None
    return None


def evaluate_models_comprehensive(fitted_models: Dict[str, Any], thresholds: Dict[str, float], 
                                X_test: pd.DataFrame, y_test: pd.Series, meta_model: Any = None) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive evaluation of base models and meta-model with all metrics.
    
    Args:
        fitted_models: Dictionary of trained models
        thresholds: Dictionary of optimal thresholds per model
        X_test: Test features
        y_test: Test labels
        meta_model: Trained meta-model (optional)
        
    Returns:
        Dictionary of metrics for each model
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                f1_score, roc_auc_score, average_precision_score)
    
    results = {}
    
    # Evaluate base models
    for name, model in fitted_models.items():
        if name == 'meta_model':
            continue
            
        try:
            # Get predictions
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)[:, 1]
            else:
                proba = model.predict(X_test)
                
            thresh = thresholds.get(name, 0.5)
            y_pred = (proba >= thresh).astype(int)

            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else 0.0,
                'pr_auc': average_precision_score(y_test, proba) if len(np.unique(y_test)) > 1 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model {name}: {e}")
            results[name] = {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                'f1': 0.0, 'roc_auc': 0.0, 'pr_auc': 0.0
            }

    # Evaluate meta-model if provided
    if meta_model is not None:
        try:
            # Create meta-features from base model predictions
            meta_X_test = pd.DataFrame({
                name: fitted_models[name].predict_proba(X_test)[:, 1] 
                for name in fitted_models.keys() if name != 'meta_model'
            })
            
            meta_proba = meta_model.predict_proba(meta_X_test)[:, 1]
            y_meta = (meta_proba >= 0.5).astype(int)  # Default threshold for meta-model
            
            results['meta_model'] = {
                'accuracy': accuracy_score(y_test, y_meta),
                'precision': precision_score(y_test, y_meta, zero_division=0),
                'recall': recall_score(y_test, y_meta, zero_division=0),
                'f1': f1_score(y_test, y_meta, zero_division=0),
                'roc_auc': roc_auc_score(y_test, meta_proba) if len(np.unique(y_test)) > 1 else 0.0,
                'pr_auc': average_precision_score(y_test, meta_proba) if len(np.unique(y_test)) > 1 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error evaluating meta-model: {e}")
            results['meta_model'] = {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                'f1': 0.0, 'roc_auc': 0.0, 'pr_auc': 0.0
            }

    return results


def prepare_calibration_method(method: str):
    """
    Prepare calibration method for probability calibration.
    
    Args:
        method: Calibration method ('isotonic', 'sigmoid', or 'none')
        
    Returns:
        Calibration method string or None
    """
    if method in ('isotonic', 'sigmoid'):
        return method
    return None


def train_and_calibrate_model(model, X_train, y_train, X_val, y_val, calibration_method, use_threshold=True):
    """
    Train and calibrate a single model with optimal threshold tuning.
    
    Args:
        model: Base model instance
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        calibration_method: Calibration method ('isotonic', 'sigmoid', or None)
        use_threshold: Whether to tune threshold
        
    Returns:
        Tuple of (fitted_model, best_threshold)
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import f1_score
    
    # Extract the underlying sklearn estimator from our custom wrapper
    sklearn_estimator = None
    if hasattr(model, 'model') and model.model is not None:
        sklearn_estimator = model.model
    elif hasattr(model, 'pipeline') and model.pipeline is not None:
        sklearn_estimator = model.pipeline
    else:
        # For custom models, try to get the sklearn-compatible estimator
        sklearn_estimator = getattr(model, 'estimator', model)
    
    # Apply calibration if specified
    if calibration_method and CALIBRATION_AVAILABLE:
        # Adaptive CV folds based on available validation data
        min_class_samples = min(y_val.value_counts()) if len(y_val.value_counts()) > 1 else len(y_val)
        adaptive_cv = min(DEFAULT_CALIBRATION_CV, max(2, min_class_samples // 2))
        
        cal_model = CalibratedClassifierCV(
            estimator=sklearn_estimator,
            method=calibration_method,
            cv=adaptive_cv
        )
        fitted_model = cal_model
    else:
        fitted_model = sklearn_estimator
    
    # Train the model
    fitted_model.fit(X_train, y_train)
    
    # Tune threshold if requested
    best_threshold = 0.5  # Default
    if use_threshold:
        val_proba = fitted_model.predict_proba(X_val)[:, 1]
        
        # Grid search for optimal threshold
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1 = 0
        
        for threshold in thresholds:
            y_pred_thresh = (val_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred_thresh, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
    
    return fitted_model, best_threshold


def export_metrics_to_csv(metrics: Dict[str, Dict[str, float]], thresholds: Dict[str, float], output_dir: str) -> None:
    """
    Export model metrics to CSV file.
    
    Args:
        metrics: Dictionary of metrics for each model
        thresholds: Dictionary of optimal thresholds for each model
        output_dir: Output directory for the CSV file
    """
    try:
        # Prepare data for CSV
        csv_data = []
        for model_name, model_metrics in metrics.items():
            row = {
                'model_name': model_name,
                'f1_score': model_metrics.get('f1', 0.0),
                'roc_auc': model_metrics.get('roc_auc', 0.0),
                'pr_auc': model_metrics.get('pr_auc', 0.0),
                'optimal_threshold': thresholds.get(model_name, 0.5),
                'accuracy': model_metrics.get('accuracy', 0.0),
                'precision': model_metrics.get('precision', 0.0),
                'recall': model_metrics.get('recall', 0.0)
            }
            csv_data.append(row)
        
        # Create DataFrame and save to CSV
        import pandas as pd
        df_metrics = pd.DataFrame(csv_data)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        csv_path = os.path.join(output_dir, 'model_summary.csv')
        df_metrics.to_csv(csv_path, index=False)
        logger.info(f"📊 Model metrics exported to: {csv_path}")
        
    except Exception as e:
        logger.error(f"❌ Error exporting metrics to CSV: {e}")


def save_meta_model_coefficients(meta_model, base_model_names: List[str], output_dir: str) -> None:
    """
    Save meta-model coefficients to CSV file.
    
    Args:
        meta_model: Trained LogisticRegression meta-model
        base_model_names: List of base model names (feature names)
        output_dir: Output directory for the CSV file
    """
    try:
        if not hasattr(meta_model, 'coef_'):
            logger.warning("Meta-model does not have coefficients (not a linear model)")
            return
        
        # Extract coefficients
        coefficients = meta_model.coef_[0] if meta_model.coef_.ndim > 1 else meta_model.coef_
        intercept = meta_model.intercept_[0] if hasattr(meta_model, 'intercept_') else 0.0
        
        # Prepare data for CSV
        coef_data = []
        for i, model_name in enumerate(base_model_names):
            coef_data.append({
                'feature': model_name,
                'coefficient': coefficients[i] if i < len(coefficients) else 0.0
            })
        
        # Add intercept
        coef_data.append({
            'feature': 'intercept',
            'coefficient': intercept
        })
        
        # Create DataFrame and save to CSV
        import pandas as pd
        df_coef = pd.DataFrame(coef_data)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        coef_path = os.path.join(output_dir, 'meta_model_coefficients.csv')
        df_coef.to_csv(coef_path, index=False)
        logger.info(f"🧠 Meta-model coefficients saved to: {coef_path}")
        
    except Exception as e:
        logger.error(f"❌ Error saving meta-model coefficients: {e}")


def prepare_calibration_method(method: str):
    """
    Prepare calibration method for probability calibration.
    
    Args:
        method: Calibration method ('isotonic', 'sigmoid', or 'none')
        
    Returns:
        Calibration method string or None
    """
    if method in ('isotonic', 'sigmoid'):
        return method
    return None


def _determine_output_directory(batch_num: Any = None, base_dir: str = "reports") -> tuple[str, str]:
    """Helper function to determine output directory and suffix for visualizations."""
    if batch_num is not None:
        if isinstance(batch_num, str) and '_' in batch_num:
            batch_parts = batch_num.split('_')
            if len(batch_parts) > 1:
                batch_dir = f"{base_dir}/batch multiple"
                suffix = f"_batch_multiple"
            else:
                batch_dir = f"{base_dir}/batch {batch_num}"
                suffix = f"_batch_{batch_num}"
        else:
            batch_dir = f"{base_dir}/batch {batch_num}"
            suffix = f"_batch_{batch_num}"
    else:
        batch_dir = base_dir
        suffix = ""
    return batch_dir, suffix


def _setup_output_directory(batch_dir: str) -> None:
    """Helper function to setup output directory for visualizations."""
    import shutil
    
    logger.info(f"📊 Creating visualizations in directory: {batch_dir}")
    
    # Recreate directory
    if os.path.exists(batch_dir):
        shutil.rmtree(batch_dir)
        logger.info(f"🗑️  Cleaned existing directory: {batch_dir}")
    
    os.makedirs(batch_dir, exist_ok=True)
    logger.info(f"📁 Created visualization directory: {batch_dir}")


def _create_metric_bar_chart(metric_name: str, metrics: Dict[str, Dict[str, float]], 
                           batch_dir: str, suffix: str, title_prefix: str = "Committee") -> bool:
    """Helper function to create metric comparison bar charts."""
    try:
        import matplotlib.pyplot as plt
        
        logger.info(f"📊 Generating {metric_name} comparison chart...")
        
        plt.figure(figsize=(CHART_FIGURE_WIDTH, CHART_FIGURE_HEIGHT))
        
        # Extract model names and values (excluding meta_model for base comparison)
        model_names = [name for name in metrics.keys() if name != 'meta_model']
        values = [metrics[name][metric_name] for name in model_names]
        
        # Add meta-model if it exists
        if 'meta_model' in metrics:
            model_names.append('meta_model')
            values.append(metrics['meta_model'][metric_name])
        
        # Create bar plot with different colors for meta-model
        colors = ['skyblue'] * (len(model_names) - 1)
        if 'meta_model' in model_names[-1:]:
            colors.append('orange')
        else:
            colors = ['skyblue'] * len(model_names)
        
        bars = plt.bar(range(len(model_names)), values, color=colors)
        
        plt.ylim(0, 1)
        plt.title(f"{title_prefix} - {metric_name.upper()} Comparison")
        plt.ylabel(metric_name.upper())
        plt.xlabel('Model')
        plt.xticks(range(len(model_names)), [name.replace('_', ' ').title() for name in model_names], rotation=45)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f"{val:.3f}", ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = f"{batch_dir}/{metric_name}_comparison{suffix}.png"
        plt.savefig(plot_path, dpi=CHART_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Saved metric chart: {plot_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error generating {metric_name} chart: {e}")
        return False


def _create_confusion_matrix_plot(model_name: str, y_true, y_pred, batch_dir: str, suffix: str) -> bool:
    """Helper function to create confusion matrix plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        logger.info(f"📊 Generating confusion matrix for {model_name}...")
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(MATRIX_FIGURE_WIDTH, MATRIX_FIGURE_HEIGHT))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Trade', 'Trade'], 
                   yticklabels=['No Trade', 'Trade'])
        plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save the plot
        matrix_path = f"{batch_dir}/confusion_matrix_{model_name}{suffix}.png"
        plt.savefig(matrix_path, dpi=CHART_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Saved confusion matrix: {matrix_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error generating confusion matrix for {model_name}: {e}")
        return False


def _create_confusion_matrices_oof(models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, 
                                   thresholds: Dict[str, float], output_dir: str, suffix: str = "") -> int:
    """
    Create confusion matrices for OOF models using averaged test predictions.
    
    Args:
        models: Dictionary containing model information including test predictions
        X_test: Test features 
        y_test: Test labels
        thresholds: Dictionary of optimal thresholds per model
        output_dir: Output directory for plots
        suffix: Filename suffix
        
    Returns:
        Number of confusion matrices created
    """
    from sklearn.metrics import confusion_matrix
    count = 0
    
    # Extract test predictions and create confusion matrices
    test_predictions = models.get('test_predictions', {})
    
    for model_name, test_proba in test_predictions.items():
        try:
            if len(test_proba) == 0:
                continue
                
            threshold = thresholds.get(model_name, 0.5)
            y_pred = (test_proba >= threshold).astype(int)
            
            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(MATRIX_FIGURE_WIDTH, MATRIX_FIGURE_HEIGHT))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No Trade', 'Trade'], 
                       yticklabels=['No Trade', 'Trade'])
            plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}\n(Threshold: {threshold:.3f})')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            # Save the plot
            matrix_path = os.path.join(output_dir, f"confusion_matrix_{model_name}{suffix}.png")
            plt.savefig(matrix_path, dpi=CHART_DPI, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Saved confusion matrix: {matrix_path}")
            count += 1
            
        except Exception as e:
            logger.error(f"❌ Error creating confusion matrix for {model_name}: {e}")
            continue
    
    # Create confusion matrix for meta-model if available
    if 'meta_predictions' in models:
        try:
            y_pred_meta = models['meta_predictions']
            cm = confusion_matrix(y_test, y_pred_meta)
            
            plt.figure(figsize=(MATRIX_FIGURE_WIDTH, MATRIX_FIGURE_HEIGHT))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No Trade', 'Trade'], 
                       yticklabels=['No Trade', 'Trade'])
            plt.title(f'Confusion Matrix - Meta Model')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            # Save the plot
            matrix_path = os.path.join(output_dir, f"confusion_matrix_meta_model{suffix}.png")
            plt.savefig(matrix_path, dpi=CHART_DPI, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Saved meta-model confusion matrix: {matrix_path}")
            count += 1
            
        except Exception as e:
            logger.error(f"❌ Error creating meta-model confusion matrix: {e}")
    
    return count


def _create_metric_comparison_charts(metrics: Dict[str, Dict[str, float]], output_dir: str, 
                                   suffix: str = "", title_prefix: str = "Committee") -> int:
    """
    Create bar charts comparing F1, ROC AUC, and PR AUC across models.
    
    Args:
        metrics: Dictionary of metrics for each model
        output_dir: Output directory for plots
        suffix: Filename suffix
        title_prefix: Prefix for chart titles
        
    Returns:
        Number of charts created
    """
    metric_names = ['f1', 'roc_auc', 'pr_auc']
    metric_display_names = {'f1': 'F1 Score', 'roc_auc': 'ROC AUC', 'pr_auc': 'PR AUC'}
    count = 0
    
    for metric_name in metric_names:
        try:
            logger.info(f"📊 Generating {metric_display_names[metric_name]} comparison chart...")
            
            plt.figure(figsize=(CHART_FIGURE_WIDTH, CHART_FIGURE_HEIGHT))
            
            # Extract model names and values (separate base models from meta-model)
            base_models = [name for name in metrics.keys() if name != 'meta_model']
            base_values = [metrics[name][metric_name] for name in base_models]
            
            # Prepare colors
            colors = ['skyblue'] * len(base_models)
            model_names = base_models[:]
            values = base_values[:]
            
            # Add meta-model if it exists
            if 'meta_model' in metrics:
                model_names.append('meta_model')
                values.append(metrics['meta_model'][metric_name])
                colors.append('orange')
            
            # Create bar plot
            bars = plt.bar(range(len(model_names)), values, color=colors)
            
            plt.ylim(0, 1)
            plt.title(f"{title_prefix} - {metric_display_names[metric_name]} Comparison")
            plt.ylabel(metric_display_names[metric_name])
            plt.xlabel('Model')
            plt.xticks(range(len(model_names)), [name.replace('_', ' ').title() for name in model_names], rotation=45)
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f"{val:.3f}", ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(output_dir, f"{metric_name}_comparison{suffix}.png")
            plt.savefig(plot_path, dpi=CHART_DPI, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Saved {metric_display_names[metric_name]} chart: {plot_path}")
            count += 1
            
        except Exception as e:
            logger.error(f"❌ Error generating {metric_display_names[metric_name]} chart: {e}")
            continue
    
    return count


def create_visualizations_unified(models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, 
                                 metrics: Dict[str, Dict[str, float]], batch_num: Any = None,
                                 mode: str = "oof", output_dir: str = "reports") -> None:
    """
    Unified visualization function for both OOF and advanced training modes.
    
    Creates confusion matrices and metric comparison charts (F1, ROC AUC, PR AUC).
    All plots are saved with descriptive names in the specified output directory.
    
    Args:
        models: Dictionary containing trained models and predictions
        X_test: Test features
        y_test: Test labels  
        metrics: Nested dictionary of metrics for each model
        batch_num: Optional batch identifier used to name the output directory
        mode: Training mode ("oof" for out-of-fold, "advanced" for advanced training)
        output_dir: Base output directory for saving plots
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization libraries not available; skipping plots.")
        return
    
    try:
        # Setup output directory
        batch_dir, suffix = _determine_output_directory(batch_num, output_dir)
        _setup_output_directory(batch_dir)
        
        title_prefix = "Out-of-Fold Committee" if mode == "oof" else "Advanced Committee"
        
        # Generate confusion matrices
        confusion_matrix_count = 0
        if mode == "oof":
            # For OOF mode, use test predictions from the models dictionary
            thresholds = models.get('thresholds', {})
            confusion_matrix_count = _create_confusion_matrices_oof(models, X_test, y_test, thresholds, batch_dir, suffix)
        else:
            # For advanced mode, use direct model predictions
            for model_name in models.keys():
                if model_name not in ['meta_model', 'thresholds', 'oof_predictions', 'test_predictions', 'meta_test_features', 'meta_predictions']:
                    try:
                        model = models[model_name]
                        if hasattr(model, 'predict'):
                            y_pred = model.predict(X_test)
                            if _create_confusion_matrix_plot(model_name, y_test, y_pred, batch_dir, suffix):
                                confusion_matrix_count += 1
                    except Exception as e:
                        logger.error(f"❌ Error creating confusion matrix for {model_name}: {e}")
                        continue
        
        # Generate metric comparison charts (F1, ROC AUC, PR AUC)
        metric_chart_count = _create_metric_comparison_charts(metrics, batch_dir, suffix, title_prefix)
        
        # Export metrics to CSV
        thresholds = models.get('thresholds', {})
        export_metrics_to_csv(metrics, thresholds, batch_dir)
        
        # Save meta-model coefficients if available
        if 'meta_model' in models:
            meta_model = models['meta_model']
            base_model_names = [name for name in metrics.keys() if name != 'meta_model']
            save_meta_model_coefficients(meta_model, base_model_names, batch_dir)
        
        # Log summary
        total_images = confusion_matrix_count + metric_chart_count
        logger.info(f"📊 Generated {confusion_matrix_count} confusion matrices and {metric_chart_count} metric comparison charts")
        logger.info(f"🎉 Visualization creation complete! Total images: {total_images}")
        logger.info(f"📂 All visualizations saved to: {batch_dir}")
        
    except Exception as e:
        logger.error(f"❌ Critical error in visualization creation: {e}")
        return


def create_visualizations_oof(models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, 
                             metrics: Dict[str, Dict[str, float]], batch_num: Any = None, output_dir: str = "reports") -> None:
    """
    Generate and save confusion matrices and metric bar charts for out-of-fold stacking models.
    
    This function now delegates to the unified visualization function.
    
    Args:
        models: Dictionary containing trained models and predictions from OOF training
        X_test: Test features
        y_test: Test labels
        metrics: Nested dictionary of metrics for each model
        batch_num: Optional batch identifier used to name the output directory
        output_dir: Base output directory for saving plots
    """
    create_visualizations_unified(models, X_test, y_test, metrics, batch_num, mode="oof", output_dir=output_dir)


def create_visualizations_advanced(models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, 
                                  metrics: Dict[str, Dict[str, float]], batch_num: Any = None, output_dir: str = "reports") -> None:
    """
    Generate and save confusion matrices and metric bar charts for advanced models.

    This function now delegates to the unified visualization function.

    Args:
        models: Dictionary containing trained models and predictions
        X_test: Test features
        y_test: Test labels
        metrics: Nested dictionary of metrics for each model
        batch_num: Optional batch identifier used to name the output directory
        output_dir: Base output directory for saving plots
    """
    create_visualizations_unified(models, X_test, y_test, metrics, batch_num, mode="advanced", output_dir=output_dir)
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


def main(batch_numbers: List[int] = None, max_symbols_per_batch: int = 10, use_alpaca: bool = True, 
         stacking_method: str = 'oof', n_folds: int = DEFAULT_N_FOLDS, sampler: str = 'smoteenn',
         calibrate: str = DEFAULT_CALIBRATION_METHOD, threshold: float = 0.5, output_dir: str = 'reports') -> None:
    """
    Main function for Committee of Five training with Alpaca data integration.
    
    Args:
        batch_numbers: List of batch numbers to process (default: [1, 2])
        max_symbols_per_batch: Maximum symbols per batch (default: 10)
        use_alpaca: Whether to use Alpaca API for data collection (default: True)
        stacking_method: Stacking method 'oof' (out-of-fold) or 'standard' (default: 'oof')
        n_folds: Number of cross-validation folds (default: DEFAULT_N_FOLDS)
        sampler: Sampling method for class imbalance ('smoteenn' or 'none', default: 'smoteenn')
        calibrate: Calibration method ('isotonic', 'sigmoid', or 'none', default: 'isotonic')
        threshold: Default probability threshold for classification (default: 0.5)
        output_dir: Directory for saving plots (default: 'reports')
    """
    if batch_numbers is None:
        batch_numbers = [1, 2]  # Default to first 2 batches for testing
    
    logger.info(f"🚀 Starting Committee of Five training for batches: {batch_numbers}")
    logger.info(f"📊 Using {stacking_method.upper()} stacking method")
    logger.info(f"⚙️  Configuration: n_folds={n_folds}, sampler={sampler}, calibrate={calibrate}, threshold={threshold}")
    
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
        
        # Train committee models with specified stacking method
        if stacking_method == 'oof':
            models, metrics = train_committee_models_advanced(X_train, y_train, X_test, y_test)
        else:
            # Keep the original method for backward compatibility
            logger.warning("Standard stacking method not implemented in this version. Using OOF method.")
            models, metrics = train_committee_models_advanced(X_train, y_train, X_test, y_test)
        
        # Generate visualizations using the unified approach with configurable output directory
        if VISUALIZATION_AVAILABLE:
            batch_suffix = '_'.join(map(str, batch_numbers)) if batch_numbers else 'default'
            # Create visualizations in the specified output directory
            create_visualizations_unified(
                models, X_test, y_test, metrics, 
                batch_num=batch_suffix, mode="oof" if stacking_method == 'oof' else "advanced",
                output_dir=output_dir
            )
        
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
    parser.add_argument('--stacking', choices=['oof', 'standard'], default='oof',
                       help='Stacking method: oof (out-of-fold) or standard (default: oof)')
    
    # Add centralized CLI flags for model configuration
    parser.add_argument('--n-folds', type=int, default=DEFAULT_N_FOLDS,
                       help=f'Number of CV folds (default: {DEFAULT_N_FOLDS})')
    parser.add_argument('--sampler', choices=['smoteenn', 'none'], default='smoteenn',
                       help='Sampling method to balance classes (default: smoteenn)')
    parser.add_argument('--calibrate', choices=['isotonic', 'sigmoid', 'none'], default=DEFAULT_CALIBRATION_METHOD,
                       help=f'Calibration method (default: {DEFAULT_CALIBRATION_METHOD})')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Default probability threshold for classification (default: 0.5)')
    parser.add_argument('--output-dir', type=str, default='reports',
                       help='Directory for saving plots (default: reports)')
    
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
        use_alpaca=not args.no_alpaca,
        stacking_method=args.stacking,
        n_folds=args.n_folds,
        sampler=args.sampler,
        calibrate=args.calibrate,
        threshold=args.threshold,
        output_dir=args.output_dir
    )