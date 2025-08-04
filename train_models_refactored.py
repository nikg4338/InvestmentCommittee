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
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd

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
    export_training_results, create_confusion_matrices, generate_training_report
)
from utils.visualization import create_training_report as create_visual_report
from utils.helpers import compute_classification_metrics

# Import Alpaca data collection
from data_collection_alpaca import AlpacaDataCollector

# Enhanced utilities for extreme imbalance (keep these for backward compatibility)
from utils.data_splitting import ensure_minority_samples
from utils.evaluation import find_optimal_threshold

logger = logging.getLogger(__name__)

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
                         config: Optional[TrainingConfig] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare training data with robust splitting and cleaning.
    
    Args:
        df: Input DataFrame with features and target
        feature_columns: List of feature column names
        target_column: Name of target column
        config: Training configuration
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    if config is None:
        config = get_default_config()
    
    logger.info("Preparing training data...")
    
    # Extract features and target
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    # Clean data
    X_clean = clean_data_for_ml(X)
    
    # Remove rows with NaN values
    mask = ~(X_clean.isnull().any(axis=1) | y.isnull())
    X_final = X_clean[mask]
    y_final = y[mask]
    
    logger.info(f"Data after cleaning: {len(X_final)} samples")
    logger.info(f"Class distribution: {y_final.value_counts().to_dict()}")
    
    # Perform robust stratified split
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X_final, y_final,
        test_size=config.test_size,
        random_state=config.random_state,
        min_minority_samples=config.cross_validation.min_minority_samples
    )
    
    # Validate split quality
    split_quality = validate_split_quality(X_train, X_test, y_train, y_test)
    logger.info(f"Split quality: {split_quality}")
    
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
    
    # Step 1: Out-of-fold stacking
    logger.info("\n📊 Phase 1: Out-of-fold stacking...")
    
    train_meta_features, test_meta_features, trained_models = out_of_fold_stacking(
        X_train, y_train, X_test, config
    )
    
    # Check if OOF was successful
    if train_meta_features is None:
        logger.info("OOF stacking fell back to simple train/test - using base models only")
        use_meta_model = False
        meta_model = None
        meta_test_proba = None
    else:
        use_meta_model = True
        
        # Step 2: Train meta-model
        logger.info("\n🧠 Phase 2: Training meta-model...")
        
        meta_model = train_meta_model(train_meta_features, y_train, config)
        
        # Get meta-model predictions
        meta_test_proba = meta_model.predict_proba(test_meta_features)[:, -1]
        logger.info(f"Meta-model test predictions range: [{meta_test_proba.min():.4f}, {meta_test_proba.max():.4f}]")
    
    # Step 3: Create ensemble predictions
    logger.info("\n🎯 Phase 3: Creating ensemble predictions...")
    
    ensemble_results = create_ensemble_predictions(
        trained_models, X_test, meta_model, config
    )
    
    # Step 4: Rank-and-vote ensemble (production approach)
    if config.ensemble.voting_strategy == 'rank_and_vote':
        logger.info("\n🗳️ Phase 4: Rank-and-vote ensemble...")
        
        final_predictions = create_rank_vote_ensemble(
            ensemble_results['base_predictions'],
            meta_test_proba,
            config
        )
        
        # Convert to probabilities for evaluation (simple approach)
        final_probabilities = np.mean(list(ensemble_results['base_predictions'].values()), axis=0)
        if meta_test_proba is not None:
            final_probabilities = (final_probabilities + meta_test_proba) / 2
    else:
        # Use simple ensemble
        final_predictions = (ensemble_results['simple_ensemble'] >= 0.5).astype(int)
        final_probabilities = ensemble_results['simple_ensemble']
    
    # Step 5: Comprehensive evaluation
    logger.info("\n📈 Phase 5: Evaluation...")
    
    # Prepare evaluation data
    all_predictions = ensemble_results['base_predictions'].copy()
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
    
    # Step 6: Export and visualize results
    logger.info("\n💾 Phase 6: Export and visualization...")
    
    # Export results
    exported_files = export_training_results(evaluation_results, config)
    
    # Create visualizations
    training_results = {
        'evaluation_results': evaluation_results,
        'confusion_matrices': create_confusion_matrices(evaluation_results),
        'model_names': list(ensemble_results['base_predictions'].keys()) + (['meta_model'] if use_meta_model else []),
        'metrics_df': create_performance_summary(evaluation_results),
        'y_train': y_train,
        'y_test': y_test
    }
    
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
        'performance_summary': create_performance_summary(evaluation_results),
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
                       choices=['xgboost', 'lightgbm', 'catboost', 'random_forest', 'svm'],
                       help='Models to train (default: all)')
    parser.add_argument('--data-file', type=str, help='Path to training data CSV')
    parser.add_argument('--target-column', type=str, default='target', 
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
    
    # Data collection or loading
    if args.collect_data:
        logger.info("🔄 Collecting fresh data using Alpaca...")
        try:
            collector = AlpacaDataCollector()
            df = collector.collect_and_engineer_features()
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
    
    # Get feature columns (all except target)
    feature_columns = [col for col in df.columns if col != args.target_column]
    logger.info(f"📊 Using {len(feature_columns)} features")
    
    # Prepare training data
    try:
        X_train, X_test, y_train, y_test = prepare_training_data(
            df, feature_columns, args.target_column, config
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
