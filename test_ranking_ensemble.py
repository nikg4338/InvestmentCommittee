#!/usr/bin/env python3
"""
Test ranking-based ensemble approach for extreme imbalance
"""

import pandas as pd
import numpy as np
from train_models import collect_alpaca_training_data, prepare_training_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ranking_ensemble():
    """Test ranking-based ensemble approach instead of threshold-based"""
    
    logger.info("🔍 Testing ranking-based ensemble approach...")
    
    # Mock some model probabilities for our test case
    # Based on the real results we saw in the logs
    test_size = 119
    
    # Create mock probabilities similar to what we observed
    np.random.seed(42)
    
    # Mock model probabilities (based on the ranges we saw in logs)
    xgboost_proba = np.random.uniform(0.002, 0.042, test_size)
    lightgbm_proba = np.random.uniform(0.002, 0.012, test_size)  
    catboost_proba = np.random.uniform(0.000, 0.039, test_size)
    rf_proba = np.random.uniform(0.000, 0.016, test_size)
    svm_proba = np.random.uniform(0.000, 0.001, test_size)
    
    # Make the positive case (index 22) have distinctive characteristics
    positive_idx = 22
    xgboost_proba[positive_idx] = 0.035  # High for XGBoost
    lightgbm_proba[positive_idx] = 0.010  # Medium for LightGBM  
    catboost_proba[positive_idx] = 0.025  # Medium for CatBoost
    rf_proba[positive_idx] = 0.012  # High for RF
    svm_proba[positive_idx] = 0.0008  # Medium for SVM
    
    # Create true labels (only index 22 is positive)
    y_test = np.zeros(test_size)
    y_test[positive_idx] = 1
    
    models = {
        'xgboost': xgboost_proba,
        'lightgbm': lightgbm_proba,
        'catboost': catboost_proba,
        'random_forest': rf_proba,
        'svm': svm_proba
    }
    
    logger.info(f"Testing with positive case at index {positive_idx}")
    logger.info(f"Positive case probabilities: XGB={xgboost_proba[positive_idx]:.6f}, LGB={lightgbm_proba[positive_idx]:.6f}, CAT={catboost_proba[positive_idx]:.6f}, RF={rf_proba[positive_idx]:.6f}, SVM={svm_proba[positive_idx]:.6f}")
    
    # 🎯 RANKING-BASED APPROACH
    model_top_indices = {}
    num_candidates = 5  # Top 5 candidates per model
    
    for model_name, test_proba in models.items():
        # Get indices of top-ranked samples for this model
        top_indices = np.argsort(test_proba)[-num_candidates:]  # Highest probabilities
        model_top_indices[model_name] = top_indices
        
        logger.info(f"🏆 {model_name} top {num_candidates} candidates: indices {top_indices} with probabilities {test_proba[top_indices]}")
        
        # Check if positive case is in top candidates
        if positive_idx in top_indices:
            rank = len(top_indices) - list(top_indices).index(positive_idx)  # 1-based rank
            logger.info(f"   ✅ Positive case (idx {positive_idx}) is ranked #{rank} by {model_name}")
        else:
            # Find actual rank of positive case
            sorted_indices = np.argsort(test_proba)
            actual_rank = len(sorted_indices) - list(sorted_indices).index(positive_idx)
            logger.info(f"   ❌ Positive case (idx {positive_idx}) is ranked #{actual_rank} by {model_name} (outside top {num_candidates})")
    
    # Count votes for each sample across all models
    vote_counts = {}
    for model_name, top_indices in model_top_indices.items():
        for idx in top_indices:
            if idx not in vote_counts:
                vote_counts[idx] = 0
            vote_counts[idx] += 1
    
    # Sort samples by number of votes (most voted = most likely positive)
    sorted_by_votes = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
    logger.info(f"🗳️  Sample vote counts (top 10): {sorted_by_votes[:10]}")
    
    # Check how many votes the positive case got
    positive_votes = vote_counts.get(positive_idx, 0)
    logger.info(f"🎯 Positive case (idx {positive_idx}) received {positive_votes} votes out of {len(models)} models")
    
    # Test different consensus thresholds
    for min_votes in [1, 2, 3, 4, 5]:
        consensus_positives = [idx for idx, votes in sorted_by_votes if votes >= min_votes]
        
        if len(consensus_positives) == 0 and sorted_by_votes:
            # If no consensus, take top voted sample
            consensus_positives = [sorted_by_votes[0][0]]
        
        # Create binary predictions
        y_pred = np.zeros(test_size, dtype=int)
        for idx in consensus_positives:
            y_pred[idx] = 1
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        positive_in_consensus = positive_idx in consensus_positives
        
        logger.info(f"📊 Min votes={min_votes}: {len(consensus_positives)} predictions, F1={f1:.3f}, P={precision:.3f}, R={recall:.3f}, positive_found={positive_in_consensus}")
    
    logger.info("\n🎯 Analysis complete!")
    logger.info("💡 If ranking approach finds the positive case, it should achieve F1 > 0")

if __name__ == "__main__":
    test_ranking_ensemble()
