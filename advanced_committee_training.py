#!/usr/bin/env python3
"""
Advanced Committee of Five Training with Improvements
====================================================

This script implements the advanced improvements for handling extreme class imbalance:
1. Per-model threshold tuning
2. Probability calibration
3. SMOTE for synthetic minority sampling
4. Out-of-fold stacking
5. Combined over/under sampling
"""

import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Advanced imports
try:
    from sklearn.calibration import CalibratedClassifierCV
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

# Import our existing components
from data_collection_alpaca import AlpacaDataCollector
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.catboost_model import CatBoostModel
from models.random_forest_model import RandomForestModel
from models.svc_model import SVMClassifier
from utils.helpers import compute_classification_metrics_with_threshold, find_optimal_threshold

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AdvancedCommitteeTrainer:
    """Advanced trainer implementing all the improvements for class imbalance."""
    
    def __init__(self, use_calibration: bool = True, use_smote: bool = True):
        self.use_calibration = use_calibration and CALIBRATION_AVAILABLE
        self.use_smote = use_smote and IMBLEARN_AVAILABLE
        self.models = {}
        self.thresholds = {}
        self.metrics = {}
        
        logger.info(f"Advanced trainer initialized:")
        logger.info(f"  Calibration: {self.use_calibration}")
        logger.info(f"  SMOTE: {self.use_smote}")
    
    def prepare_balanced_data(self, X_train: pd.DataFrame, y_train: pd.Series, method: str = 'smoteenn') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare balanced training data using advanced techniques."""
        
        original_counts = y_train.value_counts().to_dict()
        logger.info(f"Original distribution: {original_counts}")
        
        if not self.use_smote:
            logger.info("Using original data without balancing")
            return X_train, y_train
        
        try:
            # Ensure we have enough samples for SMOTE
            min_samples = min(y_train.value_counts())
            if min_samples < 2:
                logger.warning("Too few minority samples for SMOTE, using original data")
                return X_train, y_train
            
            if method == 'smote':
                # Basic SMOTE
                k_neighbors = min(5, min_samples - 1)
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
                logger.info("Applied SMOTE")
                
            elif method == 'smoteenn':
                # Combined over/under sampling
                smoteenn = SMOTEENN(random_state=42)
                X_balanced, y_balanced = smoteenn.fit_resample(X_train, y_train)
                logger.info("Applied SMOTEENN")
                
            # Convert back to pandas
            X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
            y_balanced = pd.Series(y_balanced, name=y_train.name)
            
            balanced_counts = y_balanced.value_counts().to_dict()
            logger.info(f"Balanced distribution: {balanced_counts}")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            logger.error(f"Balancing failed: {e}, using original data")
            return X_train, y_train
    
    def create_calibrated_model(self, base_model, X_train, y_train, model_name: str):
        """Wrap model in calibration if available and feasible."""
        if self.use_calibration and hasattr(base_model, 'predict_proba'):
            try:
                # Check if we have enough samples for calibration
                min_class_count = min(y_train.value_counts())
                if min_class_count < 3:
                    logger.warning(f"Too few minority samples ({min_class_count}) for calibration of {model_name}")
                    return base_model.model if hasattr(base_model, 'model') else base_model
                
                # Use 2-fold for small datasets
                cv_folds = min(2, min_class_count)
                base_estimator = base_model.model if hasattr(base_model, 'model') else base_model
                calibrated = CalibratedClassifierCV(base_estimator, method="isotonic", cv=cv_folds)
                calibrated.fit(X_train, y_train)
                logger.info(f"Applied calibration to {model_name} with {cv_folds} folds")
                return calibrated
            except Exception as e:
                logger.warning(f"Calibration failed for {model_name}: {e}")
                return base_model.model if hasattr(base_model, 'model') else base_model
        return base_model.model if hasattr(base_model, 'model') else base_model
    
    def train_with_threshold_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                  X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Train all models with threshold tuning."""
        
        models = {}
        thresholds = {}
        metrics = {}
        
        # 1. XGBoost with calibration
        logger.info("Training XGBoost...")
        xgb_model = XGBoostModel()
        
        # Balance data for XGBoost
        X_balanced, y_balanced = self.prepare_balanced_data(X_train, y_train, method='smote')
        
        # Train base model
        xgb_model.train(X_balanced, y_balanced)
        
        # Apply calibration
        models['xgboost'] = self.create_calibrated_model(xgb_model, X_balanced, y_balanced, 'xgboost')
        
        # Tune threshold
        val_proba = models['xgboost'].predict_proba(X_val)[:, 1]
        threshold, _ = find_optimal_threshold(y_val, val_proba, metric='f1')
        thresholds['xgboost'] = threshold
        metrics['xgboost'] = compute_classification_metrics_with_threshold(y_val, val_proba, threshold)
        
        logger.info(f"XGBoost optimal threshold: {threshold:.4f}, F1: {metrics['xgboost']['f1']:.4f}")
        
        # 2. LightGBM with calibration
        logger.info("Training LightGBM...")
        lgb_model = LightGBMModel()
        X_balanced, y_balanced = self.prepare_balanced_data(X_train, y_train, method='smote')
        lgb_model.train(X_balanced, y_balanced)
        
        models['lightgbm'] = self.create_calibrated_model(lgb_model, X_balanced, y_balanced, 'lightgbm')
        
        val_proba = models['lightgbm'].predict_proba(X_val)[:, 1]
        threshold, _ = find_optimal_threshold(y_val, val_proba, metric='f1')
        thresholds['lightgbm'] = threshold
        metrics['lightgbm'] = compute_classification_metrics_with_threshold(y_val, val_proba, threshold)
        
        logger.info(f"LightGBM optimal threshold: {threshold:.4f}, F1: {metrics['lightgbm']['f1']:.4f}")
        
        # 3. CatBoost (already has auto_class_weights)
        logger.info("Training CatBoost...")
        cb_model = CatBoostModel()
        cb_model.train(X_train, y_train)  # Use original data since it has built-in balancing
        models['catboost'] = cb_model.model
        
        val_proba = models['catboost'].predict_proba(X_val)[:, 1]
        threshold, _ = find_optimal_threshold(y_val, val_proba, metric='f1')
        thresholds['catboost'] = threshold
        metrics['catboost'] = compute_classification_metrics_with_threshold(y_val, val_proba, threshold)
        
        logger.info(f"CatBoost optimal threshold: {threshold:.4f}, F1: {metrics['catboost']['f1']:.4f}")
        
        # 4. Random Forest with calibration
        logger.info("Training Random Forest...")
        rf_model = RandomForestModel()
        X_balanced, y_balanced = self.prepare_balanced_data(X_train, y_train, method='smoteenn')
        rf_model.train(X_balanced, y_balanced)
        
        models['random_forest'] = self.create_calibrated_model(rf_model, X_balanced, y_balanced, 'random_forest')
        
        val_proba = models['random_forest'].predict_proba(X_val)[:, 1]
        threshold, _ = find_optimal_threshold(y_val, val_proba, metric='f1')
        thresholds['random_forest'] = threshold
        metrics['random_forest'] = compute_classification_metrics_with_threshold(y_val, val_proba, threshold)
        
        logger.info(f"Random Forest optimal threshold: {threshold:.4f}, F1: {metrics['random_forest']['f1']:.4f}")
        
        # 5. SVM with calibration
        logger.info("Training SVM...")
        svc_model = SVMClassifier()
        X_balanced, y_balanced = self.prepare_balanced_data(X_train, y_train, method='smote')
        svc_model.train(X_balanced, y_balanced)
        
        models['svc'] = self.create_calibrated_model(svc_model, X_balanced, y_balanced, 'svc')
        
        val_proba = models['svc'].predict_proba(X_val)[:, 1]
        threshold, _ = find_optimal_threshold(y_val, val_proba, metric='f1')
        thresholds['svc'] = threshold
        metrics['svc'] = compute_classification_metrics_with_threshold(y_val, val_proba, threshold)
        
        logger.info(f"SVM optimal threshold: {threshold:.4f}, F1: {metrics['svc']['f1']:.4f}")
        
        return models, thresholds, metrics
    
    def out_of_fold_stacking(self, X: pd.DataFrame, y: pd.Series, n_folds: int = 5):
        """Implement out-of-fold stacking to prevent overfitting."""
        
        logger.info(f"Starting {n_folds}-fold out-of-fold stacking...")
        
        # Initialize arrays for out-of-fold predictions
        n_samples = len(X)
        n_models = 5  # XGBoost, LightGBM, CatBoost, RF, SVM
        oof_predictions = np.zeros((n_samples, n_models))
        
        # Stratified K-Fold split
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Processing fold {fold + 1}/{n_folds}...")
            
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train base models on this fold
            fold_models, fold_thresholds, fold_metrics = self.train_with_threshold_tuning(
                X_fold_train, y_fold_train, X_fold_val, y_fold_val
            )
            
            # Generate out-of-fold predictions
            model_names = ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'svc']
            for i, model_name in enumerate(model_names):
                if model_name in fold_models:
                    pred_proba = fold_models[model_name].predict_proba(X_fold_val)[:, 1]
                    oof_predictions[val_idx, i] = pred_proba
        
        # Train meta-model on out-of-fold predictions
        logger.info("Training meta-model on out-of-fold predictions...")
        meta_model = LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight='balanced',  # ← FIXED: Added balanced class weights
            solver='liblinear'  # Better for small datasets with balanced weights
        )
        meta_model.fit(oof_predictions, y)
        
        # ← NEW: Find optimal threshold for meta-model
        logger.info("Finding optimal threshold for meta-model...")
        meta_proba_train = meta_model.predict_proba(oof_predictions)[:, 1]
        
        # Simple threshold optimization (you can import from utils.evaluation for advanced version)
        from sklearn.metrics import f1_score
        thresholds = np.arange(0.01, 1.0, 0.01)
        best_threshold = 0.5
        best_f1 = 0.0
        
        for threshold in thresholds:
            y_pred = (meta_proba_train >= threshold).astype(int)
            f1 = f1_score(y, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        logger.info(f"Optimal meta-model threshold: {best_threshold:.3f} (F1: {best_f1:.3f})")
        
        # Final validation split for evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train final models on full training set
        final_models, final_thresholds, final_metrics = self.train_with_threshold_tuning(
            X_train, y_train, X_test, y_test
        )
        
        # Generate test predictions for meta-model
        test_predictions = np.zeros((len(X_test), n_models))
        model_names = ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'svc']
        for i, model_name in enumerate(model_names):
            if model_name in final_models:
                test_predictions[:, i] = final_models[model_name].predict_proba(X_test)[:, 1]
        
        # Meta-model predictions
        meta_proba = meta_model.predict_proba(test_predictions)[:, 1]
        meta_threshold, _ = find_optimal_threshold(y_test, meta_proba, metric='f1')
        meta_metrics = compute_classification_metrics_with_threshold(y_test, meta_proba, meta_threshold)
        
        final_models['stacked'] = meta_model
        final_thresholds['stacked'] = meta_threshold
        final_metrics['stacked'] = meta_metrics
        
        logger.info(f"Meta-model optimal threshold: {meta_threshold:.4f}, F1: {meta_metrics['f1']:.4f}")
        
        return final_models, final_thresholds, final_metrics


def main():
    """Main function demonstrating the advanced improvements."""
    
    logger.info("🚀 Starting Advanced Committee of Five training...")
    
    # Collect data
    collector = AlpacaDataCollector()
    df = collector.collect_training_data(batch_numbers=[1], max_symbols_per_batch=10)
    
    if len(df) == 0:
        logger.error("No training data available")
        return
    
    # Prepare features
    feature_columns = [col for col in df.columns if col not in ['target', 'ticker']]
    X = df[feature_columns]
    y = df['target']
    
    logger.info(f"Dataset: {len(X)} samples, {len(feature_columns)} features")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Initialize advanced trainer
    trainer = AdvancedCommitteeTrainer(use_calibration=True, use_smote=True)
    
    # Run out-of-fold stacking training
    models, thresholds, metrics = trainer.out_of_fold_stacking(X, y, n_folds=3)
    
    # Print final results
    logger.info("\n🎉 Advanced Training Complete!")
    logger.info("📊 Final Model Performance Summary:")
    for model_name, model_metrics in metrics.items():
        f1 = model_metrics.get('f1', 0)
        roc_auc = model_metrics.get('roc_auc', 0)
        threshold = thresholds.get(model_name, 0.5)
        logger.info(f"   {model_name}: F1={f1:.4f}, ROC-AUC={roc_auc:.4f}, Threshold={threshold:.4f}")


if __name__ == '__main__':
    main()
