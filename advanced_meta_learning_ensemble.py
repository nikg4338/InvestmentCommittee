#!/usr/bin/env python3
"""
Advanced Meta-Learning Ensemble System
====================================

This module implements true meta-learning with sophisticated stacked ensemble
architecture, going beyond simple weighted averaging to use ML for meta-learning.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.model_selection import cross_val_predict, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import catboost as cb
import pickle
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedMetaLearningEnsemble:
    """Advanced meta-learning ensemble with sophisticated feature engineering."""
    
    def __init__(self, use_time_series_cv: bool = True, cv_folds: int = 5):
        """Initialize advanced meta-learning ensemble."""
        self.use_time_series_cv = use_time_series_cv
        self.cv_folds = cv_folds
        self.base_models = {}
        self.meta_learners = {}
        self.best_meta_learner = None
        self.meta_features_scaler = StandardScaler()
        self.is_fitted = False
        
        # Initialize meta-learner candidates
        self._initialize_meta_learners()
        
    def _initialize_meta_learners(self):
        """Initialize various meta-learner candidates."""
        self.meta_learner_candidates = {
            'meta_lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=31,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                random_state=42,
                verbose=-1
            ),
            'meta_catboost': cb.CatBoostClassifier(
                iterations=100,
                learning_rate=0.05,
                depth=4,
                random_state=42,
                verbose=0
            ),
            'meta_rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            'meta_gbm': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                random_state=42
            ),
            'meta_logistic': LogisticRegression(
                C=0.1,
                max_iter=1000,
                random_state=42
            ),
            'meta_ridge': RidgeClassifier(
                alpha=1.0,
                random_state=42
            ),
            'meta_mlp': MLPClassifier(
                hidden_layer_sizes=(64, 32),
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42
            )
        }
    
    def add_base_model(self, name: str, model: Any, predictions: np.ndarray, 
                      probabilities: Optional[np.ndarray] = None):
        """Add a base model with its predictions."""
        self.base_models[name] = {
            'model': model,
            'predictions': predictions,
            'probabilities': probabilities if probabilities is not None else predictions
        }
        logger.debug(f"Added base model: {name}")
    
    def _create_meta_features(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Create sophisticated meta-features from base model predictions."""
        try:
            if not self.base_models:
                raise ValueError("No base models available for meta-feature creation")
            
            logger.info("🔧 Creating advanced meta-features...")
            
            # Get cross-validated predictions from base models
            meta_features_dict = {}
            
            # Set up cross-validation
            if self.use_time_series_cv:
                cv_splitter = TimeSeriesSplit(n_splits=self.cv_folds)
            else:
                from sklearn.model_selection import StratifiedKFold
                cv_splitter = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            
            # Generate CV predictions for each base model
            for name, model_info in self.base_models.items():
                model = model_info['model']
                
                try:
                    # Get cross-validated probability predictions
                    if hasattr(model, 'predict_proba'):
                        cv_probs = cross_val_predict(model, X, y, cv=cv_splitter, method='predict_proba')
                        if cv_probs.shape[1] > 1:
                            meta_features_dict[f'{name}_proba'] = cv_probs[:, 1]  # Positive class probability
                        else:
                            meta_features_dict[f'{name}_proba'] = cv_probs[:, 0]
                    else:
                        # For models without predict_proba, use decision_function or predict
                        if hasattr(model, 'decision_function'):
                            cv_scores = cross_val_predict(model, X, y, cv=cv_splitter, method='decision_function')
                            meta_features_dict[f'{name}_score'] = cv_scores
                        else:
                            cv_preds = cross_val_predict(model, X, y, cv=cv_splitter)
                            meta_features_dict[f'{name}_pred'] = cv_preds
                    
                    logger.debug(f"✅ Generated CV predictions for {name}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to generate CV predictions for {name}: {e}")
                    # Fallback to using stored predictions
                    meta_features_dict[f'{name}_fallback'] = model_info['probabilities']
            
            # Convert to DataFrame
            meta_features_df = pd.DataFrame(meta_features_dict)
            
            # Create interaction features
            model_names = list(meta_features_dict.keys())
            logger.info(f"🔄 Creating interaction features from {len(model_names)} base predictions...")
            
            # Pairwise interactions
            interaction_count = 0
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i < j:  # Only create each interaction once
                        # Multiplicative interaction
                        meta_features_df[f'{model1}_x_{model2}'] = meta_features_df[model1] * meta_features_df[model2]
                        
                        # Additive interaction
                        meta_features_df[f'{model1}_plus_{model2}'] = meta_features_df[model1] + meta_features_df[model2]
                        
                        # Difference
                        meta_features_df[f'{model1}_minus_{model2}'] = meta_features_df[model1] - meta_features_df[model2]
                        
                        interaction_count += 3
            
            logger.info(f"✅ Created {interaction_count} interaction features")
            
            # Statistical aggregations
            logger.info("📊 Creating statistical aggregation features...")
            
            base_predictions = meta_features_df[model_names].values
            
            # Basic statistics
            meta_features_df['pred_mean'] = np.mean(base_predictions, axis=1)
            meta_features_df['pred_std'] = np.std(base_predictions, axis=1)
            meta_features_df['pred_min'] = np.min(base_predictions, axis=1)
            meta_features_df['pred_max'] = np.max(base_predictions, axis=1)
            meta_features_df['pred_median'] = np.median(base_predictions, axis=1)
            meta_features_df['pred_range'] = meta_features_df['pred_max'] - meta_features_df['pred_min']
            
            # Advanced statistics
            meta_features_df['pred_skew'] = pd.DataFrame(base_predictions).skew(axis=1)
            meta_features_df['pred_kurtosis'] = pd.DataFrame(base_predictions).kurtosis(axis=1)
            meta_features_df['pred_var'] = np.var(base_predictions, axis=1)
            
            # Quantiles
            meta_features_df['pred_q25'] = np.percentile(base_predictions, 25, axis=1)
            meta_features_df['pred_q75'] = np.percentile(base_predictions, 75, axis=1)
            meta_features_df['pred_iqr'] = meta_features_df['pred_q75'] - meta_features_df['pred_q25']
            
            # Confidence metrics
            meta_features_df['prediction_confidence'] = 1.0 - meta_features_df['pred_std']
            meta_features_df['prediction_consensus'] = (base_predictions > 0.5).sum(axis=1) / len(model_names)
            
            # Agreement metrics
            meta_features_df['high_agreement'] = (meta_features_df['pred_std'] < 0.1).astype(int)
            meta_features_df['low_agreement'] = (meta_features_df['pred_std'] > 0.3).astype(int)
            
            # Add original feature context (top-k most important features)
            if hasattr(self, 'top_features') and self.top_features:
                logger.info("🎯 Adding top feature context...")
                for i, feature in enumerate(self.top_features[:10]):  # Top 10 features
                    if feature in X.columns:
                        meta_features_df[f'original_{feature}'] = X[feature].values
            
            # Handle any NaN values
            meta_features_df = meta_features_df.fillna(0)
            
            logger.info(f"✅ Created {len(meta_features_df.columns)} meta-features")
            logger.debug(f"Meta-features: {list(meta_features_df.columns)}")
            
            return meta_features_df
            
        except Exception as e:
            logger.error(f"Failed to create meta-features: {e}")
            # Fallback to basic meta-features
            basic_features = pd.DataFrame()
            for name, model_info in self.base_models.items():
                basic_features[name] = model_info['probabilities']
            return basic_features
    
    def _train_meta_learners(self, meta_features: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Train multiple meta-learner candidates and select the best one."""
        try:
            logger.info("🎓 Training meta-learner candidates...")
            
            # Scale meta-features
            meta_features_scaled = self.meta_features_scaler.fit_transform(meta_features)
            meta_features_scaled_df = pd.DataFrame(meta_features_scaled, columns=meta_features.columns)
            
            # Set up cross-validation for meta-learner evaluation
            if self.use_time_series_cv:
                cv_splitter = TimeSeriesSplit(n_splits=3)  # Fewer splits for meta-learning
            else:
                from sklearn.model_selection import StratifiedKFold
                cv_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            meta_results = {}
            
            # Train and evaluate each meta-learner candidate
            for name, model in self.meta_learner_candidates.items():
                try:
                    logger.debug(f"Training meta-learner: {name}")
                    
                    # Clone the model to avoid issues
                    from sklearn.base import clone
                    model_clone = clone(model)
                    
                    # Train the model
                    model_clone.fit(meta_features_scaled_df, y)
                    
                    # Get cross-validated predictions for evaluation
                    if hasattr(model_clone, 'predict_proba'):
                        cv_probs = cross_val_predict(model_clone, meta_features_scaled_df, y, 
                                                   cv=cv_splitter, method='predict_proba')
                        if cv_probs.shape[1] > 1:
                            y_pred_proba = cv_probs[:, 1]
                        else:
                            y_pred_proba = cv_probs[:, 0]
                    else:
                        y_pred_proba = cross_val_predict(model_clone, meta_features_scaled_df, y, cv=cv_splitter)
                    
                    # Calculate metrics
                    y_pred = (y_pred_proba > 0.5).astype(int)
                    
                    accuracy = accuracy_score(y, y_pred)
                    roc_auc = roc_auc_score(y, y_pred_proba)
                    pr_auc = average_precision_score(y, y_pred_proba)
                    f1 = f1_score(y, y_pred)
                    
                    meta_results[name] = {
                        'model': model_clone,
                        'accuracy': accuracy,
                        'roc_auc': roc_auc,
                        'pr_auc': pr_auc,
                        'f1': f1,
                        'cv_predictions': y_pred_proba
                    }
                    
                    logger.debug(f"✅ {name}: PR-AUC={pr_auc:.4f}, ROC-AUC={roc_auc:.4f}, F1={f1:.4f}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to train meta-learner {name}: {e}")
                    continue
            
            if not meta_results:
                raise ValueError("No meta-learners were successfully trained")
            
            # Select best meta-learner based on PR-AUC (most important for imbalanced data)
            best_name = max(meta_results.keys(), key=lambda x: meta_results[x]['pr_auc'])
            best_result = meta_results[best_name]
            
            self.best_meta_learner = best_result['model']
            self.meta_learners = meta_results
            
            logger.info(f"🏆 Best meta-learner: {best_name}")
            logger.info(f"📊 Performance - PR-AUC: {best_result['pr_auc']:.4f}, ROC-AUC: {best_result['roc_auc']:.4f}")
            
            return {
                'best_learner': best_name,
                'best_performance': best_result,
                'all_results': meta_results
            }
            
        except Exception as e:
            logger.error(f"Failed to train meta-learners: {e}")
            # Fallback to simple logistic regression
            self.best_meta_learner = LogisticRegression(random_state=42)
            meta_features_scaled = self.meta_features_scaler.fit_transform(meta_features)
            self.best_meta_learner.fit(meta_features_scaled, y)
            
            return {
                'best_learner': 'fallback_logistic',
                'best_performance': {'pr_auc': 0.0},
                'all_results': {}
            }
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, base_models: Dict[str, Any], 
            top_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Fit the meta-learning ensemble."""
        try:
            logger.info("🚀 Training advanced meta-learning ensemble...")
            
            # Store base models
            self.base_models = {}
            for name, model_info in base_models.items():
                if isinstance(model_info, dict):
                    self.add_base_model(name, model_info.get('model'), 
                                      model_info.get('predictions', np.array([])),
                                      model_info.get('probabilities'))
                else:
                    # Assume it's just the model, generate predictions
                    if hasattr(model_info, 'predict_proba'):
                        probs = model_info.predict_proba(X)[:, 1] if hasattr(model_info, 'predict_proba') else model_info.predict(X)
                    else:
                        probs = model_info.predict(X)
                    self.add_base_model(name, model_info, probs, probs)
            
            # Store top features for context
            self.top_features = top_features
            
            # Create meta-features
            meta_features = self._create_meta_features(X, y)
            
            # Train meta-learners
            meta_results = self._train_meta_learners(meta_features, y)
            
            # Generate final ensemble predictions
            meta_features_scaled = self.meta_features_scaler.transform(meta_features)
            final_predictions = self.best_meta_learner.predict_proba(meta_features_scaled)
            
            if final_predictions.shape[1] > 1:
                final_probs = final_predictions[:, 1]
            else:
                final_probs = final_predictions[:, 0]
            
            # Calculate final ensemble metrics
            final_preds = (final_probs > 0.5).astype(int)
            ensemble_metrics = {
                'accuracy': accuracy_score(y, final_preds),
                'roc_auc': roc_auc_score(y, final_probs),
                'pr_auc': average_precision_score(y, final_probs),
                'f1': f1_score(y, final_preds)
            }
            
            self.is_fitted = True
            
            logger.info("✅ Meta-learning ensemble training completed!")
            logger.info(f"🎯 Final ensemble performance - PR-AUC: {ensemble_metrics['pr_auc']:.4f}")
            
            return {
                'ensemble_metrics': ensemble_metrics,
                'meta_learner_results': meta_results,
                'meta_features_count': len(meta_features.columns),
                'final_predictions': final_probs
            }
            
        except Exception as e:
            logger.error(f"Failed to fit meta-learning ensemble: {e}")
            return {'error': str(e)}
    
    def predict(self, X: pd.DataFrame, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Make predictions using the trained meta-learning ensemble."""
        try:
            if not self.is_fitted:
                raise ValueError("Meta-learning ensemble is not fitted")
            
            # Update base model predictions
            for name, predictions in base_predictions.items():
                if name in self.base_models:
                    self.base_models[name]['predictions'] = predictions
                    self.base_models[name]['probabilities'] = predictions
            
            # Create meta-features for prediction
            # Note: We can't use the same CV approach for prediction, so we use the stored predictions
            meta_features_dict = {}
            
            for name, model_info in self.base_models.items():
                meta_features_dict[f'{name}_proba'] = model_info['probabilities']
            
            meta_features_df = pd.DataFrame(meta_features_dict)
            
            # Create the same interaction and statistical features
            model_names = list(meta_features_dict.keys())
            
            # Pairwise interactions
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i < j:
                        meta_features_df[f'{model1}_x_{model2}'] = meta_features_df[model1] * meta_features_df[model2]
                        meta_features_df[f'{model1}_plus_{model2}'] = meta_features_df[model1] + meta_features_df[model2]
                        meta_features_df[f'{model1}_minus_{model2}'] = meta_features_df[model1] - meta_features_df[model2]
            
            # Statistical aggregations
            base_predictions_array = meta_features_df[model_names].values
            meta_features_df['pred_mean'] = np.mean(base_predictions_array, axis=1)
            meta_features_df['pred_std'] = np.std(base_predictions_array, axis=1)
            meta_features_df['pred_min'] = np.min(base_predictions_array, axis=1)
            meta_features_df['pred_max'] = np.max(base_predictions_array, axis=1)
            meta_features_df['pred_median'] = np.median(base_predictions_array, axis=1)
            meta_features_df['pred_range'] = meta_features_df['pred_max'] - meta_features_df['pred_min']
            meta_features_df['pred_skew'] = pd.DataFrame(base_predictions_array).skew(axis=1)
            meta_features_df['pred_kurtosis'] = pd.DataFrame(base_predictions_array).kurtosis(axis=1)
            meta_features_df['pred_var'] = np.var(base_predictions_array, axis=1)
            meta_features_df['pred_q25'] = np.percentile(base_predictions_array, 25, axis=1)
            meta_features_df['pred_q75'] = np.percentile(base_predictions_array, 75, axis=1)
            meta_features_df['pred_iqr'] = meta_features_df['pred_q75'] - meta_features_df['pred_q25']
            meta_features_df['prediction_confidence'] = 1.0 - meta_features_df['pred_std']
            meta_features_df['prediction_consensus'] = (base_predictions_array > 0.5).sum(axis=1) / len(model_names)
            meta_features_df['high_agreement'] = (meta_features_df['pred_std'] < 0.1).astype(int)
            meta_features_df['low_agreement'] = (meta_features_df['pred_std'] > 0.3).astype(int)
            
            # Add original feature context if available
            if hasattr(self, 'top_features') and self.top_features:
                for i, feature in enumerate(self.top_features[:10]):
                    if feature in X.columns:
                        meta_features_df[f'original_{feature}'] = X[feature].values
            
            # Fill NaN values
            meta_features_df = meta_features_df.fillna(0)
            
            # Scale features
            meta_features_scaled = self.meta_features_scaler.transform(meta_features_df)
            
            # Make predictions
            predictions = self.best_meta_learner.predict_proba(meta_features_scaled)
            
            if predictions.shape[1] > 1:
                return predictions[:, 1]
            else:
                return predictions[:, 0]
                
        except Exception as e:
            logger.error(f"Failed to make predictions with meta-learning ensemble: {e}")
            # Fallback to simple average
            if base_predictions:
                return np.mean(list(base_predictions.values()), axis=0)
            else:
                return np.zeros(len(X))
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the best meta-learner."""
        try:
            if not self.is_fitted or not self.best_meta_learner:
                return {}
            
            if hasattr(self.best_meta_learner, 'feature_importances_'):
                # Tree-based models
                importances = self.best_meta_learner.feature_importances_
            elif hasattr(self.best_meta_learner, 'coef_'):
                # Linear models
                importances = np.abs(self.best_meta_learner.coef_[0])
            else:
                return {}
            
            # Get feature names (we need to reconstruct them)
            # This is a simplified version - in practice, you'd store the feature names
            feature_names = [f'meta_feature_{i}' for i in range(len(importances))]
            
            return dict(zip(feature_names, importances))
            
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {}
    
    def save(self, filepath: str):
        """Save the trained meta-learning ensemble."""
        try:
            ensemble_data = {
                'base_models': self.base_models,
                'best_meta_learner': self.best_meta_learner,
                'meta_features_scaler': self.meta_features_scaler,
                'meta_learners': self.meta_learners,
                'is_fitted': self.is_fitted,
                'use_time_series_cv': self.use_time_series_cv,
                'cv_folds': self.cv_folds
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(ensemble_data, f)
            
            logger.info(f"✅ Meta-learning ensemble saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save meta-learning ensemble: {e}")
    
    def load(self, filepath: str):
        """Load a trained meta-learning ensemble."""
        try:
            with open(filepath, 'rb') as f:
                ensemble_data = pickle.load(f)
            
            self.base_models = ensemble_data['base_models']
            self.best_meta_learner = ensemble_data['best_meta_learner']
            self.meta_features_scaler = ensemble_data['meta_features_scaler']
            self.meta_learners = ensemble_data['meta_learners']
            self.is_fitted = ensemble_data['is_fitted']
            self.use_time_series_cv = ensemble_data['use_time_series_cv']
            self.cv_folds = ensemble_data['cv_folds']
            
            logger.info(f"✅ Meta-learning ensemble loaded: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load meta-learning ensemble: {e}")


# Global advanced ensemble instance
advanced_meta_ensemble = AdvancedMetaLearningEnsemble()
