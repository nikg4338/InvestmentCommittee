#!/usr/bin/env python3
"""
Stacked Ensemble Classifier with Meta-Learning
==============================================

Advanced ensemble architecture that uses a meta-learner to combine base model predictions.
This approach is superior to simple weighted averaging as it can learn complex non-linear
combinations of base model outputs.

Features:
- Stacked generalization with cross-validation
- Diversity-promoting ensemble selection
- Meta-learner with sophisticated combining logic
- Uncertainty quantification and confidence scoring
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class StackedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Advanced stacked ensemble classifier with meta-learning and diversity promotion.
    """
    
    def __init__(self, 
                 base_models: Dict[str, Any],
                 meta_learner: Optional[Any] = None,
                 cv_folds: int = 5,
                 diversity_threshold: float = 0.1,
                 use_class_weights: bool = True,
                 positive_threshold: float = 0.35):
        """
        Initialize the stacked ensemble.
        
        Args:
            base_models: Dictionary of {name: model} base classifiers
            meta_learner: Meta-learner model (default: LogisticRegression)
            cv_folds: Number of cross-validation folds for stacking
            diversity_threshold: Minimum correlation difference between models to include
            use_class_weights: Whether to use class weights in meta-learner
            positive_threshold: Threshold for positive class prediction
        """
        self.base_models = base_models
        self.meta_learner = meta_learner or LogisticRegression(
            random_state=42,
            class_weight='balanced' if use_class_weights else None
        )
        self.cv_folds = cv_folds
        self.diversity_threshold = diversity_threshold
        self.use_class_weights = use_class_weights
        self.positive_threshold = positive_threshold
        
        # Fitted attributes
        self.fitted_base_models_ = {}
        self.meta_features_ = None
        self.model_weights_ = {}
        self.diversity_scores_ = {}
        self.is_fitted_ = False
        
        logger.info(f"🎭 Stacked Ensemble initialized with {len(base_models)} base models")
        logger.info(f"   Meta-learner: {type(self.meta_learner).__name__}")
        logger.info(f"   CV folds: {cv_folds}, Diversity threshold: {diversity_threshold}")
    
    def _calculate_model_diversity(self, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate diversity scores between models to promote ensemble diversity."""
        logger.info("📊 Calculating model diversity...")
        
        diversity_scores = {}
        model_names = list(predictions.keys())
        
        for i, model1 in enumerate(model_names):
            diversity_sum = 0
            count = 0
            
            for j, model2 in enumerate(model_names):
                if i != j:
                    # Calculate correlation between predictions
                    correlation = np.corrcoef(predictions[model1], predictions[model2])[0, 1]
                    diversity_sum += 1 - abs(correlation)  # Higher score for lower correlation
                    count += 1
            
            diversity_scores[model1] = diversity_sum / count if count > 0 else 0
        
        logger.info(f"   Diversity scores: {diversity_scores}")
        return diversity_scores
    
    def _select_diverse_models(self, 
                             predictions: Dict[str, np.ndarray],
                             y_true: np.ndarray) -> Dict[str, Any]:
        """Select diverse models that complement each other well."""
        logger.info("🎯 Selecting diverse models for ensemble...")
        
        # Calculate individual model performance
        model_scores = {}
        for name, preds in predictions.items():
            model_scores[name] = roc_auc_score(y_true, preds)
        
        # Calculate diversity scores
        diversity_scores = self._calculate_model_diversity(predictions)
        
        # Combined score: performance + diversity
        combined_scores = {}
        for name in predictions.keys():
            performance_score = model_scores[name]
            diversity_score = diversity_scores[name]
            combined_scores[name] = 0.7 * performance_score + 0.3 * diversity_score
        
        # Select top models
        sorted_models = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        selected_models = {}
        
        for name, score in sorted_models:
            if name in self.base_models:
                selected_models[name] = self.base_models[name]
                logger.info(f"   Selected {name}: score={score:.4f} (perf={model_scores[name]:.4f}, div={diversity_scores[name]:.4f})")
        
        self.diversity_scores_ = diversity_scores
        return selected_models
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'StackedEnsembleClassifier':
        """
        Fit the stacked ensemble using cross-validation for meta-features.
        """
        logger.info(f"🚀 Training stacked ensemble with {X.shape[0]} samples...")
        
        # Step 1: Generate meta-features using cross-validation
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        base_predictions = {}
        
        # Generate cross-validated predictions for each base model
        for i, (name, model) in enumerate(self.base_models.items()):
            logger.info(f"   Training base model: {name}")
            
            try:
                # Get cross-validated predictions
                cv_preds = cross_val_predict(
                    model, X, y, cv=skf, method='predict_proba'
                )
                
                if cv_preds.ndim > 1:
                    cv_preds = cv_preds[:, 1]  # Get positive class probabilities
                
                meta_features[:, i] = cv_preds
                base_predictions[name] = cv_preds
                
                # Fit model on full data for later use
                model.fit(X, y)
                self.fitted_base_models_[name] = model
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to train {name}: {e}")
                meta_features[:, i] = 0.5  # Neutral predictions
        
        # Step 2: Select diverse models
        self.fitted_base_models_ = self._select_diverse_models(base_predictions, y)
        
        # Step 3: Train meta-learner on meta-features
        logger.info("🧠 Training meta-learner...")
        feature_names = list(self.fitted_base_models_.keys())
        meta_df = pd.DataFrame(meta_features[:, :len(feature_names)], columns=feature_names)
        
        # Add interaction features for meta-learner
        if len(feature_names) >= 2:
            # Add pairwise interactions
            for i in range(len(feature_names)):
                for j in range(i+1, len(feature_names)):
                    interaction_name = f"{feature_names[i]}_x_{feature_names[j]}"
                    meta_df[interaction_name] = meta_df[feature_names[i]] * meta_df[feature_names[j]]
            
            # Add agreement features
            meta_df['avg_confidence'] = meta_df[feature_names].mean(axis=1)
            meta_df['max_confidence'] = meta_df[feature_names].max(axis=1)
            meta_df['min_confidence'] = meta_df[feature_names].min(axis=1)
            meta_df['confidence_std'] = meta_df[feature_names].std(axis=1)
        
        self.meta_features_ = meta_df
        self.meta_learner.fit(meta_df, y)
        
        # Calculate model weights from meta-learner coefficients (if available)
        if hasattr(self.meta_learner, 'coef_'):
            base_coeffs = self.meta_learner.coef_[0][:len(feature_names)]
            total_weight = np.sum(np.abs(base_coeffs))
            if total_weight > 0:
                self.model_weights_ = {
                    name: abs(coeff) / total_weight 
                    for name, coeff in zip(feature_names, base_coeffs)
                }
            else:
                self.model_weights_ = {name: 1/len(feature_names) for name in feature_names}
        else:
            self.model_weights_ = {name: 1/len(feature_names) for name in feature_names}
        
        logger.info(f"✅ Stacked ensemble training complete!")
        logger.info(f"   Model weights: {self.model_weights_}")
        
        self.is_fitted_ = True
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions using the stacked ensemble."""
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get base model predictions
        base_preds = np.zeros((X.shape[0], len(self.fitted_base_models_)))
        
        for i, (name, model) in enumerate(self.fitted_base_models_.items()):
            try:
                if hasattr(model, 'predict_proba'):
                    preds = model.predict_proba(X)
                    if preds.ndim > 1:
                        preds = preds[:, 1]  # Get positive class probabilities
                else:
                    preds = model.predict(X)
                
                base_preds[:, i] = preds
                
            except Exception as e:
                logger.warning(f"⚠️ Prediction failed for {name}: {e}")
                base_preds[:, i] = 0.5  # Neutral predictions
        
        # Create meta-features
        feature_names = list(self.fitted_base_models_.keys())
        meta_df = pd.DataFrame(base_preds, columns=feature_names)
        
        # Add interaction features (same as training)
        if len(feature_names) >= 2:
            for i in range(len(feature_names)):
                for j in range(i+1, len(feature_names)):
                    interaction_name = f"{feature_names[i]}_x_{feature_names[j]}"
                    meta_df[interaction_name] = meta_df[feature_names[i]] * meta_df[feature_names[j]]
            
            meta_df['avg_confidence'] = meta_df[feature_names].mean(axis=1)
            meta_df['max_confidence'] = meta_df[feature_names].max(axis=1)
            meta_df['min_confidence'] = meta_df[feature_names].min(axis=1)
            meta_df['confidence_std'] = meta_df[feature_names].std(axis=1)
        
        # Get meta-learner predictions
        if hasattr(self.meta_learner, 'predict_proba'):
            ensemble_probs = self.meta_learner.predict_proba(meta_df)[:, 1]
        else:
            ensemble_probs = self.meta_learner.predict(meta_df)
        
        # Return as standard sklearn format
        return np.column_stack([1 - ensemble_probs, ensemble_probs])
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate binary predictions using the stacked ensemble."""
        probs = self.predict_proba(X)[:, 1]
        return (probs > self.positive_threshold).astype(int)
    
    def get_uncertainty_scores(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate uncertainty scores for predictions."""
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before uncertainty calculation")
        
        # Get individual model predictions
        base_preds = {}
        for name, model in self.fitted_base_models_.items():
            try:
                if hasattr(model, 'predict_proba'):
                    preds = model.predict_proba(X)[:, 1]
                else:
                    preds = model.predict(X)
                base_preds[name] = preds
            except Exception as e:
                logger.warning(f"⚠️ Uncertainty calculation failed for {name}: {e}")
                base_preds[name] = np.full(X.shape[0], 0.5)
        
        # Calculate uncertainty metrics
        all_preds = np.column_stack(list(base_preds.values()))
        
        uncertainty_scores = {
            'prediction_variance': np.var(all_preds, axis=1),
            'prediction_std': np.std(all_preds, axis=1),
            'prediction_range': np.max(all_preds, axis=1) - np.min(all_preds, axis=1),
            'agreement_score': 1 - np.std(all_preds, axis=1),  # Higher = more agreement
            'ensemble_confidence': np.abs(self.predict_proba(X)[:, 1] - 0.5) * 2
        }
        
        return uncertainty_scores
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from meta-learner."""
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before getting feature importance")
        
        if hasattr(self.meta_learner, 'feature_importances_'):
            # For tree-based meta-learners
            feature_names = list(self.meta_features_.columns)
            importances = self.meta_learner.feature_importances_
            return dict(zip(feature_names, importances))
        
        elif hasattr(self.meta_learner, 'coef_'):
            # For linear meta-learners
            feature_names = list(self.meta_features_.columns)
            coefficients = np.abs(self.meta_learner.coef_[0])
            return dict(zip(feature_names, coefficients))
        
        else:
            return self.model_weights_
    
    def evaluate_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Comprehensive ensemble evaluation."""
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before evaluation")
        
        # Get predictions
        y_pred_proba = self.predict_proba(X)[:, 1]
        y_pred = self.predict(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba),
            'f1_score': f1_score(y, y_pred),
            'threshold_used': self.positive_threshold
        }
        
        # Add uncertainty metrics
        uncertainty_scores = self.get_uncertainty_scores(X)
        metrics.update({
            'avg_prediction_variance': np.mean(uncertainty_scores['prediction_variance']),
            'avg_agreement_score': np.mean(uncertainty_scores['agreement_score']),
            'avg_ensemble_confidence': np.mean(uncertainty_scores['ensemble_confidence'])
        })
        
        return metrics


def create_enhanced_ensemble(base_models: Dict[str, Any], 
                           X_train: pd.DataFrame, 
                           y_train: pd.Series,
                           use_stacking: bool = True) -> StackedEnsembleClassifier:
    """
    Create and train an enhanced ensemble classifier.
    
    Args:
        base_models: Dictionary of {name: model} base classifiers
        X_train: Training features
        y_train: Training targets
        use_stacking: Whether to use stacking (True) or simple averaging (False)
    
    Returns:
        Trained StackedEnsembleClassifier
    """
    logger.info(f"🎭 Creating enhanced ensemble with {len(base_models)} base models...")
    
    if use_stacking:
        # Use sophisticated meta-learner
        meta_learner = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42
        )
    else:
        # Use simple linear combination
        meta_learner = LogisticRegression(
            random_state=42,
            class_weight='balanced'
        )
    
    ensemble = StackedEnsembleClassifier(
        base_models=base_models,
        meta_learner=meta_learner,
        cv_folds=5,
        diversity_threshold=0.1,
        use_class_weights=True,
        positive_threshold=0.35
    )
    
    # Train the ensemble
    ensemble.fit(X_train, y_train)
    
    logger.info("✅ Enhanced ensemble training complete!")
    return ensemble


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from catboost import CatBoostClassifier
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, 
                             weights=[0.8, 0.2], random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    # Create base models
    base_models = {
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'svm': SVC(probability=True, random_state=42),
        'catboost': CatBoostClassifier(iterations=100, verbose=False, random_state=42)
    }
    
    # Train ensemble
    ensemble = create_enhanced_ensemble(base_models, X, y)
    
    # Evaluate
    metrics = ensemble.evaluate_ensemble(X, y)
    print("Ensemble metrics:", metrics)
