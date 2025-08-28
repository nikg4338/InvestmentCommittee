#!/usr/bin/env python3
"""
Enhanced Ensemble Classifier
============================

Advanced ensemble system with:
- Uncertainty quantification
- Dynamic weighting based on confidence
- Out-of-fold predictions for stacking
- Calibrated probability outputs
- Real-time performance monitoring
"""

import json
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Statistics and ML
from scipy import stats
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

# Neural Networks
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UncertaintyQuantifier:
    """Quantify prediction uncertainty using multiple methods."""
    
    @staticmethod
    def entropy_uncertainty(probabilities: np.ndarray) -> np.ndarray:
        """Calculate entropy-based uncertainty."""
        # Clip probabilities to avoid log(0)
        probs = np.clip(probabilities, 1e-8, 1-1e-8)
        entropy = -probs * np.log2(probs) - (1-probs) * np.log2(1-probs)
        return entropy
    
    @staticmethod
    def confidence_interval_uncertainty(probabilities: np.ndarray, n_samples: int = 1000) -> np.ndarray:
        """Calculate uncertainty based on confidence intervals."""
        # Simulate beta distribution for each prediction
        alpha = probabilities * n_samples
        beta = (1 - probabilities) * n_samples
        
        # Calculate 95% confidence interval width
        ci_lower = stats.beta.ppf(0.025, alpha, beta)
        ci_upper = stats.beta.ppf(0.975, alpha, beta)
        uncertainty = ci_upper - ci_lower
        
        return uncertainty
    
    @staticmethod
    def prediction_variance(predictions_list: List[np.ndarray]) -> np.ndarray:
        """Calculate prediction variance across models."""
        predictions_array = np.array(predictions_list)
        variance = np.var(predictions_array, axis=0)
        return variance
    
    @staticmethod
    def epistemic_uncertainty(predictions_list: List[np.ndarray]) -> np.ndarray:
        """Calculate epistemic (model) uncertainty."""
        mean_pred = np.mean(predictions_list, axis=0)
        epistemic = np.mean([(pred - mean_pred)**2 for pred in predictions_list], axis=0)
        return epistemic


class ModelPerformanceMonitor:
    """Monitor individual model performance in real-time."""
    
    def __init__(self):
        self.performance_history = defaultdict(list)
        self.recent_performance = defaultdict(lambda: defaultdict(list))
        self.decay_factor = 0.95  # For exponential decay
        
    def update_performance(self, model_name: str, prediction: float, actual: Optional[float] = None,
                         confidence: float = 0.0, timestamp: Optional[datetime] = None):
        """Update performance metrics for a model."""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Store prediction info
        pred_info = {
            'timestamp': timestamp,
            'prediction': prediction,
            'confidence': confidence
        }
        
        if actual is not None:
            pred_info['actual'] = actual
            pred_info['error'] = abs(prediction - actual)
            pred_info['correct'] = abs(prediction - actual) < 0.5
        
        self.performance_history[model_name].append(pred_info)
        
        # Keep only recent history (last 1000 predictions)
        if len(self.performance_history[model_name]) > 1000:
            self.performance_history[model_name] = self.performance_history[model_name][-1000:]
    
    def get_recent_accuracy(self, model_name: str, lookback: int = 100) -> float:
        """Get recent accuracy for a model."""
        recent_preds = self.performance_history[model_name][-lookback:]
        if not recent_preds:
            return 0.5  # Default
        
        correct_preds = [p for p in recent_preds if p.get('actual') is not None and p.get('correct', False)]
        if not correct_preds:
            return 0.5
        
        return len(correct_preds) / len([p for p in recent_preds if p.get('actual') is not None])
    
    def get_confidence_accuracy_correlation(self, model_name: str, lookback: int = 100) -> float:
        """Get correlation between confidence and accuracy."""
        recent_preds = self.performance_history[model_name][-lookback:]
        if len(recent_preds) < 10:
            return 0.0
        
        confidences = [p['confidence'] for p in recent_preds if p.get('actual') is not None]
        accuracies = [1.0 if p.get('correct', False) else 0.0 for p in recent_preds if p.get('actual') is not None]
        
        if len(confidences) < 10:
            return 0.0
        
        correlation, _ = stats.pearsonr(confidences, accuracies)
        return correlation if not np.isnan(correlation) else 0.0
    
    def get_model_weight(self, model_name: str, base_weight: float = 1.0) -> float:
        """Calculate dynamic weight for model based on recent performance."""
        recent_accuracy = self.get_recent_accuracy(model_name)
        confidence_correlation = self.get_confidence_accuracy_correlation(model_name)
        
        # Adjust weight based on performance
        accuracy_multiplier = (recent_accuracy - 0.5) * 2  # Scale to [-1, 1]
        confidence_multiplier = max(0, confidence_correlation)  # Only positive correlation helps
        
        # Combine factors
        performance_multiplier = 0.7 * accuracy_multiplier + 0.3 * confidence_multiplier
        
        # Apply multiplier to base weight
        dynamic_weight = base_weight * (1 + performance_multiplier)
        
        # Ensure positive weight
        return max(0.1, dynamic_weight)


class EnhancedEnsembleClassifier:
    """Advanced ensemble classifier with modern ML best practices."""
    
    def __init__(self, models_dir: str = "models/production"):
        """Initialize ensemble with enhanced features."""
        self.models_dir = Path(models_dir)
        self.models = {}
        self.preprocessing_pipelines = {}
        self.model_metadata = {}
        self.feature_order = []
        
        # Advanced components
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.performance_monitor = ModelPerformanceMonitor()
        self.meta_learner = None  # For stacking
        
        # Ensemble parameters
        self.base_weights = {}
        self.confidence_threshold = 0.6
        self.uncertainty_threshold = 0.3
        
        # Performance tracking
        self.prediction_history = []
        self.performance_metrics = {}
        
        logger.info(f"🎯 Enhanced Ensemble Classifier initialized")
        
    def load_models(self) -> None:
        """Load all trained models with metadata."""
        logger.info(f"📥 Loading models from {self.models_dir}")
        
        if not self.models_dir.exists():
            logger.error(f"❌ Models directory does not exist: {self.models_dir}")
            return
        
        # Load feature order from manifest
        manifest_path = Path("models/feature_order_manifest.json")
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                self.feature_order = manifest['feature_order']
        
        # Find all model files
        model_files = list(self.models_dir.glob("enhanced_*.pkl"))
        
        for model_file in model_files:
            try:
                # Extract model name
                model_name = model_file.stem.replace("enhanced_", "")
                if model_name.endswith("_preprocessing"):
                    continue  # Skip preprocessing files
                
                logger.info(f"   Loading {model_name}...")
                
                # Load model
                model = joblib.load(model_file)
                self.models[model_name] = model
                
                # Load preprocessing pipeline
                preprocessing_file = self.models_dir / f"enhanced_{model_name}_preprocessing.pkl"
                if preprocessing_file.exists():
                    preprocessing = joblib.load(preprocessing_file)
                    self.preprocessing_pipelines[model_name] = preprocessing
                
                # Load metadata
                metadata_file = self.models_dir / f"enhanced_{model_name}_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        self.model_metadata[model_name] = metadata
                        
                        # Set base weight from ROC-AUC
                        roc_auc = metadata.get('metrics', {}).get('roc_auc', 0.5)
                        self.base_weights[model_name] = max(0.1, roc_auc)
                
                logger.info(f"   ✅ {model_name} loaded successfully")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to load {model_file}: {e}")
        
        # Load neural network if exists
        nn_path = self.models_dir / "enhanced_neural_network.h5"
        if nn_path.exists() and TENSORFLOW_AVAILABLE:
            try:
                nn_model = tf.keras.models.load_model(nn_path)
                self.models['neural_network'] = nn_model
                
                # Load NN metadata
                nn_metadata_path = self.models_dir / "enhanced_neural_network_metadata.json"
                if nn_metadata_path.exists():
                    with open(nn_metadata_path, 'r') as f:
                        metadata = json.load(f)
                        self.model_metadata['neural_network'] = metadata
                        roc_auc = metadata.get('metrics', {}).get('roc_auc', 0.5)
                        self.base_weights['neural_network'] = max(0.1, roc_auc)
                
                logger.info(f"   ✅ neural_network loaded successfully")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to load neural network: {e}")
        
        logger.info(f"✅ Loaded {len(self.models)} models")
        logger.info(f"   Models: {list(self.models.keys())}")
        
        # Initialize meta-learner for stacking
        self._initialize_meta_learner()
    
    def _initialize_meta_learner(self) -> None:
        """Initialize meta-learner for stacking ensemble."""
        if len(self.models) > 1:
            self.meta_learner = LogisticRegression(
                C=1.0,
                random_state=42,
                max_iter=1000
            )
            logger.info(f"🧠 Meta-learner initialized for stacking")
    
    def _validate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate and reorder features according to manifest."""
        if not self.feature_order:
            logger.warning("⚠️ No feature order loaded, using current order")
            return X
        
        # Check available features
        available_features = [f for f in self.feature_order if f in X.columns]
        missing_features = [f for f in self.feature_order if f not in X.columns]
        
        if missing_features:
            logger.warning(f"⚠️ Missing {len(missing_features)} features")
            # Fill missing features with zeros
            for feature in missing_features:
                X[feature] = 0.0
        
        # Reorder to match training
        X_ordered = X[self.feature_order]
        
        # Handle any remaining NaN values
        X_ordered = X_ordered.fillna(0.0)
        
        return X_ordered
    
    def _get_model_prediction(self, model_name: str, X: np.ndarray) -> Tuple[float, float]:
        """Get prediction and confidence from a single model."""
        model = self.models[model_name]
        
        try:
            if model_name == 'neural_network':
                # Neural network prediction
                pred_proba = model.predict(X, verbose=0)[0][0]
                confidence = abs(pred_proba - 0.5) * 2  # Distance from 0.5, scaled to [0,1]
            else:
                # Sklearn-style prediction
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X)[0][1]
                else:
                    pred_proba = model.predict(X)[0]
                
                confidence = abs(pred_proba - 0.5) * 2  # Distance from 0.5, scaled to [0,1]
            
            return float(pred_proba), float(confidence)
            
        except Exception as e:
            logger.error(f"❌ Prediction failed for {model_name}: {e}")
            return 0.5, 0.0  # Default values
    
    def predict_single(self, X: pd.DataFrame, return_uncertainty: bool = True) -> Dict[str, Any]:
        """Make prediction with uncertainty quantification for a single sample."""
        if len(self.models) == 0:
            logger.error("❌ No models loaded")
            return {'prediction': 0.5, 'confidence': 0.0, 'uncertainty': 1.0}
        
        # Validate and prepare features
        X_validated = self._validate_features(X)
        
        # Collect predictions from all models
        individual_predictions = {}
        individual_confidences = {}
        prediction_list = []
        
        for model_name in self.models.keys():
            try:
                # Prepare data for this model
                if model_name in self.preprocessing_pipelines:
                    X_processed = self.preprocessing_pipelines[model_name].transform(X_validated)
                else:
                    X_processed = X_validated.values
                
                # Get prediction
                pred, conf = self._get_model_prediction(model_name, X_processed)
                
                individual_predictions[model_name] = pred
                individual_confidences[model_name] = conf
                prediction_list.append(pred)
                
                # Update performance monitor
                self.performance_monitor.update_performance(model_name, pred, confidence=conf)
                
            except Exception as e:
                logger.error(f"❌ Prediction failed for {model_name}: {e}")
                continue
        
        if not prediction_list:
            logger.error("❌ All model predictions failed")
            return {'prediction': 0.5, 'confidence': 0.0, 'uncertainty': 1.0}
        
        # Calculate ensemble prediction with dynamic weighting
        ensemble_prediction = self._calculate_weighted_ensemble(individual_predictions, individual_confidences)
        
        # Calculate uncertainty metrics
        uncertainty_metrics = {}
        if return_uncertainty:
            uncertainty_metrics = self._calculate_uncertainty_metrics(prediction_list, individual_confidences)
        
        # Meta-learning prediction (if available)
        meta_prediction = None
        if self.meta_learner is not None and len(prediction_list) > 1:
            try:
                meta_features = np.array(prediction_list).reshape(1, -1)
                meta_prediction = self.meta_learner.predict_proba(meta_features)[0][1]
            except Exception as e:
                logger.warning(f"⚠️ Meta-learner prediction failed: {e}")
        
        # Combine ensemble and meta predictions
        if meta_prediction is not None:
            final_prediction = 0.7 * ensemble_prediction + 0.3 * meta_prediction
        else:
            final_prediction = ensemble_prediction
        
        # Calculate overall confidence
        avg_confidence = np.mean(list(individual_confidences.values()))
        uncertainty_penalty = uncertainty_metrics.get('entropy_uncertainty', 0)
        overall_confidence = avg_confidence * (1 - uncertainty_penalty)
        
        result = {
            'prediction': float(final_prediction),
            'confidence': float(overall_confidence),
            'individual_predictions': individual_predictions,
            'individual_confidences': individual_confidences,
            'meta_prediction': meta_prediction,
            'ensemble_prediction': ensemble_prediction,
            **uncertainty_metrics
        }
        
        # Store prediction for monitoring
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'prediction': final_prediction,
            'confidence': overall_confidence,
            'individual_predictions': individual_predictions
        })
        
        return result
    
    def _calculate_weighted_ensemble(self, predictions: Dict[str, float], 
                                   confidences: Dict[str, float]) -> float:
        """Calculate weighted ensemble prediction with dynamic weights."""
        if not predictions:
            return 0.5
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for model_name, prediction in predictions.items():
            # Get base weight
            base_weight = self.base_weights.get(model_name, 1.0)
            
            # Get dynamic weight based on recent performance
            dynamic_weight = self.performance_monitor.get_model_weight(model_name, base_weight)
            
            # Adjust by confidence
            confidence = confidences.get(model_name, 0.5)
            confidence_weight = 1.0 + (confidence - 0.5)  # Scale around 1.0
            
            # Final weight
            final_weight = dynamic_weight * confidence_weight
            
            weighted_sum += prediction * final_weight
            total_weight += final_weight
        
        if total_weight == 0:
            return 0.5
        
        return weighted_sum / total_weight
    
    def _calculate_uncertainty_metrics(self, predictions: List[float], 
                                     confidences: Dict[str, float]) -> Dict[str, float]:
        """Calculate comprehensive uncertainty metrics."""
        predictions_array = np.array(predictions)
        
        # Entropy uncertainty
        mean_pred = np.mean(predictions_array)
        entropy_uncertainty = self.uncertainty_quantifier.entropy_uncertainty(np.array([mean_pred]))[0]
        
        # Prediction variance
        prediction_variance = np.var(predictions_array)
        
        # Epistemic uncertainty
        epistemic_uncertainty = self.uncertainty_quantifier.epistemic_uncertainty(
            [[p] for p in predictions]
        )[0]
        
        # Confidence interval uncertainty
        ci_uncertainty = self.uncertainty_quantifier.confidence_interval_uncertainty(
            np.array([mean_pred])
        )[0]
        
        # Overall uncertainty (weighted combination)
        overall_uncertainty = (
            0.3 * entropy_uncertainty +
            0.3 * prediction_variance +
            0.2 * epistemic_uncertainty +
            0.2 * ci_uncertainty
        )
        
        return {
            'entropy_uncertainty': float(entropy_uncertainty),
            'prediction_variance': float(prediction_variance),
            'epistemic_uncertainty': float(epistemic_uncertainty),
            'ci_uncertainty': float(ci_uncertainty),
            'overall_uncertainty': float(overall_uncertainty)
        }
    
    def predict_batch(self, X: pd.DataFrame, return_uncertainty: bool = True) -> pd.DataFrame:
        """Make batch predictions with uncertainty quantification."""
        logger.info(f"🔮 Making batch predictions for {len(X)} samples")
        
        results = []
        for idx in range(len(X)):
            sample = X.iloc[[idx]]
            result = self.predict_single(sample, return_uncertainty)
            result['sample_index'] = idx
            results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        logger.info(f"✅ Batch predictions complete")
        logger.info(f"   Mean prediction: {results_df['prediction'].mean():.3f}")
        logger.info(f"   Mean confidence: {results_df['confidence'].mean():.3f}")
        if 'overall_uncertainty' in results_df.columns:
            logger.info(f"   Mean uncertainty: {results_df['overall_uncertainty'].mean():.3f}")
        
        return results_df
    
    def update_with_feedback(self, predictions: List[Dict], actuals: List[float]) -> None:
        """Update ensemble with feedback from actual outcomes."""
        logger.info(f"📈 Updating ensemble with {len(actuals)} feedback samples")
        
        for pred_dict, actual in zip(predictions, actuals):
            # Update individual model performance
            individual_preds = pred_dict.get('individual_predictions', {})
            individual_confs = pred_dict.get('individual_confidences', {})
            
            for model_name, prediction in individual_preds.items():
                confidence = individual_confs.get(model_name, 0.5)
                self.performance_monitor.update_performance(
                    model_name, prediction, actual, confidence
                )
        
        # Update meta-learner if we have enough data
        if len(self.prediction_history) > 50 and self.meta_learner is not None:
            self._retrain_meta_learner()
        
        logger.info(f"✅ Ensemble updated with feedback")
    
    def _retrain_meta_learner(self) -> None:
        """Retrain meta-learner with recent predictions."""
        try:
            # Get recent predictions with actuals
            recent_with_actuals = [
                p for p in self.prediction_history[-200:] 
                if 'actual' in p and 'individual_predictions' in p
            ]
            
            if len(recent_with_actuals) < 20:
                return
            
            # Prepare training data
            X_meta = []
            y_meta = []
            
            for pred_info in recent_with_actuals:
                individual_preds = pred_info['individual_predictions']
                if len(individual_preds) > 1:
                    X_meta.append(list(individual_preds.values()))
                    y_meta.append(pred_info['actual'])
            
            if len(X_meta) > 10:
                X_meta = np.array(X_meta)
                y_meta = np.array(y_meta)
                
                # Retrain meta-learner
                self.meta_learner.fit(X_meta, y_meta)
                logger.info(f"🧠 Meta-learner retrained with {len(X_meta)} samples")
            
        except Exception as e:
            logger.error(f"❌ Meta-learner retraining failed: {e}")
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for all models."""
        summary = {}
        
        for model_name in self.models.keys():
            model_summary = {
                'base_weight': self.base_weights.get(model_name, 1.0),
                'recent_accuracy': self.performance_monitor.get_recent_accuracy(model_name),
                'confidence_correlation': self.performance_monitor.get_confidence_accuracy_correlation(model_name),
                'dynamic_weight': self.performance_monitor.get_model_weight(model_name),
                'total_predictions': len(self.performance_monitor.performance_history[model_name])
            }
            
            # Add metadata if available
            if model_name in self.model_metadata:
                model_summary['training_metrics'] = self.model_metadata[model_name].get('metrics', {})
                model_summary['training_date'] = self.model_metadata[model_name].get('training_date', 'Unknown')
            
            summary[model_name] = model_summary
        
        return summary
    
    def save_ensemble_state(self, filepath: str = "models/production/ensemble_state.json") -> None:
        """Save current ensemble state for persistence."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'base_weights': self.base_weights,
            'confidence_threshold': self.confidence_threshold,
            'uncertainty_threshold': self.uncertainty_threshold,
            'model_performance_summary': self.get_model_performance_summary(),
            'prediction_history_length': len(self.prediction_history)
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"💾 Ensemble state saved to {filepath}")


def main():
    """Demo script for enhanced ensemble."""
    logger.info("🎯 Enhanced Ensemble Classifier Demo")
    
    # Initialize ensemble
    ensemble = EnhancedEnsembleClassifier()
    
    # Load models
    ensemble.load_models()
    
    # Demo prediction (placeholder)
    logger.info("✅ Enhanced Ensemble ready for predictions")
    
    # Show performance summary
    if ensemble.models:
        performance_summary = ensemble.get_model_performance_summary()
        logger.info("📊 Model Performance Summary:")
        for model_name, metrics in performance_summary.items():
            logger.info(f"   {model_name}: weight={metrics['dynamic_weight']:.3f}, accuracy={metrics['recent_accuracy']:.3f}")


if __name__ == "__main__":
    main()
