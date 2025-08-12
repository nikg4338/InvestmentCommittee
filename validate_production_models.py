#!/usr/bin/env python3
"""
Model Validation and Reality Check
Check if the perfect scores are realistic or indicate overfitting.
"""

import sys
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ModelValidator:
    """Validate trained models for production readiness."""
    
    def __init__(self):
        self.models_dir = 'models/production'
        self.training_data_path = 'alpaca_training_data.csv'
        
    def load_fresh_test_data(self):
        """Load and prepare fresh test data."""
        try:
            print("📊 Loading training data for validation...")
            df = pd.read_csv(self.training_data_path)
            
            # Remove non-feature columns
            feature_cols = [col for col in df.columns if col not in ['target', 'target_enhanced', 'symbol', 'ticker', 'date', 'timestamp']]
            
            X = df[feature_cols]
            y = df['target']
            
            # Handle missing values
            X = X.fillna(X.median())
            
            # Use the last 20% as validation set (different from training test set)
            split_idx = int(len(X) * 0.8)
            X_val = X.iloc[split_idx:]
            y_val = y.iloc[split_idx:]
            
            print(f"✅ Validation set: {len(X_val)} samples")
            print(f"✅ Features: {len(feature_cols)}")
            print(f"✅ Target distribution: {y_val.value_counts().to_dict()}")
            
            return X_val, y_val, feature_cols
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None, None, None
    
    def load_production_models(self):
        """Load all production models."""
        models = {}
        
        model_files = [
            'optimized_catboost.pkl',
            'optimized_random_forest.pkl', 
            'optimized_svm.pkl',
            'optimized_xgboost.pkl',
            'optimized_lightgbm.pkl'
        ]
        
        for model_file in model_files:
            model_path = os.path.join(self.models_dir, model_file)
            if os.path.exists(model_path):
                try:
                    model_name = model_file.replace('optimized_', '').replace('.pkl', '')
                    models[model_name] = joblib.load(model_path)
                    print(f"✅ Loaded {model_name}")
                except Exception as e:
                    print(f"❌ Error loading {model_file}: {e}")
            else:
                print(f"⚠️ Model not found: {model_file}")
        
        return models
    
    def validate_model_performance(self, models, X_val, y_val):
        """Validate model performance on fresh data."""
        
        print("\n🔍 VALIDATING MODEL PERFORMANCE")
        print("="*50)
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\n📈 Testing {model_name.upper()}...")
            
            try:
                # Make predictions
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = (y_pred == y_val).mean()
                
                # Detailed classification report
                report = classification_report(y_val, y_pred, output_dict=True)
                
                # Confusion matrix
                cm = confusion_matrix(y_val, y_pred)
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'precision': report['weighted avg']['precision'],
                    'recall': report['weighted avg']['recall'],
                    'f1': report['weighted avg']['f1-score'],
                    'confusion_matrix': cm.tolist(),
                    'class_report': report
                }
                
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Precision: {report['weighted avg']['precision']:.4f}")
                print(f"  Recall: {report['weighted avg']['recall']:.4f}")
                print(f"  F1-Score: {report['weighted avg']['f1-score']:.4f}")
                
                # Check for perfect scores (potential overfitting)
                if accuracy >= 0.999:
                    print(f"  ⚠️ WARNING: Suspiciously high accuracy ({accuracy:.4f})")
                    print(f"      This may indicate data leakage or overfitting")
                elif accuracy >= 0.85:
                    print(f"  ✅ Good performance")
                else:
                    print(f"  ❌ Poor performance")
                
                # Show confusion matrix
                print(f"  Confusion Matrix:")
                print(f"    TN: {cm[0,0]}, FP: {cm[0,1]}")
                print(f"    FN: {cm[1,0]}, TP: {cm[1,1]}")
                
            except Exception as e:
                print(f"  ❌ Error testing {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def check_prediction_diversity(self, models, X_val):
        """Check if models make diverse predictions."""
        
        print(f"\n🎯 CHECKING PREDICTION DIVERSITY")
        print("="*40)
        
        predictions = {}
        
        for model_name, model in models.items():
            try:
                pred = model.predict(X_val)
                predictions[model_name] = pred
                print(f"{model_name}: {np.mean(pred):.3f} average prediction")
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
        
        if len(predictions) >= 2:
            # Check correlation between models
            pred_df = pd.DataFrame(predictions)
            corr_matrix = pred_df.corr()
            
            print(f"\n📊 Model Correlation Matrix:")
            print(corr_matrix.round(3))
            
            # Check if all models agree too much (potential overfitting)
            avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            print(f"\nAverage inter-model correlation: {avg_correlation:.3f}")
            
            if avg_correlation > 0.95:
                print(f"⚠️ WARNING: Models are too correlated ({avg_correlation:.3f})")
                print(f"   This suggests overfitting or data leakage")
            elif avg_correlation > 0.7:
                print(f"✅ Good model diversity ({avg_correlation:.3f})")
            else:
                print(f"❓ Low correlation - may indicate inconsistent training")
    
    def realistic_trading_simulation(self, models, X_val, y_val):
        """Simulate realistic trading conditions."""
        
        print(f"\n🎮 REALISTIC TRADING SIMULATION")
        print("="*40)
        
        # Simulate ensemble predictions (like production system)
        ensemble_predictions = []
        confidence_scores = []
        
        for i in range(len(X_val)):
            sample = X_val.iloc[i:i+1]
            
            # Get predictions from all models
            model_preds = []
            model_probs = []
            
            for model_name, model in models.items():
                try:
                    pred = model.predict(sample)[0]
                    model_preds.append(pred)
                    
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(sample)[0]
                        model_probs.append(max(prob))
                    else:
                        model_probs.append(0.5)  # Default confidence
                        
                except Exception:
                    continue
            
            if model_preds:
                # Ensemble prediction (majority vote)
                ensemble_pred = 1 if np.mean(model_preds) >= 0.5 else 0
                ensemble_predictions.append(ensemble_pred)
                
                # Confidence based on agreement and probability
                agreement = 1.0 - abs(np.mean(model_preds) - 0.5) * 2  # How close to unanimous
                avg_prob = np.mean(model_probs)
                confidence = (agreement + avg_prob) / 2
                confidence_scores.append(confidence)
            else:
                ensemble_predictions.append(0)
                confidence_scores.append(0.5)
        
        # Analyze ensemble performance
        ensemble_accuracy = np.mean(np.array(ensemble_predictions) == y_val.values)
        
        print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
        print(f"Average confidence: {np.mean(confidence_scores):.3f}")
        
        # Trading simulation with confidence thresholds
        thresholds = [0.6, 0.7, 0.8, 0.9]
        
        for threshold in thresholds:
            high_conf_indices = np.array(confidence_scores) >= threshold
            
            if np.sum(high_conf_indices) > 0:
                high_conf_accuracy = np.mean(
                    np.array(ensemble_predictions)[high_conf_indices] == y_val.values[high_conf_indices]
                )
                trade_percentage = np.mean(high_conf_indices) * 100
                
                print(f"  Threshold {threshold}: {high_conf_accuracy:.3f} accuracy on {trade_percentage:.1f}% of samples")
            else:
                print(f"  Threshold {threshold}: No predictions above threshold")
    
    def production_readiness_assessment(self, results, X_val, y_val):
        """Assess if models are ready for production."""
        
        print(f"\n🎯 PRODUCTION READINESS ASSESSMENT")
        print("="*50)
        
        total_models = len(results)
        good_models = 0
        excellent_models = 0
        suspicious_models = 0
        
        for model_name, result in results.items():
            if 'error' in result:
                continue
                
            accuracy = result['accuracy']
            
            if accuracy >= 0.999:
                suspicious_models += 1
                print(f"❌ {model_name}: SUSPICIOUS (accuracy={accuracy:.4f})")
            elif accuracy >= 0.85:
                if accuracy >= 0.95:
                    excellent_models += 1
                    print(f"🌟 {model_name}: EXCELLENT (accuracy={accuracy:.4f})")
                else:
                    good_models += 1
                    print(f"✅ {model_name}: GOOD (accuracy={accuracy:.4f})")
            else:
                print(f"❌ {model_name}: POOR (accuracy={accuracy:.4f})")
        
        print(f"\n📊 SUMMARY:")
        print(f"  Total models: {total_models}")
        print(f"  Good models (85-95%): {good_models}")
        print(f"  Excellent models (95-99%): {excellent_models}")
        print(f"  Suspicious models (>99%): {suspicious_models}")
        
        # Overall recommendation
        if suspicious_models >= 3:
            print(f"\n🚨 RECOMMENDATION: DO NOT USE FOR PAPER TRADING")
            print(f"   Too many models showing perfect/near-perfect scores")
            print(f"   This strongly suggests data leakage or overfitting")
            print(f"   Need to investigate training data and feature engineering")
            return False
        elif good_models + excellent_models >= 3:
            print(f"\n✅ RECOMMENDATION: SAFE FOR PAPER TRADING")
            print(f"   Models show realistic performance")
            print(f"   Good ensemble diversity expected")
            print(f"   Proceed with careful monitoring")
            return True
        else:
            print(f"\n⚠️ RECOMMENDATION: CAUTIOUS PAPER TRADING")
            print(f"   Mixed model performance")
            print(f"   Use conservative position sizing")
            print(f"   Monitor performance closely")
            return None

def main():
    """Main validation function."""
    
    print("🔍 MODEL VALIDATION AND REALITY CHECK")
    print("="*60)
    print("Checking if perfect training scores are realistic...")
    
    validator = ModelValidator()
    
    # Load data and models
    X_val, y_val, features = validator.load_fresh_test_data()
    if X_val is None:
        return
    
    models = validator.load_production_models()
    if not models:
        print("❌ No models found for validation")
        return
    
    # Run validation tests
    results = validator.validate_model_performance(models, X_val, y_val)
    validator.check_prediction_diversity(models, X_val)
    validator.realistic_trading_simulation(models, X_val, y_val)
    ready = validator.production_readiness_assessment(results, X_val, y_val)
    
    print(f"\n🎉 VALIDATION COMPLETE")
    
    if ready is True:
        print(f"✅ Models are ready for paper trading!")
    elif ready is False:
        print(f"❌ Models need further work before trading")
    else:
        print(f"⚠️ Proceed with caution in paper trading")

if __name__ == "__main__":
    main()
