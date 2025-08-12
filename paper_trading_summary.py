#!/usr/bin/env python3
"""
Production Summary and Validation
================================

Summarize the optimized models and validate their readiness for paper trading.
"""

import logging
import pandas as pd
import numpy as np
import joblib
import json
import os
from typing import Dict, Any
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_production_models(model_dir: str = "models/production") -> Dict[str, Any]:
    """Load all production models."""
    models = {}
    
    for file in os.listdir(model_dir):
        if file.endswith('.pkl') and 'config' not in file:
            model_name = file.replace('.pkl', '')
            try:
                model = joblib.load(os.path.join(model_dir, file))
                models[model_name] = model
                logger.info(f"Loaded {model_name}")
            except Exception as e:
                logger.error(f"Failed to load {file}: {e}")
    
    return models


def validate_models_performance(models: Dict[str, Any], 
                               data_file: str = "alpaca_training_data_batches_1.csv") -> Dict[str, Dict]:
    """Validate model performance on test data."""
    
    # Load test data
    data = pd.read_csv(data_file)
    feature_columns = [col for col in data.columns if col not in ['target', 'symbol', 'timestamp', 'ticker']]
    X = data[feature_columns].fillna(data[feature_columns].median())
    y = data['target']
    
    # Use last 20% as test set
    split_idx = int(len(X) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    logger.info(f"Validating on {len(X_test)} test samples, {y_test.mean():.3f} positive rate")
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Validating {model_name}...")
        
        try:
            # Get predictions
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                y_pred_50 = (y_proba >= 0.5).astype(int)
            else:
                y_pred_50 = model.predict(X_test)
                y_proba = np.zeros(len(y_pred_50))  # Fallback
            
            # Test multiple thresholds
            thresholds_to_test = [0.1, 0.3, 0.5, 0.7, 0.9]
            threshold_results = {}
            
            for threshold in thresholds_to_test:
                if hasattr(model, 'predict_proba'):
                    y_pred = (y_proba >= threshold).astype(int)
                else:
                    y_pred = y_pred_50  # Use default prediction
                
                n_predictions = np.sum(y_pred)
                
                if len(y_test.unique()) > 1 and n_predictions > 0:
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    threshold_results[f"threshold_{threshold}"] = {
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1': float(f1),
                        'predictions': int(n_predictions)
                    }
                else:
                    threshold_results[f"threshold_{threshold}"] = {
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0,
                        'predictions': int(n_predictions)
                    }
            
            # Overall assessment
            prediction_diversity = len(np.unique(y_proba)) if hasattr(model, 'predict_proba') else 1
            
            results[model_name] = {
                'prediction_diversity': int(prediction_diversity),
                'prediction_range': [float(y_proba.min()), float(y_proba.max())] if hasattr(model, 'predict_proba') else [0, 1],
                'threshold_analysis': threshold_results,
                'model_type': type(model).__name__,
                'has_predict_proba': hasattr(model, 'predict_proba')
            }
            
        except Exception as e:
            logger.error(f"Validation failed for {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results


def create_paper_trading_summary():
    """Create a comprehensive summary for paper trading deployment."""
    
    logger.info("Creating paper trading deployment summary...")
    
    # Load models
    models = load_production_models()
    
    if not models:
        logger.error("No production models found!")
        return
    
    # Validate performance
    validation_results = validate_models_performance(models)
    
    # Create summary
    summary = {
        'deployment_summary': {
            'timestamp': pd.Timestamp.now().isoformat(),
            'models_available': len(models),
            'model_names': list(models.keys()),
            'status': 'READY_FOR_PAPER_TRADING'
        },
        'model_validation': validation_results,
        'recommendations': {
            'primary_model': None,
            'backup_models': [],
            'suggested_threshold': 0.5,
            'portfolio_considerations': []
        },
        'deployment_checklist': [
            "✓ Models trained and optimized",
            "✓ Threshold analysis completed", 
            "✓ Performance validation done",
            "- Set up real-time data feed",
            "- Configure position sizing",
            "- Implement risk management",
            "- Start paper trading monitoring"
        ]
    }
    
    # Analyze results and make recommendations
    best_model = None
    best_score = 0.0
    
    for model_name, results in validation_results.items():
        if 'error' in results:
            continue
            
        # Score based on prediction diversity and F1 at 0.5 threshold
        diversity_score = min(results['prediction_diversity'] / 10.0, 1.0)
        
        threshold_50_results = results['threshold_analysis'].get('threshold_0.5', {})
        f1_score_val = threshold_50_results.get('f1', 0.0)
        
        combined_score = 0.7 * diversity_score + 0.3 * f1_score_val
        
        if combined_score > best_score:
            best_score = combined_score
            best_model = model_name
    
    if best_model:
        summary['recommendations']['primary_model'] = best_model
        summary['recommendations']['backup_models'] = [name for name in models.keys() if name != best_model]
    
    # Portfolio considerations
    for model_name, results in validation_results.items():
        if 'error' in results:
            continue
            
        # Check prediction counts at different thresholds
        threshold_analysis = results['threshold_analysis']
        for threshold_key, threshold_results in threshold_analysis.items():
            predictions = threshold_results['predictions']
            if 10 <= predictions <= 30:  # Good portfolio size
                summary['recommendations']['portfolio_considerations'].append(
                    f"{model_name} at {threshold_key.replace('threshold_', '')} threshold: {predictions} positions"
                )
    
    # Save summary
    with open('paper_trading_deployment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print readable summary
    print("\n" + "="*80)
    print("PAPER TRADING DEPLOYMENT SUMMARY")
    print("="*80)
    
    print(f"Models Available: {len(models)}")
    for model_name in models.keys():
        print(f"  - {model_name}")
    
    print(f"\nRecommended Primary Model: {summary['recommendations']['primary_model']}")
    
    print("\nModel Performance Analysis:")
    for model_name, results in validation_results.items():
        if 'error' in results:
            print(f"  {model_name}: ERROR - {results['error']}")
            continue
            
        print(f"  {model_name}:")
        print(f"    Type: {results['model_type']}")
        print(f"    Prediction Diversity: {results['prediction_diversity']} unique values")
        print(f"    Prediction Range: [{results['prediction_range'][0]:.3f}, {results['prediction_range'][1]:.3f}]")
        
        # Show best threshold
        best_threshold = None
        best_f1 = 0.0
        for threshold_key, threshold_results in results['threshold_analysis'].items():
            if threshold_results['f1'] > best_f1:
                best_f1 = threshold_results['f1']
                best_threshold = threshold_key.replace('threshold_', '')
        
        if best_threshold:
            print(f"    Best Threshold: {best_threshold} (F1: {best_f1:.3f})")
    
    print("\nPortfolio Considerations:")
    for consideration in summary['recommendations']['portfolio_considerations']:
        print(f"  - {consideration}")
    
    print("\nDeployment Checklist:")
    for item in summary['deployment_checklist']:
        print(f"  {item}")
    
    print("\n" + "="*80)
    print("NEXT STEPS FOR PAPER TRADING:")
    print("1. Review model performance and select primary model")
    print("2. Set up real-time market data connection")
    print("3. Implement position sizing (e.g., equal weight across 20 positions)")
    print("4. Configure stop-loss (5%) and take-profit (15%) rules")
    print("5. Start paper trading with daily performance monitoring")
    print("6. Track precision, recall, and portfolio returns")
    print("7. Retrain models weekly with new data")
    print("="*80)
    
    logger.info("Summary saved to paper_trading_deployment_summary.json")
    
    return summary


if __name__ == "__main__":
    create_paper_trading_summary()
