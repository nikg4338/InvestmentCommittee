#!/usr/bin/env python3
"""
Simplified Paper Trading Optimization
====================================

Production optimization without unicode characters for Windows compatibility.
"""

import logging
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score

# Set up logging without unicode
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def optimize_simple_thresholds(models: Dict[str, Any], 
                             X_test: pd.DataFrame, 
                             y_test: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Simple threshold optimization for production models.
    """
    logger.info("Starting threshold optimization...")
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Optimizing {model_name}...")
        
        try:
            # Get predictions
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'predict'):
                y_pred_raw = model.predict(X_test)
                # Convert regressor output to probabilities
                y_proba = 1 / (1 + np.exp(-y_pred_raw))
            else:
                continue
            
            # Test different thresholds
            best_threshold = 0.5
            best_f1 = 0.0
            
            # Focus on higher precision thresholds for trading
            thresholds = np.concatenate([
                np.linspace(0.1, 0.9, 50),  # Main range
                np.percentile(y_proba, [70, 80, 85, 90, 95, 99])  # Percentile-based
            ])
            
            for threshold in np.unique(thresholds):
                y_pred = (y_proba >= threshold).astype(int)
                
                if np.sum(y_pred) == 0:
                    continue
                    
                try:
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    # Portfolio constraints (aim for 10-30 positions)
                    n_positions = np.sum(y_pred)
                    if 5 <= n_positions <= 40 and precision >= 0.2:
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = threshold
                            
                except Exception:
                    continue
            
            # Final metrics with best threshold
            y_pred_final = (y_proba >= best_threshold).astype(int)
            precision_final = precision_score(y_test, y_pred_final, zero_division=0)
            recall_final = recall_score(y_test, y_pred_final, zero_division=0)
            f1_final = f1_score(y_test, y_pred_final, zero_division=0)
            
            results[model_name] = {
                'threshold': best_threshold,
                'precision': precision_final,
                'recall': recall_final,
                'f1': f1_final,
                'portfolio_size': np.sum(y_pred_final)
            }
            
            logger.info(f"  Threshold: {best_threshold:.4f}, P: {precision_final:.3f}, "
                       f"R: {recall_final:.3f}, Portfolio: {np.sum(y_pred_final)}")
            
        except Exception as e:
            logger.error(f"Error optimizing {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results


def run_quick_optimization(batch_number: int = 1):
    """
    Quick optimization for already trained models.
    """
    logger.info("Starting quick optimization...")
    
    # Load data
    try:
        data_file = f"alpaca_training_data_batches_{batch_number}.csv"
        if not os.path.exists(data_file):
            data_file = "alpaca_training_data.csv"
        
        data = pd.read_csv(data_file)
        logger.info(f"Loaded {len(data)} samples from {data_file}")
        
        # Prepare features and target
        feature_columns = [col for col in data.columns if col not in ['target', 'symbol', 'timestamp', 'ticker']]
        X = data[feature_columns]
        y = data['target'] if 'target' in data.columns else data.iloc[:, -1]
        
        # Use last 20% as test set
        split_idx = int(len(data) * 0.8)
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        
        logger.info(f"Test set: {len(X_test)} samples, {y_test.mean():.3f} positive rate")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Load models
    models = {}
    model_patterns = [
        f"models/batch_{batch_number}_*.pkl",
        "models/saved/*.pkl"
    ]
    
    import glob
    for pattern in model_patterns:
        for model_file in glob.glob(pattern):
            try:
                model = joblib.load(model_file)
                model_name = os.path.basename(model_file).replace('.pkl', '')
                models[model_name] = model
                logger.info(f"Loaded {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load {model_file}: {e}")
    
    if not models:
        logger.error("No models found!")
        return
    
    # Optimize thresholds
    threshold_results = optimize_simple_thresholds(models, X_test, y_test)
    
    # Save results
    output_dir = f"config"
    os.makedirs(output_dir, exist_ok=True)
    
    config = {
        'batch_number': batch_number,
        'optimization_timestamp': datetime.now().isoformat(),
        'test_samples': len(X_test),
        'test_positive_rate': float(y_test.mean()),
        'model_thresholds': threshold_results
    }
    
    config_file = os.path.join(output_dir, f"optimized_thresholds_batch_{batch_number}.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Results saved to {config_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION SUMMARY")
    print("="*60)
    
    for model_name, result in threshold_results.items():
        if 'error' in result:
            print(f"{model_name}: ERROR - {result['error']}")
        else:
            print(f"{model_name}:")
            print(f"  Threshold: {result['threshold']:.4f}")
            print(f"  Precision: {result['precision']:.3f}")
            print(f"  Recall: {result['recall']:.3f}")
            print(f"  F1: {result['f1']:.3f}")
            print(f"  Portfolio Size: {result['portfolio_size']}")
            print()
    
    print("="*60)
    print(f"Configuration saved to: {config_file}")


if __name__ == "__main__":
    run_quick_optimization(batch_number=1)
