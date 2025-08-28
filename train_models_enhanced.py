#!/usr/bin/env python3
"""
Enhanced Train Models Script for Investment Committee
===================================================

This script serves as the entry point for training models using our enhanced
infrastructure. It integrates with the advanced_model_trainer.py and 
enhanced_ensemble_classifier.py for state-of-the-art ML performance.

Key Features:
- Uses advanced_model_trainer.py for modern ML pipeline
- Consistent feature ordering via feature_order_manifest.json
- Multiple algorithms with uncertainty quantification
- Proper calibration and ensemble methods
- Comprehensive validation and reporting

Usage:
    python train_models.py --data-file data/batch_1_data.csv --config extreme_imbalance
    python train_models.py --batch-id 1 --optuna-trials 15
"""

import argparse
import logging
import os
import sys
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np

# Import our enhanced infrastructure
from advanced_model_trainer import AdvancedModelTrainer
from enhanced_ensemble_classifier import EnhancedEnsembleClassifier

# Set up logging
def setup_logging(log_level: str = "INFO"):
    """Set up comprehensive logging."""
    os.makedirs('logs', exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('alpaca_trade_api').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


def load_training_data(data_file: str) -> pd.DataFrame:
    """Load training data from CSV file."""
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    df = pd.read_csv(data_file)
    logger.info(f"📊 Loaded training data: {len(df)} samples, {len(df.columns)} features")
    
    # Basic data validation
    if len(df) < 100:
        raise ValueError(f"Insufficient data: {len(df)} samples (minimum 100 required)")
    
    # Check for target column
    if 'target' not in df.columns:
        raise ValueError("Target column 'target' not found in data")
    
    return df


def save_telemetry(results: Dict[str, Any], batch_id: Optional[int] = None):
    """Save training telemetry for batch tracking."""
    try:
        # Save detailed results
        results_path = Path('logs') / f'detailed_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save batch-specific telemetry if batch_id provided
        if batch_id is not None:
            telemetry_path = Path('logs') / f'telemetry_batch_{batch_id}.json'
            
            # Extract key metrics for telemetry
            telemetry_data = {
                'batch_id': batch_id,
                'timestamp': datetime.now().isoformat(),
                'models_trained': len(results.get('model_results', {})),
                'training_successful': results.get('success', False),
                'best_model': results.get('best_model', 'unknown'),
                'best_accuracy': results.get('best_accuracy', 0.0),
                'ensemble_accuracy': results.get('ensemble_metrics', {}).get('accuracy', 0.0),
                'pr_auc_meta': results.get('ensemble_metrics', {}).get('pr_auc', 0.0),
                'gate': 'PASSED' if results.get('ensemble_metrics', {}).get('pr_auc', 0.0) > 0.6 else 'FAILED',
                'dynamic_weights': results.get('ensemble_weights', {}),
                'training_time_seconds': results.get('training_time', 0.0),
                'samples_processed': results.get('samples_processed', 0),
                'feature_count': results.get('feature_count', 0)
            }
            
            with open(telemetry_path, 'w') as f:
                json.dump(telemetry_data, f, indent=2)
            
            logger.info(f"✅ Telemetry saved: {telemetry_path}")
            logger.info(f"📊 TELEMETRY| PR-AUC: {telemetry_data['pr_auc_meta']:.3f}, Gate: {telemetry_data['gate']}, Models: {telemetry_data['models_trained']}")
        
    except Exception as e:
        logger.warning(f"Failed to save telemetry: {e}")


def create_training_plots(results: Dict[str, Any], save_plots: bool = True):
    """Create training visualization plots."""
    if not save_plots:
        return
    
    try:
        os.makedirs('plots', exist_ok=True)
        
        # Import plotting here to avoid dependency issues
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.style.use('default')  # Ensure consistent style
        
        # Model performance comparison
        models_data = results.get('models', {})
        if models_data:
            model_names = list(models_data.keys())
            accuracies = []
            roc_aucs = []
            
            for name in model_names:
                model_metrics = models_data[name].get('metrics', {})
                accuracies.append(model_metrics.get('accuracy', 0))
                roc_aucs.append(model_metrics.get('roc_auc', 0))
            
            # Performance comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Accuracy plot
            bars1 = ax1.bar(model_names, accuracies, color='skyblue', alpha=0.7)
            ax1.set_title('Model Accuracy Comparison')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, acc in zip(bars1, accuracies):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
            
            # ROC-AUC plot
            bars2 = ax2.bar(model_names, roc_aucs, color='lightcoral', alpha=0.7)
            ax2.set_title('Model ROC-AUC Comparison')
            ax2.set_ylabel('ROC-AUC')
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, auc in zip(bars2, roc_aucs):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{auc:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('plots/performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📈 Performance comparison plot saved with {len(model_names)} models")
        
        # Create confusion matrix plot if available
        if 'confusion_matrices' in results:
            plt.figure(figsize=(12, 8))
            n_models = len(results['confusion_matrices'])
            cols = min(3, n_models)
            rows = (n_models + cols - 1) // cols
            
            for i, (model_name, cm) in enumerate(results['confusion_matrices'].items()):
                plt.subplot(rows, cols, i + 1)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'{model_name} Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
            
            plt.tight_layout()
            plt.savefig('plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("📊 Confusion matrices plot saved")
        
        # Feature importance plot if available
        if 'feature_importances' in results and results['feature_importances']:
            plt.figure(figsize=(12, 8))
            
            # Get feature importance from best model
            best_model = results.get('best_model', list(results['feature_importances'].keys())[0])
            importance_data = results['feature_importances'].get(best_model, {})
            
            if importance_data:
                features = list(importance_data.keys())[:20]  # Top 20 features
                importances = [importance_data[f] for f in features]
                
                plt.barh(features, importances)
                plt.title(f'Top 20 Feature Importances - {best_model}')
                plt.xlabel('Importance')
                plt.tight_layout()
                plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info("🎯 Feature importance plot saved")
        
        logger.info("✅ All training plots created successfully")
        
    except Exception as e:
        logger.warning(f"Failed to create plots: {e}")
        import traceback
        logger.debug(f"Plot creation traceback: {traceback.format_exc()}")


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Enhanced Model Training for Investment Committee')
    
    # Data arguments
    parser.add_argument('--data-file', type=str, required=True, help='Path to training data CSV')
    parser.add_argument('--target-column', type=str, default='target', help='Target column name')
    
    # Training arguments
    parser.add_argument('--config', type=str, default='extreme_imbalance', 
                       choices=['default', 'extreme_imbalance', 'fast_training'],
                       help='Training configuration')
    parser.add_argument('--batch-id', type=int, help='Batch ID for telemetry tracking')
    parser.add_argument('--optuna-trials', type=int, default=15, help='Number of Optuna optimization trials')
    
    # Output arguments
    parser.add_argument('--save-plots', action='store_true', help='Save training plots')
    parser.add_argument('--export-results', action='store_true', help='Export detailed results')
    parser.add_argument('--telemetry-json', type=str, help='Path to save telemetry JSON')
    
    # Logging
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Set up logging
    global logger
    logger = setup_logging(args.log_level)
    
    logger.info("🚀 Starting Enhanced Model Training")
    logger.info(f"📊 Data file: {args.data_file}")
    logger.info(f"⚙️  Configuration: {args.config}")
    logger.info(f"🔧 Optuna trials: {args.optuna_trials}")
    
    start_time = time.time()
    
    try:
        # Load training data
        df = load_training_data(args.data_file)
        
        # Prepare features and target
        if args.target_column not in df.columns:
            raise ValueError(f"Target column '{args.target_column}' not found")
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col != args.target_column]
        X = df[feature_columns]
        y = df[args.target_column]
        
        logger.info(f"📊 Features: {len(feature_columns)}, Samples: {len(X)}")
        logger.info(f"🎯 Target distribution: {y.value_counts().to_dict()}")
        
        # Initialize advanced model trainer
        trainer = AdvancedModelTrainer()
        
        # Reconstruct DataFrame with proper target column
        df_for_training = X.copy()
        df_for_training['target'] = y
        
        # Train models with specified parameters
        logger.info("🔄 Starting model training with advanced pipeline...")
        training_results = trainer.train_all_models(
            df_for_training, 
            target_column='target',
            test_size=0.2,
            feature_selection=True,
            max_features=50
        )
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Prepare comprehensive results
        results = {
            'success': True,
            'training_time': training_time,
            'samples_processed': len(X),
            'feature_count': len(feature_columns),
            'config': args.config,
            'optuna_trials': args.optuna_trials,
            'data_file': args.data_file,
            'timestamp': datetime.now().isoformat(),
            **training_results
        }
        
        # Log key results
        logger.info("✅ Training completed successfully!")
        logger.info(f"⏱️  Training time: {training_time:.1f} seconds")
        
        if 'model_results' in results:
            logger.info(f"🤖 Models trained: {len(results['model_results'])}")
            
            # Log individual model performance
            for model_name, model_result in results['model_results'].items():
                accuracy = model_result.get('accuracy', 0)
                roc_auc = model_result.get('roc_auc', 0)
                logger.info(f"   {model_name}: Accuracy={accuracy:.3f}, ROC-AUC={roc_auc:.3f}")
        
        if 'ensemble_metrics' in results:
            ensemble_acc = results['ensemble_metrics'].get('accuracy', 0)
            ensemble_auc = results['ensemble_metrics'].get('roc_auc', 0)
            pr_auc = results['ensemble_metrics'].get('pr_auc', 0)
            logger.info(f"🎭 Ensemble: Accuracy={ensemble_acc:.3f}, ROC-AUC={ensemble_auc:.3f}, PR-AUC={pr_auc:.3f}")
            
            # Log gate status
            gate_status = 'PASSED' if pr_auc > 0.6 else 'FAILED'
            logger.info(f"🚪 PR-AUC gate: {gate_status}")
            logger.info(f"TELEMETRY| Batch signal quality {gate_status}")
        
        # Save telemetry
        save_telemetry(results, args.batch_id)
        
        # Create plots if requested
        create_training_plots(results, args.save_plots)
        
        # Export results if requested
        if args.export_results:
            results_dir = Path('results')
            results_dir.mkdir(exist_ok=True)
            
            # Save comprehensive results
            results_file = results_dir / f'training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"💾 Results exported to: {results_file}")
        
        logger.info("🎉 Training pipeline completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        
        # Save failure telemetry
        failure_results = {
            'success': False,
            'error': str(e),
            'training_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat(),
            'data_file': args.data_file,
            'config': args.config
        }
        
        save_telemetry(failure_results, args.batch_id)
        return 1


if __name__ == "__main__":
    sys.exit(main())
