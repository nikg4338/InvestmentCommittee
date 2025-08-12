#!/usr/bin/env python3
"""
Paper Trading Preparation Pipeline
=================================

Comprehensive system to prepare optimized models for paper trading deployment.
Includes threshold optimization, hyperparameter tuning, and production validation.
"""

import logging
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('paper_trading_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our optimization modules
from utils.production_optimization import (
    optimize_regressor_thresholds, 
    create_production_threshold_config,
    validate_threshold_robustness
)
from utils.hyperparameter_optimization import ProductionHyperparameterOptimizer

# Import existing training infrastructure
from train_models import train_committee_models, prepare_training_data
try:
    from utils.evaluation import ModelEvaluator
except ImportError:
    # Create a minimal evaluator if not available
    class ModelEvaluator:
        def __init__(self):
            pass


class PaperTradingPreparation:
    """
    Complete pipeline for preparing models for paper trading deployment.
    """
    
    def __init__(self, batch_number: int = 1, 
                 portfolio_size: int = 20,
                 risk_tolerance: str = 'moderate'):
        """
        Initialize paper trading preparation pipeline.
        
        Args:
            batch_number: Training batch to optimize
            portfolio_size: Target number of positions
            risk_tolerance: Risk tolerance level
        """
        self.batch_number = batch_number
        self.portfolio_size = portfolio_size
        self.risk_tolerance = risk_tolerance
        
        # Initialize components
        self.evaluator = ModelEvaluator()
        self.hp_optimizer = ProductionHyperparameterOptimizer(
            portfolio_size=portfolio_size,
            risk_tolerance=risk_tolerance
        )
        
        # Storage for optimization results
        self.current_models = {}
        self.optimized_models = {}
        self.threshold_results = {}
        self.hyperparameter_results = {}
        
        logger.info("🚀 Paper Trading Preparation Pipeline Initialized")
        logger.info(f"   Batch: {batch_number}")
        logger.info(f"   Portfolio Size: {portfolio_size}")
        logger.info(f"   Risk Tolerance: {risk_tolerance}")
    
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load and prepare training data for the specified batch.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"📊 Loading training data for batch {self.batch_number}...")
        
        try:
            # Load batch data
            batch_file = f"data/batch_{self.batch_number}_data.csv"
            if not os.path.exists(batch_file):
                # Try alternative naming
                batch_file = f"alpaca_training_data_batches_{self.batch_number}.csv"
            
            if os.path.exists(batch_file):
                data = pd.read_csv(batch_file)
                logger.info(f"✅ Loaded {len(data)} samples from {batch_file}")
            else:
                # Fallback to main training file
                logger.warning(f"Batch file not found, using main training data")
                data = pd.read_csv("alpaca_training_data.csv")
            
            # Prepare features and target
            feature_columns = [col for col in data.columns if col not in ['target', 'symbol', 'timestamp']]
            X = data[feature_columns]
            y = data['target'] if 'target' in data.columns else data.iloc[:, -1]  # Last column as target
            
            # Train-test split (use last 20% as test set for temporal consistency)
            split_idx = int(len(data) * 0.8)
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            logger.info(f"📈 Data split: {len(X_train)} train, {len(X_test)} test")
            logger.info(f"📊 Target distribution - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"❌ Failed to load training data: {e}")
            raise
    
    def load_current_models(self) -> Dict[str, Any]:
        """
        Load currently trained models for optimization.
        
        Returns:
            Dictionary of current models
        """
        logger.info("🔧 Loading current trained models...")
        
        try:
            # Load models from the standard training pipeline
            model_files = [
                f"models/batch_{self.batch_number}_xgboost_classifier.pkl",
                f"models/batch_{self.batch_number}_lightgbm_classifier.pkl", 
                f"models/batch_{self.batch_number}_catboost_classifier.pkl",
                f"models/batch_{self.batch_number}_randomforest_classifier.pkl",
                f"models/batch_{self.batch_number}_xgboost_regressor.pkl",
                f"models/batch_{self.batch_number}_lightgbm_regressor.pkl",
                f"models/batch_{self.batch_number}_catboost_regressor.pkl",
                f"models/batch_{self.batch_number}_randomforest_regressor.pkl",
                f"models/batch_{self.batch_number}_svm_classifier.pkl"
            ]
            
            models = {}
            
            for model_file in model_files:
                if os.path.exists(model_file):
                    try:
                        model = joblib.load(model_file)
                        model_name = os.path.basename(model_file).replace('.pkl', '')
                        models[model_name] = model
                        logger.info(f"✅ Loaded {model_name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load {model_file}: {e}")
                else:
                    logger.warning(f"⚠️ Model file not found: {model_file}")
            
            if not models:
                logger.error("❌ No models found! Please train models first.")
                raise FileNotFoundError("No trained models available")
            
            self.current_models = models
            logger.info(f"📚 Loaded {len(models)} models for optimization")
            
            return models
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise
    
    def optimize_thresholds(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Optimize prediction thresholds for all models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of optimized thresholds
        """
        logger.info("🎯 Starting threshold optimization...")
        
        try:
            # Focus on regressor models first (they had the 0.01 threshold issue)
            threshold_results = optimize_regressor_thresholds(
                self.current_models, X_test, y_test, 
                portfolio_size=self.portfolio_size
            )
            
            # Extend to classifier models with portfolio focus
            for model_name, model in self.current_models.items():
                if 'classifier' in model_name and model_name not in threshold_results:
                    logger.info(f"🎯 Optimizing threshold for {model_name}...")
                    
                    try:
                        # Get predictions
                        if hasattr(model, 'predict_proba'):
                            y_proba = model.predict_proba(X_test)[:, 1]
                        else:
                            continue
                        
                        # Use portfolio-focused optimization
                        from utils.production_optimization import find_portfolio_optimal_threshold
                        threshold, metrics = find_portfolio_optimal_threshold(
                            y_test.values, y_proba,
                            portfolio_size=self.portfolio_size,
                            risk_tolerance=self.risk_tolerance
                        )
                        
                        threshold_results[model_name] = {
                            'threshold': threshold,
                            'precision': metrics.get('precision', 0),
                            'recall': metrics.get('recall', 0),
                            'f1': metrics.get('f1', 0),
                            'portfolio_size': metrics.get('portfolio_size', 0)
                        }
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Threshold optimization failed for {model_name}: {e}")
            
            self.threshold_results = threshold_results
            
            # Save threshold configuration
            create_production_threshold_config(
                threshold_results, 
                f"config/production_thresholds_batch_{self.batch_number}.json"
            )
            
            logger.info("✅ Threshold optimization complete!")
            return threshold_results
            
        except Exception as e:
            logger.error(f"❌ Threshold optimization failed: {e}")
            raise
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                n_trials: int = 50) -> Dict[str, Dict[str, Any]]:
        """
        Optimize model hyperparameters using advanced techniques.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_trials: Number of optimization trials per model
            
        Returns:
            Dictionary of optimization results
        """
        logger.info("🧠 Starting hyperparameter optimization...")
        
        try:
            # Select models to optimize (focus on best performers)
            models_to_optimize = ['xgboost', 'lightgbm', 'catboost']
            
            hyperparameter_results = self.hp_optimizer.optimize_all_models(
                X_train, y_train, 
                models_to_optimize=models_to_optimize,
                n_trials=n_trials
            )
            
            self.hyperparameter_results = hyperparameter_results
            
            # Save optimization results
            self.hp_optimizer.save_optimization_results(
                f"models/batch_{self.batch_number}_optimized_params"
            )
            
            logger.info("✅ Hyperparameter optimization complete!")
            return hyperparameter_results
            
        except Exception as e:
            logger.error(f"❌ Hyperparameter optimization failed: {e}")
            raise
    
    def create_production_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Create production-ready models with optimized parameters.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of production models
        """
        logger.info("🏭 Creating production models...")
        
        try:
            # Create optimized models
            optimized_models = self.hp_optimizer.create_production_models(
                self.hyperparameter_results, X_train, y_train
            )
            
            self.optimized_models = optimized_models
            
            # Save production models
            model_dir = f"models/production_batch_{self.batch_number}"
            os.makedirs(model_dir, exist_ok=True)
            
            for model_name, model in optimized_models.items():
                model_file = os.path.join(model_dir, f"{model_name}.pkl")
                joblib.dump(model, model_file)
                logger.info(f"💾 Saved {model_name} to {model_file}")
            
            logger.info("✅ Production models created!")
            return optimized_models
            
        except Exception as e:
            logger.error(f"❌ Production model creation failed: {e}")
            raise
    
    def validate_production_readiness(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Validate that production models meet trading requirements.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Validation results
        """
        logger.info("✅ Validating production readiness...")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'batch_number': self.batch_number,
            'portfolio_size': self.portfolio_size,
            'risk_tolerance': self.risk_tolerance,
            'models_validated': {},
            'overall_status': 'UNKNOWN'
        }
        
        try:
            # Combine optimized and current models for validation
            all_models = {**self.current_models, **self.optimized_models}
            
            # Validate threshold robustness
            if self.threshold_results:
                thresholds = {name: result['threshold'] for name, result in self.threshold_results.items()}
                robustness_results = validate_threshold_robustness(
                    all_models, X_test, y_test, thresholds
                )
                validation_results['threshold_robustness'] = robustness_results
            
            # Validate individual models
            min_precision = 0.25
            min_stability = 0.7
            production_ready_count = 0
            
            for model_name, model in all_models.items():
                logger.info(f"🔍 Validating {model_name}...")
                
                try:
                    # Get predictions
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_test)[:, 1]
                    elif hasattr(model, 'predict'):
                        y_pred_raw = model.predict(X_test)
                        y_proba = 1 / (1 + np.exp(-y_pred_raw))  # Sigmoid for regressors
                    else:
                        continue
                    
                    # Apply threshold if available
                    threshold = self.threshold_results.get(model_name, {}).get('threshold', 0.5)
                    y_pred = (y_proba >= threshold).astype(int)
                    
                    # Calculate validation metrics
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    portfolio_size = np.sum(y_pred)
                    
                    # Get stability score if available
                    stability = robustness_results.get(model_name, {}).get('stability_score', 0)
                    
                    # Production readiness criteria
                    meets_precision = precision >= min_precision
                    meets_stability = stability >= min_stability
                    reasonable_portfolio = 5 <= portfolio_size <= 50  # Reasonable range
                    
                    is_production_ready = meets_precision and meets_stability and reasonable_portfolio
                    
                    if is_production_ready:
                        production_ready_count += 1
                    
                    validation_results['models_validated'][model_name] = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'portfolio_size': portfolio_size,
                        'stability_score': stability,
                        'threshold': threshold,
                        'meets_precision': meets_precision,
                        'meets_stability': meets_stability,
                        'reasonable_portfolio': reasonable_portfolio,
                        'production_ready': is_production_ready
                    }
                    
                    status_emoji = "✅" if is_production_ready else "⚠️"
                    logger.info(f"{status_emoji} {model_name}: P={precision:.3f}, R={recall:.3f}, "
                               f"Portfolio={portfolio_size}, Stable={stability:.3f}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Validation failed for {model_name}: {e}")
                    validation_results['models_validated'][model_name] = {'error': str(e)}
            
            # Overall status determination
            total_models = len(validation_results['models_validated'])
            if production_ready_count >= max(1, total_models // 2):
                validation_results['overall_status'] = 'PRODUCTION_READY'
                logger.info(f"🎉 PRODUCTION READY: {production_ready_count}/{total_models} models meet criteria")
            else:
                validation_results['overall_status'] = 'NEEDS_IMPROVEMENT'
                logger.warning(f"⚠️ NEEDS IMPROVEMENT: Only {production_ready_count}/{total_models} models ready")
            
            # Save validation results
            validation_file = f"reports/production_validation_batch_{self.batch_number}.json"
            os.makedirs(os.path.dirname(validation_file), exist_ok=True)
            with open(validation_file, 'w') as f:
                json.dump(validation_results, f, indent=2)
            
            logger.info(f"📊 Validation results saved to {validation_file}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Production validation failed: {e}")
            validation_results['overall_status'] = 'VALIDATION_FAILED'
            validation_results['error'] = str(e)
            return validation_results
    
    def create_paper_trading_config(self) -> Dict[str, Any]:
        """
        Create complete configuration for paper trading deployment.
        
        Returns:
            Paper trading configuration
        """
        logger.info("📋 Creating paper trading configuration...")
        
        config = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'batch_number': self.batch_number,
                'portfolio_size': self.portfolio_size,
                'risk_tolerance': self.risk_tolerance,
                'preparation_version': '1.0'
            },
            'model_paths': {},
            'thresholds': {},
            'hyperparameters': {},
            'trading_parameters': {
                'max_positions': self.portfolio_size,
                'position_sizing': 'equal_weight',
                'rebalance_frequency': 'daily',
                'risk_limits': {
                    'max_position_size': 0.1,  # 10% max per position
                    'max_sector_exposure': 0.3,  # 30% max per sector
                    'stop_loss': 0.05,  # 5% stop loss
                    'take_profit': 0.15  # 15% take profit
                }
            },
            'monitoring': {
                'performance_metrics': ['precision', 'recall', 'portfolio_return', 'sharpe_ratio'],
                'alert_thresholds': {
                    'min_precision': 0.25,
                    'max_drawdown': 0.1,
                    'min_daily_volume': 100000
                }
            }
        }
        
        # Add model paths
        for model_name in self.optimized_models.keys():
            model_file = f"models/production_batch_{self.batch_number}/{model_name}.pkl"
            config['model_paths'][model_name] = model_file
        
        # Add optimized thresholds
        for model_name, threshold_info in self.threshold_results.items():
            config['thresholds'][model_name] = threshold_info['threshold']
        
        # Add hyperparameters
        for model_type, hp_info in self.hyperparameter_results.items():
            if 'best_params' in hp_info:
                config['hyperparameters'][model_type] = hp_info['best_params']
        
        # Save configuration
        config_file = f"config/paper_trading_config_batch_{self.batch_number}.json"
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"📋 Paper trading config saved to {config_file}")
        return config
    
    def run_complete_preparation(self, n_trials: int = 50) -> Dict[str, Any]:
        """
        Run the complete paper trading preparation pipeline.
        
        Args:
            n_trials: Number of hyperparameter optimization trials
            
        Returns:
            Complete preparation results
        """
        logger.info("🚀 Starting complete paper trading preparation pipeline...")
        
        try:
            # 1. Load data
            X_train, X_test, y_train, y_test = self.load_training_data()
            
            # 2. Load current models
            self.load_current_models()
            
            # 3. Optimize thresholds
            self.optimize_thresholds(X_test, y_test)
            
            # 4. Optimize hyperparameters
            self.optimize_hyperparameters(X_train, y_train, n_trials)
            
            # 5. Create production models
            self.create_production_models(X_train, y_train)
            
            # 6. Validate production readiness
            validation_results = self.validate_production_readiness(X_test, y_test)
            
            # 7. Create paper trading configuration
            trading_config = self.create_paper_trading_config()
            
            # Prepare final results
            final_results = {
                'preparation_complete': True,
                'batch_number': self.batch_number,
                'models_optimized': len(self.optimized_models),
                'thresholds_optimized': len(self.threshold_results),
                'production_status': validation_results['overall_status'],
                'trading_config_path': f"config/paper_trading_config_batch_{self.batch_number}.json",
                'next_steps': []
            }
            
            # Determine next steps
            if validation_results['overall_status'] == 'PRODUCTION_READY':
                final_results['next_steps'] = [
                    "Deploy models to paper trading environment",
                    "Set up real-time data feeds", 
                    "Configure position sizing and risk management",
                    "Start paper trading with daily monitoring"
                ]
                logger.info("🎉 READY FOR PAPER TRADING DEPLOYMENT!")
            else:
                final_results['next_steps'] = [
                    "Review model performance issues",
                    "Increase hyperparameter optimization trials",
                    "Consider ensemble methods",
                    "Retrain with more data if available"
                ]
                logger.warning("⚠️ Additional optimization recommended before deployment")
            
            # Save summary report
            summary_file = f"reports/paper_trading_preparation_summary_batch_{self.batch_number}.json"
            os.makedirs(os.path.dirname(summary_file), exist_ok=True)
            with open(summary_file, 'w') as f:
                json.dump(final_results, f, indent=2)
            
            logger.info(f"📊 Preparation summary saved to {summary_file}")
            logger.info("✅ Paper trading preparation pipeline complete!")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Paper trading preparation failed: {e}")
            raise


def main():
    """Main execution function."""
    
    # Configuration
    BATCH_NUMBER = 1
    PORTFOLIO_SIZE = 20
    RISK_TOLERANCE = 'moderate'  # conservative, moderate, aggressive
    N_TRIALS = 50  # Hyperparameter optimization trials
    
    logger.info("🚀 Paper Trading Preparation - Starting...")
    
    try:
        # Initialize preparation pipeline
        preparation = PaperTradingPreparation(
            batch_number=BATCH_NUMBER,
            portfolio_size=PORTFOLIO_SIZE,
            risk_tolerance=RISK_TOLERANCE
        )
        
        # Run complete preparation
        results = preparation.run_complete_preparation(n_trials=N_TRIALS)
        
        # Print final status
        print("\n" + "="*80)
        print("📊 PAPER TRADING PREPARATION COMPLETE")
        print("="*80)
        print(f"Batch Number: {results['batch_number']}")
        print(f"Models Optimized: {results['models_optimized']}")
        print(f"Thresholds Optimized: {results['thresholds_optimized']}")
        print(f"Production Status: {results['production_status']}")
        print(f"Trading Config: {results['trading_config_path']}")
        print("\nNext Steps:")
        for i, step in enumerate(results['next_steps'], 1):
            print(f"  {i}. {step}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"❌ Preparation pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
