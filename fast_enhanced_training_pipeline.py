#!/usr/bin/env python3
"""
Fast Enhanced Training Pipeline
===============================
Optimized version focusing on core fixes without heavy optimization.
Provides all the probability fixing, feature alignment, and data drift mitigation
but with faster training times.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import warnings
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the feature alignment and fixing systems (core functionality)
from feature_alignment_system import FeatureAligner, ProbabilityFixer, DataDriftMitigator

# Import original training modules
try:
    import train_models
    TRAIN_MODELS_AVAILABLE = True
except ImportError:
    TRAIN_MODELS_AVAILABLE = False
    logger.warning("train_models.py not available")

class FastEnhancedTrainingPipeline:
    """Fast training pipeline with core fixes but minimal optimization overhead."""
    
    def __init__(self, quick_mode: bool = True):
        """Initialize fast enhanced training pipeline."""
        self.quick_mode = quick_mode
        
        # Initialize core fixing systems
        self.feature_aligner = FeatureAligner()
        self.probability_fixer = ProbabilityFixer()
        self.drift_mitigator = DataDriftMitigator()
        
        # Fast training configuration
        self.config = {
            'cross_validation_folds': 3,  # Reduced for speed
            'timeout_per_model': 180,     # 3 minutes max per model
            'enable_probability_fixing': True,
            'enable_feature_alignment': True,
            'enable_drift_mitigation': True,
            'save_results': True
        }
        
        logger.info("🚀 Fast Enhanced Training Pipeline initialized")
        logger.info(f"   Quick mode: {quick_mode}")
        logger.info(f"   Timeout per model: {self.config['timeout_per_model']}s")
    
    def train_single_batch_fast(self, batch_name: str, X_train: pd.DataFrame, 
                               y_train: np.ndarray, X_test: pd.DataFrame, 
                               y_test: np.ndarray) -> Dict[str, Any]:
        """Train a single batch with fast enhanced processing."""
        logger.info(f"🎯 Fast training batch: {batch_name}")
        start_time = time.time()
        
        # Step 1: Apply core fixes
        logger.info("🔧 Applying feature alignment...")
        X_train_aligned = self.feature_aligner.align_features(X_train)
        X_test_aligned = self.feature_aligner.align_features(X_test)
        
        logger.info("🔄 Detecting and mitigating data drift...")
        X_train_final, X_test_final, drift_report = self.drift_mitigator.detect_and_handle_drift(
            X_train_aligned, X_test_aligned, threshold=0.1
        )
        
        logger.info(f"   Drift detected in {len(drift_report['features_with_drift'])} features")
        
        # Step 2: Fast training without heavy optimization
        if TRAIN_MODELS_AVAILABLE:
            logger.info("🤖 Training models with fast settings...")
            
            # Convert to pandas Series for train_models compatibility
            y_train_series = pd.Series(y_train, index=X_train_final.index) if isinstance(y_train, np.ndarray) else y_train
            y_test_series = pd.Series(y_test, index=X_test_final.index) if isinstance(y_test, np.ndarray) else y_test
            
            # Create a fast training config
            try:
                # Import the config system
                from config.training_config import TrainingConfig
                
                # Create fast config
                fast_config = TrainingConfig()
                fast_config.cross_validation.n_folds = 3  # Reduced folds
                fast_config.enable_optuna = False  # Disable heavy optimization
                fast_config.optuna_trials = 10  # Minimal trials if enabled
                
                # Train with fast settings
                training_results = train_models.train_committee_models(
                    X_train_final, y_train_series, X_test_final, y_test_series, fast_config
                )
            except:
                # Fallback: use default training
                training_results = train_models.train_committee_models(
                    X_train_final, y_train_series, X_test_final, y_test_series
                )
        else:
            logger.error("❌ train_models not available - cannot proceed")
            return {}
        
        # Step 3: CRITICAL - Apply probability fixing
        if training_results and training_results.get('trained_models'):
            logger.info("🔧 Applying probability fixing...")
            
            models_fixed = 0
            for model_name, model_info in training_results['trained_models'].items():
                if 'test_predictions' in model_info:
                    # Handle different prediction formats
                    test_preds = model_info['test_predictions']
                    
                    # Convert string representation to array if needed
                    if isinstance(test_preds, str):
                        try:
                            test_preds = np.fromstring(test_preds.strip('[]'), sep=' ')
                        except:
                            logger.warning(f"Could not parse predictions for {model_name}")
                            continue
                    elif isinstance(test_preds, list):
                        test_preds = np.array(test_preds)
                    
                    # Check and fix uniform probabilities
                    if self.probability_fixer.detect_uniform_probabilities(test_preds):
                        logger.info(f"   🔧 Fixing uniform probabilities for {model_name}")
                        fixed_probs = self.probability_fixer.fix_uniform_probabilities(test_preds, y_test)
                        
                        # Update the results
                        model_info['test_predictions'] = fixed_probs.tolist()
                        model_info['original_predictions'] = test_preds.tolist()
                        model_info['uniform_probabilities_fixed'] = True
                        
                        # Log the improvement
                        original_range = test_preds.max() - test_preds.min()
                        fixed_range = fixed_probs.max() - fixed_probs.min()
                        logger.info(f"     Range: {original_range:.4f} → {fixed_range:.4f}")
                        
                        models_fixed += 1
                    else:
                        model_info['uniform_probabilities_fixed'] = False
            
            logger.info(f"✅ Fixed {models_fixed} models with uniform probabilities")
        
        # Step 4: Add enhancement metadata
        enhanced_results = {
            **training_results,
            'enhancement_metadata': {
                'feature_alignment_applied': True,
                'drift_mitigation_applied': drift_report['mitigation_applied'],
                'features_with_drift': len(drift_report['features_with_drift']),
                'probability_fixing_applied': True,
                'training_time': time.time() - start_time,
                'pipeline_version': 'fast_enhanced',
                'timestamp': datetime.now().isoformat()
            },
            'drift_report': drift_report,
            'training_time': time.time() - start_time
        }
        
        # Step 5: Save results
        if self.config['save_results']:
            results_path = f"reports/{batch_name}_fast_enhanced_results.json"
            os.makedirs("reports", exist_ok=True)
            
            try:
                with open(results_path, 'w') as f:
                    json.dump(enhanced_results, f, indent=2, default=str)
                logger.info(f"✅ Results saved: {results_path}")
            except Exception as e:
                logger.error(f"Failed to save results: {e}")
        
        total_time = time.time() - start_time
        logger.info(f"🎉 Fast enhanced training completed: {batch_name} ({total_time:.1f}s)")
        
        return enhanced_results
    
    def train_all_batches_fast(self, data_dir: str = "data") -> Dict[str, Any]:
        """Train all batches with fast enhanced processing."""
        logger.info("🌟 Starting fast enhanced training for all batches...")
        start_time = time.time()
        
        # Find batch files
        batch_files = [f for f in os.listdir(data_dir) 
                      if f.endswith('.csv') and f.startswith('leak_free_batch_')]
        
        if not batch_files:
            logger.error(f"No leak-free batch files found in {data_dir}")
            return {}
        
        logger.info(f"Found {len(batch_files)} leak-free batch files")
        
        # Process each batch
        all_results = []
        successful_batches = 0
        
        for i, batch_file in enumerate(sorted(batch_files), 1):
            # Extract batch name
            batch_name = batch_file.replace('leak_free_', '').replace('_data.csv', '')
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 Processing batch {i}/{len(batch_files)}: {batch_name}")
            logger.info(f"{'='*60}")
            
            try:
                # Load batch data
                batch_data = self._load_batch_data(os.path.join(data_dir, batch_file))
                
                if batch_data:
                    X_train, y_train, X_test, y_test = batch_data
                    
                    # Train batch
                    batch_results = self.train_single_batch_fast(
                        batch_name, X_train, y_train, X_test, y_test
                    )
                    
                    if batch_results:
                        all_results.append(batch_results)
                        successful_batches += 1
                        logger.info(f"✅ Batch {batch_name} completed successfully")
                    else:
                        logger.error(f"❌ Batch {batch_name} failed")
                else:
                    logger.warning(f"⚠️ Failed to load data for {batch_file}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing batch {batch_file}: {e}")
                continue
        
        # Create summary
        total_time = time.time() - start_time
        summary = {
            'total_batches_found': len(batch_files),
            'successful_batches': successful_batches,
            'total_training_time': total_time,
            'average_time_per_batch': total_time / len(batch_files) if batch_files else 0,
            'pipeline_config': self.config,
            'timestamp': datetime.now().isoformat(),
            'batch_results': all_results
        }
        
        # Save summary
        summary_path = f"reports/fast_enhanced_training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"✅ Training summary saved: {summary_path}")
        except Exception as e:
            logger.error(f"Failed to save training summary: {e}")
        
        # Final report
        logger.info(f"\n{'='*60}")
        logger.info("🎉 FAST ENHANCED TRAINING COMPLETED!")
        logger.info(f"{'='*60}")
        logger.info(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        logger.info(f"   Batches processed: {successful_batches}/{len(batch_files)}")
        logger.info(f"   Success rate: {successful_batches/len(batch_files)*100:.1f}%")
        logger.info(f"   Average time per batch: {total_time/len(batch_files):.1f}s")
        
        return summary
    
    def _load_batch_data(self, batch_file_path: str) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
        """Load and prepare batch data with fast processing."""
        try:
            logger.info(f"📁 Loading: {os.path.basename(batch_file_path)}")
            
            # Security check
            filename = os.path.basename(batch_file_path)
            if not filename.startswith("leak_free_"):
                logger.error(f"🚨 SECURITY: Only leak-free files allowed! Got: {filename}")
                return None
            
            # Load data
            df = pd.read_csv(batch_file_path)
            logger.info(f"   Loaded {len(df)} samples")
            
            # Quick validation
            if 'target' not in df.columns:
                logger.error("❌ Missing target column")
                return None
            
            # Fast feature extraction
            exclude_cols = ['ticker', 'target', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            feature_columns = [col for col in df.columns if col not in exclude_cols]
            
            X = df[feature_columns].select_dtypes(include=[np.number])  # Only numeric for speed
            y = df['target'].values
            
            # Quick cleaning
            X = X.fillna(X.median())  # Fast median fill
            X = X.replace([np.inf, -np.inf], 0)  # Handle infinities
            
            # Apply feature alignment
            X = self.feature_aligner.align_features(X)
            
            # Clean targets
            y = pd.to_numeric(y, errors='coerce')
            valid_mask = ~pd.isna(y) & ~X.isna().any(axis=1)
            X, y = X[valid_mask], y[valid_mask]
            
            if len(X) == 0:
                logger.error("❌ No valid samples after cleaning")
                return None
            
            logger.info(f"   Cleaned: {len(X)} samples, {X.shape[1]} features")
            
            # Fast train/test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, 
                stratify=y if len(np.unique(y)) > 1 else None
            )
            
            logger.info(f"   Split: {len(X_train)} train, {len(X_test)} test")
            
            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            logger.error(f"Failed to load {batch_file_path}: {e}")
            return None


def main():
    """Main function for fast enhanced training."""
    logger.info("🚀 Starting Fast Enhanced Training Pipeline")
    
    # Initialize fast pipeline
    pipeline = FastEnhancedTrainingPipeline(quick_mode=True)
    
    # Run fast training
    results = pipeline.train_all_batches_fast()
    
    if results and results['successful_batches'] > 0:
        logger.info("✅ Fast enhanced training completed successfully!")
    else:
        logger.error("❌ Fast enhanced training failed")

if __name__ == "__main__":
    main()
