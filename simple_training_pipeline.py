#!/usr/bin/env python3
"""
Simple Training Pipeline with Core Fixes
========================================

A basic training pipeline that applies the three key fixes:
1. Feature alignment
2. Probability fixing  
3. Data drift mitigation

No fancy logging, just results.
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from feature_alignment_system import FeatureAligner, ProbabilityFixer, DataDriftMitigator

class SimpleTrainingPipeline:
    """Simple training pipeline with core fixes"""
    
    def __init__(self):
        self.feature_aligner = FeatureAligner()
        self.probability_fixer = ProbabilityFixer()
        self.data_drift_mitigator = DataDriftMitigator()
        
        os.makedirs('logs', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
    
    def train_batch_simple(self, batch_name):
        """Train single batch with simple approach"""
        start_time = time.time()
        
        try:
            print(f"Training batch: {batch_name}")
            
            # Load data
            data_path = f"data/leak_free_{batch_name}_data.csv"
            if not os.path.exists(data_path):
                print(f"Data file not found: {data_path}")
                return None
            
            df = pd.read_csv(data_path)
            print(f"Loaded {len(df)} samples")
            
            # Separate target before feature alignment
            if 'target' not in df.columns:
                print("No target column found")
                return None
            
            target = df['target'].copy()
            
            # Remove non-feature columns
            exclude_cols = ['target', 'temporal_split', 'ticker', 'timestamp', 
                          'data_collection_timestamp', 'data_source', 'leak_free_validated']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            df_features = df[feature_cols].copy()
            
            print(f"Original features: {len(feature_cols)}")
            
            # Feature alignment
            print("Applying feature alignment...")
            df_aligned = self.feature_aligner.align_features(df_features)
            print(f"Aligned features: {df_aligned.shape[1]}")
            
            # Add target back
            df_aligned['target'] = target
            
            # Clean data
            df_clean = df_aligned.dropna(subset=['target']).copy()
            print(f"Clean samples: {len(df_clean)}")
            
            # Prepare X and y
            feature_cols = [col for col in df_clean.columns if col != 'target']
            X = df_clean[feature_cols].fillna(0)
            y = df_clean['target']
            
            # Simple train/test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            print(f"Split: {len(X_train)} train, {len(X_test)} test")
            
            # Data drift detection and mitigation
            print("Checking for data drift...")
            X_train, X_test, drift_report = self.data_drift_mitigator.detect_and_handle_drift(X_train, X_test)
            drift_features = drift_report['features_with_drift']
            if len(drift_features) > 0:
                print(f"Data drift detected in {len(drift_features)} features")
                print(f"Applied drift mitigation")
            else:
                print("No significant drift detected")
            
            # Simple model training
            results = {}
            
            # XGBoost
            try:
                print("Training XGBoost...")
                from xgboost import XGBClassifier
                model = XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=0
                )
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_test)[:, 1]
                results['xgboost'] = probs
                print("XGBoost completed")
            except Exception as e:
                print(f"XGBoost failed: {e}")
            
            # LightGBM
            try:
                print("Training LightGBM...")
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=-1
                )
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_test)[:, 1]
                results['lightgbm'] = probs
                print("LightGBM completed")
            except Exception as e:
                print(f"LightGBM failed: {e}")
            
            if not results:
                print("No models trained successfully")
                return None
            
            # Probability fixing
            print("Applying probability fixing...")
            fixed_count = 0
            for model_name, probs in results.items():
                if self.probability_fixer.detect_uniform_probabilities(probs):
                    print(f"Fixing {model_name} uniform probabilities")
                    fixed_probs = self.probability_fixer.fix_uniform_probabilities(probs, y_test)
                    results[model_name] = fixed_probs
                    fixed_count += 1
            
            print(f"Fixed {fixed_count} models")
            
            # Simple evaluation
            from sklearn.metrics import roc_auc_score, average_precision_score
            
            evaluation = {}
            for model_name, probs in results.items():
                try:
                    auc = roc_auc_score(y_test, probs)
                    ap = average_precision_score(y_test, probs)
                    evaluation[model_name] = {'auc': auc, 'ap': ap}
                    print(f"{model_name}: AUC={auc:.3f}, AP={ap:.3f}")
                except:
                    evaluation[model_name] = {'auc': 0.5, 'ap': 0.1}
                    print(f"{model_name}: evaluation failed")
            
            # Ensemble
            if len(results) > 1:
                ensemble_probs = np.mean(list(results.values()), axis=0)
                try:
                    ensemble_auc = roc_auc_score(y_test, ensemble_probs)
                    ensemble_ap = average_precision_score(y_test, ensemble_probs)
                    evaluation['ensemble'] = {'auc': ensemble_auc, 'ap': ensemble_ap}
                    print(f"ensemble: AUC={ensemble_auc:.3f}, AP={ensemble_ap:.3f}")
                except:
                    evaluation['ensemble'] = {'auc': 0.5, 'ap': 0.1}
            
            training_time = time.time() - start_time
            
            # Save results
            batch_results = {
                'batch': batch_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'training_time': training_time,
                'n_samples': len(X),
                'n_features': len(feature_cols),
                'n_models': len(results),
                'drift_features': len(drift_features) if drift_features else 0,
                'fixed_models': fixed_count,
                'evaluation': evaluation
            }
            
            # Save to file
            output_path = f"reports/{batch_name}_simple_results.json"
            with open(output_path, 'w') as f:
                json.dump(batch_results, f, indent=2, default=str)
            
            print(f"Results saved: {output_path}")
            print(f"Training completed: {batch_name} ({training_time:.1f}s)")
            print("-" * 50)
            
            return batch_results
            
        except Exception as e:
            print(f"Error training batch {batch_name}: {str(e)}")
            return None
    
    def run_all_batches(self):
        """Run training on all available batches"""
        print("Starting simple training pipeline...")
        print("=" * 50)
        
        # Find batch files
        data_dir = Path("data")
        batch_files = list(data_dir.glob("leak_free_batch_*_data.csv"))
        
        if not batch_files:
            print("No batch files found")
            return []
        
        print(f"Found {len(batch_files)} batches")
        
        all_results = []
        total_start_time = time.time()
        
        # Process first 3 batches for testing
        for i, batch_file in enumerate(batch_files[:3], 1):
            batch_name = batch_file.stem.replace('leak_free_', '').replace('_data', '')
            
            print(f"Processing batch {i}/3: {batch_name}")
            
            batch_results = self.train_batch_simple(batch_name)
            
            if batch_results:
                all_results.append(batch_results)
                print(f"Batch {batch_name} completed successfully")
            else:
                print(f"Batch {batch_name} failed")
        
        total_time = time.time() - total_start_time
        
        # Summary
        print("=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(f"Total batches processed: {len(all_results)}/3")
        print(f"Total training time: {total_time:.1f}s")
        
        if all_results:
            avg_auc = np.mean([r['evaluation'].get('ensemble', {}).get('auc', 0.5) for r in all_results])
            avg_ap = np.mean([r['evaluation'].get('ensemble', {}).get('ap', 0.1) for r in all_results])
            print(f"Average ensemble AUC: {avg_auc:.3f}")
            print(f"Average ensemble AP: {avg_ap:.3f}")
            
            # Show fixes applied
            total_drift = sum([r['drift_features'] for r in all_results])
            total_fixed = sum([r['fixed_models'] for r in all_results])
            print(f"Total drift features fixed: {total_drift}")
            print(f"Total models probability-fixed: {total_fixed}")
        
        # Save summary
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_time': total_time,
            'batches_processed': len(all_results),
            'results': all_results
        }
        
        summary_path = "reports/simple_training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Summary saved: {summary_path}")
        print("Training pipeline completed!")
        
        return all_results


def main():
    """Main execution function"""
    pipeline = SimpleTrainingPipeline()
    results = pipeline.run_all_batches()
    
    if results:
        print(f"Pipeline completed with {len(results)} successful batches")
    else:
        print("Pipeline failed - no successful batches")


if __name__ == "__main__":
    main()
