#!/usr/bin/env python3
"""
Comprehensive Model Training and Validation Script
==================================================

Complete end-to-end script that:
1. Loads training data with proper feature ordering
2. Trains all models using advanced pipeline
3. Validates models with comprehensive metrics
4. Tests production integration
5. Generates detailed reports

This script addresses all identified ML infrastructure issues.
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our components
from data_collection_alpaca import AlpacaDataCollector
from advanced_model_trainer import AdvancedModelTrainer
from enhanced_ensemble_classifier import EnhancedEnsembleClassifier
from enhanced_production_trading_engine import EnhancedProductionTradingEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveTrainingValidator:
    """Comprehensive training and validation system."""
    
    def __init__(self):
        """Initialize the comprehensive training system."""
        self.data_collector = AlpacaDataCollector()
        self.model_trainer = AdvancedModelTrainer()
        self.validation_results = {}
        self.training_summary = {}
        
        logger.info("🚀 Comprehensive Training Validator initialized")
    
    def collect_training_data(self, num_batches: int = 3, save_data: bool = True) -> pd.DataFrame:
        """Collect comprehensive training data."""
        logger.info(f"📊 Collecting training data (num_batches={num_batches})")
        
        try:
            # Load stock batches
            batches = self.data_collector.load_stock_batches()
            if not batches:
                logger.error("❌ No batches available for data collection")
                return pd.DataFrame()
            
            # Select first few batches
            batch_names = list(batches.keys())[:num_batches]
            logger.info(f"   Selected batches: {batch_names}")
            
            # Collect data for all selected batches
            all_data = []
            successful_collections = 0
            
            for batch_name in batch_names:
                try:
                    logger.info(f"   Collecting data for batch {batch_name}")
                    symbols = batches[batch_name]
                    
                    # Collect batch data (limit symbols per batch)
                    batch_data = self.data_collector.collect_batch_data(
                        batch_name=batch_name,
                        symbols=symbols,
                        max_symbols=20  # Limit to 20 symbols per batch
                    )
                    
                    if batch_data is not None and not batch_data.empty:
                        all_data.append(batch_data)
                        successful_collections += 1
                        logger.info(f"     ✅ {batch_name}: {len(batch_data)} samples")
                    else:
                        logger.warning(f"     ⚠️ {batch_name}: No data collected")
                
                except Exception as e:
                    logger.error(f"     ❌ {batch_name}: Data collection failed - {e}")
                    continue
            
            if not all_data:
                logger.error("❌ No data collected for any batch")
                return pd.DataFrame()
            
            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)
            
            logger.info(f"✅ Data collection complete:")
            logger.info(f"   Successful batches: {successful_collections}/{len(batch_names)}")
            logger.info(f"   Total samples: {len(combined_data)}")
            logger.info(f"   Features: {len(combined_data.columns)}")
            
            # Check for target columns
            target_columns = [col for col in combined_data.columns if 'target' in col.lower()]
            logger.info(f"   Target columns found: {target_columns}")
            
            # Save data if requested
            if save_data:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                data_path = f"data/comprehensive_training_data_{timestamp}.csv"
                Path(data_path).parent.mkdir(exist_ok=True)
                combined_data.to_csv(data_path, index=False)
                logger.info(f"💾 Training data saved to {data_path}")
            
            return combined_data
            
        except Exception as e:
            logger.error(f"❌ Training data collection failed: {e}")
            return pd.DataFrame()
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the quality of training data."""
        logger.info("🔍 Validating data quality...")
        
        validation = {
            'total_samples': len(df),
            'total_features': len(df.columns),
            'missing_data_ratio': df.isnull().sum().sum() / (len(df) * len(df.columns)),
            'duplicate_rows': df.duplicated().sum(),
            'feature_order_check': True,
            'target_distribution': {},
            'data_quality_score': 0.0,
            'issues': []
        }
        
        # Check feature ordering
        try:
            with open("models/feature_order_manifest.json", 'r') as f:
                manifest = json.load(f)
                expected_features = manifest['feature_order']
            
            available_features = [f for f in expected_features if f in df.columns]
            missing_features = [f for f in expected_features if f not in df.columns]
            
            validation['expected_features'] = len(expected_features)
            validation['available_features'] = len(available_features)
            validation['missing_features'] = len(missing_features)
            validation['feature_coverage'] = len(available_features) / len(expected_features)
            
            if missing_features:
                validation['issues'].append(f"Missing {len(missing_features)} expected features")
                logger.warning(f"⚠️ Missing features: {missing_features[:10]}")
        
        except Exception as e:
            validation['feature_order_check'] = False
            validation['issues'].append(f"Feature order validation failed: {e}")
        
        # Check target distributions
        target_columns = [col for col in df.columns if 'target' in col.lower()]
        for target_col in target_columns:
            if target_col in df.columns:
                target_dist = df[target_col].value_counts().to_dict()
                validation['target_distribution'][target_col] = target_dist
                
                # Check for class imbalance
                if len(target_dist) == 2:
                    values = list(target_dist.values())
                    imbalance_ratio = min(values) / max(values)
                    if imbalance_ratio < 0.3:
                        validation['issues'].append(f"Class imbalance in {target_col}: {imbalance_ratio:.2f}")
        
        # Calculate overall quality score
        quality_factors = [
            (1 - validation['missing_data_ratio']) * 0.3,  # Low missing data
            (1 - validation['duplicate_rows'] / len(df)) * 0.2,  # Low duplicates
            validation.get('feature_coverage', 0.5) * 0.3,  # Good feature coverage
            (1.0 if validation['total_samples'] > 1000 else validation['total_samples'] / 1000) * 0.2  # Sufficient samples
        ]
        validation['data_quality_score'] = sum(quality_factors)
        
        logger.info(f"✅ Data quality validation complete:")
        logger.info(f"   Quality score: {validation['data_quality_score']:.3f}")
        logger.info(f"   Issues found: {len(validation['issues'])}")
        
        return validation
    
    def run_comprehensive_training(self, df: pd.DataFrame, target_column: str = 'target_1d_enhanced') -> Dict[str, Any]:
        """Run comprehensive model training pipeline."""
        logger.info(f"🚀 Starting comprehensive training pipeline...")
        
        try:
            # Validate target column exists
            if target_column not in df.columns:
                available_targets = [col for col in df.columns if 'target' in col.lower()]
                if available_targets:
                    target_column = available_targets[0]
                    logger.warning(f"⚠️ Using available target: {target_column}")
                else:
                    logger.error(f"❌ No target column found")
                    return {'error': 'No target column available'}
            
            # Run training
            training_results = self.model_trainer.train_all_models(
                df=df,
                target_column=target_column,
                test_size=0.2,
                val_size=0.1,
                feature_selection=True,
                max_features=50
            )
            
            # Store results
            self.training_summary = training_results['summary']
            
            logger.info(f"✅ Comprehensive training complete:")
            logger.info(f"   Models trained: {len(training_results['models'])}")
            logger.info(f"   Best model: {self.training_summary.get('best_model', 'None')}")
            
            return training_results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive training failed: {e}")
            return {'error': str(e)}
    
    def validate_model_consistency(self) -> Dict[str, Any]:
        """Validate that models produce consistent predictions."""
        logger.info("🔍 Validating model consistency...")
        
        try:
            # Initialize ensemble
            ensemble = EnhancedEnsembleClassifier()
            ensemble.load_models()
            
            if not ensemble.models:
                return {'error': 'No models loaded for validation'}
            
            # Create test data (using feature order)
            np.random.seed(42)
            n_samples = 100
            n_features = len(ensemble.feature_order) if ensemble.feature_order else 50
            
            test_features = pd.DataFrame(
                np.random.randn(n_samples, n_features),
                columns=ensemble.feature_order[:n_features] if ensemble.feature_order else [f'feature_{i}' for i in range(n_features)]
            )
            
            # Get predictions from ensemble
            predictions_batch = ensemble.predict_batch(test_features, return_uncertainty=True)
            
            # Analyze consistency
            validation = {
                'total_test_samples': n_samples,
                'successful_predictions': len(predictions_batch),
                'mean_prediction': predictions_batch['prediction'].mean(),
                'mean_confidence': predictions_batch['confidence'].mean(),
                'mean_uncertainty': predictions_batch.get('overall_uncertainty', pd.Series([0.5])).mean(),
                'prediction_std': predictions_batch['prediction'].std(),
                'models_loaded': list(ensemble.models.keys()),
                'feature_consistency': True
            }
            
            # Check prediction variance
            if validation['prediction_std'] > 0.4:
                validation['consistency_warning'] = f"High prediction variance: {validation['prediction_std']:.3f}"
            
            # Check individual model agreement
            if 'individual_predictions' in predictions_batch.columns:
                individual_preds = predictions_batch['individual_predictions'].apply(pd.Series)
                model_agreement = individual_preds.std(axis=1).mean()
                validation['model_agreement'] = model_agreement
                
                if model_agreement > 0.3:
                    validation['agreement_warning'] = f"Low model agreement: {model_agreement:.3f}"
            
            logger.info(f"✅ Model consistency validation complete:")
            logger.info(f"   Models loaded: {len(validation['models_loaded'])}")
            logger.info(f"   Prediction variance: {validation['prediction_std']:.3f}")
            logger.info(f"   Mean confidence: {validation['mean_confidence']:.3f}")
            
            return validation
            
        except Exception as e:
            logger.error(f"❌ Model consistency validation failed: {e}")
            return {'error': str(e)}
    
    def validate_production_integration(self) -> Dict[str, Any]:
        """Validate production trading engine integration."""
        logger.info("🔍 Validating production integration...")
        
        try:
            # Initialize production engine
            engine = EnhancedProductionTradingEngine()
            
            # Initialize models
            if not engine.initialize_models():
                return {'error': 'Failed to initialize production models'}
            
            # Test with sample symbols
            test_symbols = ['AAPL', 'MSFT', 'GOOGL']
            
            validation = {
                'engine_initialized': True,
                'models_loaded': len(engine.ensemble.models),
                'test_symbols': test_symbols,
                'successful_predictions': 0,
                'failed_predictions': 0,
                'recommendations_generated': 0,
                'feature_processing_errors': 0
            }
            
            # Test predictions for each symbol
            for symbol in test_symbols:
                try:
                    # Get current market data
                    current_data = engine.data_collector.get_current_market_data(symbol)
                    
                    if current_data is None:
                        logger.warning(f"⚠️ No market data for {symbol}")
                        validation['failed_predictions'] += 1
                        continue
                    
                    # Make prediction
                    prediction_result = engine.make_enhanced_prediction(symbol, current_data)
                    
                    if 'error' in prediction_result:
                        validation['failed_predictions'] += 1
                        if 'feature' in prediction_result.get('error', '').lower():
                            validation['feature_processing_errors'] += 1
                    else:
                        validation['successful_predictions'] += 1
                        if prediction_result.get('recommendation', 'HOLD') != 'HOLD':
                            validation['recommendations_generated'] += 1
                
                except Exception as e:
                    logger.error(f"❌ Prediction test failed for {symbol}: {e}")
                    validation['failed_predictions'] += 1
            
            # Calculate success rate
            total_tests = validation['successful_predictions'] + validation['failed_predictions']
            validation['success_rate'] = validation['successful_predictions'] / max(total_tests, 1)
            
            # Check critical issues
            validation['critical_issues'] = []
            if validation['models_loaded'] == 0:
                validation['critical_issues'].append("No models loaded")
            if validation['success_rate'] < 0.5:
                validation['critical_issues'].append(f"Low success rate: {validation['success_rate']:.2%}")
            if validation['feature_processing_errors'] > 0:
                validation['critical_issues'].append(f"Feature processing errors: {validation['feature_processing_errors']}")
            
            logger.info(f"✅ Production integration validation complete:")
            logger.info(f"   Models loaded: {validation['models_loaded']}")
            logger.info(f"   Success rate: {validation['success_rate']:.2%}")
            logger.info(f"   Critical issues: {len(validation['critical_issues'])}")
            
            return validation
            
        except Exception as e:
            logger.error(f"❌ Production integration validation failed: {e}")
            return {'error': str(e)}
    
    def generate_comprehensive_report(self, training_results: Dict[str, Any], 
                                    data_validation: Dict[str, Any],
                                    model_consistency: Dict[str, Any],
                                    production_validation: Dict[str, Any]) -> None:
        """Generate comprehensive training and validation report."""
        logger.info("📊 Generating comprehensive report...")
        
        # Create report
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'training_summary': {
                'models_trained': training_results.get('summary', {}).get('models_trained', []),
                'best_model': training_results.get('summary', {}).get('best_model'),
                'total_samples': training_results.get('summary', {}).get('total_samples', 0),
                'selected_features': training_results.get('summary', {}).get('selected_features', []),
                'training_metrics': training_results.get('metrics', {})
            },
            'data_quality': {
                'quality_score': data_validation.get('data_quality_score', 0.0),
                'total_samples': data_validation.get('total_samples', 0),
                'feature_coverage': data_validation.get('feature_coverage', 0.0),
                'missing_data_ratio': data_validation.get('missing_data_ratio', 1.0),
                'issues_found': data_validation.get('issues', [])
            },
            'model_consistency': {
                'models_loaded': model_consistency.get('models_loaded', []),
                'prediction_variance': model_consistency.get('prediction_std', 1.0),
                'mean_confidence': model_consistency.get('mean_confidence', 0.0),
                'model_agreement': model_consistency.get('model_agreement', 0.0),
                'consistency_warnings': [
                    model_consistency.get('consistency_warning', ''),
                    model_consistency.get('agreement_warning', '')
                ]
            },
            'production_integration': {
                'success_rate': production_validation.get('success_rate', 0.0),
                'models_loaded': production_validation.get('models_loaded', 0),
                'critical_issues': production_validation.get('critical_issues', []),
                'feature_processing_errors': production_validation.get('feature_processing_errors', 0)
            },
            'overall_assessment': self._generate_overall_assessment(
                training_results, data_validation, model_consistency, production_validation
            ),
            'recommendations': self._generate_recommendations(
                training_results, data_validation, model_consistency, production_validation
            )
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/comprehensive_training_report_{timestamp}.json"
        Path(report_path).parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save human-readable summary
        summary_path = f"reports/training_summary_{timestamp}.md"
        self._save_markdown_summary(report, summary_path)
        
        logger.info(f"📋 Comprehensive report saved:")
        logger.info(f"   Full report: {report_path}")
        logger.info(f"   Summary: {summary_path}")
        
        # Print key findings
        self._print_key_findings(report)
    
    def _generate_overall_assessment(self, training_results: Dict, data_validation: Dict,
                                   model_consistency: Dict, production_validation: Dict) -> Dict[str, Any]:
        """Generate overall assessment of the training pipeline."""
        assessment = {
            'training_success': len(training_results.get('models', {})) > 0,
            'data_quality_acceptable': data_validation.get('data_quality_score', 0) > 0.7,
            'models_consistent': model_consistency.get('prediction_std', 1.0) < 0.3,
            'production_ready': production_validation.get('success_rate', 0) > 0.8,
            'critical_issues_count': len(production_validation.get('critical_issues', [])),
            'overall_score': 0.0
        }
        
        # Calculate overall score
        score_components = [
            1.0 if assessment['training_success'] else 0.0,
            data_validation.get('data_quality_score', 0.0),
            (1.0 - min(model_consistency.get('prediction_std', 1.0), 1.0)),
            production_validation.get('success_rate', 0.0)
        ]
        assessment['overall_score'] = sum(score_components) / len(score_components)
        
        # Determine readiness
        if assessment['overall_score'] > 0.8 and assessment['critical_issues_count'] == 0:
            assessment['readiness'] = 'PRODUCTION_READY'
        elif assessment['overall_score'] > 0.6:
            assessment['readiness'] = 'NEEDS_MINOR_FIXES'
        else:
            assessment['readiness'] = 'NEEDS_MAJOR_IMPROVEMENTS'
        
        return assessment
    
    def _generate_recommendations(self, training_results: Dict, data_validation: Dict,
                                model_consistency: Dict, production_validation: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Data quality recommendations
        if data_validation.get('data_quality_score', 0) < 0.7:
            recommendations.append("🔧 Improve data quality: address missing data and feature coverage issues")
        
        if data_validation.get('missing_data_ratio', 0) > 0.1:
            recommendations.append("📊 Reduce missing data ratio below 10%")
        
        # Model consistency recommendations
        if model_consistency.get('prediction_std', 1.0) > 0.3:
            recommendations.append("🎯 Improve model consistency: high prediction variance detected")
        
        if model_consistency.get('model_agreement', 1.0) > 0.3:
            recommendations.append("🤝 Improve model agreement through better ensemble weighting")
        
        # Production integration recommendations
        if production_validation.get('success_rate', 0) < 0.8:
            recommendations.append("🚀 Fix production integration issues before deployment")
        
        if production_validation.get('feature_processing_errors', 0) > 0:
            recommendations.append("⚙️ Fix feature processing errors in production pipeline")
        
        # Model performance recommendations
        best_model_auc = 0.0
        if training_results.get('metrics'):
            aucs = [metrics.get('roc_auc', 0) for metrics in training_results['metrics'].values()]
            best_model_auc = max(aucs) if aucs else 0.0
        
        if best_model_auc < 0.6:
            recommendations.append("📈 Improve model performance: best ROC-AUC is below 0.6")
        
        # Feature recommendations
        if data_validation.get('feature_coverage', 0) < 0.8:
            recommendations.append("🔍 Add missing features to improve coverage")
        
        if not recommendations:
            recommendations.append("✅ System appears ready for production deployment")
        
        return recommendations
    
    def _save_markdown_summary(self, report: Dict[str, Any], filepath: str) -> None:
        """Save human-readable markdown summary."""
        md_content = f"""# Comprehensive Training Report
        
Generated: {report['report_timestamp']}

## Overall Assessment
- **Readiness**: {report['overall_assessment']['readiness']}
- **Overall Score**: {report['overall_assessment']['overall_score']:.3f}/1.0
- **Critical Issues**: {report['overall_assessment']['critical_issues_count']}

## Training Summary
- **Models Trained**: {len(report['training_summary']['models_trained'])}
- **Best Model**: {report['training_summary']['best_model']}
- **Total Samples**: {report['training_summary']['total_samples']}
- **Selected Features**: {len(report['training_summary']['selected_features'])}

## Data Quality
- **Quality Score**: {report['data_quality']['quality_score']:.3f}/1.0
- **Feature Coverage**: {report['data_quality']['feature_coverage']:.2%}
- **Missing Data**: {report['data_quality']['missing_data_ratio']:.2%}
- **Issues Found**: {len(report['data_quality']['issues_found'])}

## Model Consistency
- **Models Loaded**: {len(report['model_consistency']['models_loaded'])}
- **Prediction Variance**: {report['model_consistency']['prediction_variance']:.3f}
- **Mean Confidence**: {report['model_consistency']['mean_confidence']:.3f}

## Production Integration
- **Success Rate**: {report['production_integration']['success_rate']:.2%}
- **Models Loaded**: {report['production_integration']['models_loaded']}
- **Feature Processing Errors**: {report['production_integration']['feature_processing_errors']}

## Recommendations
"""
        
        for i, rec in enumerate(report['recommendations'], 1):
            md_content += f"{i}. {rec}\n"
        
        with open(filepath, 'w') as f:
            f.write(md_content)
    
    def _print_key_findings(self, report: Dict[str, Any]) -> None:
        """Print key findings to console."""
        logger.info("🎯 KEY FINDINGS:")
        logger.info(f"   Overall Readiness: {report['overall_assessment']['readiness']}")
        logger.info(f"   Overall Score: {report['overall_assessment']['overall_score']:.3f}/1.0")
        
        if report['recommendations']:
            logger.info("🔧 TOP RECOMMENDATIONS:")
            for rec in report['recommendations'][:3]:
                logger.info(f"   • {rec}")


def main():
    """Main training and validation pipeline."""
    logger.info("🚀 Starting Comprehensive Training and Validation Pipeline")
    
    # Initialize validator
    validator = ComprehensiveTrainingValidator()
    
    try:
        # Step 1: Collect training data
        logger.info("\n" + "="*60)
        logger.info("STEP 1: DATA COLLECTION")
        logger.info("="*60)
        
        training_data = validator.collect_training_data(num_batches=3)
        if training_data.empty:
            logger.error("❌ No training data collected - aborting pipeline")
            return
        
        # Step 2: Validate data quality
        logger.info("\n" + "="*60)
        logger.info("STEP 2: DATA QUALITY VALIDATION")
        logger.info("="*60)
        
        data_validation = validator.validate_data_quality(training_data)
        
        # Step 3: Run comprehensive training
        logger.info("\n" + "="*60)
        logger.info("STEP 3: COMPREHENSIVE MODEL TRAINING")
        logger.info("="*60)
        
        training_results = validator.run_comprehensive_training(training_data)
        if 'error' in training_results:
            logger.error(f"❌ Training failed: {training_results['error']}")
            return
        
        # Step 4: Validate model consistency
        logger.info("\n" + "="*60)
        logger.info("STEP 4: MODEL CONSISTENCY VALIDATION")
        logger.info("="*60)
        
        model_consistency = validator.validate_model_consistency()
        
        # Step 5: Validate production integration
        logger.info("\n" + "="*60)
        logger.info("STEP 5: PRODUCTION INTEGRATION VALIDATION")
        logger.info("="*60)
        
        production_validation = validator.validate_production_integration()
        
        # Step 6: Generate comprehensive report
        logger.info("\n" + "="*60)
        logger.info("STEP 6: COMPREHENSIVE REPORTING")
        logger.info("="*60)
        
        validator.generate_comprehensive_report(
            training_results, data_validation, model_consistency, production_validation
        )
        
        logger.info("\n" + "="*60)
        logger.info("🎉 COMPREHENSIVE TRAINING PIPELINE COMPLETE!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
