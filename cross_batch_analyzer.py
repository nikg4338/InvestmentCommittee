#!/usr/bin/env python3
"""
Cross-Batch Performance Analysis and Consistency Checker
======================================================

This module analyzes performance consistency across all trained batches,
identifies outliers, and provides recommendations for model improvements.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from scipy import stats

logger = logging.getLogger(__name__)

class CrossBatchAnalyzer:
    """Analyzes performance consistency across multiple training batches."""
    
    def __init__(self, reports_dir: str = 'reports'):
        """Initialize cross-batch analyzer."""
        self.reports_dir = Path(reports_dir)
        self.batch_data = []
        self.performance_metrics = {}
        
    def load_batch_data(self) -> bool:
        """Load telemetry data from all batch reports."""
        try:
            self.batch_data = []
            
            if not self.reports_dir.exists():
                logger.error(f"Reports directory not found: {self.reports_dir}")
                return False
            
            # Scan for batch directories
            batch_dirs = sorted([d for d in self.reports_dir.glob('batch_*') if d.is_dir()],
                              key=lambda x: int(x.name.replace('batch_', '')))
            
            for batch_dir in batch_dirs:
                batch_num = int(batch_dir.name.replace('batch_', ''))
                
                # Load telemetry JSON
                telemetry_file = batch_dir / f'batch_{batch_num}_telemetry.json'
                if telemetry_file.exists():
                    try:
                        with open(telemetry_file, 'r') as f:
                            data = json.load(f)
                        
                        # Extract key metrics
                        batch_info = {
                            'batch': batch_num,
                            'pr_auc': data.get('pr_auc_meta', 0.0),
                            'gate': data.get('gate', 'UNKNOWN'),
                            'models_trained': data.get('models_trained', 0),
                            'training_time': data.get('training_time_seconds', 0),
                            'samples': data.get('samples_processed', 0),
                            'features': data.get('feature_count', 0),
                            'success': data.get('training_successful', False),
                            'best_model': data.get('best_model', 'unknown'),
                            'ensemble_accuracy': data.get('ensemble_accuracy', 0.0),
                            'dynamic_weights': data.get('dynamic_weights', {}),
                            'timestamp': data.get('timestamp', '')
                        }
                        
                        self.batch_data.append(batch_info)
                        logger.info(f"✅ Loaded data for batch {batch_num}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to load telemetry for batch {batch_num}: {e}")
                else:
                    logger.warning(f"Telemetry file not found for batch {batch_num}")
            
            if self.batch_data:
                logger.info(f"📊 Loaded data for {len(self.batch_data)} batches")
                return True
            else:
                logger.warning("No batch data found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load batch data: {e}")
            return False
    
    def calculate_performance_statistics(self) -> Dict[str, Any]:
        """Calculate statistical measures of performance consistency."""
        if not self.batch_data:
            return {}
        
        try:
            # Extract metrics
            pr_aucs = [d['pr_auc'] for d in self.batch_data]
            training_times = [d['training_time'] for d in self.batch_data]
            samples = [d['samples'] for d in self.batch_data]
            
            # Calculate statistics
            stats_data = {
                'pr_auc_stats': {
                    'mean': np.mean(pr_aucs),
                    'std': np.std(pr_aucs),
                    'min': np.min(pr_aucs),
                    'max': np.max(pr_aucs),
                    'median': np.median(pr_aucs),
                    'cv': np.std(pr_aucs) / np.mean(pr_aucs) if np.mean(pr_aucs) > 0 else 0,
                    'values': pr_aucs
                },
                'training_time_stats': {
                    'mean': np.mean(training_times),
                    'std': np.std(training_times),
                    'min': np.min(training_times),
                    'max': np.max(training_times),
                    'median': np.median(training_times),
                    'values': training_times
                },
                'sample_stats': {
                    'mean': np.mean(samples),
                    'std': np.std(samples),
                    'min': np.min(samples),
                    'max': np.max(samples),
                    'total': np.sum(samples),
                    'values': samples
                },
                'success_rate': {
                    'total_batches': len(self.batch_data),
                    'successful': sum(1 for d in self.batch_data if d['success']),
                    'passed_gate': sum(1 for d in self.batch_data if d['gate'] == 'PASSED'),
                    'rate': sum(1 for d in self.batch_data if d['success']) / len(self.batch_data)
                }
            }
            
            # Identify outliers using IQR method
            q1, q3 = np.percentile(pr_aucs, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = []
            for i, value in enumerate(pr_aucs):
                if value < lower_bound or value > upper_bound:
                    outliers.append({
                        'batch': self.batch_data[i]['batch'],
                        'pr_auc': value,
                        'type': 'low' if value < lower_bound else 'high'
                    })
            
            stats_data['outliers'] = outliers
            self.performance_metrics = stats_data
            return stats_data
            
        except Exception as e:
            logger.error(f"Failed to calculate performance statistics: {e}")
            return {}
    
    def create_consistency_report(self) -> str:
        """Create a comprehensive consistency analysis report."""
        try:
            if not self.batch_data:
                return "No batch data available for analysis"
            
            stats = self.calculate_performance_statistics()
            if not stats:
                return "Failed to calculate performance statistics"
            
            # Generate report
            report_lines = [
                "CROSS-BATCH PERFORMANCE CONSISTENCY REPORT",
                "=" * 60,
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Total Batches Analyzed: {len(self.batch_data)}",
                "",
                "📊 PR-AUC PERFORMANCE ANALYSIS",
                "-" * 40,
                f"Mean PR-AUC: {stats['pr_auc_stats']['mean']:.4f}",
                f"Standard Deviation: {stats['pr_auc_stats']['std']:.4f}",
                f"Coefficient of Variation: {stats['pr_auc_stats']['cv']:.4f}",
                f"Range: {stats['pr_auc_stats']['min']:.4f} - {stats['pr_auc_stats']['max']:.4f}",
                f"Median: {stats['pr_auc_stats']['median']:.4f}",
                "",
                "🎯 CONSISTENCY ASSESSMENT",
                "-" * 40
            ]
            
            # Consistency assessment
            cv = stats['pr_auc_stats']['cv']
            if cv < 0.1:
                consistency = "EXCELLENT"
                assessment = "Very consistent performance across batches"
            elif cv < 0.2:
                consistency = "GOOD"
                assessment = "Good consistency with minor variations"
            elif cv < 0.3:
                consistency = "MODERATE"
                assessment = "Moderate consistency, some investigation recommended"
            else:
                consistency = "POOR"
                assessment = "Poor consistency, requires immediate attention"
            
            report_lines.extend([
                f"Consistency Rating: {consistency}",
                f"Assessment: {assessment}",
                ""
            ])
            
            # Success rate analysis
            success_stats = stats['success_rate']
            report_lines.extend([
                "✅ SUCCESS RATE ANALYSIS",
                "-" * 40,
                f"Total Batches: {success_stats['total_batches']}",
                f"Successful Training: {success_stats['successful']} ({success_stats['rate']*100:.1f}%)",
                f"Passed Quality Gate: {success_stats['passed_gate']} ({success_stats['passed_gate']/success_stats['total_batches']*100:.1f}%)",
                ""
            ])
            
            # Outlier analysis
            outliers = stats.get('outliers', [])
            if outliers:
                report_lines.extend([
                    "🚨 OUTLIER DETECTION",
                    "-" * 40
                ])
                for outlier in outliers:
                    report_lines.append(f"Batch {outlier['batch']}: PR-AUC {outlier['pr_auc']:.4f} ({outlier['type']} outlier)")
                report_lines.append("")
            else:
                report_lines.extend([
                    "✅ NO OUTLIERS DETECTED",
                    "-" * 40,
                    "All batches fall within expected performance range",
                    ""
                ])
            
            # Performance recommendations
            report_lines.extend([
                "💡 RECOMMENDATIONS",
                "-" * 40
            ])
            
            if consistency == "POOR":
                report_lines.extend([
                    "• Investigate data quality and feature engineering consistency",
                    "• Review hyperparameter optimization settings",
                    "• Consider more robust cross-validation strategies",
                    "• Check for data leakage or temporal inconsistencies"
                ])
            elif consistency == "MODERATE":
                report_lines.extend([
                    "• Fine-tune hyperparameter optimization ranges",
                    "• Consider ensemble methods for more stable predictions",
                    "• Review feature selection consistency across batches"
                ])
            else:
                report_lines.extend([
                    "• Current performance is stable and reliable",
                    "• Consider optimizing for higher absolute performance",
                    "• Maintain current data preprocessing pipeline"
                ])
            
            # Training efficiency analysis
            time_stats = stats['training_time_stats']
            report_lines.extend([
                "",
                "⏱️ TRAINING EFFICIENCY ANALYSIS",
                "-" * 40,
                f"Average Training Time: {time_stats['mean']/60:.1f} minutes",
                f"Time Variability: {time_stats['std']/60:.1f} minutes (±)",
                f"Fastest Batch: {time_stats['min']/60:.1f} minutes",
                f"Slowest Batch: {time_stats['max']/60:.1f} minutes"
            ])
            
            # Sample distribution analysis
            sample_stats = stats['sample_stats']
            report_lines.extend([
                "",
                "📈 DATA DISTRIBUTION ANALYSIS",
                "-" * 40,
                f"Total Samples Processed: {sample_stats['total']:,}",
                f"Average Samples per Batch: {sample_stats['mean']:.0f}",
                f"Sample Count Variability: {sample_stats['std']:.0f} (±)",
                f"Range: {sample_stats['min']:.0f} - {sample_stats['max']:.0f} samples"
            ])
            
            # Final assessment
            report_lines.extend([
                "",
                "🏁 FINAL ASSESSMENT",
                "=" * 60
            ])
            
            if success_stats['rate'] >= 0.8 and consistency in ["EXCELLENT", "GOOD"]:
                final_assessment = "PRODUCTION READY - System shows excellent stability and performance"
            elif success_stats['rate'] >= 0.6:
                final_assessment = "NEEDS IMPROVEMENT - Address consistency issues before production"
            else:
                final_assessment = "NOT READY - Significant issues require resolution"
            
            report_lines.append(final_assessment)
            
            # Create report
            report_content = "\n".join(report_lines)
            
            # Save report
            report_path = self.reports_dir / 'CONSISTENCY_ANALYSIS.txt'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"✅ Consistency report saved: {report_path}")
            return report_content
            
        except Exception as e:
            logger.error(f"Failed to create consistency report: {e}")
            return f"Error generating report: {e}"
    
    def create_performance_visualization(self) -> str:
        """Create comprehensive performance visualization across batches."""
        try:
            if not self.batch_data:
                logger.warning("No batch data available for visualization")
                return ""
            
            # Set up the plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Extract data for plotting
            batches = [d['batch'] for d in self.batch_data]
            pr_aucs = [d['pr_auc'] for d in self.batch_data]
            gates = [d['gate'] for d in self.batch_data]
            training_times = [d['training_time'] / 60 for d in self.batch_data]  # Convert to minutes
            samples = [d['samples'] for d in self.batch_data]
            
            # 1. PR-AUC performance across batches
            colors = ['green' if gate == 'PASSED' else 'red' for gate in gates]
            bars1 = ax1.bar(batches, pr_aucs, color=colors, alpha=0.7)
            ax1.axhline(0.05, color='orange', linestyle='--', label='Minimum Threshold')
            ax1.set_xlabel('Batch Number')
            ax1.set_ylabel('PR-AUC Score')
            ax1.set_title('PR-AUC Performance Across Batches')
            ax1.legend()
            
            # Add value labels
            for bar, value in zip(bars1, pr_aucs):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            # 2. Training time analysis
            ax2.plot(batches, training_times, marker='o', linewidth=2, markersize=6)
            ax2.set_xlabel('Batch Number')
            ax2.set_ylabel('Training Time (minutes)')
            ax2.set_title('Training Time Across Batches')
            ax2.grid(True, alpha=0.3)
            
            # 3. Sample count per batch
            ax3.bar(batches, samples, color='lightblue', alpha=0.7)
            ax3.set_xlabel('Batch Number')
            ax3.set_ylabel('Sample Count')
            ax3.set_title('Sample Count per Batch')
            
            # 4. Success rate summary
            passed_count = sum(1 for gate in gates if gate == 'PASSED')
            failed_count = len(gates) - passed_count
            
            ax4.pie([passed_count, failed_count], labels=['Passed', 'Failed'], 
                   colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%', startangle=90)
            ax4.set_title(f'Overall Success Rate\n({passed_count}/{len(gates)} batches passed)')
            
            plt.suptitle('Cross-Batch Performance Analysis', fontsize=16, y=0.98)
            plt.tight_layout()
            
            # Save plot
            plot_path = self.reports_dir / 'cross_batch_analysis.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Cross-batch analysis plot saved: {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to create cross-batch visualization: {e}")
            if 'plt' in locals():
                plt.close()
            return ""
    
    def analyze_cross_batch_performance(self) -> Dict[str, Any]:
        """Run complete cross-batch analysis and return results."""
        logger.info("🔍 Starting comprehensive cross-batch analysis...")
        
        # Load data
        if not self.load_batch_data():
            return {"error": "Failed to load batch data"}
        
        # Calculate statistics
        stats = self.calculate_performance_statistics()
        
        # Generate report
        report = self.create_consistency_report()
        
        # Generate visualization
        plot_path = self.create_performance_visualization()
        
        results = {
            'statistics': stats,
            'report': report,
            'plot_path': plot_path,
            'batch_count': len(self.batch_data),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Cross-batch analysis completed for {len(self.batch_data)} batches")
        return results
    
    def run_full_analysis(self) -> Dict[str, str]:
        """Run complete cross-batch analysis and generate all outputs."""
        logger.info("🔍 Starting comprehensive cross-batch analysis...")
        
        results = {}
        
        # Load data
        if not self.load_batch_data():
            results['error'] = "Failed to load batch data"
            return results
        
        # Generate consistency report
        report = self.create_consistency_report()
        results['consistency_report'] = report
        
        # Generate performance visualization
        plot_path = self.create_performance_visualization()
        if plot_path:
            results['visualization_path'] = plot_path
        
        # Save analysis summary
        analysis_summary = {
            'batch_count': len(self.batch_data),
            'performance_metrics': self.performance_metrics,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        summary_path = self.reports_dir / 'cross_batch_analysis_summary.json'
        try:
            with open(summary_path, 'w') as f:
                json.dump(analysis_summary, f, indent=2, default=str)
            results['summary_path'] = str(summary_path)
            logger.info(f"✅ Analysis summary saved: {summary_path}")
        except Exception as e:
            logger.error(f"Failed to save analysis summary: {e}")
        
        logger.info("✅ Full cross-batch analysis completed")
        return results


# Global analyzer instance
cross_batch_analyzer = CrossBatchAnalyzer()
