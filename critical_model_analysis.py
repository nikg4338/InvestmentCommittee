#!/usr/bin/env python3
"""
Critical Model Analysis
Quick assessment of training results and data leakage concerns.
"""

import json
import pandas as pd
import numpy as np

def analyze_training_results():
    """Analyze the training results for data leakage indicators."""
    
    print("🚨 CRITICAL ANALYSIS: TRAINING RESULTS")
    print("="*60)
    
    # Load training summary
    with open('models/production/training_summary.json', 'r') as f:
        summary = json.load(f)
    
    print("📊 TRAINING METRICS SUMMARY:")
    print("-" * 40)
    
    suspicious_count = 0
    perfect_count = 0
    
    for model_name, results in summary['results'].items():
        if 'error' in results:
            continue
            
        cv_score = results['best_cv_score']
        test_roc = results['test_metrics']['roc_auc']
        test_acc = results['test_metrics']['accuracy']
        
        print(f"\n{model_name.upper()}:")
        print(f"  CV Score: {cv_score:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test ROC-AUC: {test_roc:.4f}")
        
        if test_acc >= 1.0:
            perfect_count += 1
            print(f"  🚨 PERFECT ACCURACY - HIGHLY SUSPICIOUS")
        elif test_acc >= 0.999:
            suspicious_count += 1
            print(f"  ⚠️ NEAR-PERFECT ACCURACY - SUSPICIOUS")
        elif test_acc >= 0.95:
            print(f"  ✅ HIGH ACCURACY - ACCEPTABLE")
        else:
            print(f"  ❌ POOR ACCURACY")
    
    print(f"\n🎯 ANALYSIS SUMMARY:")
    print(f"  Perfect models (100%): {perfect_count}")
    print(f"  Suspicious models (>99.9%): {suspicious_count}")
    print(f"  Total models: {len(summary['results'])}")
    
    # Data leakage assessment
    if perfect_count >= 3:
        print(f"\n🚨 CRITICAL WARNING: DATA LEAKAGE DETECTED")
        print(f"   {perfect_count} models achieving perfect accuracy is statistically impossible")
        print(f"   in real financial markets without data leakage")
        return "LEAKAGE"
    elif suspicious_count + perfect_count >= 4:
        print(f"\n⚠️ HIGH SUSPICION: POSSIBLE DATA LEAKAGE")
        print(f"   {suspicious_count + perfect_count} models with >99% accuracy is highly unusual")
        return "SUSPICIOUS"
    else:
        print(f"\n✅ ACCEPTABLE: Models show realistic performance")
        return "ACCEPTABLE"

def check_data_structure():
    """Check for potential data leakage sources."""
    
    print(f"\n🔍 DATA LEAKAGE INVESTIGATION")
    print("="*40)
    
    try:
        # Load training data
        df = pd.read_csv('alpaca_training_data.csv')
        
        print(f"Training data shape: {df.shape}")
        print(f"Features: {df.shape[1] - 4}")  # Excluding target, ticker, timestamp, etc.
        
        # Check for potential leakage indicators
        feature_cols = [col for col in df.columns if col not in ['target', 'target_enhanced', 'ticker', 'timestamp']]
        
        # Look for suspicious features
        suspicious_features = []
        for col in feature_cols:
            if any(keyword in col.lower() for keyword in ['future', 'next', 'tomorrow', 'ahead', 'forward']):
                suspicious_features.append(col)
        
        if suspicious_features:
            print(f"\n⚠️ SUSPICIOUS FEATURES FOUND:")
            for feature in suspicious_features:
                print(f"  - {feature}")
        
        # Check target distribution
        target_dist = df['target'].value_counts()
        print(f"\nTarget distribution: {target_dist.to_dict()}")
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        print(f"Duplicate rows: {duplicates}")
        
        # Check correlation between features and target
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        target_corr = df[numeric_cols].corr()['target'].abs().sort_values(ascending=False)
        
        print(f"\nTop 5 features correlated with target:")
        for feature, corr in target_corr.head(6).items():  # Top 6 (includes target itself)
            if feature != 'target':
                print(f"  {feature}: {corr:.4f}")
                if corr > 0.95:
                    print(f"    🚨 EXTREMELY HIGH CORRELATION - POSSIBLE LEAKAGE")
                elif corr > 0.8:
                    print(f"    ⚠️ HIGH CORRELATION - INVESTIGATE")
        
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")

def recommend_action():
    """Provide recommendations based on analysis."""
    
    print(f"\n🎯 RECOMMENDATIONS")
    print("="*30)
    
    # Based on validation results
    assessment = analyze_training_results()
    
    if assessment == "LEAKAGE":
        print(f"\n❌ DO NOT USE FOR PAPER TRADING")
        print(f"   1. 🔍 Investigate data pipeline for leakage")
        print(f"   2. 🧹 Remove future-looking features")
        print(f"   3. ✅ Validate train/test split timing")
        print(f"   4. 🔄 Retrain with clean data")
        print(f"   5. 🎯 Target 70-85% accuracy as realistic")
        
    elif assessment == "SUSPICIOUS":
        print(f"\n⚠️ USE WITH EXTREME CAUTION")
        print(f"   1. ✅ Start with VERY small position sizes")
        print(f"   2. 📊 Monitor performance closely")
        print(f"   3. 🔍 Investigate top-performing models")
        print(f"   4. 🚫 Stop trading if results don't match expectations")
        
    else:
        print(f"\n✅ PROCEED WITH PAPER TRADING")
        print(f"   1. 💰 Use normal position sizing")
        print(f"   2. 📈 Monitor ensemble performance")
        print(f"   3. 🎯 Expect 70-85% real-world accuracy")
        
    print(f"\n💡 ADDITIONAL RECOMMENDATIONS:")
    print(f"   • Use ensemble voting instead of single models")
    print(f"   • Set confidence thresholds (>80%) for trading")
    print(f"   • Implement stop-loss mechanisms")
    print(f"   • Track real performance vs backtested results")

def main():
    """Main analysis function."""
    
    print("🔍 CRITICAL MODEL ASSESSMENT")
    print("Evaluating if 100% accuracy models are safe for trading")
    print("="*70)
    
    recommend_action()
    check_data_structure()
    
    print(f"\n🎉 ANALYSIS COMPLETE")
    print(f"Review recommendations above before proceeding with trading")

if __name__ == "__main__":
    main()
