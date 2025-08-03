#!/usr/bin/env python3
"""
Comprehensive edge case testing for the improved training system
============================================================

This script tests various extreme scenarios to ensure robust handling
of challenging datasets.
"""

import numpy as np
import pandas as pd
from train_models import prepare_training_data, train_committee_models_advanced

def test_edge_cases():
    """Test various edge cases that could break the system."""
    print("🔬 Testing comprehensive edge cases...")
    
    test_cases = [
        {
            'name': 'All Class 0',
            'class_0_count': 30,
            'class_1_count': 0,
            'description': 'Dataset with only majority class'
        },
        {
            'name': 'All Class 1', 
            'class_0_count': 0,
            'class_1_count': 30,
            'description': 'Dataset with only minority class'
        },
        {
            'name': 'Tiny Dataset',
            'class_0_count': 5,
            'class_1_count': 0,
            'description': 'Very small single-class dataset'
        },
        {
            'name': 'Singleton Minority',
            'class_0_count': 25,
            'class_1_count': 1,
            'description': 'Single minority sample (previously working)'
        }
    ]
    
    results = []
    feature_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    
    for i, test_case in enumerate(test_cases):
        print(f"\n📊 Test {i+1}: {test_case['name']}")
        print(f"   Description: {test_case['description']}")
        
        try:
            # Create test data
            np.random.seed(42 + i)  # Different seed for each test
            
            total_samples = test_case['class_0_count'] + test_case['class_1_count']
            if total_samples == 0:
                print("   ⚠️ Skipping: No samples to create")
                continue
                
            # Create features
            X = pd.DataFrame({
                'feature_1': np.random.randn(total_samples),
                'feature_2': np.random.randn(total_samples), 
                'feature_3': np.random.randn(total_samples),
                'feature_4': np.random.randn(total_samples),
                'feature_5': np.random.randn(total_samples)
            })
            
            # Create target
            y = []
            y.extend([0] * test_case['class_0_count'])
            y.extend([1] * test_case['class_1_count'])
            y = pd.Series(y, name='target', dtype=float)
            
            # Combine
            df = pd.concat([X, y], axis=1)
            
            print(f"   📈 Original distribution: {y.value_counts().to_dict()}")
            
            # Test data preparation
            X_train, X_test, y_train, y_test = prepare_training_data(
                df, feature_columns, 'target', test_size=0.2, random_state=42
            )
            
            print(f"   ✅ Data preparation succeeded")
            print(f"   📊 Train: {y_train.value_counts().to_dict()}")
            print(f"   📊 Test: {y_test.value_counts().to_dict()}")
            
            # Test training (quick test with subset of models)
            if len(y_train.unique()) >= 1:  # At least one class in training
                print(f"   🚀 Testing model training...")
                
                # Test with a reduced set for speed
                models, metrics = train_committee_models_advanced(
                    X_train, y_train, X_test, y_test
                )
                
                # Check if we got reasonable metrics
                if metrics and len(metrics) > 0:
                    avg_f1 = np.mean([m['f1'] for m in metrics.values()])
                    print(f"   ✅ Training succeeded! Average F1: {avg_f1:.3f}")
                    results.append({
                        'test_case': test_case['name'],
                        'status': 'PASSED',
                        'avg_f1': avg_f1,
                        'model_count': len(metrics)
                    })
                else:
                    print(f"   ⚠️ Training completed but no metrics returned")
                    results.append({
                        'test_case': test_case['name'],
                        'status': 'PARTIAL',
                        'avg_f1': 0.0,
                        'model_count': 0
                    })
            else:
                print(f"   ⚠️ Training data has no valid classes")
                results.append({
                    'test_case': test_case['name'],
                    'status': 'FAILED',
                    'avg_f1': 0.0,
                    'model_count': 0
                })
                
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            results.append({
                'test_case': test_case['name'],
                'status': 'FAILED',
                'avg_f1': 0.0,
                'model_count': 0
            })
    
    return results

def main():
    """Run comprehensive edge case testing."""
    print("🚀 Starting comprehensive edge case testing...")
    
    results = test_edge_cases()
    
    print("\n" + "="*60)
    print("🎯 COMPREHENSIVE TEST RESULTS")
    print("="*60)
    
    passed = sum(1 for r in results if r['status'] == 'PASSED')
    total = len(results)
    
    for result in results:
        status_emoji = {
            'PASSED': '✅',
            'PARTIAL': '⚠️', 
            'FAILED': '❌'
        }.get(result['status'], '❓')
        
        print(f"{status_emoji} {result['test_case']:<20} | "
              f"F1: {result['avg_f1']:.3f} | "
              f"Models: {result['model_count']}")
    
    print(f"\n🏆 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All edge cases handled successfully!")
        print("🚀 The system is ready for production with extreme datasets!")
    else:
        print("⚠️ Some edge cases still need attention.")

if __name__ == "__main__":
    main()
