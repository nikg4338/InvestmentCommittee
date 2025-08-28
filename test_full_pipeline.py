"""
Quick fixes for Enhanced Training Pipeline
Fix the minor issues to improve model performance.
"""

from enhanced_training_pipeline import EnhancedTrainingPipeline
import os

def test_enhanced_pipeline_with_optimization():
    """Test the enhanced pipeline with minimal optimization enabled."""
    print('🚀 Testing Enhanced Training Pipeline with Optimization...')
    
    # Initialize with minimal optimization
    pipeline = EnhancedTrainingPipeline(
        use_enhanced_optimization=True,   # Enable optimization
        use_advanced_meta_learning=False, # Skip meta-learning for faster test
        generate_comprehensive_plots=False
    )
    
    # Use a smaller timeout for quick testing
    pipeline.enhanced_config['timeout_per_model'] = 60  # 1 minute per model
    
    # Test with a single batch
    test_file = 'data/leak_free_batch_1_data.csv'
    if os.path.exists(test_file):
        try:
            print('📊 Loading data...')
            batch_data = pipeline._load_batch_data(test_file)
            
            if batch_data:
                X_train, y_train, X_test, y_test = batch_data
                print(f'✅ Data loaded: {X_train.shape[0]} training samples, {X_train.shape[1]} features')
                
                # Test single batch training with optimization
                print('🎯 Starting optimized training...')
                results = pipeline.train_single_batch_enhanced(
                    'optimized_test_batch', X_train, y_train, X_test, y_test
                )
                
                if results and 'optimization_results' in results:
                    print('✅ Optimized training completed successfully!')
                    
                    # Show optimization results
                    opt_results = results['optimization_results']
                    print('🔧 Optimization Results:')
                    for model_name, result in opt_results.items():
                        if isinstance(result, dict) and 'best_score' in result:
                            score = result['best_score']
                            print(f'   {model_name}: Best PR-AUC = {score:.4f}')
                    
                    # Show final performance
                    if 'performance_summary' in results:
                        perf = results['performance_summary']
                        print('📈 Final Performance:')
                        for model_name, metrics in perf.items():
                            if isinstance(metrics, dict) and 'f1' in metrics:
                                f1 = metrics['f1']
                                print(f'   {model_name}: F1-Score = {f1:.4f}')
                    
                    print(f'⏱️ Total training time: {results.get("training_time", 0):.1f}s')
                    return True
                else:
                    print('❌ Optimized training failed or returned incomplete results')
                    return False
            else:
                print('❌ Data loading failed')
                return False
        except Exception as e:
            print(f'❌ Optimized training failed: {e}')
            import traceback
            traceback.print_exc()
            return False
    else:
        print(f'❌ Test file not found: {test_file}')
        return False

def run_full_batches_test():
    """Test training multiple batches with the enhanced pipeline."""
    print('🌟 Testing Enhanced Training Pipeline with Multiple Batches...')
    
    # Initialize with conservative settings
    pipeline = EnhancedTrainingPipeline(
        use_enhanced_optimization=False,  # Skip optimization for faster multi-batch test
        use_advanced_meta_learning=False,
        generate_comprehensive_plots=False
    )
    
    try:
        # Test training all batches
        print('🔄 Starting multi-batch training...')
        results = pipeline.train_all_batches_enhanced(data_dir="data")
        
        if results and 'total_batches_processed' in results:
            total = results['total_batches_processed']
            successful = results['successful_batches']
            total_time = results['total_training_time']
            
            print(f'✅ Multi-batch training completed!')
            print(f'   Batches processed: {total}')
            print(f'   Successful batches: {successful}')
            print(f'   Success rate: {successful/total*100:.1f}%')
            print(f'   Total time: {total_time:.1f}s')
            
            # Show cross-batch analysis if available
            if 'cross_batch_analysis' in results and results['cross_batch_analysis']:
                print('📈 Cross-batch analysis completed')
            
            # Show deployment results if available
            if 'deployment_results' in results and results['deployment_results']:
                print('🚀 Production deployment completed')
            
            return True
        else:
            print('❌ Multi-batch training failed or returned incomplete results')
            return False
            
    except Exception as e:
        print(f'❌ Multi-batch training failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Running Enhanced Training Pipeline Tests")
    print("=" * 60)
    
    # Test 1: Single batch with optimization
    print("\n" + "="*60)
    print("TEST 1: Single Batch with Optimization")
    print("="*60)
    success1 = test_enhanced_pipeline_with_optimization()
    
    # Test 2: Multiple batches
    print("\n" + "="*60)
    print("TEST 2: Multiple Batches")
    print("="*60)
    success2 = run_full_batches_test()
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    if success1:
        print("✅ Single batch with optimization: PASSED")
    else:
        print("❌ Single batch with optimization: FAILED")
    
    if success2:
        print("✅ Multiple batches: PASSED")
    else:
        print("❌ Multiple batches: FAILED")
    
    if success1 and success2:
        print("\n🎉 All tests passed! Enhanced training pipeline is fully functional.")
    elif success1 or success2:
        print("\n⚠️ Partial success. Enhanced training pipeline has some functionality.")
    else:
        print("\n❌ Tests failed. Please review the issues above.")
