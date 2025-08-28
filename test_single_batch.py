"""
Test Enhanced Training Pipeline Functionality
"""
from enhanced_training_pipeline import EnhancedTrainingPipeline
import os

print('🚀 Testing Enhanced Training Pipeline with Single Batch...')

# Initialize with minimal settings
pipeline = EnhancedTrainingPipeline(
    use_enhanced_optimization=False,
    use_advanced_meta_learning=False,
    generate_comprehensive_plots=False
)

# Test single batch training
test_file = 'data/leak_free_batch_1_data.csv'
if os.path.exists(test_file):
    try:
        batch_data = pipeline._load_batch_data(test_file)
        if batch_data:
            X_train, y_train, X_test, y_test = batch_data
            print(f'✅ Data loaded: {X_train.shape[0]} training samples')
            
            # Test single batch training
            print('🎯 Starting single batch training...')
            results = pipeline.train_single_batch_enhanced(
                'test_batch_1', X_train, y_train, X_test, y_test
            )
            
            if results:
                print('✅ Single batch training completed successfully!')
                if 'models' in results:
                    model_names = list(results['models'].keys())
                    print('   Models trained:', model_names)
                if 'training_time' in results:
                    training_time = results['training_time']
                    print(f'   Training time: {training_time:.1f}s')
                print('   Result keys:', list(results.keys()))
            else:
                print('❌ Single batch training returned no results')
                
        else:
            print('❌ Data loading failed')
    except Exception as e:
        print(f'❌ Single batch training failed: {e}')
        import traceback
        traceback.print_exc()
else:
    print(f'❌ Test file not found: {test_file}')
