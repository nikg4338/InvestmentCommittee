#!/usr/bin/env python3
"""Quick test of the training pipeline."""

from comprehensive_training_validator import ComprehensiveTrainingValidator
import pandas as pd
import numpy as np

def main():
    """Quick test with existing data."""
    try:
        # Create validator
        validator = ComprehensiveTrainingValidator()
        
        # Load existing training data if available
        df = pd.read_csv('data/comprehensive_training_data_20250826_005032.csv')
        print(f'✅ Loaded existing data: {len(df)} samples')
        
        # Clean infinite values  
        df = df.replace([np.inf, -np.inf], 0.0)
        df = df.fillna(0.0)
        
        # Take a smaller sample for testing
        df_sample = df.sample(n=min(3000, len(df)), random_state=42)
        print(f'📊 Using sample: {len(df_sample)} samples')
        
        # Run training on sample
        print("🚀 Starting training...")
        training_results = validator.run_comprehensive_training(df_sample)
        
        if 'error' not in training_results:
            print('🎉 Training successful!')
            models_count = len(training_results.get('models', {}))
            print(f'   Models trained: {models_count}')
            
            # List model names
            model_names = list(training_results.get('models', {}).keys())
            print(f'   Model names: {model_names}')
            
        else:
            print(f'❌ Training failed: {training_results["error"]}')
            
    except Exception as e:
        print(f'❌ Test failed: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
