#!/usr/bin/env python3
"""
Debug script to trace where train_models.py fails with real data
"""

import sys
import os
import pandas as pd
import traceback
import logging

# Setup basic logging to see output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("DEBUG: Starting real data test")
        
        # Test data loading
        data_file = 'data/batch_1_data.csv'
        target_column = 'target'
        
        logger.info(f"Loading data from {data_file}")
        df = pd.read_csv(data_file)
        logger.info(f"✓ Loaded {len(df)} samples from file")
        
        # Check target column
        if target_column not in df.columns:
            logger.error(f"❌ Target column '{target_column}' not found in data")
            return 1
        logger.info(f"✓ Target column '{target_column}' found")
        
        # Check target distribution
        target_dist = df[target_column].value_counts().to_dict()
        logger.info(f"Target distribution: {target_dist}")
        
        # Now try to import training modules
        logger.info("Importing training modules...")
        from config.training_config import get_default_config
        config = get_default_config()
        logger.info("✓ Config loaded")
        
        # Test data preparation function import
        logger.info("Testing data preparation imports...")
        
        # Check if the prepare_training_data function exists
        try:
            import train_models
            if hasattr(train_models, 'prepare_training_data'):
                logger.info("✓ prepare_training_data function found")
            else:
                logger.error("❌ prepare_training_data function not found")
                return 1
        except Exception as e:
            logger.error(f"Failed to import train_models: {e}")
            traceback.print_exc()
            return 1
            
        logger.info("DEBUG: All imports successful, calling actual script...")
        
        # Now try the actual script call
        original_argv = sys.argv.copy()
        sys.argv = [
            'train_models.py', 
            '--data-file', data_file,
            '--target-column', target_column,
            '--log-level', 'INFO'
        ]
        
        # Call main directly
        result = train_models.main()
        logger.info(f"Script returned: {result}")
        
    except Exception as e:
        logger.error(f"Exception: {e}")
        traceback.print_exc()
        return 1
    finally:
        if 'original_argv' in locals():
            sys.argv = original_argv
    
    logger.info("DEBUG: Test completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
