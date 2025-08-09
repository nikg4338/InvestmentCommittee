#!/usr/bin/env python3
"""
Simple clean data collection using known-good symbols.
"""
import pandas as pd
import logging
import sys
import os

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collection_alpaca import AlpacaDataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Collect data from known authentic symbols only."""
    
    # Known authentic symbols (excluding EURKU)
    authentic_symbols = ['SVAL', 'EUM', 'SXQG', 'EUFN', 'EUDV']
    
    logger.info(f"🎯 Collecting data for authentic symbols: {authentic_symbols}")
    
    # Initialize collector with validation enabled
    collector = AlpacaDataCollector(enable_validation=True)
    
    # Create a mock batch data structure
    batch_data = {
        'batches': {
            'clean_batch': authentic_symbols
        },
        'metadata': {
            'description': 'Clean batch without synthetic tickers',
            'created': '2025-01-08'
        }
    }
    
    # Collect data using the batch collection method
    try:
        logger.info("🚀 Starting clean data collection...")
        result = collector.collect_enhanced_data(
            batch_numbers=['clean_batch'],
            max_symbols_per_batch=5,
            lookback_days=730,
            batch_data_override=batch_data
        )
        
        logger.info(f"✅ Collection complete: {result['data_shape']} samples")
        
        # Save to file
        output_file = "data/truly_clean_batch.csv"
        result['data'].to_csv(output_file, index=False)
        
        # Summary
        print(f"""
🎉 Clean Data Collection Success:
   📊 Samples: {result['data_shape'][0]}
   🏢 Tickers: {result['data']['ticker'].nunique()}
   🎯 Target distribution: {result['data']['target'].value_counts().to_dict()}
   📁 File: {output_file}
   
   Tickers collected: {list(result['data']['ticker'].unique())}
""")
        
    except Exception as e:
        logger.error(f"❌ Collection failed: {e}")
        raise

if __name__ == "__main__":
    main()
