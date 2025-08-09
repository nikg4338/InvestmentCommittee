#!/usr/bin/env python3
"""
Clean data collection script that excludes known synthetic tickers.
"""
import pandas as pd
import logging
from data_collection_alpaca import AlpacaDataCollector
from utils.data_validation import SyntheticDataDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_authentic_data_only(batch_nums=[1], max_symbols=5, output_file="data/authentic_batch.csv"):
    """Collect data and filter out synthetic tickers."""
    
    # Initialize collector and validator
    collector = AlpacaDataCollector(enable_validation=False)  # We'll validate manually
    detector = SyntheticDataDetector()
    
    # Known synthetic tickers to exclude
    known_synthetic = ['EURKU']
    
    logger.info(f"🚫 Excluding known synthetic tickers: {known_synthetic}")
    
    # Load all batches
    batches = collector.load_stock_batches()
    
    # Collect raw data first
    all_data = []
    
    for batch_num in batch_nums:
        batch_key = f"batch_{batch_num}"
        
        if batch_key not in batches:
            logger.error(f"❌ Batch {batch_key} not found in batches")
            continue
            
        logger.info(f"🔄 Processing {batch_key}...")
        
        # Get symbol list for this batch
        batch_symbols = batches[batch_key]
        
        # Filter out synthetic tickers
        clean_symbols = [s for s in batch_symbols if s not in known_synthetic]
        
        if max_symbols:
            clean_symbols = clean_symbols[:max_symbols]
        
        logger.info(f"📋 Original symbols: {len(batch_symbols)}, Clean symbols: {len(clean_symbols)}")
        logger.info(f"🎯 Collecting data for: {clean_symbols}")
        
        # Collect data for each clean symbol
        for symbol in clean_symbols:
            try:
                logger.info(f"📊 Processing {symbol}...")
                symbol_data = collector._collect_single_symbol(symbol, lookback_days=730)
                
                if symbol_data is not None and len(symbol_data) > 0:
                    all_data.append(symbol_data)
                    logger.info(f"✅ {symbol}: {len(symbol_data)} samples collected")
                else:
                    logger.warning(f"⚠️ {symbol}: No data collected")
                    
            except Exception as e:
                logger.error(f"❌ Error collecting {symbol}: {e}")
                continue
    
    if not all_data:
        raise ValueError("No authentic data could be collected!")
    
    # Combine all data
    final_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"📊 Combined data: {len(final_df)} samples from {final_df['ticker'].nunique()} tickers")
    
    # Final validation check
    validation_results = detector.detect_synthetic_patterns(final_df)
    
    if validation_results.get('is_synthetic', False):
        logger.error(f"❌ Validation failed: {validation_results['issues']}")
        raise ValueError("Collected data still contains synthetic patterns!")
    
    # Save clean data
    final_df.to_csv(output_file, index=False)
    logger.info(f"💾 Clean authenticated data saved to {output_file}")
    
    # Report summary
    print(f"""
🎉 Clean Data Collection Complete:
   📊 Samples: {len(final_df)}
   🏢 Tickers: {final_df['ticker'].nunique()}
   🎯 Target distribution: {final_df['target'].value_counts().to_dict()}
   ✅ Validation: PASSED (no synthetic data detected)
   📁 File: {output_file}
""")
    
    return final_df

if __name__ == "__main__":
    # Collect 5 clean symbols from batch 1
    collect_authentic_data_only(batch_nums=[1], max_symbols=5, output_file="data/clean_batch_1.csv")
