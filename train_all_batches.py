#!/usr/bin/env python3
"""
Automated Batch Training Script for Investment Committee
=======================================================

This script processes all batches from filtered_iex_batches.json with:
- Extreme imbalance configuration optimized for financial data
- Complete visual plots and reports
- Detailed CSV exports and logging
- Organized output in reports folder by batch

Usage:
    python train_all_batches.py                    # Process all non-empty batches
    python train_all_batches.py --batch 1          # Process specific batch
    python train_all_batches.py --start 1 --end 5 # Process batch range
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Setup logging for this script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/batch_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def ensure_directories():
    """Ensure required directories exist"""
    dirs = ['logs', 'reports', 'data', 'models/saved']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    logger.info(f"✓ Created directories: {', '.join(dirs)}")

def load_batch_data() -> Dict[str, Any]:
    """Load batch information from filtered_iex_batches.json"""
    try:
        with open("filtered_iex_batches.json", "r") as f:
            batch_data = json.load(f)
        logger.info(f"✓ Loaded batch data: {batch_data['metadata']['total_symbols']} symbols across {batch_data['metadata']['total_batches']} batches")
        return batch_data
    except FileNotFoundError:
        logger.error("❌ filtered_iex_batches.json not found!")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in filtered_iex_batches.json: {e}")
        sys.exit(1)

def get_non_empty_batches(batch_data: Dict[str, Any]) -> List[int]:
    """Get list of batch numbers that contain symbols"""
    non_empty = []
    for batch_name, symbols in batch_data["batches"].items():
        if symbols:  # Skip empty batches
            batch_num = int(batch_name.split("_")[1])
            non_empty.append(batch_num)
    logger.info(f"✓ Found {len(non_empty)} non-empty batches: {non_empty}")
    return non_empty

def collect_batch_data(batch_num: int, max_retries: int = 3) -> str:
    """
    Collect data for a specific batch using data_collection_alpaca.py
    
    Args:
        batch_num: Batch number to collect
        max_retries: Maximum retry attempts
        
    Returns:
        Path to created CSV file
    """
    data_file = f"data/batch_{batch_num}_data.csv"
    
    for attempt in range(max_retries):
        try:
            logger.info(f"📊 Collecting data for batch {batch_num} (attempt {attempt + 1}/{max_retries})...")
            
            # Run data collection
            result = subprocess.run([
                sys.executable, "data_collection_alpaca.py", 
                "--batches", str(batch_num),
                "--max-symbols", "50",
                "--output-file", data_file
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                if os.path.exists(data_file) and os.path.getsize(data_file) > 0:
                    logger.info(f"✓ Data collection successful: {data_file}")
                    return data_file
                else:
                    logger.warning(f"⚠️ Data file created but empty: {data_file}")
            else:
                logger.warning(f"⚠️ Data collection failed (attempt {attempt + 1}): {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.warning(f"⚠️ Data collection timeout (attempt {attempt + 1})")
        except Exception as e:
            logger.warning(f"⚠️ Data collection error (attempt {attempt + 1}): {e}")
    
    logger.error(f"❌ Failed to collect data for batch {batch_num} after {max_retries} attempts")
    return None

def train_batch_models(batch_num: int, data_file: str) -> bool:
    """
    Train models for a specific batch using the refactored train_models.py
    
    Args:
        batch_num: Batch number being trained
        data_file: Path to the CSV data file
        
    Returns:
        True if training successful, False otherwise
    """
    try:
        logger.info(f"🚀 Training models for batch {batch_num}...")
        
        # Run training with all required flags
        result = subprocess.run([
            sys.executable, "train_models.py",
            "--data-file", data_file,
            "--config", "extreme_imbalance",
            "--target-column", "target", 
            "--save-plots",
            "--export-results",
            "--log-level", "INFO",
            "--batch-id", str(batch_num)
        ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
        
        if result.returncode == 0:
            logger.info(f"✓ Training completed successfully for batch {batch_num}")
            
            # Print key output for monitoring
            if "Performance Summary:" in result.stdout:
                summary_start = result.stdout.find("Performance Summary:")
                summary_section = result.stdout[summary_start:summary_start+500]
                logger.info(f"📊 {summary_section}")
            
            return True
        else:
            logger.error(f"❌ Training failed for batch {batch_num}:")
            logger.error(f"STDERR: {result.stderr}")
            logger.error(f"STDOUT: {result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Training timeout for batch {batch_num} (30 minutes)")
        return False
    except Exception as e:
        logger.error(f"❌ Training error for batch {batch_num}: {e}")
        return False

def organize_batch_results(batch_num: int, data_file: str) -> bool:
    """
    Organize results into reports folder structure
    
    Args:
        batch_num: Batch number
        data_file: Path to the original data file
        
    Returns:
        True if organization successful, False otherwise
    """
    try:
        batch_reports_dir = Path("reports") / f"batch_{batch_num}"
        batch_reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Organizing results for batch {batch_num} in {batch_reports_dir}")
        
        # Move results folder contents
        if Path("results").exists():
            results_dest = batch_reports_dir / "results"
            if results_dest.exists():
                shutil.rmtree(results_dest)
            shutil.move("results", results_dest)
            logger.info(f"✓ Moved results to {results_dest}")
        
        # Move plots folder contents  
        if Path("plots").exists():
            plots_dest = batch_reports_dir / "plots"
            if plots_dest.exists():
                shutil.rmtree(plots_dest)
            shutil.move("plots", plots_dest)
            logger.info(f"✓ Moved plots to {plots_dest}")
        
        # Move any performance plots from reports folder to batch folder
        reports_plots = []
        for plot_pattern in ["performance_comparison_*.png", "confusion_matrices_*.png", "class_distribution_*.png"]:
            reports_plots.extend(Path("reports").glob(plot_pattern))
        
        if reports_plots:
            # Ensure plots directory exists in batch folder
            plots_dest = batch_reports_dir / "plots" 
            plots_dest.mkdir(exist_ok=True)
            
            for plot_file in reports_plots:
                # Rename to include batch number
                new_name = f"batch_{batch_num}_{plot_file.name}"
                plot_dest = plots_dest / new_name
                shutil.move(str(plot_file), str(plot_dest))
                logger.info(f"✓ Moved and renamed {plot_file.name} to {new_name}")
        
        # Copy training log
        training_log = Path("logs/training.log")
        if training_log.exists():
            log_dest = batch_reports_dir / f"batch_{batch_num}_training.log"
            shutil.copy2(training_log, log_dest)
            logger.info(f"✓ Copied training log to {log_dest}")
        
        # Move data file to batch directory
        if data_file and Path(data_file).exists():
            data_dest = batch_reports_dir / f"batch_{batch_num}_data.csv"
            shutil.copy2(data_file, data_dest)
            logger.info(f"✓ Copied data file to {data_dest}")
        
        # Create batch summary
        create_batch_summary(batch_num, batch_reports_dir)
        
        logger.info(f"✅ Results organized successfully for batch {batch_num}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to organize results for batch {batch_num}: {e}")
        return False

def create_batch_summary(batch_num: int, batch_dir: Path):
    """Create a summary file for the batch results"""
    try:
        summary_file = batch_dir / "BATCH_SUMMARY.md"
        
        # Try to read performance summary if it exists
        performance_summary = ""
        results_dir = batch_dir / "results"
        if results_dir.exists():
            summary_csv = results_dir / "performance_summary.csv"
            if summary_csv.exists():
                with open(summary_csv, 'r') as f:
                    performance_summary = f.read()
        
        # Count available files
        plots_count = len(list((batch_dir / "plots").glob("*.png"))) if (batch_dir / "plots").exists() else 0
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"""# Batch {batch_num} Training Summary

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- **Batch Number:** {batch_num}
- **Training Configuration:** extreme_imbalance
- **Target Column:** target

## Files Generated
- **Plots Created:** {plots_count} visualization files
- **Results:** CSV files with detailed metrics
- **Training Log:** Complete training log with timestamps

## Performance Summary
```
{performance_summary}
```

## Directory Structure
```
batch_{batch_num}/
├── results/                    # CSV files with metrics
├── plots/                      # Visualization plots
├── batch_{batch_num}_data.csv  # Training data
├── batch_{batch_num}_training.log # Training log
└── BATCH_SUMMARY.md           # This summary
```

## Next Steps
1. Review plots in the `plots/` folder
2. Analyze metrics in `results/performance_summary.csv`
3. Check training log for any warnings or issues
4. Compare results with other batches
""")
        
        logger.info(f"✓ Created batch summary: {summary_file}")
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to create batch summary: {e}")

def process_batch(batch_num: int) -> bool:
    """
    Process a single batch: collect data, train models, organize results
    
    Args:
        batch_num: Batch number to process
        
    Returns:
        True if successful, False otherwise
    """
    batch_start_time = time.time()
    logger.info(f"🎯 Starting batch {batch_num} processing...")
    
    # Step 1: Collect data
    data_file = collect_batch_data(batch_num)
    if not data_file:
        logger.error(f"❌ Skipping batch {batch_num} - data collection failed")
        return False
    
    # Step 2: Train models
    if not train_batch_models(batch_num, data_file):
        logger.error(f"❌ Skipping batch {batch_num} - training failed")
        return False
    
    # Step 3: Organize results
    if not organize_batch_results(batch_num, data_file):
        logger.warning(f"⚠️ Batch {batch_num} training completed but result organization failed")
    
    batch_duration = time.time() - batch_start_time
    logger.info(f"✅ Batch {batch_num} completed in {batch_duration:.1f} seconds")
    return True

def create_master_summary(successful_batches: List[int], failed_batches: List[int], total_time: float):
    """Create a master summary of all batch processing"""
    try:
        summary_file = Path("reports") / "MASTER_SUMMARY.md"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"""# Investment Committee - Master Training Summary

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Processing Time:** {total_time:.1f} seconds ({total_time/60:.1f} minutes)

## Batch Processing Results

### Successful Batches ({len(successful_batches)})
{', '.join(map(str, successful_batches))}

### Failed Batches ({len(failed_batches)})
{', '.join(map(str, failed_batches)) if failed_batches else 'None'}

## Configuration Used
- **Training Config:** extreme_imbalance (optimized for financial data)
- **Target Column:** target (buy/sell signals)
- **Visualization:** All plots saved to reports/batch_X/plots/
- **Metrics Export:** CSV files in reports/batch_X/results/
- **Logging Level:** INFO (detailed progress tracking)

## Next Steps
1. **Review Individual Batches:** Check `reports/batch_X/BATCH_SUMMARY.md` for each batch
2. **Compare Performance:** Use `reports/batch_X/results/performance_summary.csv` files
3. **Analyze Plots:** Visual analysis in `reports/batch_X/plots/` folders
4. **Aggregate Results:** Consider combining successful batches for meta-analysis

## Directory Structure
```
reports/
├── MASTER_SUMMARY.md          # This file
├── batch_1/                   # Batch 1 results
├── batch_2/                   # Batch 2 results
├── ...
└── batch_N/                   # Batch N results
```

## Model Performance Overview
Each batch was trained with the Committee of Five ensemble:
- XGBoost
- LightGBM  
- CatBoost
- Random Forest
- Support Vector Machine
- Meta-model (LogisticRegression)
- Final Ensemble (rank-and-vote)

For detailed performance metrics, see individual batch result files.
""")
        
        logger.info(f"✓ Created master summary: {summary_file}")
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to create master summary: {e}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Train models on all batches with comprehensive reporting')
    parser.add_argument('--batch', type=int, help='Process specific batch number only')
    parser.add_argument('--start', type=int, help='Start batch number (inclusive)')
    parser.add_argument('--end', type=int, help='End batch number (inclusive)')
    parser.add_argument('--skip-data-collection', action='store_true', 
                       help='Skip data collection (use existing CSV files)')
    
    args = parser.parse_args()
    
    # Setup
    start_time = time.time()
    ensure_directories()
    
    logger.info("🏁 Starting Investment Committee batch training...")
    logger.info(f"Configuration: extreme_imbalance, save_plots=True, export_results=True")
    
    # Load batch information
    batch_data = load_batch_data()
    available_batches = get_non_empty_batches(batch_data)
    
    # Determine which batches to process
    if args.batch:
        if args.batch in available_batches:
            batches_to_process = [args.batch]
            logger.info(f"🎯 Processing single batch: {args.batch}")
        else:
            logger.error(f"❌ Batch {args.batch} not found in available batches: {available_batches}")
            sys.exit(1)
    elif args.start is not None and args.end is not None:
        batches_to_process = [b for b in available_batches if args.start <= b <= args.end]
        logger.info(f"🎯 Processing batch range {args.start}-{args.end}: {batches_to_process}")
    else:
        batches_to_process = available_batches
        logger.info(f"🎯 Processing all {len(batches_to_process)} non-empty batches")
    
    if not batches_to_process:
        logger.error("❌ No batches to process!")
        sys.exit(1)
    
    # Process each batch
    successful_batches = []
    failed_batches = []
    
    for i, batch_num in enumerate(batches_to_process, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 Processing batch {batch_num} ({i}/{len(batches_to_process)})")
        logger.info(f"{'='*60}")
        
        if process_batch(batch_num):
            successful_batches.append(batch_num)
        else:
            failed_batches.append(batch_num)
        
        # Progress update
        logger.info(f"📊 Progress: {i}/{len(batches_to_process)} batches completed")
    
    # Final summary
    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"🏁 BATCH TRAINING COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"⏱️  Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"✅ Successful: {len(successful_batches)} batches {successful_batches}")
    logger.info(f"❌ Failed: {len(failed_batches)} batches {failed_batches}")
    logger.info(f"📁 Results saved in: reports/ folder")
    
    # Create master summary
    create_master_summary(successful_batches, failed_batches, total_time)
    
    # Return appropriate exit code
    if failed_batches:
        logger.warning("⚠️ Some batches failed - check logs for details")
        sys.exit(1)
    else:
        logger.info("🎉 All batches completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
