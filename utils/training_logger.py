import os
import csv
from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def log_training_summary(batch_number: int, 
                        symbols_trained: int,
                        xgb_accuracy: float,
                        nn_accuracy: float,
                        training_time_seconds: float,
                        timeframe: str = "1Day",
                        log_file: str = "logs/training_summary.csv"):
    """
    Log training summary to CSV file.
    
    Args:
        batch_number (int): Batch number that was trained
        symbols_trained (int): Number of symbols successfully trained
        xgb_accuracy (float): XGBoost model accuracy
        nn_accuracy (float): Neural Network model accuracy  
        training_time_seconds (float): Time taken for training in seconds
        timeframe (str): Bar timeframe used for training
        log_file (str): Path to CSV log file
    """
    
    # Ensure logs directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(log_file)
    
    # Prepare row data
    row_data = {
        'timestamp': datetime.now().isoformat(),
        'batch_number': batch_number,
        'symbols_trained': symbols_trained,
        'timeframe': timeframe,
        'xgb_accuracy': round(xgb_accuracy, 4),
        'nn_accuracy': round(nn_accuracy, 4),
        'training_time_seconds': round(training_time_seconds, 2),
        'training_time_minutes': round(training_time_seconds / 60, 2)
    }
    
    try:
        with open(log_file, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'batch_number', 'symbols_trained', 'timeframe',
                         'xgb_accuracy', 'nn_accuracy', 'training_time_seconds', 'training_time_minutes']
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
                logger.info(f"Created new training summary log: {log_file}")
            
            # Write the data row
            writer.writerow(row_data)
            
        logger.info(f"Training summary logged to {log_file}")
        logger.info(f"  Batch {batch_number}: {symbols_trained} symbols, XGB: {xgb_accuracy:.3f}, NN: {nn_accuracy:.3f}, Time: {training_time_seconds / 60:.1f}m")
        
    except Exception as e:
        logger.error(f"Failed to log training summary to {log_file}: {e}")

def get_training_history(log_file: str = "logs/training_summary.csv") -> list:
    """
    Read training history from CSV file.
    
    Args:
        log_file (str): Path to CSV log file
        
    Returns:
        list: List of training records
    """
    if not os.path.exists(log_file):
        return []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            return list(reader)
    except Exception as e:
        logger.error(f"Failed to read training history from {log_file}: {e}")
        return []

def show_training_summary(log_file: str = "logs/training_summary.csv"):
    """
    Display a summary of all training sessions.
    
    Args:
        log_file (str): Path to CSV log file
    """
    history = get_training_history(log_file)
    
    if not history:
        logger.info("No training history found.")
        return
    
    logger.info("=" * 80)
    logger.info("TRAINING HISTORY SUMMARY")
    logger.info("=" * 80)
    
    total_symbols = 0
    total_time = 0
    
    for record in history:
        batch_num = record.get('batch_number', 'N/A')
        symbols = int(record.get('symbols_trained', 0))
        timeframe = record.get('timeframe', 'N/A')
        xgb_acc = float(record.get('xgb_accuracy', 0))
        nn_acc = float(record.get('nn_accuracy', 0))
        time_min = float(record.get('training_time_minutes', 0))
        timestamp = record.get('timestamp', 'N/A')[:19]  # Remove microseconds
        
        total_symbols += symbols
        total_time += time_min
        
        logger.info(f"Batch {batch_num:>2}: {symbols:>3} symbols | {timeframe:>4} | XGB: {xgb_acc:.3f} | NN: {nn_acc:.3f} | {time_min:>5.1f}m | {timestamp}")
    
    logger.info("=" * 80)
    logger.info(f"TOTALS: {len(history)} batches, {total_symbols} symbols, {total_time:.1f} minutes")
    logger.info("=" * 80) 