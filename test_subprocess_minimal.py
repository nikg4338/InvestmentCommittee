#!/usr/bin/env python3
"""
Minimal subprocess test to diagnose the silent exit issue
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    # Try the exact command from orchestrator
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    cmd_args = [
        sys.executable, '-u', 'train_models.py',
        '--data-file', 'data/batch_1_data.csv',
        '--config', 'default',
        '--target-column', 'target',
        '--save-plots',
        '--export-results',
        '--log-level', 'DEBUG',  # Use DEBUG to get more output
        '--batch-id', '1',
        '--optuna-trials', '5',
        '--telemetry-json', 'logs/telemetry_subprocess_test.json',
    ]
    
    print(f"Command: {' '.join(cmd_args)}")
    print(f"Working dir: {os.getcwd()}")
    
    # Check if the data file exists
    if not Path('data/batch_1_data.csv').exists():
        print("ERROR: Data file does not exist!")
        return
    
    print("Running subprocess with direct output...")
    try:
        # Run without capture_output to see output directly
        result = subprocess.run(cmd_args, env=env, timeout=600)
        print(f"Exit code: {result.returncode}")
    except subprocess.TimeoutExpired:
        print("Subprocess timed out")
    except Exception as e:
        print(f"Exception: {e}")
    
    # Check for any output files
    files_to_check = [
        'logs/telemetry_subprocess_test.json',
        'logs/training.log',
    ]
    
    print("\nChecking for output files:")
    for file_path in files_to_check:
        p = Path(file_path)
        if p.exists():
            size = p.stat().st_size
            mtime = p.stat().st_mtime
            print(f"  {file_path}: EXISTS ({size} bytes, mtime: {mtime})")
        else:
            print(f"  {file_path}: NOT FOUND")

if __name__ == "__main__":
    main()
