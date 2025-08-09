#!/usr/bin/env python3
"""
Minimal wrapper to debug train_models.py execution issues
"""

import sys
import subprocess
import os
from pathlib import Path

def main():
    # Set up environment exactly like the orchestrator
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    # Use the exact command from the orchestrator
    cmd_args = [
        sys.executable, '-u', 'train_models.py',
        '--data-file', 'data/batch_1_data.csv',
        '--config', 'default',
        '--target-column', 'target',
        '--save-plots',
        '--export-results',
        '--log-level', 'INFO',
        '--batch-id', '1',
        '--optuna-trials', '5',
        '--telemetry-json', 'logs/telemetry_debug.json',
    ]
    
    print(f"Running command: {' '.join(cmd_args)}")
    print(f"Environment PYTHONUNBUFFERED: {env.get('PYTHONUNBUFFERED')}")
    print(f"Working directory: {os.getcwd()}")
    
    # Try with capture_output=False to see output directly
    try:
        result = subprocess.run(cmd_args, env=env, timeout=120)
        print(f"Return code: {result.returncode}")
    except Exception as e:
        print(f"Exception: {e}")
    
    # Check for output files
    print("\nChecking for output files...")
    for pattern in ['telemetry_debug.json', 'training_summary_*.csv', 'detailed_results_*.json']:
        files = list(Path('logs').glob(pattern))
        if files:
            latest = max(files, key=lambda p: p.stat().st_mtime)
            print(f"Found {pattern}: {latest} ({latest.stat().st_size} bytes)")
        else:
            print(f"No files found for {pattern}")

if __name__ == "__main__":
    main()
