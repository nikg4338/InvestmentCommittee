#!/usr/bin/env python3
"""
Debug wrapper that prints debug statements as the train_models.py script progresses
"""

import sys
import os

# Add debug prints to trace execution
print("DEBUG: Script starting")

# Patch the train_models module to add debug logging
import train_models

# Wrap the main function to add debug tracing
original_main = train_models.main

def debug_main():
    print("DEBUG: Entering main function")
    try:
        result = original_main()
        print(f"DEBUG: main() returned: {result}")
        return result
    except Exception as e:
        print(f"DEBUG: Exception in main(): {e}")
        import traceback
        traceback.print_exc()
        return 1

# Replace the main function
train_models.main = debug_main

# Mock sys.argv to simulate orchestrator call
sys.argv = [
    'train_models.py',
    '--data-file', 'data/batch_1_data.csv',
    '--config', 'default',
    '--target-column', 'target',
    '--save-plots',
    '--export-results',
    '--log-level', 'INFO',
    '--batch-id', '1',
    '--optuna-trials', '5',
    '--telemetry-json', 'logs/telemetry_debug2.json',
]

print(f"DEBUG: sys.argv set to: {sys.argv}")

# Run main
exit_code = train_models.main()
print(f"DEBUG: Final exit code: {exit_code}")
