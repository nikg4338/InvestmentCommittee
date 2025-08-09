#!/usr/bin/env python3
"""
Debug script to test train_models.py entry points
"""

import sys
import os

print("DEBUG: Starting debug script")
print(f"DEBUG: Python version: {sys.version}")
print(f"DEBUG: Working directory: {os.getcwd()}")
print(f"DEBUG: Python path: {sys.path[:3]}")

try:
    print("DEBUG: Attempting to import train_models")
    import train_models
    print("DEBUG: Import successful")
    
    print("DEBUG: Checking if main function exists")
    if hasattr(train_models, 'main'):
        print("DEBUG: main function found")
    else:
        print("DEBUG: main function NOT found")
        
    print("DEBUG: About to call main with debug args")
    
    # Mock sys.argv for debugging
    original_argv = sys.argv.copy()
    sys.argv = ['train_models.py', '--quick-test', '--log-level', 'DEBUG']
    
    print(f"DEBUG: sys.argv set to: {sys.argv}")
    
    # Try to call main
    train_models.main()
    
    print("DEBUG: main() completed successfully")
    
except ImportError as e:
    print(f"DEBUG: Import error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"DEBUG: Exception during execution: {e}")
    import traceback
    traceback.print_exc()
finally:
    if 'original_argv' in locals():
        sys.argv = original_argv
        print("DEBUG: Restored original sys.argv")

print("DEBUG: Debug script completed")
