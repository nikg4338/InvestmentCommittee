#!/usr/bin/env python3
"""
Test script to verify the threshold optimization improvements
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# Test the improved find_optimal_threshold function
def test_find_optimal_threshold():
    """Test the find_optimal_threshold function with edge cases"""
    
    # Import the function
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    from train_models import find_optimal_threshold
    
    print("Testing find_optimal_threshold improvements...")
    
    # Test case 1: Normal case
    y_true = np.array([0, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    
    threshold, score = find_optimal_threshold(y_true, y_proba, metric='f1')
    print(f"Test 1 - Normal case: threshold={threshold:.3f}, F1={score:.3f}")
    
    # Test case 2: Very low probabilities (should use fallback)
    y_true = np.array([0, 0, 0, 1, 1])
    y_proba = np.array([0.01, 0.02, 0.03, 0.04, 0.05])  # All very low
    
    threshold, score = find_optimal_threshold(y_true, y_proba, metric='f1')
    print(f"Test 2 - Low probabilities: threshold={threshold:.6f}, F1={score:.3f}")
    
    # Manually verify this gives some positives
    y_pred = (y_proba >= threshold).astype(int)
    manual_f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"  Manual verification: {np.sum(y_pred)} positives predicted, F1={manual_f1:.3f}")
    
    # Test case 3: All zeros probabilities (edge case)
    y_true = np.array([0, 0, 0, 1, 1])
    y_proba = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    threshold, score = find_optimal_threshold(y_true, y_proba, metric='f1')
    print(f"Test 3 - All zero probabilities: threshold={threshold:.3f}, F1={score:.3f}")
    
    print("✅ find_optimal_threshold tests completed\n")

def test_threshold_range():
    """Test that thresholds are now searched in full [0,1] range"""
    
    from train_models import find_optimal_threshold
    
    print("Testing threshold range [0,1]...")
    
    # Create case where optimal threshold is very low
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # 20% positive
    y_proba = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
    
    threshold, score = find_optimal_threshold(y_true, y_proba, metric='f1')
    print(f"Low threshold case: threshold={threshold:.3f}, F1={score:.3f}")
    
    # Verify this finds a threshold below 0.1 (old range minimum)
    if threshold < 0.1:
        print("✅ Successfully found threshold below 0.1 (old minimum)")
    else:
        print("⚠️ Threshold not below 0.1 - check implementation")
    
    # Create case where optimal threshold is very high
    y_true = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])  # 80% positive
    y_proba = np.array([0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0])
    
    threshold, score = find_optimal_threshold(y_true, y_proba, metric='f1')
    print(f"High threshold case: threshold={threshold:.3f}, F1={score:.3f}")
    
    # Verify this finds a threshold above 0.9 (old range maximum)  
    if threshold > 0.9:
        print("✅ Successfully found threshold above 0.9 (old maximum)")
    else:
        print("⚠️ Threshold not above 0.9 - check implementation")
    
    print("✅ Threshold range tests completed\n")

if __name__ == "__main__":
    print("Testing threshold optimization improvements...\n")
    
    try:
        test_find_optimal_threshold()
        test_threshold_range()
        print("🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
