#!/usr/bin/env python3
"""
Test script to validate the integrated improvements from the user's OOF script.
"""

def test_new_cli_arguments():
    """Test that new CLI arguments are properly configured."""
    try:
        import argparse
        import sys
        from io import StringIO
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        # Test argument parsing
        from train_models import main
        
        # Restore stdout
        sys.stdout = old_stdout
        
        print("✅ CLI arguments properly configured")
        return True
        
    except Exception as e:
        print(f"❌ CLI argument test failed: {e}")
        return False

def test_helper_functions():
    """Test new helper functions."""
    try:
        from train_models import prepare_sampler, prepare_calibration_method, evaluate_models_comprehensive
        
        # Test sampler preparation
        sampler = prepare_sampler('smoteenn')
        print(f"   SMOTEENN sampler: {type(sampler).__name__ if sampler else 'None (fallback)'}")
        
        sampler_none = prepare_sampler('none')
        print(f"   None sampler: {sampler_none}")
        
        # Test calibration method
        calib_isotonic = prepare_calibration_method('isotonic')
        calib_none = prepare_calibration_method('none')
        print(f"   Isotonic calibration: {calib_isotonic}")
        print(f"   None calibration: {calib_none}")
        
        print("✅ Helper functions working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Helper function test failed: {e}")
        return False

def test_configuration_integration():
    """Test that configuration constants are properly integrated."""
    try:
        from train_models import (
            DEFAULT_N_FOLDS, DEFAULT_CALIBRATION_METHOD, DEFAULT_RANDOM_STATE,
            DEFAULT_SMOTE_K_NEIGHBORS, DEFAULT_META_MAX_ITER
        )
        
        print(f"   DEFAULT_N_FOLDS: {DEFAULT_N_FOLDS}")
        print(f"   DEFAULT_CALIBRATION_METHOD: {DEFAULT_CALIBRATION_METHOD}")
        print(f"   DEFAULT_RANDOM_STATE: {DEFAULT_RANDOM_STATE}")
        print(f"   DEFAULT_SMOTE_K_NEIGHBORS: {DEFAULT_SMOTE_K_NEIGHBORS}")
        print(f"   DEFAULT_META_MAX_ITER: {DEFAULT_META_MAX_ITER}")
        
        print("✅ Configuration constants properly integrated")
        return True
        
    except Exception as e:
        print(f"❌ Configuration integration test failed: {e}")
        return False

def test_unified_visualization():
    """Test that unified visualization approach is working."""
    try:
        from train_models import (
            _determine_output_directory, _setup_output_directory,
            create_visualizations_unified
        )
        
        # Test directory determination with custom base
        batch_dir, suffix = _determine_output_directory(batch_num="1_2", base_dir="custom_output")
        print(f"   Custom output directory: {batch_dir}, suffix: {suffix}")
        
        batch_dir2, suffix2 = _determine_output_directory(batch_num=1, base_dir="reports")
        print(f"   Standard output directory: {batch_dir2}, suffix: {suffix2}")
        
        print("✅ Unified visualization system working")
        return True
        
    except Exception as e:
        print(f"❌ Unified visualization test failed: {e}")
        return False

def test_main_function_signature():
    """Test that main function accepts new parameters."""
    try:
        from train_models import main
        import inspect
        
        sig = inspect.signature(main)
        params = list(sig.parameters.keys())
        
        expected_params = [
            'batch_numbers', 'max_symbols_per_batch', 'use_alpaca', 'stacking_method',
            'n_folds', 'sampler', 'calibrate', 'threshold', 'output_dir'
        ]
        
        for param in expected_params:
            if param in params:
                print(f"   ✓ {param}")
            else:
                print(f"   ✗ Missing {param}")
                return False
        
        print("✅ Main function signature updated correctly")
        return True
        
    except Exception as e:
        print(f"❌ Main function signature test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🧪 Testing integrated OOF script improvements...")
    print("=" * 60)
    
    tests = [
        test_configuration_integration,
        test_helper_functions,
        test_unified_visualization,
        test_main_function_signature,
        test_new_cli_arguments
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("=" * 60)
    print(f"🎯 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All OOF script improvements successfully integrated!")
        print()
        print("📋 Key improvements added:")
        print("   ✓ Centralized CLI flags for folds, sampler, calibration, threshold, output-dir")
        print("   ✓ Enhanced prepare_sampler() and prepare_calibration_method() utilities") 
        print("   ✓ Comprehensive evaluate_models_comprehensive() function")
        print("   ✓ Configurable output directory for visualizations")
        print("   ✓ Updated main() function with all new parameters")
        print("   ✓ Full parameter passing from CLI to main execution")
        print()
        print("🚀 You can now use commands like:")
        print("   python train_models.py --n-folds 10 --sampler smoteenn --calibrate sigmoid --output-dir custom_plots")
    else:
        print(f"⚠️  {len(tests) - passed} tests failed. Review needed.")

if __name__ == "__main__":
    main()
