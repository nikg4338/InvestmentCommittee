#!/usr/bin/env python3
"""
Simple test script for Gemini API integration
Verifies that the GeminiAnalyzer works correctly with proper error handling
Does not require pytest - runs as a standalone script
"""

import sys
import os
from unittest.mock import Mock, patch
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llm_analyzer import (
    GeminiAnalyzer, 
    AnalysisType, 
    AnalysisResult, 
    RateLimiter, 
    create_gemini_analyzer,
    quick_analysis
)
from config.settings import validate_gemini_api_key

class TestRunner:
    """Simple test runner that doesn't require pytest"""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
    
    def run_test(self, test_name: str, test_func):
        """Run a single test"""
        try:
            test_func()
            print(f"✅ {test_name}")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ {test_name}: {e}")
            self.tests_failed += 1
            self.failures.append((test_name, str(e)))
    
    def assert_equals(self, actual, expected, message=""):
        """Simple assertion"""
        if actual != expected:
            raise AssertionError(f"Expected {expected}, got {actual}. {message}")
    
    def assert_true(self, condition, message=""):
        """Assert condition is true"""
        if not condition:
            raise AssertionError(f"Expected True, got False. {message}")
    
    def assert_isinstance(self, obj, expected_type, message=""):
        """Assert object is instance of type"""
        if not isinstance(obj, expected_type):
            raise AssertionError(f"Expected {expected_type}, got {type(obj)}. {message}")
    
    def assert_in(self, item, container, message=""):
        """Assert item is in container"""
        if item not in container:
            raise AssertionError(f"Expected {item} to be in {container}. {message}")
    
    def expect_exception(self, exception_type, func, *args, **kwargs):
        """Expect an exception to be raised"""
        try:
            func(*args, **kwargs)
            raise AssertionError(f"Expected {exception_type.__name__} to be raised")
        except exception_type:
            pass  # Expected exception
    
    def summary(self):
        """Print test summary"""
        print(f"\n{'='*50}")
        print(f"Tests passed: {self.tests_passed}")
        print(f"Tests failed: {self.tests_failed}")
        print(f"Total tests: {self.tests_passed + self.tests_failed}")
        
        if self.failures:
            print(f"\nFailures:")
            for test_name, error in self.failures:
                print(f"  - {test_name}: {error}")
        
        return self.tests_failed == 0

def test_rate_limiter_initialization():
    """Test rate limiter initialization"""
    runner = TestRunner()
    
    limiter = RateLimiter()
    runner.assert_equals(limiter.requests_per_minute, 15)
    runner.assert_equals(limiter.requests_per_day, 1500)
    runner.assert_equals(limiter.backoff_time, 1)
    runner.assert_equals(limiter.max_backoff_time, 300)

def test_rate_limiter_backoff():
    """Test exponential backoff"""
    runner = TestRunner()
    
    limiter = RateLimiter()
    initial_backoff = limiter.backoff_time
    
    # Simulate rate limit error
    limiter.handle_rate_limit_error()
    runner.assert_equals(limiter.backoff_time, initial_backoff * 2)
    
    # Test reset
    limiter.reset_backoff()
    runner.assert_equals(limiter.backoff_time, 1)

def test_analyzer_initialization_without_api_key():
    """Test analyzer initialization without API key"""
    runner = TestRunner()
    
    with patch('models.llm_analyzer.validate_gemini_api_key', return_value=False):
        runner.expect_exception(ValueError, GeminiAnalyzer)

def test_analyzer_initialization_with_api_key():
    """Test analyzer initialization with valid API key"""
    runner = TestRunner()
    
    with patch('models.llm_analyzer.validate_gemini_api_key', return_value=True), \
         patch('models.llm_analyzer.get_gemini_config') as mock_config, \
         patch('models.llm_analyzer.genai.configure') as mock_configure, \
         patch('models.llm_analyzer.genai.GenerativeModel') as mock_model:
        
        mock_config.return_value = {"api_key": "test_key"}
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        analyzer = GeminiAnalyzer()
        
        runner.assert_equals(analyzer.model_name, "gemini-1.5-flash")
        runner.assert_equals(analyzer.model, mock_model_instance)
        runner.assert_isinstance(analyzer.rate_limiter, RateLimiter)
        mock_configure.assert_called_once_with(api_key="test_key")

def test_analyze_macro_conditions():
    """Test macro conditions analysis"""
    runner = TestRunner()
    
    with patch('models.llm_analyzer.validate_gemini_api_key', return_value=True), \
         patch('models.llm_analyzer.get_gemini_config') as mock_config, \
         patch('models.llm_analyzer.genai.configure'), \
         patch('models.llm_analyzer.genai.GenerativeModel') as mock_model:
        
        mock_config.return_value = {"api_key": "test_key"}
        
        # Mock response
        mock_response = Mock()
        mock_response.text = "Macro analysis result"
        mock_candidate = Mock()
        mock_candidate.finish_reason.name = "STOP"
        mock_response.candidates = [mock_candidate]
        
        mock_model_instance = Mock()
        mock_model_instance.generate_content.return_value = mock_response
        mock_model.return_value = mock_model_instance
        
        analyzer = GeminiAnalyzer()
        analyzer.rate_limiter.wait_if_needed = Mock()
        
        economic_data = {
            "gdp_growth": 2.1,
            "inflation_rate": 3.2,
            "unemployment_rate": 3.8
        }
        
        result = analyzer.analyze_macro_conditions(economic_data)
        
        runner.assert_isinstance(result, AnalysisResult)
        runner.assert_equals(result.analysis_type, AnalysisType.MACRO_ANALYSIS)
        runner.assert_equals(result.content, "Macro analysis result")
        runner.assert_equals(result.confidence, 0.85)
        runner.assert_isinstance(result.timestamp, datetime)
        runner.assert_equals(result.metadata["economic_data"], economic_data)

def test_usage_stats():
    """Test usage statistics"""
    runner = TestRunner()
    
    with patch('models.llm_analyzer.validate_gemini_api_key', return_value=True), \
         patch('models.llm_analyzer.get_gemini_config') as mock_config, \
         patch('models.llm_analyzer.genai.configure'), \
         patch('models.llm_analyzer.genai.GenerativeModel'):
        
        mock_config.return_value = {"api_key": "test_key"}
        
        analyzer = GeminiAnalyzer()
        stats = analyzer.get_usage_stats()
        
        runner.assert_in("requests_this_minute", stats)
        runner.assert_in("requests_this_day", stats)
        runner.assert_in("minute_limit", stats)
        runner.assert_in("day_limit", stats)
        runner.assert_in("current_backoff_time", stats)
        runner.assert_in("model_name", stats)
        runner.assert_equals(stats["model_name"], "gemini-1.5-flash")

def test_api_key_validation():
    """Test API key validation"""
    runner = TestRunner()
    
    # This will test the actual validation function
    result = validate_gemini_api_key()
    runner.assert_isinstance(result, bool)

def main():
    """Run all tests"""
    print("=== Simple Gemini API Integration Tests ===\n")
    
    runner = TestRunner()
    
    # Run all tests
    runner.run_test("Rate Limiter Initialization", test_rate_limiter_initialization)
    runner.run_test("Rate Limiter Backoff", test_rate_limiter_backoff)
    runner.run_test("Analyzer Init Without API Key", test_analyzer_initialization_without_api_key)
    runner.run_test("Analyzer Init With API Key", test_analyzer_initialization_with_api_key)
    runner.run_test("Analyze Macro Conditions", test_analyze_macro_conditions)
    runner.run_test("Usage Stats", test_usage_stats)
    runner.run_test("API Key Validation", test_api_key_validation)
    
    # Print summary
    success = runner.summary()
    
    # Integration test if API key is available
    if validate_gemini_api_key():
        print(f"\n{'='*50}")
        print("=== Integration Test ===")
        
        try:
            analyzer = create_gemini_analyzer()
            print("✅ Analyzer creation successful")
            
            stats = analyzer.get_usage_stats()
            print(f"✅ Usage stats retrieved: {stats}")
            
            print("\n✅ Integration test passed!")
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            success = False
    else:
        print("\n⚠️  API key not configured - skipping integration test")
        print("Please set GEMINI_API_KEY in your .env file for full testing")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 