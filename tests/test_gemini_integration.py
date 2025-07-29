#!/usr/bin/env python3
"""
Test script for Gemini API integration
Verifies that the GeminiAnalyzer works correctly with proper error handling
"""

import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Handle pytest import gracefully
try:
    import pytest  # type: ignore
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    
    # Create a dummy pytest for when it's not available
    class DummyPytest:
        @staticmethod
        def raises(exception_type, **kwargs):
            def decorator(func):
                def wrapper(*args, **kwargs):
                    try:
                        func(*args, **kwargs)
                        raise AssertionError(f"Expected {exception_type.__name__} to be raised")
                    except exception_type:
                        pass  # Expected exception
                return wrapper
            return decorator
    
    pytest = DummyPytest()

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

class TestRateLimiter:
    """Test the rate limiter functionality"""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization"""
        limiter = RateLimiter()
        assert limiter.requests_per_minute == 15
        assert limiter.requests_per_day == 1500
        assert limiter.backoff_time == 1
        assert limiter.max_backoff_time == 300
    
    def test_rate_limiter_backoff(self):
        """Test exponential backoff"""
        limiter = RateLimiter()
        initial_backoff = limiter.backoff_time
        
        # Simulate rate limit error
        limiter.handle_rate_limit_error()
        assert limiter.backoff_time == initial_backoff * 2
        
        # Test reset
        limiter.reset_backoff()
        assert limiter.backoff_time == 1

class TestGeminiAnalyzer:
    """Test the Gemini analyzer functionality"""
    
    def test_analyzer_initialization_without_api_key(self):
        """Test analyzer initialization without API key"""
        with patch('models.llm_analyzer.validate_gemini_api_key', return_value=False):
            try:
                GeminiAnalyzer()
                assert False, "Expected ValueError to be raised"
            except ValueError as e:
                assert "Gemini API key is not configured" in str(e)
    
    @patch('models.llm_analyzer.validate_gemini_api_key', return_value=True)
    @patch('models.llm_analyzer.get_gemini_config')
    @patch('models.llm_analyzer.genai.configure')
    @patch('models.llm_analyzer.genai.GenerativeModel')
    def test_analyzer_initialization_with_api_key(self, mock_model, mock_configure, mock_config, mock_validate):
        """Test analyzer initialization with valid API key"""
        mock_config.return_value = {"api_key": "test_key"}
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        analyzer = GeminiAnalyzer()
        
        assert analyzer.model_name == "gemini-1.5-flash"
        assert analyzer.model == mock_model_instance
        assert isinstance(analyzer.rate_limiter, RateLimiter)
        mock_configure.assert_called_once_with(api_key="test_key")
    
    @patch('models.llm_analyzer.validate_gemini_api_key', return_value=True)
    @patch('models.llm_analyzer.get_gemini_config')
    @patch('models.llm_analyzer.genai.configure')
    @patch('models.llm_analyzer.genai.GenerativeModel')
    def test_make_request_with_retry_success(self, mock_model, mock_configure, mock_config, mock_validate):
        """Test successful request with retry"""
        mock_config.return_value = {"api_key": "test_key"}
        
        # Mock response
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_candidate = Mock()
        mock_candidate.finish_reason.name = "STOP"
        mock_response.candidates = [mock_candidate]
        
        mock_model_instance = Mock()
        mock_model_instance.generate_content.return_value = mock_response
        mock_model.return_value = mock_model_instance
        
        analyzer = GeminiAnalyzer()
        
        # Mock rate limiter to avoid actual delays
        analyzer.rate_limiter.wait_if_needed = Mock()
        
        result = analyzer._make_request_with_retry("test prompt")
        assert result == "Test response"
    
    @patch('models.llm_analyzer.validate_gemini_api_key', return_value=True)
    @patch('models.llm_analyzer.get_gemini_config')
    @patch('models.llm_analyzer.genai.configure')
    @patch('models.llm_analyzer.genai.GenerativeModel')
    def test_analyze_macro_conditions(self, mock_model, mock_configure, mock_config, mock_validate):
        """Test macro conditions analysis"""
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
        
        assert isinstance(result, AnalysisResult)
        assert result.analysis_type == AnalysisType.MACRO_ANALYSIS
        assert result.content == "Macro analysis result"
        assert result.confidence == 0.85
        assert isinstance(result.timestamp, datetime)
        assert result.metadata["economic_data"] == economic_data
    
    @patch('models.llm_analyzer.validate_gemini_api_key', return_value=True)
    @patch('models.llm_analyzer.get_gemini_config')
    @patch('models.llm_analyzer.genai.configure')
    @patch('models.llm_analyzer.genai.GenerativeModel')
    def test_analyze_sentiment(self, mock_model, mock_configure, mock_config, mock_validate):
        """Test sentiment analysis"""
        mock_config.return_value = {"api_key": "test_key"}
        
        # Mock response
        mock_response = Mock()
        mock_response.text = "Sentiment analysis result"
        mock_candidate = Mock()
        mock_candidate.finish_reason.name = "STOP"
        mock_response.candidates = [mock_candidate]
        
        mock_model_instance = Mock()
        mock_model_instance.generate_content.return_value = mock_response
        mock_model.return_value = mock_model_instance
        
        analyzer = GeminiAnalyzer()
        analyzer.rate_limiter.wait_if_needed = Mock()
        
        news_data = [
            {
                "headline": "Test headline",
                "source": "Test source",
                "sentiment": "positive"
            }
        ]
        
        result = analyzer.analyze_sentiment(news_data)
        
        assert isinstance(result, AnalysisResult)
        assert result.analysis_type == AnalysisType.SENTIMENT_ANALYSIS
        assert result.content == "Sentiment analysis result"
        assert result.confidence == 0.80
        assert result.metadata["news_count"] == 1
    
    @patch('models.llm_analyzer.validate_gemini_api_key', return_value=True)
    @patch('models.llm_analyzer.get_gemini_config')
    @patch('models.llm_analyzer.genai.configure')
    @patch('models.llm_analyzer.genai.GenerativeModel')
    def test_error_handling(self, mock_model, mock_configure, mock_config, mock_validate):
        """Test error handling in analysis"""
        mock_config.return_value = {"api_key": "test_key"}
        
        # Mock exception
        mock_model_instance = Mock()
        mock_model_instance.generate_content.side_effect = Exception("Test error")
        mock_model.return_value = mock_model_instance
        
        analyzer = GeminiAnalyzer()
        analyzer.rate_limiter.wait_if_needed = Mock()
        
        result = analyzer.analyze_macro_conditions({"test": "data"})
        
        assert isinstance(result, AnalysisResult)
        assert result.analysis_type == AnalysisType.MACRO_ANALYSIS
        assert "Error analyzing macro conditions" in result.content
        assert result.confidence == 0.0
        assert "error" in result.metadata
    
    @patch('models.llm_analyzer.validate_gemini_api_key', return_value=True)
    @patch('models.llm_analyzer.get_gemini_config')
    @patch('models.llm_analyzer.genai.configure')
    @patch('models.llm_analyzer.genai.GenerativeModel')
    def test_usage_stats(self, mock_model, mock_configure, mock_config, mock_validate):
        """Test usage statistics"""
        mock_config.return_value = {"api_key": "test_key"}
        mock_model.return_value = Mock()
        
        analyzer = GeminiAnalyzer()
        stats = analyzer.get_usage_stats()
        
        assert "requests_this_minute" in stats
        assert "requests_this_day" in stats
        assert "minute_limit" in stats
        assert "day_limit" in stats
        assert "current_backoff_time" in stats
        assert "model_name" in stats
        assert stats["model_name"] == "gemini-1.5-flash"

class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @patch('models.llm_analyzer.GeminiAnalyzer')
    def test_create_gemini_analyzer(self, mock_analyzer_class):
        """Test analyzer creation function"""
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        
        result = create_gemini_analyzer("test-model")
        
        mock_analyzer_class.assert_called_once_with(model_name="test-model")
        assert result == mock_analyzer
    
    @patch('models.llm_analyzer.create_gemini_analyzer')
    def test_quick_analysis(self, mock_create_analyzer):
        """Test quick analysis function"""
        mock_analyzer = Mock()
        mock_analyzer._make_request_with_retry.return_value = "Quick analysis result"
        mock_create_analyzer.return_value = mock_analyzer
        
        result = quick_analysis("test prompt", AnalysisType.MARKET_SUMMARY)
        
        assert isinstance(result, AnalysisResult)
        assert result.analysis_type == AnalysisType.MARKET_SUMMARY
        assert result.content == "Quick analysis result"
        assert result.confidence == 0.70

def test_api_key_validation():
    """Test API key validation"""
    # This will test the actual validation function
    # The result depends on whether the API key is actually set
    result = validate_gemini_api_key()
    assert isinstance(result, bool)

if __name__ == "__main__":
    # Run a simple integration test
    print("=== Gemini API Integration Test ===\n")
    
    if validate_gemini_api_key():
        print("✅ API key validation passed")
        
        try:
            # Test analyzer creation
            analyzer = create_gemini_analyzer()
            print("✅ Analyzer creation successful")
            
            # Test usage stats
            stats = analyzer.get_usage_stats()
            print(f"✅ Usage stats retrieved: {stats}")
            
            print("\n✅ All basic tests passed!")
            print("\nTo run full test suite, use: python -m pytest tests/test_gemini_integration.py")
            
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
    else:
        print("❌ API key not configured")
        print("Please set GEMINI_API_KEY in your .env file") 