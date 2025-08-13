#!/usr/bin/env python3
"""
Tests for Data Validation Module
===============================

Comprehensive test suite to ensure data validation catches synthetic data
and validates authentic Alpaca API data correctly.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path
from typing import Dict

# Import our validation modules
from utils.data_validation import (
    SyntheticDataDetector, AlpacaDataValidator, 
    validate_training_data, clean_synthetic_data
)

class TestSyntheticDataDetector:
    """Test the synthetic data detection functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.detector = SyntheticDataDetector()
    
    def create_realistic_market_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Create realistic market data for testing."""
        np.random.seed(42)  # For reproducible tests
        
        # Generate realistic timestamps (weekdays only)
        start_date = datetime(2023, 1, 1)
        dates = pd.bdate_range(start_date, periods=n_samples)
        
        # Generate realistic price data with random walk + volatility
        initial_price = 100.0
        returns = np.random.normal(0.0001, 0.02, n_samples)  # Realistic daily returns
        prices = [initial_price]
        for r in returns[1:]:
            prices.append(prices[-1] * (1 + r))
        
        # Generate realistic volume data
        base_volume = 1000000
        volume = np.random.lognormal(np.log(base_volume), 0.5, n_samples)
        
        # Generate OHLC data
        close_prices = np.array(prices)
        daily_range = np.random.uniform(0.005, 0.03, n_samples)  # 0.5% to 3% daily range
        high_prices = close_prices * (1 + daily_range/2)
        low_prices = close_prices * (1 - daily_range/2)
        open_prices = np.roll(close_prices, 1)  # Previous close as open
        open_prices[0] = close_prices[0]
        
        df = pd.DataFrame({
            'timestamp': dates,
            'ticker': ['AAPL'] * n_samples,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume,
            'price_change_1d': np.concatenate([[0], np.diff(close_prices) / close_prices[:-1]]),
            'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # 30% positive rate
        })
        
        return df
    
    def create_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Create obviously synthetic data for testing detection."""
        # Create perfect synthetic patterns
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=n_samples),
            'ticker': ['TESTDATA'] * n_samples,
            'close': [100.0] * n_samples,  # Identical prices
            'volume': [1000] * n_samples,  # Identical volume
            'price_change_1d': [0.005958291956305928] * n_samples,  # Identical tiny changes
            'target': [1] * n_samples  # All positive targets
        })
        
        return df
    
    def test_realistic_data_passes_validation(self):
        """Test that realistic market data passes validation."""
        df = self.create_realistic_market_data()
        results = self.detector.detect_synthetic_patterns(df)
        
        assert not results['is_synthetic'], f"Realistic data flagged as synthetic: {results['issues']}"
        assert results['synthetic_score'] < 0.3, f"High synthetic score for realistic data: {results['synthetic_score']}"
        assert len(results['issues']) <= 2, f"Too many issues with realistic data: {results['issues']}"
    
    def test_synthetic_data_detected(self):
        """Test that synthetic data is properly detected."""
        df = self.create_synthetic_data()
        results = self.detector.detect_synthetic_patterns(df)
        
        assert results['is_synthetic'], "Synthetic data not detected"
        assert results['synthetic_score'] > 0.3, f"Low synthetic score for obvious synthetic data: {results['synthetic_score']}"
        assert len(results['issues']) >= 3, f"Not enough issues detected: {results['issues']}"
    
    def test_price_realism_detection(self):
        """Test price realism detection specifically."""
        # Create data with unrealistic price patterns
        df = self.create_realistic_market_data()
        df['price_change_1d'] = [0.000001] * len(df)  # Unrealistically tiny movements
        
        results = self.detector._check_price_realism(df)
        
        assert len(results['issues']) > 0, "Price realism issues not detected"
        assert any('tiny price movements' in issue for issue in results['issues'])
    
    def test_volume_pattern_detection(self):
        """Test volume pattern detection."""
        df = self.create_realistic_market_data()
        df['volume'] = [0] * len(df)  # All zero volume
        
        results = self.detector._check_volume_patterns(df)
        
        assert len(results['issues']) > 0, "Volume issues not detected"
        assert any('zero volume' in issue for issue in results['issues'])
    
    def test_timestamp_authenticity(self):
        """Test timestamp validation."""
        df = self.create_realistic_market_data()
        # Add weekend data (markets closed)
        weekend_dates = pd.date_range('2023-01-07', periods=10, freq='D')  # Includes weekends
        weekend_df = pd.DataFrame({
            'timestamp': weekend_dates,
            'close': [100] * 10,
            'volume': [1000] * 10
        })
        df = pd.concat([df, weekend_df], ignore_index=True)
        
        results = self.detector._check_timestamp_authenticity(df)
        
        assert len(results['issues']) > 0, "Weekend trading not detected"
        assert any('weekend' in issue.lower() for issue in results['issues'])
    
    def test_target_distribution_validation(self):
        """Test target distribution checks."""
        df = self.create_realistic_market_data()
        df['target'] = [1] * len(df)  # All positive targets (unrealistic)
        
        results = self.detector._check_target_distributions(df)
        
        assert len(results['issues']) > 0, "Unrealistic target distribution not detected"
        assert any('identical target' in issue for issue in results['issues'])
    
    def test_ticker_authenticity(self):
        """Test ticker symbol validation."""
        df = self.create_realistic_market_data()
        df['ticker'] = ['TESTDATA'] * len(df)  # Obviously synthetic ticker
        
        results = self.detector._check_ticker_authenticity(df)
        
        assert len(results['issues']) > 0, "Synthetic ticker not detected"
        assert any('Suspicious ticker' in issue for issue in results['issues'])

class TestAlpacaDataValidator:
    """Test the Alpaca-specific data validation."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.validator = AlpacaDataValidator()
    
    def create_alpaca_like_data(self) -> pd.DataFrame:
        """Create data that looks like authentic Alpaca API data."""
        np.random.seed(42)
        n_samples = 500
        
        # Realistic timestamps with timezone info
        dates = pd.bdate_range('2023-01-01', periods=n_samples, tz='America/New_York')
        
        # Realistic OHLCV data
        close_prices = 100 + np.cumsum(np.random.normal(0, 1, n_samples))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'ticker': ['AAPL'] * n_samples,
            'open': close_prices + np.random.normal(0, 0.1, n_samples),
            'high': close_prices + np.random.uniform(0, 2, n_samples),
            'low': close_prices - np.random.uniform(0, 2, n_samples),
            'close': close_prices,
            'volume': np.random.lognormal(15, 0.5, n_samples).astype(int),
            'target': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        })
        
        return df
    
    def test_valid_alpaca_data(self):
        """Test that valid Alpaca-like data passes validation."""
        df = self.create_alpaca_like_data()
        results = self.validator.validate_alpaca_data(df)
        
        assert results['is_valid'], f"Valid Alpaca data failed validation: {results['errors']}"
        assert results['confidence'] > 0.7, f"Low confidence for valid data: {results['confidence']}"
    
    def test_missing_required_columns(self):
        """Test detection of missing required Alpaca columns."""
        df = self.create_alpaca_like_data()
        df = df.drop('volume', axis=1)  # Remove required column
        
        results = self.validator.validate_alpaca_data(df)
        
        assert not results['is_valid'], "Should fail with missing required columns"
        assert any('Missing required' in error for error in results['errors'])
    
    def test_timezone_warning(self):
        """Test timezone warning for timestamps."""
        df = self.create_alpaca_like_data()
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)  # Remove timezone
        
        results = self.validator.validate_alpaca_data(df)
        
        # Should warn about missing timezone but still be valid
        assert any('timezone' in warning for warning in results['warnings'])

class TestDataValidationIntegration:
    """Integration tests for the complete validation system."""
    
    def test_validate_training_data_file(self):
        """Test file-based validation."""
        # Create temporary realistic data file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = self.create_test_data()
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            results = validate_training_data(temp_file, strict_mode=False)
            
            assert 'is_valid' in results
            assert 'file_info' in results
            assert results['file_info']['records'] == len(df)
            
        finally:
            os.unlink(temp_file)
    
    def test_clean_synthetic_data_directory(self):
        """Test the data directory cleaning functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mixed data files
            good_data = self.create_test_data()
            bad_data = self.create_bad_test_data()
            
            good_file = temp_path / "good_data.csv"
            bad_file = temp_path / "bad_data.csv"
            
            good_data.to_csv(good_file, index=False)
            bad_data.to_csv(bad_file, index=False)
            
            # Run cleaning
            results = clean_synthetic_data(str(temp_path), str(temp_path / "backup"))
            
            assert results['files_scanned'] == 2
            assert results['files_quarantined'] >= 1  # Bad file should be quarantined
            
            # Check that good file still exists and bad file is moved
            assert good_file.exists()
            assert not bad_file.exists()  # Should be moved to backup
    
    def create_test_data(self) -> pd.DataFrame:
        """Create realistic test data."""
        np.random.seed(42)
        dates = pd.bdate_range('2023-01-01', periods=100)
        close_prices = 100 + np.cumsum(np.random.normal(0, 1, 100))
        
        return pd.DataFrame({
            'timestamp': dates,
            'ticker': ['AAPL'] * 100,
            'close': close_prices,
            'volume': np.random.lognormal(15, 0.3, 100).astype(int),
            'price_change_1d': np.concatenate([[0], np.diff(close_prices) / close_prices[:-1]]),
            'target': np.random.choice([0, 1], 100, p=[0.75, 0.25])
        })
    
    def create_bad_test_data(self) -> pd.DataFrame:
        """Create obviously synthetic test data."""
        return pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100),
            'ticker': ['SYNTHETIC'] * 100,
            'close': [100.0] * 100,  # Identical prices
            'volume': [1000] * 100,  # Identical volume
            'price_change_1d': [0.005958291956305928] * 100,  # Identical changes
            'target': [1] * 100  # All positive
        })

class TestRealWorldScenarios:
    """Test real-world scenarios and edge cases."""
    
    def test_eurku_batch1_pattern_detection(self):
        """Test detection of the specific EURKU batch 1 pattern we found."""
        # Recreate the problematic EURKU pattern
        df = pd.DataFrame({
            'ticker': ['EURKU'] * 1000,
            'target': [1] * 1000,  # All positive targets
            'timestamp': pd.date_range('2024-07-30', periods=1000),
            'price_change_1d': [0.005958291956305928] * 500 + [0.0] * 500,  # Identical tiny changes
            'close': [10.07] * 1000,  # Nearly identical prices
            'volume': [1682.14] * 1000,  # Identical volume
        })
        
        detector = SyntheticDataDetector()
        results = detector.detect_synthetic_patterns(df)
        
        assert results['is_synthetic'], "EURKU pattern not detected as synthetic"
        assert results['synthetic_score'] > 0.5, f"EURKU pattern score too low: {results['synthetic_score']}"
        
        # Should detect multiple specific issues
        issues_text = ' '.join(results['issues'])
        assert 'identical target' in issues_text.lower()
        assert 'tiny price movements' in issues_text.lower()
    
    def test_mixed_ticker_validation(self):
        """Test validation with multiple tickers in same dataset."""
        realistic_data = pd.DataFrame({
            'ticker': ['AAPL'] * 500 + ['MSFT'] * 500,
            'timestamp': pd.bdate_range('2023-01-01', periods=1000),
            'close': np.random.normal(150, 10, 1000),
            'volume': np.random.lognormal(15, 0.5, 1000),
            'target': np.random.choice([0, 1], 1000, p=[0.8, 0.2])
        })
        
        validator = AlpacaDataValidator()
        results = validator.validate_alpaca_data(realistic_data)
        
        assert results['is_valid'], "Multi-ticker realistic data should be valid"
    
    def test_extreme_imbalance_but_realistic(self):
        """Test that extreme but realistic imbalance doesn't trigger false positive."""
        # Create realistic data with natural extreme imbalance (rare events)
        df = pd.DataFrame({
            'ticker': ['SPY'] * 2000,
            'timestamp': pd.bdate_range('2018-01-01', periods=2000),
            'close': 300 + np.cumsum(np.random.normal(0.05, 2, 2000)),  # Realistic price evolution
            'volume': np.random.lognormal(16, 0.4, 2000),  # Realistic volume variation
            'target': [0] * 1980 + [1] * 20  # 1% positive rate (realistic for rare events)
        })
        
        # Add realistic price changes
        df['price_change_1d'] = df['close'].pct_change().fillna(0)
        
        detector = SyntheticDataDetector()
        results = detector.detect_synthetic_patterns(df)
        
        # Should NOT be flagged as synthetic despite extreme imbalance
        assert not results['is_synthetic'], "Realistic extreme imbalance incorrectly flagged"

def test_batch_1_eurku_detection():
    """Specific test for the EURKU batch 1 synthetic data we discovered."""
    # This should catch the exact pattern from batch 1
    eurku_data = pd.DataFrame({
        'ticker': ['EURKU'] * 4,
        'target': [1, 1, 1, 1],  # All positive
        'timestamp': [
            '2024-07-30 00:00:00-04:00',
            '2024-07-31 00:00:00-04:00', 
            '2024-08-01 00:00:00-04:00',
            '2024-08-02 00:00:00-04:00'
        ],
        'price_change_1d': [0.0, 0.005958291956305928, 0.0, 0.0],
        'close': [10.07, 10.082, 10.094, 10.106],
        'volume': [1682.14, 46.9, 46.9, 46.9]
    })
    
    results = validate_training_data_from_dataframe(eurku_data)
    
    assert not results['is_valid'], "EURKU synthetic pattern should be detected"
    assert any('Suspicious ticker' in error for error in results['errors'])
    assert any('identical target' in ' '.join(results['validation_details']['synthetic_detection']['issues']))

def validate_training_data_from_dataframe(df: pd.DataFrame) -> Dict:
    """Helper function to validate DataFrame directly (for testing)."""
    validator = AlpacaDataValidator()
    return validator.validate_alpaca_data(df)

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
