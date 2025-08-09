#!/usr/bin/env python3
"""
Data Validation Module for Investment Committee
==============================================

This module ensures that only authentic Alpaca API data is used for training,
preventing synthetic or test data contamination that can lead to overfitting.

Key Functions:
- Detect synthetic/artificial data patterns
- Validate Alpaca API data authenticity
- Remove contaminated datasets
- Ensure data quality and realism
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import json
import warnings

logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass

class SyntheticDataDetector:
    """
    Detects synthetic or artificial data patterns that indicate 
    non-market data contamination.
    """
    
    def __init__(self):
        self.validation_rules = {
            'price_realism': True,
            'volume_realism': True,
            'timestamp_continuity': True,
            'feature_distribution': True,
            'target_distribution': True,
            'ticker_authenticity': True,
        }
    
    def detect_synthetic_patterns(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Comprehensive synthetic data detection.
        
        Returns:
            Dict with validation results and detected issues
        """
        results = {
            'is_synthetic': False,
            'synthetic_score': 0.0,
            'issues': [],
            'warnings': [],
            'details': {}
        }
        
        # Run all validation checks
        results.update(self._check_price_realism(df))
        results.update(self._check_volume_patterns(df))
        results.update(self._check_timestamp_authenticity(df))
        results.update(self._check_feature_distributions(df))
        results.update(self._check_target_distributions(df))
        results.update(self._check_ticker_authenticity(df))
        results.update(self._check_unrealistic_patterns(df))
        
        # Calculate overall synthetic score
        synthetic_score = len(results['issues']) / 10.0  # Normalize to 0-1
        results['synthetic_score'] = min(synthetic_score, 1.0)
        results['is_synthetic'] = synthetic_score > 0.3  # Threshold for flagging
        
        return results
    
    def _check_price_realism(self, df: pd.DataFrame) -> Dict[str, any]:
        """Check if price movements are realistic for market data."""
        issues = []
        details = {}
        
        if 'price_change_1d' in df.columns:
            price_changes = df['price_change_1d'].dropna()
            
            # Check for unrealistically small movements
            tiny_moves = (abs(price_changes) < 1e-6).sum()
            if tiny_moves > len(price_changes) * 0.1:  # >10% tiny moves
                issues.append(f"Excessive tiny price movements: {tiny_moves}/{len(price_changes)}")
            
            # Check for identical repeated values
            if len(price_changes.unique()) < len(price_changes) * 0.1:
                issues.append("Too many identical price changes (synthetic pattern)")
            
            # Check for unrealistic precision (too many decimal places)
            precision_check = price_changes.apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
            if precision_check.max() > 10:
                issues.append("Unrealistic price precision (>10 decimal places)")
            
            details['price_change_stats'] = {
                'mean': float(price_changes.mean()),
                'std': float(price_changes.std()),
                'unique_values': len(price_changes.unique()),
                'tiny_moves_pct': tiny_moves / len(price_changes) * 100
            }
        
        return {'issues': issues, 'details': {'price_realism': details}}
    
    def _check_volume_patterns(self, df: pd.DataFrame) -> Dict[str, any]:
        """Check volume patterns for realism."""
        issues = []
        details = {}
        
        if 'volume' in df.columns:
            volume = df['volume'].dropna()
            
            # Check for zero volume (unrealistic for liquid stocks)
            zero_volume = (volume == 0).sum()
            if zero_volume > len(volume) * 0.05:  # >5% zero volume
                issues.append(f"Excessive zero volume days: {zero_volume}/{len(volume)}")
            
            # Check for identical volume values (synthetic pattern)
            if len(volume.unique()) < len(volume) * 0.2:
                issues.append("Too many identical volume values")
            
            # Check for unrealistic volume spikes
            volume_ratio = volume / volume.rolling(20).mean()
            extreme_spikes = (volume_ratio > 100).sum()
            if extreme_spikes > len(volume) * 0.01:
                issues.append(f"Unrealistic volume spikes: {extreme_spikes}")
            
            details['volume_stats'] = {
                'mean': float(volume.mean()),
                'std': float(volume.std()),
                'zero_volume_pct': zero_volume / len(volume) * 100,
                'unique_values': len(volume.unique())
            }
        
        return {'issues': issues, 'details': {'volume_patterns': details}}
    
    def _check_timestamp_authenticity(self, df: pd.DataFrame) -> Dict[str, any]:
        """Check timestamp patterns for market authenticity."""
        issues = []
        details = {}
        
        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'])
            
            # Check for weekend data (markets are closed)
            weekends = timestamps.dt.weekday >= 5
            weekend_count = weekends.sum()
            if weekend_count > 0:
                issues.append(f"Market data on weekends: {weekend_count} records")
            
            # Check for future dates
            now = datetime.now()
            if timestamps.dt.tz is not None:
                import pytz
                # Make datetime.now() timezone-aware to match the data
                now = now.replace(tzinfo=pytz.UTC)
                if str(timestamps.dt.tz) != 'UTC':
                    # Convert to the same timezone as the data
                    now = now.astimezone(timestamps.dt.tz)
            future_dates = timestamps > now
            if future_dates.sum() > 0:
                issues.append(f"Future timestamps found: {future_dates.sum()}")
            
            # Check for regular time intervals (too perfect = synthetic)
            time_diffs = timestamps.diff().dt.days.dropna()
            if len(time_diffs.unique()) == 1 and time_diffs.iloc[0] == 1:
                # Perfectly spaced daily data might be synthetic
                if len(time_diffs) > 100:  # Only flag if substantial dataset
                    issues.append("Perfectly regular daily intervals (possible synthetic data)")
            
            details['timestamp_stats'] = {
                'date_range': f"{timestamps.min()} to {timestamps.max()}",
                'weekend_count': int(weekend_count),
                'future_count': int(future_dates.sum()),
                'total_records': len(timestamps)
            }
        
        return {'issues': issues, 'details': {'timestamp_authenticity': details}}
    
    def _check_feature_distributions(self, df: pd.DataFrame) -> Dict[str, any]:
        """Check feature distributions for realistic patterns."""
        issues = []
        details = {}
        
        # Check for features with too many identical values
        suspicious_features = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in ['timestamp', 'target']:
                continue
            
            unique_ratio = len(df[col].dropna().unique()) / len(df[col].dropna())
            if unique_ratio < 0.1:  # <10% unique values
                suspicious_features.append(col)
        
        if len(suspicious_features) > 5:
            issues.append(f"Too many features with low variance: {len(suspicious_features)}")
        
        # Check for perfect correlations (synthetic indicator)
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr().abs()
            perfect_corrs = (corr_matrix == 1.0).sum().sum() - len(corr_matrix)  # Exclude diagonal
            if perfect_corrs > 2:  # Allow some perfect correlations
                issues.append(f"Too many perfect correlations: {perfect_corrs}")
        
        details['feature_stats'] = {
            'suspicious_features': suspicious_features,
            'perfect_correlations': int(perfect_corrs) if 'perfect_corrs' in locals() else 0
        }
        
        return {'issues': issues, 'details': {'feature_distributions': details}}
    
    def _check_target_distributions(self, df: pd.DataFrame) -> Dict[str, any]:
        """Check target variable for unrealistic patterns."""
        issues = []
        details = {}
        
        if 'target' in df.columns:
            target = df['target'].dropna()
            
            # Check for extreme class imbalance that suggests synthetic data
            if len(target.unique()) == 2:  # Binary classification
                value_counts = target.value_counts()
                minority_ratio = value_counts.min() / len(target)
                
                if minority_ratio < 0.001:  # <0.1% minority class
                    issues.append(f"Extreme class imbalance: {minority_ratio:.4f}")
                
                # Check if all samples have same target (perfect separation)
                if len(value_counts) == 1:
                    issues.append("All samples have identical target (perfect synthetic data)")
            
            details['target_stats'] = {
                'unique_values': len(target.unique()),
                'value_counts': target.value_counts().to_dict(),
                'class_distribution': target.value_counts(normalize=True).to_dict()
            }
        
        return {'issues': issues, 'details': {'target_distributions': details}}
    
    def _check_ticker_authenticity(self, df: pd.DataFrame) -> Dict[str, any]:
        """Check if ticker symbols are realistic."""
        issues = []
        details = {}
        
        if 'ticker' in df.columns:
            tickers = df['ticker'].unique()
            
            # Known synthetic/test tickers that should be rejected
            known_synthetic = ['EURKU', 'TEST', 'FAKE', 'SYNTH', 'DEMO', 'MOCK']
            synthetic_patterns = ['TEST', 'FAKE', 'SYNTH', 'DEMO', 'MOCK']
            suspicious_tickers = []
            
            for ticker in tickers:
                ticker_str = str(ticker).upper()
                
                # Check for known synthetic tickers
                if ticker_str in known_synthetic:
                    suspicious_tickers.append(ticker)
                    issues.append(f"Known synthetic ticker detected: {ticker}")
                
                # Check for pattern-based synthetic tickers
                elif any(pattern in ticker_str for pattern in synthetic_patterns):
                    suspicious_tickers.append(ticker)
                
                # Check for unrealistic ticker formats
                elif len(ticker_str) > 5 or not ticker_str.isalpha():
                    if not any(c in ticker_str for c in ['.', '-']):  # Allow some special chars
                        suspicious_tickers.append(ticker)
                
                # Per-ticker suspicious pattern check
                ticker_data = df[df['ticker'] == ticker]
                if self._check_ticker_specific_patterns(ticker_data, ticker):
                    suspicious_tickers.append(ticker)
                    issues.append(f"Ticker {ticker} shows synthetic data patterns")
            
            if suspicious_tickers:
                issues.append(f"Suspicious ticker symbols: {suspicious_tickers}")
            
            details['ticker_stats'] = {
                'total_tickers': len(tickers),
                'suspicious_tickers': suspicious_tickers,
                'sample_tickers': list(tickers[:10])
            }
        
        return {'issues': issues, 'details': {'ticker_authenticity': details}}
    
    def _check_ticker_specific_patterns(self, ticker_df: pd.DataFrame, ticker: str) -> bool:
        """Check individual ticker for synthetic patterns."""
        # Check target distribution - synthetic data often has >95% positive
        if 'target' in ticker_df.columns:
            target_mean = ticker_df['target'].mean()
            if target_mean > 0.95:  # >95% positive is highly suspicious
                return True
        
        # Check for identical tiny price movements (EURKU pattern)
        if 'daily_return' in ticker_df.columns:
            returns = ticker_df['daily_return'].dropna()
            if len(returns) > 10:
                # Check for excessive tiny identical movements
                tiny_moves = (abs(returns) < 0.001).sum()
                zero_moves = (returns == 0.0).sum()
                if (tiny_moves + zero_moves) > len(returns) * 0.7:  # >70% tiny/zero moves
                    return True
                
                # Check for repeated identical values
                unique_ratio = len(returns.unique()) / len(returns)
                if unique_ratio < 0.1:  # <10% unique values
                    return True
        
        return False
    
    def _check_unrealistic_patterns(self, df: pd.DataFrame) -> Dict[str, any]:
        """Check for other unrealistic patterns."""
        issues = []
        details = {}
        
        # Check for identical rows (copy-paste synthetic data)
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > len(df) * 0.1:  # >10% duplicates
            issues.append(f"Excessive duplicate rows: {duplicate_rows}/{len(df)}")
        
        # Check for sequential patterns in supposedly random data
        if 'target' in df.columns:
            target_runs = (df['target'] != df['target'].shift()).cumsum()
            max_run = target_runs.value_counts().max()
            if max_run > len(df) * 0.5:  # >50% same target in sequence
                issues.append("Suspicious sequential target patterns")
        
        details['pattern_stats'] = {
            'duplicate_rows': int(duplicate_rows),
            'duplicate_pct': duplicate_rows / len(df) * 100
        }
        
        return {'issues': issues, 'details': {'unrealistic_patterns': details}}

class AlpacaDataValidator:
    """
    Validates that data comes from authentic Alpaca API sources
    and meets quality standards for financial market data.
    """
    
    def __init__(self):
        self.detector = SyntheticDataDetector()
        
    def validate_alpaca_data(self, df: pd.DataFrame, source_info: Optional[Dict] = None) -> Dict[str, any]:
        """
        Comprehensive validation of Alpaca data authenticity.
        
        Args:
            df: DataFrame to validate
            source_info: Optional metadata about data source
            
        Returns:
            Validation results with pass/fail status and details
        """
        results = {
            'is_valid': True,
            'confidence': 1.0,
            'errors': [],
            'warnings': [],
            'validation_details': {}
        }
        
        # Run synthetic data detection
        synthetic_results = self.detector.detect_synthetic_patterns(df)
        results['validation_details']['synthetic_detection'] = synthetic_results
        
        if synthetic_results['is_synthetic']:
            results['is_valid'] = False
            results['errors'].append("Synthetic data patterns detected")
            results['confidence'] *= 0.1
        
        # Additional Alpaca-specific validations
        alpaca_results = self._validate_alpaca_specific(df)
        results['validation_details']['alpaca_specific'] = alpaca_results
        
        for error in alpaca_results['errors']:
            results['errors'].append(error)
            results['is_valid'] = False
            results['confidence'] *= 0.5
        
        for warning in alpaca_results['warnings']:
            results['warnings'].append(warning)
            results['confidence'] *= 0.9
        
        # Market data quality checks
        quality_results = self._validate_market_quality(df)
        results['validation_details']['market_quality'] = quality_results
        
        for error in quality_results['errors']:
            results['errors'].append(error)
            results['is_valid'] = False
        
        return results
    
    def _validate_alpaca_specific(self, df: pd.DataFrame) -> Dict[str, any]:
        """Alpaca API specific validation checks."""
        errors = []
        warnings = []
        
        # Check for required Alpaca data columns or their feature-engineered equivalents
        required_cols = ['timestamp']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        # For close and volume, check if we have the raw columns OR feature-engineered variants
        has_price_data = any(col in df.columns for col in ['close', 'close_price', 'open', 'high', 'low'] + 
                            [c for c in df.columns if any(price_term in c.lower() for price_term in ['price', 'ohlc', 'close', 'open'])])
        has_volume_data = any(col in df.columns for col in ['volume', 'volume_sma', 'volume_ratio'] + 
                             [c for c in df.columns if 'volume' in c.lower()])
        
        if not has_price_data:
            errors.append("Missing price data (no close, open, high, low, or price-related columns found)")
        if not has_volume_data:
            errors.append("Missing volume data (no volume or volume-related columns found)")
        
        if missing_cols:
            errors.append(f"Missing required Alpaca columns: {missing_cols}")
        
        # Check timestamp format matches Alpaca API
        if 'timestamp' in df.columns:
            try:
                timestamps = pd.to_datetime(df['timestamp'])
                # Alpaca typically provides timezone-aware timestamps
                if timestamps.dt.tz is None:
                    warnings.append("Timestamps missing timezone info (Alpaca typically includes tz)")
            except Exception:
                errors.append("Invalid timestamp format")
        
        # Check for volume characteristics if we have volume data
        volume_cols = [col for col in df.columns if 'volume' in col.lower()]
        if volume_cols:
            # Use the first volume-related column for validation
            vol_col = volume_cols[0]
            if vol_col in df.columns:
                zero_volume_pct = (df[vol_col] == 0).mean() * 100
                if zero_volume_pct > 20:  # Alpaca rarely has >20% zero volume days
                    warnings.append(f"High zero volume percentage in {vol_col}: {zero_volume_pct:.1f}%")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'checks_performed': ['required_columns', 'timestamp_format', 'volume_patterns']
        }
    
    def _validate_market_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """Validate overall market data quality."""
        errors = []
        warnings = []
        
        # Check for reasonable data size
        if len(df) < 50:
            warnings.append(f"Small dataset: {len(df)} records")
        elif len(df) > 10000:
            warnings.append(f"Very large dataset: {len(df)} records")
        
        # Check for reasonable date range
        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'])
            date_range = (timestamps.max() - timestamps.min()).days
            
            if date_range < 30:
                warnings.append(f"Short time range: {date_range} days")
            elif date_range > 3650:  # >10 years
                warnings.append(f"Very long time range: {date_range} days")
        
        # Check for missing data patterns
        missing_pct = df.isnull().mean() * 100
        high_missing = missing_pct[missing_pct > 50]
        if len(high_missing) > 0:
            warnings.append(f"High missing data in columns: {high_missing.to_dict()}")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'data_quality_score': max(0, 1.0 - len(warnings) * 0.1 - len(errors) * 0.5)
        }

def validate_training_data(data_file: str, strict_mode: bool = True) -> Dict[str, any]:
    """
    Main validation function for training data files.
    
    Args:
        data_file: Path to CSV data file
        strict_mode: Whether to use strict validation (fail on warnings)
        
    Returns:
        Validation results with recommendations
    """
    try:
        df = pd.read_csv(data_file)
        logger.info(f"Validating data file: {data_file} ({len(df)} records)")
        
        validator = AlpacaDataValidator()
        results = validator.validate_alpaca_data(df)
        
        # Add file-specific information
        results['file_info'] = {
            'path': data_file,
            'size_mb': Path(data_file).stat().st_size / (1024 * 1024),
            'records': len(df),
            'columns': list(df.columns),
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Determine final validation status
        if strict_mode and results['warnings']:
            results['is_valid'] = False
            results['errors'].extend([f"STRICT MODE: {w}" for w in results['warnings']])
        
        # Generate recommendations
        recommendations = []
        if not results['is_valid']:
            recommendations.append("❌ DO NOT USE this data for training")
            recommendations.append("🔄 Re-collect data from Alpaca API")
            recommendations.append("🧹 Clean data pipeline to remove synthetic sources")
        else:
            if results['confidence'] < 0.8:
                recommendations.append("⚠️  Use with caution - low confidence score")
            else:
                recommendations.append("✅ Data appears authentic and suitable for training")
        
        results['recommendations'] = recommendations
        
        # Log summary
        status = "✅ VALID" if results['is_valid'] else "❌ INVALID"
        logger.info(f"Validation result: {status} (confidence: {results['confidence']:.2f})")
        
        return results
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {
            'is_valid': False,
            'confidence': 0.0,
            'errors': [f"Validation error: {str(e)}"],
            'warnings': [],
            'recommendations': ["❌ File could not be validated", "🔄 Check file format and content"]
        }

def clean_synthetic_data(data_dir: str = "data", backup_dir: str = "data/backup") -> Dict[str, any]:
    """
    Scan data directory and remove/quarantine synthetic data files.
    
    Args:
        data_dir: Directory containing data files
        backup_dir: Directory to move suspicious files
        
    Returns:
        Summary of cleaning operation
    """
    data_path = Path(data_dir)
    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'files_scanned': 0,
        'files_valid': 0,
        'files_quarantined': 0,
        'files_removed': 0,
        'validation_results': {}
    }
    
    logger.info(f"🧹 Cleaning synthetic data from: {data_dir}")
    
    # Scan all CSV files in data directory
    for csv_file in data_path.glob("*.csv"):
        results['files_scanned'] += 1
        
        # Validate file
        validation = validate_training_data(str(csv_file), strict_mode=True)
        results['validation_results'][csv_file.name] = validation
        
        if validation['is_valid']:
            results['files_valid'] += 1
            logger.info(f"✅ {csv_file.name}: Valid Alpaca data")
        else:
            # Quarantine suspicious files
            backup_file = backup_path / csv_file.name
            csv_file.rename(backup_file)
            results['files_quarantined'] += 1
            
            logger.warning(f"🚨 {csv_file.name}: Quarantined to {backup_file}")
            logger.warning(f"   Issues: {validation['errors']}")
    
    # Generate summary report
    summary_file = backup_path / "cleaning_report.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"🧹 Cleaning complete:")
    logger.info(f"   Scanned: {results['files_scanned']} files")
    logger.info(f"   Valid: {results['files_valid']} files")
    logger.info(f"   Quarantined: {results['files_quarantined']} files")
    logger.info(f"   Report: {summary_file}")
    
    return results

if __name__ == "__main__":
    # Command-line interface for data validation
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate training data authenticity')
    parser.add_argument('--file', type=str, help='Validate specific data file')
    parser.add_argument('--clean', action='store_true', help='Clean data directory')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory path')
    parser.add_argument('--strict', action='store_true', help='Use strict validation mode')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.clean:
        results = clean_synthetic_data(args.data_dir)
        print(f"Cleaning results: {results}")
    elif args.file:
        results = validate_training_data(args.file, strict_mode=args.strict)
        print(f"Validation results: {results}")
        
        if not results['is_valid']:
            exit(1)
    else:
        print("Use --file to validate a specific file or --clean to clean data directory")
