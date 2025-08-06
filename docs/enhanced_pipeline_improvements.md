# Enhanced ML Pipeline - Comprehensive Improvements

This document describes the comprehensive enhancements made to the investment committee ML pipeline to address extreme class imbalance and improve model performance.

## 🎯 Three Key Improvements Implemented

### 1. Enhanced Target Labeling
**Problem**: Extreme class imbalance (99.8% negative samples) makes it difficult for models to learn meaningful patterns.

**Solutions Implemented**:

#### A. Top-Percentile Strategy (`top_percentile`)
- Labels top 10% of returns as positive (increases positive rate to ~10%)
- Uses 90th percentile threshold for binary classification
- Provides better signal-to-noise ratio for learning

#### B. Multi-Class Strategy (`multi_class`) 
- Creates 3 classes: Strong Loss, Neutral, Strong Gain
- Bottom 20% → Strong Loss (0)
- Middle 60% → Neutral (1) 
- Top 20% → Strong Gain (2)
- Allows models to learn gradual patterns

#### C. Quantile Buckets Strategy (`quantile_buckets`)
- Divides returns into 5 quintiles (0-4)
- More granular classification for regression-like patterns
- Better for models that can handle ordinal relationships

### 2. Advanced Threshold Optimization
**Problem**: Simple 0.5 thresholds are suboptimal for imbalanced data and investment objectives.

**Solutions Implemented**:

#### A. F1-Score Optimization
```python
def find_optimal_threshold_advanced(y_true, y_pred_proba, optimization_metric='f1')
```
- Searches threshold space to maximize F1 score
- Balances precision and recall optimally
- Handles class imbalance better than accuracy

#### B. Top-K% Selection
- Optimizes for selecting top K% of predictions
- Directly addresses investment goal of finding best opportunities
- Portfolio-aware ranking approach

#### C. Portfolio-Aware Optimization
```python
def portfolio_aware_threshold_optimization(y_pred_proba, y_returns, target_positions=20)
```
- Optimizes thresholds for portfolio construction
- Considers position limits and risk tolerance
- Maximizes expected portfolio returns

#### D. Robustness Evaluation
```python
def evaluate_threshold_robustness(y_true, y_pred_proba, base_threshold=0.5)
```
- Tests threshold stability across different data splits
- Prevents overfitting to specific validation sets
- Ensures reliable performance

### 3. Enriched Feature Engineering
**Problem**: Limited feature set provides insufficient information for complex market patterns.

**Solutions Implemented**:

#### A. Greeks-Inspired Features (20 new categories)
1. **Theta Features** (Time Decay):
   - `theta_1d`, `theta_5d`, `theta_20d`: Price acceleration patterns
   - Captures momentum changes and trend exhaustion

2. **Vega Features** (Volatility Sensitivity):
   - `vega_5d`, `vega_10d`, `vega_20d`: Volatility regime changes
   - Identifies periods of increasing/decreasing uncertainty

3. **Delta Features** (Price Sensitivity):
   - `delta_momentum`, `delta_trend`: Directional momentum
   - Measures price sensitivity to market moves

4. **Gamma Features** (Acceleration):
   - `gamma_5d`, `gamma_10d`: Second-order price changes
   - Captures acceleration in momentum

#### B. Market Regime Detection Features
5. **Volatility Regimes**:
   - `volatility_regime`: Low/Medium/High volatility periods
   - `volatility_percentile`: Current volatility vs historical

6. **Trend Regimes**:
   - `trend_strength`: Strength of current trend
   - `trend_consistency`: Consistency of trend direction

7. **Correlation Features**:
   - `correlation_spy_20d`: Correlation with market (SPY)
   - Identifies market-relative behavior

#### C. Advanced Technical Indicators
8. **Enhanced Bollinger Bands**:
   - `bb_squeeze`: Identifies volatility compression
   - `bb_expansion`: Identifies volatility expansion

9. **Advanced RSI**:
   - `rsi_divergence`: Price vs RSI divergence
   - `rsi_momentum`: Rate of RSI change

10. **Volume Analysis**:
    - `volume_trend`: Volume trend analysis
    - `price_volume_trend`: Price-volume relationship

#### D. Time-Based Features
11. **Cyclical Patterns**:
    - `day_of_week`, `week_of_month`: Cyclical effects
    - `month_of_year`: Seasonal patterns

12. **Market Structure**:
    - `overnight_gap`: Gap between sessions
    - `intraday_range`: Daily range patterns

#### E. Cross-Asset Features
13. **Sector Rotation**:
    - Features comparing performance across sectors
    - Relative strength indicators

14. **Market Breadth**:
    - Features measuring overall market participation
    - Advance/decline style indicators

#### F. Risk-Adjusted Features
15. **Sharpe-Style Ratios**:
    - Risk-adjusted return metrics
    - Volatility-normalized performance

16. **Drawdown Features**:
    - Maximum drawdown metrics
    - Recovery patterns

#### G. Sentiment Proxies
17. **Volatility Structure**:
    - VIX-style features for individual stocks
    - Fear/greed indicators

18. **Price Action Quality**:
    - Quality of price movements
    - Conviction indicators

#### H. Multi-Timeframe Features
19. **Cross-Timeframe Analysis**:
    - Short vs long-term signal alignment
    - Timeframe consistency metrics

20. **Regime-Aware Features**:
    - Features that adapt to current market regime
    - Context-sensitive indicators

## 🚀 Implementation Details

### Enhanced Data Collection
```python
# New function signature with enhanced options
def engineer_features_for_symbol(
    self, 
    symbol: str, 
    days: int = 730,  # Extended to 24-month lookback
    use_enhanced_targets: bool = True,
    target_strategy: str = 'top_percentile'
) -> Optional[pd.DataFrame]:
```

**Key Changes**:
- Extended lookback period to 730 days (24 months) for better feature calculation
- Multiple target strategy options
- Comprehensive feature set (60+ features vs previous 15)
- Enhanced data quality checks and logging

### Enhanced Training Pipeline
```python
# Enhanced preprocessing with multiple strategies
def prepare_training_data(
    self,
    df: pd.DataFrame,
    use_enhanced_preprocessing: bool = True,
    target_strategy: str = 'top_percentile'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
```

**Key Changes**:
- Advanced threshold optimization integration
- Enhanced SMOTE for regression targets
- Portfolio-aware ensemble methods
- Multi-strategy target handling

### Advanced Optimization Utils
```python
# New utility functions in utils/advanced_optimization.py
- find_optimal_threshold_advanced()
- enhanced_smote_for_regression()
- portfolio_aware_threshold_optimization()
- evaluate_threshold_robustness()
- top_k_percent_threshold()
- pr_auc_optimization()
```

## 📊 Expected Improvements

### Class Balance
- **Before**: 99.8% negative samples
- **After**: 10-20% positive samples (depending on strategy)
- **Impact**: Models can actually learn positive patterns

### Feature Richness
- **Before**: 15 basic technical features
- **After**: 60+ advanced features across 20 categories
- **Impact**: Much richer information for pattern detection

### Threshold Optimization
- **Before**: Fixed 0.5 threshold
- **After**: Optimized thresholds (F1, top-K%, portfolio-aware)
- **Impact**: Better precision/recall balance for investment goals

### Target Diversity
- **Before**: Single binary target
- **After**: Multiple target strategies and horizons
- **Impact**: Ensemble diversity and better signal capture

## 🧪 Testing and Validation

### Comprehensive Test Suite
The `test_enhanced_pipeline.py` script validates:

1. **Enhanced Data Collection**:
   - Tests all 3 target strategies
   - Validates feature engineering
   - Checks data quality

2. **Advanced Optimization**:
   - Tests threshold optimization functions
   - Validates SMOTE for regression
   - Checks portfolio-aware methods

3. **Enhanced Training Pipeline**:
   - Tests integration of all components
   - Validates preprocessing
   - Checks model training

### Running Tests
```bash
python test_enhanced_pipeline.py
```

## 🔧 Usage Examples

### Basic Enhanced Data Collection
```python
from data_collection_alpaca import EnhancedDataCollector

collector = EnhancedDataCollector()

# Collect data with top-percentile targets
training_data = collector.collect_training_data(
    batch_numbers=[1, 2],
    use_enhanced_targets=True,
    target_strategy='top_percentile',
    days=730
)
```

### Advanced Model Training
```python
from train_models import ModelTrainer

trainer = ModelTrainer()

# Train with enhanced features and optimization
results = trainer.train_committee_models(
    training_data,
    use_enhanced_features=True,
    use_advanced_optimization=True,
    target_strategy='top_percentile'
)
```

### Custom Threshold Optimization
```python
from utils.advanced_optimization import find_optimal_threshold_advanced

# Optimize for F1 score
best_threshold, best_f1 = find_optimal_threshold_advanced(
    y_true, y_pred_proba, 
    optimization_metric='f1'
)

# Portfolio-aware optimization
portfolio_threshold = portfolio_aware_threshold_optimization(
    y_pred_proba, y_returns, 
    target_positions=20,
    risk_tolerance=0.5
)
```

## 📈 Performance Monitoring

### Key Metrics to Track
1. **Positive Sample Rate**: Should be 10-20% instead of 0.2%
2. **F1 Score**: Should improve significantly with optimized thresholds
3. **Precision at K%**: Better top-K% selection
4. **Portfolio Returns**: Improved risk-adjusted returns

### Logging and Monitoring
- Enhanced logging throughout the pipeline
- Detailed statistics for each enhancement
- Performance tracking across different strategies
- Robustness evaluation results

## 🔄 Next Steps

1. **Run Comprehensive Tests**: Execute `test_enhanced_pipeline.py`
2. **Validate on Historical Data**: Test on multiple batches
3. **Compare Strategies**: Evaluate all 3 target strategies
4. **Optimize Hyperparameters**: Fine-tune threshold optimization
5. **Monitor Production Performance**: Track real-world results

---

*This enhanced pipeline addresses the fundamental issues with extreme class imbalance while providing much richer feature sets and optimized decision thresholds for better investment performance.*
