# Production Trading System

A sophisticated automated trading system that integrates optimized machine learning models with Gemini LLM analysis for enhanced trading decisions.

## 🚀 Key Features

### Machine Learning Integration
- **Optimized Production Models**: CatBoost, Random Forest, and SVM models with hyperparameter tuning
- **Ensemble Predictions**: Weighted combination of multiple models for robust signals
- **Calibrated Thresholds**: Optimized decision thresholds for each model
- **Meta Model Integration**: Higher-level model that combines ML predictions with LLM analysis

### LLM Analysis (Gemini AI)
- **Trade Decision Analysis**: Comprehensive analysis of individual trading opportunities
- **Market Context Integration**: Incorporates macro conditions and sentiment
- **Risk Assessment**: AI-powered risk factor identification
- **Consensus Scoring**: Combines ML and LLM recommendations

### Production Trading Features
- **Alpaca Integration**: Paper and live trading through Alpaca API
- **Risk Management**: Position sizing, stop losses, portfolio limits
- **Real-time Data**: Technical indicators and market data processing
- **Performance Tracking**: Comprehensive trade and portfolio monitoring

## 📁 System Architecture

```
production_trading_system/
├── config/
│   ├── production_config.py         # Main configuration system
│   └── optimized_thresholds_batch_1.json  # Optimized model thresholds
├── models/
│   ├── production/
│   │   ├── optimized_catboost.pkl   # Primary production model
│   │   ├── random_forest.pkl        # Backup model 1
│   │   └── svm.pkl                  # Backup model 2
│   └── enhanced_llm_analyzer.py     # Gemini LLM integration
├── trading/
│   ├── production_trading_engine.py # Main trading engine
│   ├── execution/
│   │   └── alpaca_client.py         # Alpaca API client
│   └── portfolio/
│       ├── position_manager.py      # Position management
│       └── performance_tracker.py   # Performance tracking
├── run_production_trading.py        # Main execution script
└── start_trading.bat               # Windows startup script
```

## 🔧 Setup Instructions

### 1. Environment Setup

Create a `.env` file in the project root with your API credentials:

```bash
# Alpaca API (Paper Trading)
ALPACA_API_KEY_ID=your_alpaca_key_id
ALPACA_SECRET_KEY=your_alpaca_secret_key

# Gemini AI API
GEMINI_API_KEY=your_gemini_api_key

# Trading Environment (paper/live)
TRADING_ENVIRONMENT=paper
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Install Google AI SDK for Gemini
pip install google-generativeai
```

### 3. Verify Setup

```bash
# Test configuration and API connections
python run_production_trading.py --validate-only
```

## 🎯 Usage

### Quick Start (Windows)

```bash
# Run in paper trading mode (safe, no real money)
start_trading.bat

# Run with debug logging
start_trading.bat --debug

# Validate configuration only
start_trading.bat --validate
```

### Command Line Usage

```bash
# Paper trading (recommended for testing)
python run_production_trading.py --dry-run

# Live trading (real money!)
python run_production_trading.py

# Specific symbols only
python run_production_trading.py --symbols AAPL MSFT GOOGL

# Debug mode
python run_production_trading.py --log-level DEBUG

# Validate configuration
python run_production_trading.py --validate-only
```

### Configuration Options

The system uses `config/production_config.py` for all settings:

```python
# Key configuration parameters
portfolio_size = 20                    # Target number of positions
max_position_size = 0.10              # 10% max per position
min_signal_confidence = 0.6           # Minimum confidence for trades
paper_trading = True                  # Safe mode by default

# Model settings
primary_model = "optimized_catboost"  # Primary model
model_thresholds = {                  # Optimized thresholds
    "optimized_catboost": 0.6,
    "random_forest": 0.55,
    "svm": 0.5
}

# LLM settings
enable_llm = True                     # Enable Gemini analysis
llm_weight = 0.3                     # LLM influence on decisions
```

## 🤖 How It Works

### 1. Signal Generation
- **Technical Indicators**: RSI, MACD, Bollinger Bands, moving averages
- **ML Predictions**: Ensemble of optimized models predicts price movements
- **LLM Analysis**: Gemini AI analyzes market context and provides recommendations

### 2. Decision Making
- **Consensus Scoring**: Combines ML predictions and LLM analysis
- **Risk Assessment**: Evaluates position size and risk factors
- **Filtering**: Only trades with sufficient confidence are executed

### 3. Execution
- **Order Management**: Market orders through Alpaca API
- **Position Sizing**: Dynamic sizing based on signal strength and risk
- **Portfolio Management**: Maintains target portfolio composition

### 4. Monitoring
- **Real-time Logging**: Comprehensive logging of all decisions
- **Performance Tracking**: Trade-by-trade and portfolio performance
- **Session Reports**: Detailed session summaries saved to JSON

## 📊 Model Performance

### Optimized Models (Batch 1)
- **CatBoost**: Primary model with 60 unique prediction values
- **Random Forest**: Backup model with strong generalization
- **SVM**: Alternative approach for market regime changes

### Performance Metrics
- **Precision**: Balanced for portfolio construction
- **Recall**: Optimized for trade opportunity capture
- **ROC-AUC**: Validated on out-of-sample data

## 🔒 Risk Management

### Position Limits
- **Maximum Position**: 10% of portfolio per stock
- **Minimum Position**: 1% to maintain diversification
- **Cash Reserve**: 5% maintained for opportunities

### Risk Controls
- **Stop Losses**: 8% maximum loss per position
- **Take Profits**: 20% target gains
- **Sector Limits**: 30% maximum per sector
- **Volatility Limits**: Position sizing based on VIX levels

### Safety Features
- **Paper Trading Default**: System defaults to paper trading
- **Dry Run Mode**: Test without executing trades
- **Configuration Validation**: Prevents invalid settings
- **API Connection Testing**: Validates all connections before trading

## 📈 Performance Monitoring

### Real-time Metrics
- **Signal Generation**: Number and quality of signals
- **Execution Rate**: Percentage of signals successfully executed
- **Portfolio Composition**: Current positions and allocations
- **Risk Metrics**: VaR, maximum drawdown, Sharpe ratio

### Reporting
- **Session Reports**: Detailed JSON reports for each session
- **Trade Logs**: Complete trade history and rationale
- **Performance Analytics**: Portfolio vs benchmark tracking

## 🛠 Troubleshooting

### Common Issues

1. **API Connection Errors**
   ```bash
   # Check API credentials in .env file
   python run_production_trading.py --validate-only
   ```

2. **Model Loading Errors**
   ```bash
   # Verify model files exist
   ls -la models/production/
   ```

3. **Missing Dependencies**
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt
   pip install google-generativeai
   ```

### Debug Mode
```bash
# Run with full debug logging
python run_production_trading.py --log-level DEBUG --dry-run
```

### Support Files
- **Logs**: Check `logs/trading_engine.log` for detailed information
- **Results**: Session results saved in `results/trading_sessions/`
- **Configuration**: Validate settings in `config/production_config.py`

## ⚠️ Important Warnings

### Before Live Trading
1. **Test Thoroughly**: Always test in paper trading mode first
2. **Verify Capital**: Ensure you can afford any losses
3. **Understand Risks**: Automated trading can lose money quickly
4. **Monitor Actively**: Don't run unattended until fully validated
5. **Start Small**: Begin with small position sizes

### Legal Disclaimer
This software is for educational and research purposes. Users are responsible for their own trading decisions and any resulting profits or losses. No warranty is provided regarding performance or accuracy.

## 🔮 Future Enhancements

### Planned Features
- **Multi-timeframe Analysis**: Incorporate multiple time horizons
- **Options Integration**: Extend to options trading strategies
- **Advanced Risk Models**: More sophisticated risk management
- **Web Dashboard**: Real-time monitoring interface
- **Backtesting Framework**: Historical strategy validation

### Model Improvements
- **Deep Learning Models**: Integrate neural networks
- **Alternative Data**: Incorporate news sentiment and social media
- **Regime Detection**: Automatically adjust to market conditions
- **Continuous Learning**: Models that adapt to new market data

---

## 🚀 Quick Start Guide

1. **Set up API keys in `.env` file**
2. **Run validation**: `python run_production_trading.py --validate-only`
3. **Test in paper mode**: `python run_production_trading.py --dry-run`
4. **Monitor results**: Check `results/trading_sessions/latest_session.json`
5. **Go live when ready**: Remove `--dry-run` flag

Happy trading! 🎯📈
