# 🤖 Autonomous Investment Committee

## Fully Automated Trading System with "The Closer"

This system provides **completely autonomous trading** from market open (9:30 AM) to close (4:00 PM), Monday through Friday. The system makes intelligent trading decisions using machine learning models and manages trades automatically with "The Closer" - our intelligent trade management system.

## 🚀 Quick Start

### 1. Test the System
```bash
python test_autonomous_system.py
```

### 2. Launch Autonomous Trading
```bash
python autonomous_trading_launcher.py
```

### 3. Monitor with Dashboard
```bash
streamlit run trading_dashboard.py
```

## 🎯 Key Features

### Autonomous Committee
- **Fully Automated**: No human input required during market hours
- **ML-Driven Decisions**: Uses clean optimized models (no data leakage)
- **Bull Put Spread Focus**: Specializes in credit spread strategies
- **Risk Management**: Built-in position sizing and portfolio limits
- **Market Hours Only**: Automatically starts/stops with market

### The Closer - Intelligent Trade Management
- **Autonomous Trade Monitoring**: Continuously monitors all open positions
- **Multiple Exit Criteria**:
  - ✅ **Profit Target**: Closes at 50% profit (configurable)
  - 🛑 **Stop Loss**: Closes at $200 max loss (configurable)
  - ⏰ **Time Decay**: Closes 7 days before expiry (configurable)
  - 🔍 **Risk Management**: Dynamic risk assessment
- **Good & Bad Scenarios**: Handles both profitable and losing trades intelligently
- **No New Models Needed**: Uses predefined parameters from the committee

## 📊 System Architecture

```
Autonomous Committee
    ├── Market Hours Detection
    ├── ML Model Ensemble (Clean Data)
    ├── Symbol Scanning (529 symbols)
    ├── Risk Management
    └── Trade Execution
            ↓
    The Closer (Trade Manager)
    ├── Profit Target Monitoring
    ├── Stop Loss Protection
    ├── Time Decay Management
    ├── Risk Assessment
    └── Automatic Exit Execution
```

## ⚙️ Configuration

### Trading Parameters
- **Market Hours**: 9:30 AM - 4:00 PM ET, Monday-Friday
- **Max Daily Trades**: 10 trades per day
- **Max Portfolio Positions**: 20 concurrent positions
- **Risk Per Trade**: 2% of account value
- **Profit Target**: 50% of credit received
- **Stop Loss**: $200 maximum loss per spread
- **Time Decay Close**: 7 days before expiration

### The Closer Parameters
- **Monitoring Interval**: 30 seconds during market hours
- **Profit Target**: 50% (trades close when 50% profitable)
- **Stop Loss**: $200 (trades close if loss exceeds $200)
- **Time Management**: Closes positions 7 days before expiry
- **Risk Triggers**: Multiple risk conditions for early exit

## 📈 Strategy: Bull Put Spreads

The system focuses on **bull put spreads** - a credit strategy that profits when:
- Stock price stays above the short put strike
- Time decay reduces option values
- Implied volatility decreases

### Typical Trade Structure
- **Short Put**: ~3% out-of-the-money (collect premium)
- **Long Put**: ~6% out-of-the-money (protection)
- **Expiration**: 45 days to expiration
- **Credit Target**: Minimum $0.30 per spread

## 🔍 Monitoring & Logging

### Real-Time Dashboard
Access the dashboard at: `http://localhost:8501` (after running streamlit)

**Dashboard Features**:
- 📊 Live trade monitoring
- 📈 Performance analytics
- 🎯 The Closer status
- 📝 Recent activity log
- ⚙️ System configuration

### Log Files
- `logs/autonomous_trading.log` - Main system log
- `logs/trade_completions.jsonl` - Completed trades
- `data/managed_trades.json` - Active trades
- `logs/performance.csv` - Performance metrics

## 🛡️ Risk Management

### Portfolio Level
- Maximum 20 concurrent positions
- Maximum 10 new trades per day
- 2% risk per trade (of total account value)
- Position sizing based on account value

### Trade Level
- Stop loss at $200 per spread
- Profit target at 50% of credit
- Time-based exits (7 days before expiry)
- Dynamic risk assessment by The Closer

### Market Conditions
- Only trades during market hours
- Respects weekends and holidays
- Automatic shutdown outside trading hours

## 🔧 Technical Details

### Models Used
- **Clean Optimized Models**: Located in `models/clean/`
- **No Data Leakage**: Uses properly cleaned training data
- **Ensemble Approach**: Combines multiple model predictions
- **Confidence Threshold**: Only trades with >75% model confidence

### Data Sources
- **Symbol Universe**: 529 filtered symbols from `filtered_iex_batches.json`
- **Market Data**: Real-time quotes via Alpaca API
- **Features**: Technical indicators and market metrics

### Dependencies
```bash
# Main dependencies
pandas numpy scikit-learn
catboost xgboost lightgbm
alpaca-trade-api
streamlit plotly
asyncio schedule pytz
```

## 🚨 Important Notes

### Before Running
1. **Paper Trading**: System is configured for paper trading initially
2. **Alpaca API**: Ensure Alpaca credentials are properly configured
3. **Clean Models**: Verify clean models exist in `models/clean/`
4. **Disk Space**: Ensure adequate space for logs and data

### During Operation
- **No Manual Intervention**: System operates autonomously
- **Monitor Dashboard**: Check progress via Streamlit dashboard
- **Log Monitoring**: Watch logs for any issues
- **Market Hours Only**: System automatically pauses outside market hours

### Emergency Stop
- **Ctrl+C**: Graceful shutdown of autonomous system
- **The Closer**: Continues managing existing positions
- **Manual Override**: Can manually close positions if needed

## 📋 Daily Workflow

### Market Open (9:30 AM ET)
1. System automatically detects market open
2. Loads clean models and symbol universe
3. Begins scanning for opportunities
4. The Closer starts monitoring existing positions

### During Market Hours
1. **Every 5 minutes**: Scan for new opportunities
2. **Every 30 seconds**: The Closer monitors active trades
3. **Continuous**: Risk management and position tracking
4. **As needed**: Execute new trades or close existing ones

### Market Close (4:00 PM ET)
1. Stop scanning for new opportunities
2. Generate end-of-day summary
3. The Closer continues monitoring until resolution
4. Update performance metrics and logs

## 🎯 Expected Performance

Based on clean model results:
- **Model Accuracy**: 85-90% (realistic, no data leakage)
- **Win Rate Target**: 70-80% for bull put spreads
- **Average Return**: 10-30% return on margin per trade
- **Time Frame**: 45-day average holding period

## 📞 Support & Troubleshooting

### Common Issues
1. **Models Not Found**: Run `train_clean_models.py` first
2. **API Errors**: Check Alpaca credentials and connection
3. **No Trades**: Verify market hours and model confidence levels
4. **Dashboard Issues**: Ensure Streamlit dependencies installed

### Log Analysis
```bash
# View recent activity
tail -f logs/autonomous_trading.log

# Check trade completions
cat logs/trade_completions.jsonl | jq .

# Monitor performance
python -c "import pandas as pd; print(pd.read_csv('logs/performance.csv').tail())"
```

---

## 🏁 Getting Started

1. **Test Everything**: `python test_autonomous_system.py`
2. **Launch System**: `python autonomous_trading_launcher.py`
3. **Open Dashboard**: `streamlit run trading_dashboard.py`
4. **Monitor & Enjoy**: Watch your autonomous committee trade!

The system is designed to run **completely hands-off** during market hours while providing full transparency through comprehensive logging and real-time monitoring.

**Happy Autonomous Trading! 🚀**
