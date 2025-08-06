# Investment Committee Trading System

AI-driven trading system for bull put spread strategies using ML models, LLM analysis, and live market data through Alpaca API.

## Features

### Core Trading System
- **Bull Put Spread Strategy**: Automated option spread selection and execution
- **Multi-Model ML Pipeline**: XGBoost + Neural Networks for trade predictions  
- **LLM Analysis**: Gemini AI integration for market sentiment and risk assessment
- **Real-time Execution**: Alpaca API integration for paper and live trading
- **Risk Management**: Position sizing, stop-loss, and portfolio risk controls

### Data & Symbols
- **IEX Filtered Universe**: 529 pre-filtered stocks across 11 batches
- **Alpaca Integration**: Real-time and historical market data (free tier compatible)
- **Multiple Timeframes**: Support for 1Min, 5Min, 15Min, 1Hour, 1Day bars
- **Technical Indicators**: RSI, MACD, Bollinger Bands, momentum, volatility metrics

### Training & Model Management
- **Batch Training**: Process symbols in manageable batches (prevents system overload)
- **Progress Tracking**: Resume training from where you left off
- **Force Retraining**: Override completed batches with `--force` flag
- **CSV Logging**: Comprehensive training history with accuracy and timing metrics
- **Retry Logic**: Automatic retries with exponential backoff for failed API calls

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Environment Setup
Create a `.env` file with your API keys:
```env
# Alpaca API (Paper Trading)
APCA_API_KEY_ID=your_alpaca_key
APCA_API_SECRET_KEY=your_alpaca_secret  
APCA_API_BASE_URL=https://paper-api.alpaca.markets

# LLM APIs (Optional)
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
```

### Training Models

#### Basic Usage
```bash
# Train on batch 1 (recommended starting point)
python train_models.py --batch 1

# Train with hourly data 
python train_models.py --batch 2 --bar-timeframe 1Hour

# Train subset of batch (useful for testing)
python train_models.py --batch 3 --limit-tickers 10

# Force retrain completed batch
python train_models.py --batch 1 --force
```

#### Training Options
```bash
# List available batches
python train_models.py --list-batches

# Show training progress  
python train_models.py --show-progress

# Help with all options
python train_models.py --help
```

#### Timeframe Options
- `1Min`: 1-minute bars (high frequency, large datasets)
- `5Min`: 5-minute bars (intraday trading)  
- `15Min`: 15-minute bars (short-term patterns)
- `1Hour`: 1-hour bars (medium-term trends)
- `1Day`: Daily bars (default, most stable)

### System Architecture

```
├── train_models.py          # Main training pipeline
├── main.py                  # System orchestrator
├── config/
│   └── settings.py          # Configuration management
├── models/
│   ├── xgboost_model.py     # XGBoost implementation
│   ├── neural_predictor.py  # Neural network models
│   └── saved/               # Trained model files
├── trading/
│   ├── execution/           # Alpaca API client
│   ├── strategy/            # Bull put spread logic
│   └── portfolio/           # Position & performance tracking
├── analysis/
│   ├── technical_analysis.py
│   ├── sentiment_analysis.py
│   └── macro_analysis.py
├── utils/
│   ├── helpers.py           # IEX batch management
│   ├── training_logger.py   # CSV logging utilities
│   └── logging.py           # System logging
└── logs/
    ├── training_summary.csv # Training history & metrics
    └── trading_system_*.log # System logs
```

### Training Pipeline Flow

1. **Load IEX Batch**: Select symbols from pre-filtered batch
2. **Fetch Market Data**: Get historical bars via Alpaca API with retry logic
3. **Feature Engineering**: Generate 28+ technical indicators
4. **Data Preparation**: Clean, balance, and split datasets
5. **Model Training**: Train XGBoost and Neural Network models
6. **Evaluation**: Generate accuracy metrics and confusion matrices  
7. **Visualization**: Create performance charts and save to `reports/`
8. **Progress Logging**: Update JSON progress and CSV training history

### Monitoring & Logs

#### Training History
```bash
# View training summary CSV
cat logs/training_summary.csv

# Training progress JSON
cat training_progress.json
```

#### Log Files
- `logs/training_summary.csv`: Batch training metrics
- `logs/trading_system_*.log`: Detailed system logs
- `training_progress.json`: Completed batch tracking

### Advanced Usage

#### Custom Batch Files
```bash
# Use custom batch configuration
python train_models.py --batch 1 --batch-file custom_batches.json
```

#### Performance Tuning
```bash
# Reduce memory usage with smaller batches
python train_models.py --batch 1 --limit-tickers 25

# Use daily bars for faster processing
python train_models.py --batch 1 --bar-timeframe 1Day
```

## API Integration

### Alpaca Markets
- **Free Tier**: IEX data with 15-minute delay
- **Paper Trading**: Risk-free testing environment
- **Real-time Quotes**: For live trading (paid plans)

### Supported Operations
- Market data retrieval (multiple timeframes)
- Account management and positions
- Order placement and tracking
- Portfolio performance monitoring

## Model Performance

### Typical Results
- **XGBoost**: 60-75% accuracy on directional predictions
- **Neural Networks**: 60-70% accuracy with regularization
- **Combined**: Meta-model ensemble for improved stability

### Evaluation Metrics
- Classification accuracy and F1-scores
- Precision/recall for both bull and bear signals
- Confusion matrices and ROC curves
- Training time and convergence metrics

## Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd investment-committee

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Code Organization
- Follow existing patterns for new models/strategies
- Add comprehensive logging and error handling
- Test with small batches before full runs
- Document new parameters and options

## License

This project is for educational and research purposes. Not financial advice.

## Troubleshooting

### Common Issues

**"Subscription does not permit querying recent SIP data"**
- Using delayed data automatically (15-minute delay)
- Upgrade Alpaca plan for real-time access

**"No market data was successfully fetched"** 
- Check API keys in `.env` file
- Verify symbols exist in IEX filtered batches
- Try with `--limit-tickers 5` for testing

**Training fails with memory errors**
- Reduce `--limit-tickers` to smaller number
- Use `1Day` timeframe instead of intraday
- Close other applications to free RAM

### Getting Help
- Check logs in `logs/` directory for detailed errors
- Use `--help` flag for available options  
- Test with small batches first (`--limit-tickers 3`) 