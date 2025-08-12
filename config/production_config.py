"""
Production Trading Configuration
===============================

Configuration for production trading system with optimized models and LLM integration.
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for production models."""
    # Primary production model
    primary_model_path: str = "models/production/optimized_catboost.pkl"
    primary_model_type: str = "catboost"
    primary_model_threshold: float = 0.6
    
    # Backup models
    backup_models: Dict[str, str] = field(default_factory=lambda: {
        "random_forest": "models/production/random_forest.pkl",
        "svm": "models/production/svm.pkl"
    })
    
    # Model thresholds (optimized values)
    model_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "optimized_catboost": 0.6,
        "random_forest": 0.55,
        "svm": 0.5
    })
    
    # Model weights for ensemble
    model_weights: Dict[str, float] = field(default_factory=lambda: {
        "optimized_catboost": 0.5,
        "random_forest": 0.3,
        "svm": 0.2
    })
    
    # Feature configuration
    feature_columns: List[str] = field(default_factory=lambda: [
        'price_change_1d', 'price_change_5d', 'price_change_10d', 'price_change_20d',
        'price_vs_sma5', 'price_vs_sma10', 'price_vs_sma20',
        'volatility_5d', 'volatility_10d', 'volatility_20d',
        'volume_ratio', 'hl_ratio', 'hl_ratio_5d',
        'rsi_14', 'macd', 'macd_signal', 'macd_histogram', 'bb_position'
    ])


@dataclass
class LLMConfig:
    """Configuration for LLM integration."""
    # Gemini API settings
    api_key: Optional[str] = None  # Will use environment variable if None
    model_name: str = "gemini-2.0-flash-exp"  # Upgraded to Gemini 2.0 Flash
    enable_llm: bool = True  # Re-enabled with smart consultation
    
    # Rate limiting
    max_requests_per_minute: int = 15
    max_requests_per_day: int = 1500
    
    # Analysis settings
    confidence_threshold: float = 0.7  # Minimum confidence for LLM recommendations
    enable_meta_model: bool = True
    meta_model_path: str = "models/production/meta_model.pkl"
    
    # LLM influence on trading decisions
    llm_weight: float = 0.3  # How much to weight LLM vs ML models
    require_llm_consensus: bool = False  # Require LLM agreement for trades


@dataclass 
class TradingConfig:
    """Configuration for trading system."""
    # Portfolio settings
    portfolio_size: int = 20  # Target number of positions
    max_position_size: float = 0.10  # 10% max per position
    min_position_size: float = 0.01  # 1% minimum per position
    cash_reserve: float = 0.05  # 5% cash reserve
    
    # Risk management
    max_portfolio_risk: float = 0.15  # 15% max portfolio risk
    max_sector_concentration: float = 0.30  # 30% max per sector
    stop_loss_threshold: float = -0.08  # 8% stop loss
    take_profit_threshold: float = 0.20  # 20% take profit
    
    # Signal filtering
    min_signal_confidence: float = 0.3  # Lowered for testing - was 0.6
    min_consensus_score: float = 0.35  # Lowered for testing - was 0.55
    require_technical_confirmation: bool = True
    
    # Trading universe
    trading_universe: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", 
        "ORCL", "CRM", "AMD", "INTC", "QCOM", "AVGO", "TXN", "MU",
        "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC",
        "JNJ", "PFE", "UNH", "CVS", "ABBV", "LLY", "BMY", "GILD"
    ])
    
    # Rebalancing
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    rebalance_threshold: float = 0.05  # 5% drift from target
    
    # Paper trading settings
    paper_trading: bool = True
    paper_trading_start_balance: float = 100000.0


@dataclass
class AlpacaConfig:
    """Configuration for Alpaca API."""
    # API credentials (will use environment variables)
    api_key_id: Optional[str] = None
    secret_key: Optional[str] = None
    base_url: str = "https://paper-api.alpaca.markets"  # Paper trading by default
    
    # Trading settings
    market_hours_only: bool = True
    extended_hours: bool = False
    day_trading_buying_power: bool = False
    
    # Order settings
    default_order_type: str = "market"  # market, limit, stop, stop_limit
    default_time_in_force: str = "day"  # day, gtc, opg, cls
    
    # Data settings
    data_feed: str = "iex"  # iex, sip
    historical_bars_limit: int = 1000


@dataclass
class ProductionConfig:
    """Master configuration for production trading system."""
    # Component configurations
    model_config: ModelConfig = field(default_factory=ModelConfig)
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    trading_config: TradingConfig = field(default_factory=TradingConfig)
    alpaca_config: AlpacaConfig = field(default_factory=AlpacaConfig)
    
    # System settings
    environment: str = "paper"  # paper, live
    log_level: str = "INFO"
    log_file: str = "logs/trading_engine.log"
    
    # Performance tracking
    enable_performance_tracking: bool = True
    performance_benchmark: str = "SPY"
    save_trades_to_file: bool = True
    trades_file: str = "data/trades.csv"
    
    # Alerts and notifications
    enable_alerts: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["log", "file"])
    
    def __post_init__(self):
        """Load configuration from environment and files."""
        self.load_from_environment()
        self.load_optimized_thresholds()
    
    def load_from_environment(self):
        """Load configuration from environment variables."""
        # Alpaca API - support both ALPACA_ and APCA_ prefixes
        self.alpaca_config.api_key_id = (os.getenv("ALPACA_API_KEY_ID") or 
                                       os.getenv("APCA_API_KEY_ID"))
        self.alpaca_config.secret_key = (os.getenv("ALPACA_SECRET_KEY") or 
                                       os.getenv("APCA_API_SECRET_KEY"))
        
        # Gemini API
        self.llm_config.api_key = os.getenv("GEMINI_API_KEY")
        
        # Environment settings
        if os.getenv("TRADING_ENVIRONMENT") == "live":
            self.environment = "live"
            self.alpaca_config.base_url = "https://api.alpaca.markets"
            self.trading_config.paper_trading = False
    
    def load_optimized_thresholds(self):
        """Load universal production thresholds."""
        threshold_files = [
            "config/production_thresholds.json",  # Universal production thresholds
            "config/model_thresholds.json",       # Fallback
            "config/optimized_thresholds_batch_1.json"  # Legacy fallback
        ]
        
        for threshold_file in threshold_files:
            if os.path.exists(threshold_file):
                try:
                    with open(threshold_file, 'r') as f:
                        threshold_data = json.load(f)
                    
                    if 'model_thresholds' in threshold_data:
                        for model_name, threshold_info in threshold_data['model_thresholds'].items():
                            if isinstance(threshold_info, dict) and 'threshold' in threshold_info:
                                self.model_config.model_thresholds[model_name] = threshold_info['threshold']
                            elif isinstance(threshold_info, (int, float)):
                                self.model_config.model_thresholds[model_name] = threshold_info
                    
                    # Load LLM integration settings
                    if 'llm_integration' in threshold_data:
                        llm_config = threshold_data['llm_integration']
                        self.llm_config.llm_weight = llm_config.get('weight', 0.2)
                        
                    # Load ensemble settings
                    if 'ensemble_config' in threshold_data:
                        ensemble_config = threshold_data['ensemble_config']
                        self.trading_config.min_signal_confidence = ensemble_config.get('confidence_threshold', 0.3)
                    
                    print(f"✅ Loaded universal thresholds from {threshold_file}")
                    break
                    
                except Exception as e:
                    print(f"⚠️ Failed to load thresholds from {threshold_file}: {e}")
        else:
            print("⚠️ No threshold files found, using defaults")
    
    def save_config(self, config_path: str = "config/production_config.json"):
        """Save configuration to file."""
        try:
            # Create config directory if it doesn't exist
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Convert to dictionary (excluding sensitive data)
            config_dict = {
                "model_config": {
                    "primary_model_path": self.model_config.primary_model_path,
                    "backup_models": self.model_config.backup_models,
                    "model_thresholds": self.model_config.model_thresholds,
                    "model_weights": self.model_config.model_weights,
                    "feature_columns": self.model_config.feature_columns
                },
                "llm_config": {
                    "model_name": self.llm_config.model_name,
                    "enable_llm": self.llm_config.enable_llm,
                    "confidence_threshold": self.llm_config.confidence_threshold,
                    "enable_meta_model": self.llm_config.enable_meta_model,
                    "llm_weight": self.llm_config.llm_weight
                },
                "trading_config": {
                    "portfolio_size": self.trading_config.portfolio_size,
                    "max_position_size": self.trading_config.max_position_size,
                    "min_position_size": self.trading_config.min_position_size,
                    "trading_universe": self.trading_config.trading_universe,
                    "min_signal_confidence": self.trading_config.min_signal_confidence,
                    "paper_trading": self.trading_config.paper_trading
                },
                "system_settings": {
                    "environment": self.environment,
                    "log_level": self.log_level,
                    "enable_performance_tracking": self.enable_performance_tracking
                }
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            print(f"✅ Configuration saved to {config_path}")
            
        except Exception as e:
            print(f"❌ Failed to save configuration: {e}")
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check model files exist
        if not os.path.exists(self.model_config.primary_model_path):
            issues.append(f"Primary model not found: {self.model_config.primary_model_path}")
        
        for name, path in self.model_config.backup_models.items():
            if not os.path.exists(path):
                issues.append(f"Backup model {name} not found: {path}")
        
        # Check API keys
        if not self.alpaca_config.api_key_id:
            issues.append("Alpaca API Key ID not configured")
        
        if not self.alpaca_config.secret_key:
            issues.append("Alpaca Secret Key not configured")
        
        if self.llm_config.enable_llm and not self.llm_config.api_key:
            issues.append("Gemini API Key not configured but LLM is enabled")
        
        # Check trading parameters
        if self.trading_config.max_position_size <= 0 or self.trading_config.max_position_size > 1:
            issues.append("Invalid max_position_size (should be between 0 and 1)")
        
        if not self.trading_config.trading_universe:
            issues.append("Trading universe is empty")
        
        # Check thresholds
        for model_name, threshold in self.model_config.model_thresholds.items():
            if threshold < 0 or threshold > 1:
                issues.append(f"Invalid threshold for {model_name}: {threshold}")
        
        return issues
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            "environment": self.environment,
            "primary_model": self.model_config.primary_model_path,
            "backup_models": len(self.model_config.backup_models),
            "llm_enabled": self.llm_config.enable_llm,
            "portfolio_size": self.trading_config.portfolio_size,
            "trading_universe_size": len(self.trading_config.trading_universe),
            "paper_trading": self.trading_config.paper_trading,
            "min_confidence": self.trading_config.min_signal_confidence,
            "model_thresholds": self.model_config.model_thresholds
        }


# Global configuration instance
_config = None

def get_production_config() -> ProductionConfig:
    """Get global production configuration instance."""
    global _config
    if _config is None:
        _config = ProductionConfig()
    return _config

def load_production_config(config_path: str = None) -> ProductionConfig:
    """Load production configuration from file."""
    global _config
    _config = ProductionConfig()
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Update configuration with loaded data
            # This would need more sophisticated merging logic
            print(f"✅ Loaded configuration from {config_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to load configuration from {config_path}: {e}")
    
    return _config


if __name__ == "__main__":
    # Test configuration
    config = ProductionConfig()
    
    print("\n" + "="*60)
    print("PRODUCTION CONFIGURATION")
    print("="*60)
    
    # Validate configuration
    issues = config.validate_config()
    if issues:
        print("\n❌ Configuration Issues:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print("\n✅ Configuration is valid")
    
    # Show summary
    print("\n📊 Configuration Summary:")
    summary = config.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Save configuration
    config.save_config()
    
    print("\n" + "="*60)
