# Configuration settings for Investment Committee trading system
# Contains trading parameters, model settings, and system configurations 

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_config() -> Dict[str, Any]:
    """
    Load configuration settings from environment variables with defaults
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    config = {
        # API Keys (using the variable names from your .env file)
        "ALPACA_API_KEY": os.getenv("APCA_API_KEY_ID", ""),
        "ALPACA_SECRET_KEY": os.getenv("APCA_API_SECRET_KEY", ""),
        "ALPACA_BASE_URL": os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets"),
        
        # LLM API Keys
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
        "GOOGLE_API_KEY": os.getenv("GEMINI_API_KEY", ""),
        
        # Trading Parameters
        "MAX_PORTFOLIO_RISK": float(os.getenv("MAX_PORTFOLIO_RISK", "0.02")),  # 2% max risk per trade
        "MIN_DAYS_TO_EXPIRATION": int(os.getenv("MIN_DAYS_TO_EXPIRATION", "30")),
        "MAX_DAYS_TO_EXPIRATION": int(os.getenv("MAX_DAYS_TO_EXPIRATION", "45")),
        "MIN_DELTA": float(os.getenv("MIN_DELTA", "0.15")),
        "MAX_DELTA": float(os.getenv("MAX_DELTA", "0.30")),
        
        # Model Settings
        "MODEL_RETRAIN_DAYS": int(os.getenv("MODEL_RETRAIN_DAYS", "7")),
        "PREDICTION_CONFIDENCE_THRESHOLD": float(os.getenv("PREDICTION_CONFIDENCE_THRESHOLD", "0.7")),
        
        # System Settings
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "DATA_DIRECTORY": os.getenv("DATA_DIRECTORY", "data"),
        "LOGS_DIRECTORY": os.getenv("LOGS_DIRECTORY", "logs"),
        
        # Symbol Universe Settings
        "USE_IEX_SYMBOLS_ONLY": os.getenv("USE_IEX_SYMBOLS_ONLY", "true").lower() == "true",
        "IEX_BATCHES_FILE": os.getenv("IEX_BATCHES_FILE", "filtered_iex_batches.json"),
        "DEFAULT_SYMBOL_LIMIT": int(os.getenv("DEFAULT_SYMBOL_LIMIT", "100")),
        
        # Database Settings
        "DATABASE_URL": os.getenv("DATABASE_URL", "sqlite:///data/investment_committee.db"),
        "REDIS_URL": os.getenv("REDIS_URL", "redis://localhost:6379"),
        
        # Trading Schedule
        "TRADING_ENABLED": os.getenv("TRADING_ENABLED", "false").lower() == "true",
        "MARKET_HOURS_ONLY": os.getenv("MARKET_HOURS_ONLY", "true").lower() == "true",
    }
    
    return config

# Default configuration instance
DEFAULT_CONFIG = load_config()

# API Key Configuration Functions
# These functions provide compatibility with the existing codebase
# while using the new .env-based configuration system

def get_alpaca_config() -> Dict[str, Any]:
    """
    Get Alpaca API configuration from environment variables
    
    Returns:
        Dict[str, Any]: Alpaca configuration dictionary
    """
    return {
        "api_key": os.getenv("APCA_API_KEY_ID", ""),
        "secret_key": os.getenv("APCA_API_SECRET_KEY", ""),
        "base_url": os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets"),
        "paper_trading": True  # Default to paper trading for safety
    }

def validate_api_keys() -> bool:
    """
    Validate that required Alpaca API keys are configured
    
    Returns:
        bool: True if API keys are valid, False otherwise
    """
    config = get_alpaca_config()
    return bool(config["api_key"] and config["secret_key"])

def get_gemini_config() -> Dict[str, Any]:
    """
    Get Gemini API configuration from environment variables
    
    Returns:
        Dict[str, Any]: Gemini configuration dictionary
    """
    return {
        "api_key": os.getenv("GEMINI_API_KEY", ""),
        "model": os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    }

def validate_gemini_api_key() -> bool:
    """
    Validate that Gemini API key is configured
    
    Returns:
        bool: True if API key is valid, False otherwise
    """
    config = get_gemini_config()
    return bool(config["api_key"]) 