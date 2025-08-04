import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from functools import lru_cache
from datetime import datetime, timedelta
import yfinance as yf

def setup_logging(log_level: str = "INFO", log_file: str = "crypto_tracker.log"):
    """Setup comprehensive logging"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

class TimedCache:
    """Cache with expiration"""
    def __init__(self, expiry_minutes: int = 60):
        self.cache = {}
        self.expiry = timedelta(minutes=expiry_minutes)
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.expiry:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        self.cache[key] = (value, datetime.now())

# Global cache
cache = TimedCache(expiry_minutes=int(os.getenv('CACHE_EXPIRY_MINUTES', 60)))

@lru_cache(maxsize=128)
def get_stock_info_cached(symbol: str) -> Dict[str, Any]:
    """Cache stock info to reduce API calls"""
    try:
        return yf.Ticker(symbol).info
    except Exception as e:
        logging.error(f"Error fetching stock info for {symbol}: {e}")
        return {}

def safe_yf_download(symbol: str, *args, **kwargs) -> Optional[pd.DataFrame]:
    """Safe wrapper for yfinance download"""
    try:
        # Set auto_adjust=False by default to avoid FutureWarning
        if 'auto_adjust' not in kwargs:
            kwargs['auto_adjust'] = False
            
        df = yf.download(symbol, *args, **kwargs)
        if df is None or df.empty:
            logging.warning(f"No data returned for {symbol}")
            return None
        return df
    except Exception as e:
        logging.error(f"yfinance error for {symbol}: {e}")
        return None

def safe_yf_ticker_history(symbol: str, *args, **kwargs) -> Optional[pd.DataFrame]:
    """Safe wrapper for yfinance ticker history"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(*args, **kwargs)
        if df is None or df.empty:
            logging.warning(f"No history for {symbol}")
            return None
        return df
    except Exception as e:
        logging.error(f"yfinance ticker error for {symbol}: {e}")
        return None

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency with proper symbols"""
    symbols = {"USD": "$", "EUR": "€", "AUD": "A$"}
    symbol = symbols.get(currency, currency)
    return f"{symbol}{amount:,.2f}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage with proper sign"""
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimals}f}%"
