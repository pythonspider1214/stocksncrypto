import pandas as pd
import numpy as np
import yfinance as yf
import logging
from typing import Dict, Any
from utils import safe_yf_ticker_history

class TechnicalIndicators:
    """Technical analysis indicators"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        if 'Close' not in df.columns:
            logging.error("No 'Close' column found in DataFrame")
            return df
            
        close = df['Close']
        df['SMA20'] = TechnicalIndicators.calculate_sma(close, 20)
        df['SMA50'] = TechnicalIndicators.calculate_sma(close, 50)
        df['RSI14'] = TechnicalIndicators.calculate_rsi(close, 14)
        return df
    
    @staticmethod
    def generate_signals(symbol: str, period: str = "6mo") -> Dict[str, Any]:
        """Generate trading signals"""
        try:
            df = safe_yf_ticker_history(symbol, period=period)
            if df is None or df.empty:
                return {'error': f'No data available for {symbol}'}
            
            df = TechnicalIndicators.calculate_indicators(df)
            
            if len(df) == 0:
                return {'error': f'No data after calculating indicators for {symbol}'}
            
            latest = df.iloc[-1]
            signals = []
            
            # Check if we have valid indicator values
            if pd.isna(latest.get('RSI14')):
                return {'error': f'Insufficient data for technical analysis of {symbol}'}
            
            # RSI signals
            rsi_value = latest['RSI14']
            if rsi_value < 30:
                signals.append("RSI Oversold - Potential Buy")
            elif rsi_value > 70:
                signals.append("RSI Overbought - Potential Sell")
            
            # Moving average signals
            if not pd.isna(latest.get('SMA20')) and not pd.isna(latest.get('SMA50')):
                current_price = latest['Close']
                sma20 = latest['SMA20']
                sma50 = latest['SMA50']
                
                if current_price > sma20 > sma50:
                    signals.append("Price Above Moving Averages - Bullish")
                elif current_price < sma20 < sma50:
                    signals.append("Price Below Moving Averages - Bearish")
            
            # Determine recommendation
            bullish = sum(1 for s in signals if any(word in s.lower() for word in ['buy', 'bullish']))
            bearish = sum(1 for s in signals if any(word in s.lower() for word in ['sell', 'bearish']))
            
            if bullish > bearish:
                recommendation = "BUY"
            elif bearish > bullish:
                recommendation = "SELL"
            else:
                recommendation = "HOLD"
            
            return {
                'symbol': symbol,
                'current_price': latest['Close'],
                'rsi': latest.get('RSI14', 0),
                'sma_20': latest.get('SMA20', 0),
                'sma_50': latest.get('SMA50', 0),
                'signals': signals,
                'recommendation': recommendation
            }
            
        except Exception as e:
            logging.error(f"Error analyzing {symbol}: {e}")
            return {'error': f'Error analyzing {symbol}: {str(e)}'}
