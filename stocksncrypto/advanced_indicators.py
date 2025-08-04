import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import talib
from scipy import stats
from sklearn.preprocessing import StandardScaler
import logging

class AdvancedIndicators:
    """Professional-grade technical indicators for crypto and stock analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    # ===========================================
    # MOMENTUM INDICATORS
    # ===========================================
    
    def stochastic_rsi(self, df: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
        """Stochastic RSI - More sensitive than regular RSI"""
        rsi = talib.RSI(df['Close'].values, timeperiod=period)
        
        # Calculate Stochastic of RSI
        rsi_min = pd.Series(rsi).rolling(window=period).min()
        rsi_max = pd.Series(rsi).rolling(window=period).max()
        
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100
        
        df['StochRSI_K'] = pd.Series(stoch_rsi).rolling(window=smooth_k).mean()
        df['StochRSI_D'] = df['StochRSI_K'].rolling(window=smooth_d).mean()
        
        return df
    
    def williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Williams %R - Momentum oscillator"""
        high_max = df['High'].rolling(window=period).max()
        low_min = df['Low'].rolling(window=period).min()
        
        df['Williams_R'] = ((high_max - df['Close']) / (high_max - low_min)) * -100
        return df
    
    def commodity_channel_index(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """CCI - Identifies cyclical trends"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        df['CCI'] = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return df
    
    def money_flow_index(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """MFI - Volume-weighted RSI"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        # Positive and negative money flow
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        # Money flow ratio
        pos_mf = positive_flow.rolling(window=period).sum()
        neg_mf = negative_flow.rolling(window=period).sum()
        
        mfr = pos_mf / neg_mf
        df['MFI'] = 100 - (100 / (1 + mfr))
        
        return df
    
    # ===========================================
    # VOLATILITY INDICATORS
    # ===========================================
    
    def average_true_range(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """ATR - Measures volatility"""
        df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=period)
        return df
    
    def bollinger_bands_advanced(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """Enhanced Bollinger Bands with additional metrics"""
        sma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        
        df['BB_Upper'] = sma + (std * std_dev)
        df['BB_Lower'] = sma - (std * std_dev)
        df['BB_Middle'] = sma
        
        # Bollinger Band metrics
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] * 100
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Squeeze detection (low volatility)
        df['BB_Squeeze'] = df['BB_Width'] < df['BB_Width'].rolling(window=20).quantile(0.1)
        
        return df
    
    def keltner_channels(self, df: pd.DataFrame, period: int = 20, multiplier: float = 2) -> pd.DataFrame:
        """Keltner Channels - Volatility-based channels"""
        ema = df['Close'].ewm(span=period).mean()
        atr = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=period)
        
        df['KC_Upper'] = ema + (multiplier * atr)
        df['KC_Lower'] = ema - (multiplier * atr)
        df['KC_Middle'] = ema
        
        return df
    
    # ===========================================
    # VOLUME INDICATORS
    # ===========================================
    
    def on_balance_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """OBV - Volume-price trend"""
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        df['OBV'] = obv
        return df
    
    def volume_weighted_average_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """VWAP - Volume weighted average price"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        return df
    
    def accumulation_distribution_line(self, df: pd.DataFrame) -> pd.DataFrame:
        """A/D Line - Volume flow indicator"""
        money_flow_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        money_flow_volume = money_flow_multiplier * df['Volume']
        df['ADL'] = money_flow_volume.cumsum()
        return df
    
    # ===========================================
    # TREND INDICATORS
    # ===========================================
    
    def adaptive_moving_average(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """KAMA - Adaptive Moving Average"""
        change = abs(df['Close'] - df['Close'].shift(period))
        volatility = abs(df['Close'] - df['Close'].shift(1)).rolling(window=period).sum()
        
        efficiency_ratio = change / volatility
        
        # Smoothing constants
        fastest_sc = 2 / (2 + 1)
        slowest_sc = 2 / (30 + 1)
        
        sc = (efficiency_ratio * (fastest_sc - slowest_sc) + slowest_sc) ** 2
        
        # Calculate KAMA
        kama = [df['Close'].iloc[0]]
        for i in range(1, len(df)):
            kama.append(kama[-1] + sc.iloc[i] * (df['Close'].iloc[i] - kama[-1]))
        
        df['KAMA'] = kama
        return df
    
    def parabolic_sar(self, df: pd.DataFrame, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.2) -> pd.DataFrame:
        """Parabolic SAR - Trend reversal indicator"""
        df['PSAR'] = talib.SAR(df['High'].values, df['Low'].values, acceleration=af_start, maximum=af_max)
        return df
    
    def ichimoku_cloud(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ichimoku Cloud - Comprehensive trend analysis"""
        # Tenkan-sen (Conversion Line)
        high_9 = df['High'].rolling(window=9).max()
        low_9 = df['Low'].rolling(window=9).min()
        df['Ichimoku_Tenkan'] = (high_9 + low_9) / 2
        
        # Kijun-sen (Base Line)
        high_26 = df['High'].rolling(window=26).max()
        low_26 = df['Low'].rolling(window=26).min()
        df['Ichimoku_Kijun'] = (high_26 + low_26) / 2
        
        # Senkou Span A (Leading Span A)
        df['Ichimoku_SpanA'] = ((df['Ichimoku_Tenkan'] + df['Ichimoku_Kijun']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        high_52 = df['High'].rolling(window=52).max()
        low_52 = df['Low'].rolling(window=52).min()
        df['Ichimoku_SpanB'] = ((high_52 + low_52) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        df['Ichimoku_Chikou'] = df['Close'].shift(-26)
        
        return df

class QuantitativeAnalysis:
    """Advanced quantitative analysis for asset evaluation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Sharpe Ratio - Risk-adjusted returns"""
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Sortino Ratio - Downside risk-adjusted returns"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        downside_std = np.sqrt(252) * downside_returns.std()
        return np.sqrt(252) * excess_returns.mean() / downside_std if downside_std != 0 else 0
    
    def calculate_maximum_drawdown(self, prices: pd.Series) -> Dict[str, float]:
        """Maximum Drawdown analysis"""
        cumulative = (1 + prices.pct_change()).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        max_dd = drawdown.min()
        max_dd_duration = 0
        current_dd_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_duration': max_dd_duration
        }
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Value at Risk calculation"""
        return np.percentile(returns, confidence_level * 100)
    
    def calculate_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Beta coefficient vs market"""
        covariance = np.cov(asset_returns.dropna(), market_returns.dropna())[0][1]
        market_variance = np.var(market_returns.dropna())
        return covariance / market_variance if market_variance != 0 else 0
    
    def calculate_correlation_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Correlation matrix for portfolio analysis"""
        return returns_df.corr()
    
    def risk_parity_weights(self, returns_df: pd.DataFrame) -> Dict[str, float]:
        """Risk Parity portfolio weights"""
        # Simplified risk parity - inverse volatility weighting
        volatilities = returns_df.std()
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()
        return weights.to_dict()

class MarketRegimeDetection:
    """Detect market regimes (bull, bear, sideways)"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_trend_regime(self, df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
        """Detect trend regimes using multiple indicators"""
        # Price trend
        df['Price_Trend'] = np.where(df['Close'] > df['Close'].rolling(lookback).mean(), 1, -1)
        
        # Volatility regime
        returns = df['Close'].pct_change()
        vol_rolling = returns.rolling(lookback).std()
        vol_threshold = vol_rolling.quantile(0.7)
        df['Vol_Regime'] = np.where(vol_rolling > vol_threshold, 1, 0)  # 1 = High vol, 0 = Low vol
        
        # Momentum regime
        momentum = df['Close'] / df['Close'].shift(lookback) - 1
        df['Momentum_Regime'] = np.where(momentum > 0, 1, -1)
        
        # Combined regime
        regime_score = df['Price_Trend'] + df['Momentum_Regime']
        df['Market_Regime'] = np.select(
            [regime_score >= 1, regime_score <= -1],
            ['Bull', 'Bear'],
            default='Sideways'
        )
        
        return df
    
    def volatility_clustering(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Detect volatility clustering (GARCH-like)"""
        squared_returns = returns ** 2
        vol_ma = squared_returns.rolling(window).mean()
        current_vol = squared_returns.rolling(5).mean()
        
        return current_vol > vol_ma * 1.5  # Volatility spike detection

class SmartSignalGenerator:
    """Advanced signal generation combining multiple indicators"""
    
    def __init__(self):
        self.indicators = AdvancedIndicators()
        self.quant = QuantitativeAnalysis()
        self.regime = MarketRegimeDetection()
        self.logger = logging.getLogger(__name__)
    
    def generate_comprehensive_signals(self, df: pd.DataFrame) -> Dict[str, any]:
        """Generate signals using multiple advanced indicators"""
        
        # Calculate all indicators
        df = self.indicators.stochastic_rsi(df)
        df = self.indicators.williams_r(df)
        df = self.indicators.commodity_channel_index(df)
        df = self.indicators.money_flow_index(df)
        df = self.indicators.average_true_range(df)
        df = self.indicators.bollinger_bands_advanced(df)
        df = self.indicators.keltner_channels(df)
        df = self.indicators.on_balance_volume(df)
        df = self.indicators.volume_weighted_average_price(df)
        df = self.indicators.adaptive_moving_average(df)
        df = self.indicators.parabolic_sar(df)
        df = self.indicators.ichimoku_cloud(df)
        
        # Detect market regime
        df = self.regime.detect_trend_regime(df)
        
        latest = df.iloc[-1]
        signals = []
        signal_strength = 0
        
        # Momentum signals
        if latest['StochRSI_K'] < 20 and latest['StochRSI_D'] < 20:
            signals.append("StochRSI Oversold - Strong Buy Signal")
            signal_strength += 2
        elif latest['StochRSI_K'] > 80 and latest['StochRSI_D'] > 80:
            signals.append("StochRSI Overbought - Strong Sell Signal")
            signal_strength -= 2
        
        if latest['Williams_R'] < -80:
            signals.append("Williams %R Oversold")
            signal_strength += 1
        elif latest['Williams_R'] > -20:
            signals.append("Williams %R Overbought")
            signal_strength -= 1
        
        if latest['CCI'] < -100:
            signals.append("CCI Oversold")
            signal_strength += 1
        elif latest['CCI'] > 100:
            signals.append("CCI Overbought")
            signal_strength -= 1
        
        if latest['MFI'] < 20:
            signals.append("Money Flow Oversold")
            signal_strength += 1
        elif latest['MFI'] > 80:
            signals.append("Money Flow Overbought")
            signal_strength -= 1
        
        # Volatility signals
        if latest['BB_Squeeze']:
            signals.append("Bollinger Band Squeeze - Breakout Expected")
        
        if latest['BB_Position'] < 0.1:
            signals.append("Near Lower Bollinger Band - Potential Bounce")
            signal_strength += 1
        elif latest['BB_Position'] > 0.9:
            signals.append("Near Upper Bollinger Band - Potential Reversal")
            signal_strength -= 1
        
        # Trend signals
        if latest['Close'] > latest['KAMA']:
            signals.append("Above Adaptive MA - Uptrend")
            signal_strength += 1
        else:
            signals.append("Below Adaptive MA - Downtrend")
            signal_strength -= 1
        
        if latest['Close'] > latest['PSAR']:
            signals.append("Above Parabolic SAR - Bullish")
            signal_strength += 1
        else:
            signals.append("Below Parabolic SAR - Bearish")
            signal_strength -= 1
        
        # Ichimoku signals
        if (latest['Close'] > latest['Ichimoku_SpanA'] and 
            latest['Close'] > latest['Ichimoku_SpanB']):
            signals.append("Above Ichimoku Cloud - Strong Bullish")
            signal_strength += 2
        elif (latest['Close'] < latest['Ichimoku_SpanA'] and 
              latest['Close'] < latest['Ichimoku_SpanB']):
            signals.append("Below Ichimoku Cloud - Strong Bearish")
            signal_strength -= 2
        
        # Volume confirmation
        if latest['OBV'] > df['OBV'].rolling(20).mean().iloc[-1]:
            signals.append("Volume Supports Trend")
            signal_strength += 1
        
        # Market regime adjustment
        regime_multiplier = 1.0
        if latest['Market_Regime'] == 'Bull':
            regime_multiplier = 1.2
            signals.append("Bull Market Regime")
        elif latest['Market_Regime'] == 'Bear':
            regime_multiplier = 0.8
            signals.append("Bear Market Regime")
        else:
            signals.append("Sideways Market Regime")
        
        # Final signal calculation
        adjusted_strength = signal_strength * regime_multiplier
        
        if adjusted_strength >= 3:
            recommendation = "STRONG BUY"
        elif adjusted_strength >= 1:
            recommendation = "BUY"
        elif adjusted_strength <= -3:
            recommendation = "STRONG SELL"
        elif adjusted_strength <= -1:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
        
        # Risk metrics
        returns = df['Close'].pct_change().dropna()
        if len(returns) > 30:
            sharpe = self.quant.calculate_sharpe_ratio(returns)
            sortino = self.quant.calculate_sortino_ratio(returns)
            max_dd = self.quant.calculate_maximum_drawdown(df['Close'])
            var_5 = self.quant.calculate_var(returns, 0.05)
        else:
            sharpe = sortino = var_5 = 0
            max_dd = {'max_drawdown': 0, 'max_drawdown_duration': 0}
        
        return {
            'recommendation': recommendation,
            'signal_strength': adjusted_strength,
            'signals': signals,
            'market_regime': latest['Market_Regime'],
            'risk_metrics': {
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown': max_dd['max_drawdown'],
                'var_5_percent': var_5,
                'current_volatility': returns.tail(20).std() * np.sqrt(252) if len(returns) > 20 else 0
            },
            'technical_levels': {
                'support': min(latest['BB_Lower'], latest['KC_Lower']),
                'resistance': max(latest['BB_Upper'], latest['KC_Upper']),
                'stop_loss': latest['PSAR'],
                'target': latest['VWAP']
            }
        }

# Global instances
advanced_indicators = AdvancedIndicators()
smart_signals = SmartSignalGenerator()
