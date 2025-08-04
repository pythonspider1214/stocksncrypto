#!/usr/bin/env python3
"""
Machine Learning Pipeline for Financial Analysis Bot
Implements advanced ML models for price prediction and risk assessment
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import joblib
import json
import os
from decimal import Decimal

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Deep Learning (if available)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available, LSTM models disabled")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """ML model configuration"""
    name: str
    model_type: str
    target_variable: str
    features: List[str]
    hyperparameters: Dict[str, Any]
    lookback_window: int = 60
    prediction_horizon: int = 1

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    mse: float
    mae: float
    r2: float
    accuracy: float
    sharpe_ratio: float
    max_drawdown: float

class FeatureEngineer:
    """Feature engineering for financial data"""
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators as features"""
        df = df.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_change'] = df['close'] - df['open']
        df['price_range'] = df['high'] - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            df[f'price_to_sma_{window}'] = df['close'] / df[f'sma_{window}']
        
        # Volatility features
        df['volatility_5'] = df['returns'].rolling(window=5).std()
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_10']
        df['price_volume'] = df['close'] * df['volume']
        
        # Momentum features
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # Support and resistance levels
        df['resistance'] = df['high'].rolling(window=20).max()
        df['support'] = df['low'].rolling(window=20).min()
        df['distance_to_resistance'] = (df['resistance'] - df['close']) / df['close']
        df['distance_to_support'] = (df['close'] - df['support']) / df['close']
        
        return df
    
    @staticmethod
    def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features"""
        df = df.copy()
        
        # Trend strength
        df['trend_strength'] = df['close'].rolling(window=20).apply(
            lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1]
        )
        
        # Market regime (bull/bear/sideways)
        returns_20 = df['returns'].rolling(window=20).mean()
        volatility_20 = df['returns'].rolling(window=20).std()
        
        df['bull_market'] = (returns_20 > 0.001) & (volatility_20 < 0.02)
        df['bear_market'] = (returns_20 < -0.001) & (volatility_20 > 0.02)
        df['sideways_market'] = ~(df['bull_market'] | df['bear_market'])
        
        # VIX-like volatility index
        df['vix_proxy'] = df['volatility_20'] * 100
        
        return df
    
    @staticmethod
    def create_sequences(data: np.ndarray, lookback: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        X, y = [], []
        for i in range(lookback, len(data) - horizon + 1):
            X.append(data[i-lookback:i])
            y.append(data[i:i+horizon])
        return np.array(X), np.array(y)

class LSTMModel:
    """LSTM model for time series prediction"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            Dense(50, activation='relu'),
            Dropout(0.1),
            Dense(self.config.prediction_horizon)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Train LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow not available for LSTM training")
        
        # Scale data
        X_scaled = self.scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).reshape(y.shape)
        
        # Split data
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
        
        # Build model
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        y_pred = self.model.predict(X_val)
        y_pred_unscaled = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
        y_val_unscaled = self.scaler_y.inverse_transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
        
        performance = self._calculate_performance(y_val_unscaled, y_pred_unscaled)
        logger.info(f"LSTM model trained - MSE: {performance.mse:.6f}, R2: {performance.r2:.4f}")
        
        return performance
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)
        
        return y_pred
    
    def _calculate_performance(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelPerformance:
        """Calculate model performance metrics"""
        mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
        mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        r2 = r2_score(y_true.flatten(), y_pred.flatten())
        
        # Financial metrics
        returns_true = np.diff(y_true.flatten()) / y_true.flatten()[:-1]
        returns_pred = np.diff(y_pred.flatten()) / y_pred.flatten()[:-1]
        
        # Direction accuracy
        direction_true = np.sign(returns_true)
        direction_pred = np.sign(returns_pred)
        accuracy = np.mean(direction_true == direction_pred)
        
        # Sharpe ratio
        sharpe_ratio = np.mean(returns_pred) / np.std(returns_pred) * np.sqrt(252) if np.std(returns_pred) > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns_pred)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return ModelPerformance(
            mse=mse,
            mae=mae,
            r2=r2,
            accuracy=accuracy,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown
        )

class XGBoostModel:
    """XGBoost model for financial prediction"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
    
    def train(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Train XGBoost model"""
        # Flatten sequences for XGBoost
        X_flat = X.reshape(X.shape[0], -1)
        y_flat = y.flatten() if len(y.shape) > 1 else y
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_flat)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_flat, test_size=0.2, shuffle=False
        )
        
        # Train model
        self.model = xgb.XGBRegressor(
            **self.config.hyperparameters,
            random_state=42
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Evaluate
        y_pred = self.model.predict(X_val)
        performance = self._calculate_performance(y_val, y_pred)
        
        logger.info(f"XGBoost model trained - MSE: {performance.mse:.6f}, R2: {performance.r2:.4f}")
        return performance
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        X_flat = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.transform(X_flat)
        return self.model.predict(X_scaled)
    
    def _calculate_performance(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelPerformance:
        """Calculate performance metrics"""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Direction accuracy
        direction_true = np.sign(np.diff(y_true))
        direction_pred = np.sign(np.diff(y_pred))
        accuracy = np.mean(direction_true == direction_pred) if len(direction_true) > 0 else 0
        
        return ModelPerformance(
            mse=mse,
            mae=mae,
            r2=r2,
            accuracy=accuracy,
            sharpe_ratio=0,  # Simplified for XGBoost
            max_drawdown=0
        )

class RiskAnalyzer:
    """Risk analysis and portfolio optimization"""
    
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk"""
        var = RiskAnalyzer.calculate_var(returns, confidence_level)
        return np.mean(returns[returns <= var])
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * np.mean(excess_returns) / np.std(returns) if np.std(returns) > 0 else 0
    
    @staticmethod
    def calculate_maximum_drawdown(prices: np.ndarray) -> Dict[str, float]:
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + np.diff(prices) / prices[:-1])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)
        
        # Find duration
        peak_idx = np.argmax(running_max[:max_dd_idx]) if max_dd_idx > 0 else 0
        duration = max_dd_idx - peak_idx
        
        return {
            'max_drawdown': max_dd,
            'duration': duration,
            'peak_idx': peak_idx,
            'trough_idx': max_dd_idx
        }

class MLPipeline:
    """Main ML pipeline orchestrator"""
    
    def __init__(self):
        self.models = {}
        self.feature_engineer = FeatureEngineer()
        self.risk_analyzer = RiskAnalyzer()
    
    def prepare_data(self, df: pd.DataFrame, config: ModelConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        # Add features
        df_features = self.feature_engineer.add_technical_indicators(df)
        df_features = self.feature_engineer.add_market_regime_features(df_features)
        
        # Select features
        feature_cols = [col for col in config.features if col in df_features.columns]
        target_col = config.target_variable
        
        # Drop NaN values
        df_clean = df_features[feature_cols + [target_col]].dropna()
        
        # Create sequences
        X_data = df_clean[feature_cols].values
        y_data = df_clean[target_col].values
        
        X, y = self.feature_engineer.create_sequences(
            np.column_stack([X_data, y_data]),
            config.lookback_window,
            config.prediction_horizon
        )
        
        # Separate features and target
        X_features = X[:, :, :-1]  # All columns except last (target)
        y_target = X[:, -config.prediction_horizon:, -1]  # Last column, last horizon steps
        
        return X_features, y_target
    
    def train_model(self, df: pd.DataFrame, config: ModelConfig) -> ModelPerformance:
        """Train a model with given configuration"""
        logger.info(f"Training {config.model_type} model: {config.name}")
        
        # Prepare data
        X, y = self.prepare_data(df, config)
        
        # Initialize model
        if config.model_type == 'lstm' and TENSORFLOW_AVAILABLE:
            model = LSTMModel(config)
        elif config.model_type == 'xgboost':
            model = XGBoostModel(config)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        
        # Train model
        performance = model.train(X, y)
        
        # Store model
        self.models[config.name] = model
        
        return performance
    
    def generate_predictions(self, df: pd.DataFrame, model_name: str, config: ModelConfig) -> Dict[str, Any]:
        """Generate predictions using trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Prepare recent data
        X, _ = self.prepare_data(df, config)
        
        # Make prediction
        prediction = model.predict(X[-1:])  # Last sequence
        
        # Calculate confidence (simplified)
        confidence = 0.7  # Placeholder - implement proper confidence calculation
        
        return {
            'prediction': float(prediction[0]) if len(prediction.shape) == 1 else float(prediction[0][0]),
            'confidence': confidence,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat()
        }
    
    def analyze_portfolio_risk(self, portfolio_returns: np.ndarray) -> Dict[str, float]:
        """Analyze portfolio risk metrics"""
        return {
            'var_5': self.risk_analyzer.calculate_var(portfolio_returns, 0.05),
            'cvar_5': self.risk_analyzer.calculate_cvar(portfolio_returns, 0.05),
            'sharpe_ratio': self.risk_analyzer.calculate_sharpe_ratio(portfolio_returns),
            'max_drawdown': self.risk_analyzer.calculate_maximum_drawdown(
                np.cumprod(1 + portfolio_returns)
            )['max_drawdown'],
            'volatility': np.std(portfolio_returns) * np.sqrt(252)
        }

async def main():
    """Test the ML pipeline"""
    logger.info("Testing ML pipeline...")
    
    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Simulate price data
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = 100 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
        'close': prices,
        'volume': np.random.lognormal(15, 1, len(dates))
    })
    
    # Initialize pipeline
    pipeline = MLPipeline()
    
    # Model configurations
    configs = [
        ModelConfig(
            name='LSTM_Price_Predictor',
            model_type='lstm' if TENSORFLOW_AVAILABLE else 'xgboost',
            target_variable='close',
            features=['close', 'volume', 'returns', 'rsi', 'macd', 'bb_position'],
            hyperparameters={},
            lookback_window=30,
            prediction_horizon=1
        ),
        ModelConfig(
            name='XGBoost_Direction_Predictor',
            model_type='xgboost',
            target_variable='returns',
            features=['rsi', 'macd', 'bb_position', 'volume_ratio', 'momentum_5'],
            hyperparameters={
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1
            },
            lookback_window=20,
            prediction_horizon=1
        )
    ]
    
    # Train models
    for config in configs:
        try:
            performance = pipeline.train_model(df, config)
            logger.info(f"Model {config.name} performance: {performance}")
            
            # Generate sample prediction
            prediction = pipeline.generate_predictions(df, config.name, config)
            logger.info(f"Sample prediction: {prediction}")
            
        except Exception as e:
            logger.error(f"Error training {config.name}: {e}")
    
    # Risk analysis
    sample_returns = np.random.normal(0.001, 0.02, 252)  # 1 year of daily returns
    risk_metrics = pipeline.analyze_portfolio_risk(sample_returns)
    logger.info(f"Portfolio risk metrics: {risk_metrics}")
    
    logger.info("ML pipeline test completed")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
