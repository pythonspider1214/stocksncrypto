import numpy as np
import pandas as pd
import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime

# Try to import ML libraries, provide fallbacks if not available
try:
    from joblib import load as joblib_load, dump as joblib_dump
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logging.warning("Joblib not available, ML features disabled")

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available, ML training disabled")

from utils import safe_yf_download

class MLModelManager:
    """Machine learning model management"""
    
    def __init__(self, model_path: str = 'ml_model.joblib'):
        self.model_path = model_path
        self.model = None
        self.logger = logging.getLogger(__name__)
        if JOBLIB_AVAILABLE:
            self.load_model()
    
    def load_model(self) -> Optional[Any]:
        """Load ML model"""
        if not JOBLIB_AVAILABLE:
            self.logger.warning("Joblib not available, cannot load ML model")
            return None
            
        try:
            if os.path.exists(self.model_path):
                self.model = joblib_load(self.model_path)
                self.logger.info(f"ML model loaded from {self.model_path}")
                return self.model
            else:
                self.logger.info(f"No ML model found at {self.model_path}")
                return None
        except Exception as e:
            self.logger.warning(f"Could not load ML model: {e}")
            return None
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for ML model"""
        features = {}
        if 'Close' in df.columns and len(df) > 20:
            close = df['Close']
            features['Close'] = close.iloc[-1]
            
            # SMA calculation
            sma20 = close.rolling(window=20).mean()
            features['sma20'] = sma20.iloc[-1] if not pd.isna(sma20.iloc[-1]) else close.iloc[-1]
            
            # RSI calculation
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features['rsi14'] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            
            # Additional features
            return_5d = close.pct_change(5)
            features['return_5d'] = return_5d.iloc[-1] if not pd.isna(return_5d.iloc[-1]) else 0
            
            volatility_5d = close.pct_change().rolling(5).std()
            features['volatility_5d'] = volatility_5d.iloc[-1] if not pd.isna(volatility_5d.iloc[-1]) else 0
        else:
            # Return default features if insufficient data
            features = {
                'Close': 100,
                'sma20': 100,
                'rsi14': 50,
                'return_5d': 0,
                'volatility_5d': 0.01
            }
        
        return pd.DataFrame([features])
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[int]:
        """Generate ML trading signal"""
        if self.model is None:
            # Return simple rule-based signal as fallback
            return self._simple_rule_based_signal(df)
        
        try:
            features = self.extract_features(df)
            if features.empty:
                return None
            
            prediction = self.model.predict(features)[0]
            return int(prediction)  # 1=Buy, 0=Hold, -1=Sell
        except Exception as e:
            self.logger.error(f"ML prediction error: {e}")
            return self._simple_rule_based_signal(df)
    
    def _simple_rule_based_signal(self, df: pd.DataFrame) -> int:
        """Simple rule-based signal as fallback"""
        try:
            if 'Close' not in df.columns or len(df) < 20:
                return 0  # Hold
            
            close = df['Close']
            sma20 = close.rolling(window=20).mean()
            
            current_price = close.iloc[-1]
            current_sma = sma20.iloc[-1]
            
            if pd.isna(current_sma):
                return 0
            
            # Simple rule: Buy if price > SMA, Sell if price < SMA * 0.95
            if current_price > current_sma:
                return 1  # Buy
            elif current_price < current_sma * 0.95:
                return -1  # Sell
            else:
                return 0  # Hold
                
        except Exception as e:
            self.logger.error(f"Error in rule-based signal: {e}")
            return 0
    
    def train_model(self, symbol: str = 'AAPL') -> bool:
        """Train a simple ML model"""
        if not SKLEARN_AVAILABLE or not JOBLIB_AVAILABLE:
            self.logger.warning("ML libraries not available, cannot train model")
            return False
        
        try:
            self.logger.info(f"Training ML model with {symbol} data...")
            
            # Download historical data with auto_adjust=False to avoid the warning
            df = safe_yf_download(symbol, start='2020-01-01', end=datetime.now().strftime('%Y-%m-%d'), auto_adjust=False)
            if df is None or len(df) < 100:
                self.logger.error("Insufficient data for training")
                return False
            
            # Debug: Print column information
            self.logger.info(f"DataFrame shape: {df.shape}")
            self.logger.info(f"DataFrame columns: {list(df.columns)}")
            self.logger.info(f"DataFrame index: {df.index.name}")
            
            # Reset index if needed
            if df.index.name == 'Date':
                df = df.reset_index()
            
            # Handle MultiIndex columns properly
            if isinstance(df.columns, pd.MultiIndex):
                self.logger.info("Found MultiIndex columns, handling properly...")
                
                # For yfinance data, we typically have ('Close', 'AAPL') format
                # Let's extract just the first level (Close, Open, High, etc.)
                new_columns = []
                for col in df.columns:
                    if isinstance(col, tuple):
                        # Take the first part of the tuple (e.g., 'Close' from ('Close', 'AAPL'))
                        new_columns.append(col[0])
                    else:
                        new_columns.append(str(col))
                
                df.columns = new_columns
                self.logger.info(f"Simplified columns: {list(df.columns)}")
            
            # Find close column
            close_col = self._find_close_column(df)
            
            if close_col is None:
                self.logger.error(f"No close price column found. Available columns: {list(df.columns)}")
                return False
            
            self.logger.info(f"Using column '{close_col}' as Close price")
            df['Close'] = df[close_col]
            
            # Verify we have numeric data
            if not pd.api.types.is_numeric_dtype(df['Close']):
                self.logger.error(f"Close column '{close_col}' is not numeric")
                return False
            
            self.logger.info(f"Close price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
            
            # Calculate features
            df = self._calculate_features(df)
            df = self._create_labels(df)
            
            # Remove any rows with NaN values
            df_clean = df.dropna()
            
            if len(df_clean) < 50:
                self.logger.error(f"Insufficient clean data for training. Had {len(df)} rows, after cleaning: {len(df_clean)}")
                return False
            
            # Prepare training data
            feature_cols = ['Close', 'sma20', 'rsi14', 'return_5d', 'volatility_5d']
            
            # Ensure all feature columns exist
            missing_cols = [col for col in feature_cols if col not in df_clean.columns]
            if missing_cols:
                self.logger.error(f"Missing feature columns: {missing_cols}")
                return False
            
            X = df_clean[feature_cols].copy()
            y = df_clean['label'].copy()
            
            # Final check for data quality
            if len(X) == 0 or len(y) == 0:
                self.logger.error("No valid training data after preprocessing")
                return False
            
            # Check for any remaining NaN values
            if X.isnull().any().any():
                self.logger.error("Found NaN values in features")
                self.logger.error(f"NaN counts: {X.isnull().sum()}")
                return False
                
            if y.isnull().any():
                self.logger.error("Found NaN values in labels")
                return False
            
            # Check label distribution
            label_counts = y.value_counts()
            self.logger.info(f"Label distribution: {dict(label_counts)}")
            
            self.logger.info(f"Training with {len(X)} samples and {len(X.columns)} features")
            
            # Train model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            
            # Save model
            joblib_dump(self.model, self.model_path)
            self.logger.info(f"Model trained and saved to {self.model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _find_close_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the close price column using multiple strategies"""
        
        # Strategy 1: Direct match
        if 'Close' in df.columns:
            return 'Close'
        
        # Strategy 2: Case insensitive search
        for col in df.columns:
            if str(col).lower() == 'close':
                return col
        
        # Strategy 3: Search for patterns in column names
        close_patterns = ['close', 'Close', 'CLOSE']
        for pattern in close_patterns:
            for col in df.columns:
                if pattern == str(col):
                    return col
        
        # Strategy 4: Partial match
        for col in df.columns:
            col_str = str(col).lower()
            if 'close' in col_str:
                return col
        
        return None
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features"""
        try:
            close = df['Close']
            
            # SMA calculation
            df['sma20'] = close.rolling(20).mean()
            
            # RSI calculation
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            # Avoid division by zero
            rs = gain / loss.replace(0, np.nan)
            df['rsi14'] = 100 - (100 / (1 + rs))
            
            # Additional features
            df['return_5d'] = close.pct_change(5)
            df['volatility_5d'] = close.pct_change().rolling(5).std()
            
            self.logger.info("Successfully calculated technical features")
            return df
        except Exception as e:
            self.logger.error(f"Error calculating features: {e}")
            return df
    
    def _create_labels(self, df: pd.DataFrame, threshold: float = 0.02) -> pd.DataFrame:
        """Create labels for supervised learning"""
        try:
            # Calculate future returns
            future_return = df['Close'].shift(-5) / df['Close'] - 1
            
            # Create labels based on future returns
            df['label'] = 0  # Hold (default)
            
            # Use .loc to avoid the ambiguous truth value error
            buy_condition = future_return > threshold
            sell_condition = future_return < -threshold
            
            df.loc[buy_condition, 'label'] = 1    # Buy
            df.loc[sell_condition, 'label'] = -1  # Sell
            
            self.logger.info("Successfully created labels")
            return df
        except Exception as e:
            self.logger.error(f"Error creating labels: {e}")
            return df

# Global ML model manager
ml_manager = MLModelManager()
