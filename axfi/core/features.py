"""
Feature Engineering Module
Produces ML-ready feature vectors for models
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for AXFI models.
    Converts technical indicators into model-ready feature vectors.
    """
    
    def __init__(self, config: dict):
        """
        Initialize feature engineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_names = []
    
    def create_features(self, df: pd.DataFrame, lookback_windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """
        Create feature vectors from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV and indicators
            lookback_windows: Windows for rolling features
            
        Returns:
            DataFrame with feature vectors
        """
        features_df = df.copy()
        
        # Rolling returns
        for window in lookback_windows:
            features_df[f'return_{window}d'] = features_df['close'].pct_change(window)
            features_df[f'volatility_{window}d'] = features_df['close'].pct_change().rolling(window).std()
        
        # Price momentum features
        features_df['momentum_5'] = features_df['close'] / features_df['close'].shift(5) - 1
        features_df['momentum_10'] = features_df['close'] / features_df['close'].shift(10) - 1
        features_df['momentum_20'] = features_df['close'] / features_df['close'].shift(20) - 1
        
        # Volatility features
        features_df['atr_pct'] = features_df.get('atr', 0) / features_df['close']
        features_df['bb_width'] = (features_df.get('bb_upper', 0) - features_df.get('bb_lower', 0)) / features_df['close']
        
        # Volume features
        if 'volume' in features_df.columns:
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume'].rolling(20).mean()
            features_df['volume_sma'] = features_df['volume'].rolling(20).mean()
        
        # Technical indicator features (if present)
        if 'rsi' in features_df.columns:
            features_df['rsi_signal'] = np.where(features_df['rsi'] < 30, 1,
                                                np.where(features_df['rsi'] > 70, -1, 0))
        
        if 'macd' in features_df.columns:
            features_df['macd_signal'] = np.where(features_df['macd'] > features_df['macd_signal'], 1, -1)
        
        if 'bb_upper' in features_df.columns and 'bb_lower' in features_df.columns:
            features_df['bb_position'] = (features_df['close'] - features_df['bb_lower']) / \
                                        (features_df['bb_upper'] - features_df['bb_lower'])
        
        # Trend strength
        if 'adx' in features_df.columns:
            features_df['trend_strength'] = features_df['adx']
        
        # Regime detection
        features_df['volatility_regime'] = self._detect_volatility_regime(features_df)
        features_df['trend_regime'] = self._detect_trend_regime(features_df)
        
        # Add to feature names list
        self.feature_names = [col for col in features_df.columns if col not in 
                             ['open', 'high', 'low', 'close', 'volume', 'date', 'symbol', 'adj_close']]
        
        logger.info(f"Created {len(self.feature_names)} features")
        
        return features_df
    
    def _detect_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect high/low volatility regime"""
        volatility = df['close'].pct_change().rolling(20).std()
        median_vol = volatility.median()
        return np.where(volatility > median_vol * 1.5, 'high_vol', 
                       np.where(volatility < median_vol * 0.5, 'low_vol', 'normal'))
    
    def _detect_trend_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect trending vs choppy regime"""
        # Use ADX if available, otherwise use price momentum
        if 'adx' in df.columns:
            adx = df['adx']
            return np.where(adx > 25, 'trending', 
                           np.where(adx < 20, 'choppy', 'neutral'))
        else:
            # Fallback: use slope of moving average
            ma_slope = df['close'].rolling(20).mean().diff()
            slope_std = ma_slope.rolling(20).std()
            return np.where(abs(ma_slope) > slope_std, 'trending', 'choppy')
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names
    
    def get_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract only numeric features for ML models.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with only numeric features
        """
        # Remove non-numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Remove raw OHLCV (keep only engineered features)
        to_remove = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
        to_remove = [col for col in to_remove if col in numeric_df.columns]
        numeric_df = numeric_df.drop(columns=to_remove)
        
        return numeric_df.fillna(0)


if __name__ == "__main__":
    # Test feature engineering
    config = {"test": True}
    engineer = FeatureEngineer(config)
    
    # Create sample data
    sample_df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    
    # Add some indicators
    sample_df['rsi'] = 50 + np.random.randn(100).cumsum() / 10
    sample_df['macd'] = np.random.randn(100) / 10
    sample_df['macd_signal'] = np.random.randn(100) / 10
    sample_df['bb_upper'] = sample_df['close'] * 1.02
    sample_df['bb_lower'] = sample_df['close'] * 0.98
    sample_df['atr'] = np.abs(np.random.randn(100)) + 1
    
    # Create features
    features_df = engineer.create_features(sample_df)
    
    print(f"Created {len(engineer.feature_names)} features")
    print(f"Feature names: {engineer.feature_names[:10]}...")
    
    numeric_features = engineer.get_numeric_features(features_df)
    print(f"\nNumeric features shape: {numeric_features.shape}")
