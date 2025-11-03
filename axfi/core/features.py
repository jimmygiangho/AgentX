"""
Feature Engineering Module
Produces ML-ready feature vectors with normalization and derived features
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for AXFI models.
    Converts technical indicators into model-ready feature vectors with normalization.
    
    Features:
    - Derived features from indicators
    - Cross-sectional normalization (z-scores, rank percentiles)
    - Regime detection
    - Multi-timeframe features
    """
    
    def __init__(self, config: dict):
        """
        Initialize feature engineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_names = []
    
    def create_features(self, df: pd.DataFrame, 
                       lookback_windows: List[int] = [3, 7, 21],
                       universe_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create feature vectors from OHLCV and indicators with derived features.
        
        Args:
            df: DataFrame with OHLCV and indicators
            lookback_windows: Windows for rolling features
            universe_data: Optional universe snapshot for normalization
            
        Returns:
            DataFrame with feature vectors
        """
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return df
        
        features_df = df.copy()
        
        # Derived features from indicators
        features_df = self._add_derived_features(features_df)
        
        # Rolling returns and volatility
        for window in lookback_windows:
            features_df[f'return_{window}d'] = features_df['close'].pct_change(window)
            features_df[f'volatility_{window}d'] = features_df['close'].pct_change().rolling(window).std()
        
        # Momentum score (weighted returns)
        features_df['momentum_score'] = self._calculate_momentum_score(features_df, windows=[3, 7, 21])
        
        # Volume features
        if 'volume' in features_df.columns:
            features_df['volume_spike'] = features_df['volume'] / features_df['volume'].rolling(20).mean()
            features_df['liquidity_score'] = features_df['volume'].rolling(20).mean() / features_df['close']
        
        # Regime detection
        features_df = self._detect_regimes(features_df)
        
        # Normalization (if universe data provided)
        if universe_data is not None:
            features_df = self._add_normalized_features(features_df, universe_data)
        
        # Add to feature names list
        base_cols = ['open', 'high', 'low', 'close', 'volume', 'date', 'symbol', 'adj_close']
        self.feature_names = [col for col in features_df.columns if col not in base_cols]
        
        logger.info(f"Created {len(self.feature_names)} features")
        
        return features_df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features from indicators.
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            DataFrame with derived features added
        """
        result_df = df.copy()
        
        # EMA slope (delta over 7 and 21 days)
        for ema_period in [10, 20, 50, 100, 200]:
            if f'ema_{ema_period}' in result_df.columns:
                result_df[f'ema_{ema_period}_slope_7d'] = result_df[f'ema_{ema_period}'].diff(7)
                result_df[f'ema_{ema_period}_slope_21d'] = result_df[f'ema_{ema_period}'].diff(21)
        
        # EMA gap (short - long) / price
        if 'ema_10' in result_df.columns and 'ema_50' in result_df.columns:
            result_df['ema_gap_10_50'] = (result_df['ema_10'] - result_df['ema_50']) / result_df['close']
        if 'ema_20' in result_df.columns and 'ema_200' in result_df.columns:
            result_df['ema_gap_20_200'] = (result_df['ema_20'] - result_df['ema_200']) / result_df['close']
        
        # Price to BB width (position within bands)
        for bb_period in [20, 50]:
            bb_width_col = f'bb_{bb_period}_width'
            bb_middle_col = f'bb_{bb_period}_middle'
            if bb_width_col in result_df.columns and bb_middle_col in result_df.columns:
                width = result_df[bb_width_col].replace(0, np.nan)
                result_df[f'price_to_bb_{bb_period}_width'] = (
                    (result_df['close'] - result_df[bb_middle_col]) / width
                )
        
        # Volatility ratio: ATR / 30-day historical volatility
        if 'atr_14' in result_df.columns:
            hist_vol_30d = result_df['close'].pct_change().rolling(30).std()
            hist_vol_30d = hist_vol_30d.replace(0, np.nan)
            result_df['volatility_ratio'] = result_df['atr_14'] / (result_df['close'] * hist_vol_30d)
        
        # MACD signals
        if 'macd' in result_df.columns and 'macd_signal' in result_df.columns:
            result_df['macd_cross'] = np.where(
                (result_df['macd'] > result_df['macd_signal']) & 
                (result_df['macd'].shift(1) <= result_df['macd_signal'].shift(1)),
                1, np.where(
                    (result_df['macd'] < result_df['macd_signal']) & 
                    (result_df['macd'].shift(1) >= result_df['macd_signal'].shift(1)),
                    -1, 0
                )
            )
        
        # RSI signals
        for rsi_col in [c for c in result_df.columns if c.startswith('rsi')]:
            period = rsi_col.split('_')[-1] if '_' in rsi_col else '14'
            result_df[f'rsi_{period}_signal'] = np.where(
                result_df[rsi_col] < 30, 1,
                np.where(result_df[rsi_col] > 70, -1, 0)
            )
        
        return result_df
    
    def _calculate_momentum_score(self, df: pd.DataFrame, windows: List[int] = [3, 7, 21]) -> pd.Series:
        """
        Calculate weighted momentum score.
        
        Args:
            df: DataFrame with price data
            windows: Windows for momentum calculation
            
        Returns:
            Momentum score series
        """
        weights = np.array([0.5, 0.3, 0.2])  # More weight to shorter-term
        if len(weights) != len(windows):
            weights = np.ones(len(windows)) / len(windows)
        
        momentum_values = []
        for window in windows:
            momentum = df['close'].pct_change(window)
            momentum_values.append(momentum)
        
        momentum_df = pd.DataFrame(momentum_values).T
        momentum_score = (momentum_df * weights).sum(axis=1)
        
        return momentum_score
    
    def _detect_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regimes: high volatility, bull trend, sideways.
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            DataFrame with regime indicators added
        """
        result_df = df.copy()
        
        # Volatility regime
        vol_30d = result_df['close'].pct_change().rolling(30).std()
        vol_percentile_30d = vol_30d.rolling(252).rank(pct=True) * 100  # Percentile over 1 year
        
        result_df['high_vol'] = (
            (vol_percentile_30d > 80) & 
            (result_df.get('adx', pd.Series(0, index=result_df.index)) > 25)
        ).astype(int)
        
        # Bull trend
        ema_20_slope = result_df.get('ema_20', result_df['close'].rolling(20).mean()).diff()
        result_df['bull_trend'] = (
            (result_df.get('adx', 0) > 25) &
            (result_df.get('+di', 0) > result_df.get('-di', 0)) &
            (ema_20_slope > 0)
        ).astype(int)
        
        # Sideways
        result_df['sideways'] = (
            result_df.get('adx', 0) < 15
        ).astype(int)
        
        return result_df
    
    def _add_normalized_features(self, df: pd.DataFrame, universe_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add cross-sectional normalized features (z-scores and rank percentiles).
        
        Args:
            df: Single symbol DataFrame
            universe_data: Universe snapshot (all symbols for same date)
            
        Returns:
            DataFrame with normalized features added
        """
        result_df = df.copy()
        
        # Get latest date from df
        if 'date' in result_df.columns:
            latest_date = result_df['date'].max()
        else:
            latest_date = result_df.index.max() if isinstance(result_df.index, pd.DatetimeIndex) else None
        
        if latest_date is None:
            logger.warning("Could not determine date for normalization")
            return result_df
        
        # Filter universe to same date
        if 'date' in universe_data.columns:
            universe_snapshot = universe_data[universe_data['date'] == latest_date].copy()
        else:
            universe_snapshot = universe_data.loc[universe_data.index == latest_date].copy()
        
        if universe_snapshot.empty:
            logger.warning(f"No universe data for date {latest_date}")
            return result_df
        
        # Get latest row from df
        if 'date' in result_df.columns:
            df_latest = result_df[result_df['date'] == latest_date].iloc[-1:]
        else:
            df_latest = result_df.iloc[-1:]
        
        # Normalize numeric features
        numeric_features = [col for col in self.feature_names 
                          if col in universe_snapshot.columns and 
                          pd.api.types.is_numeric_dtype(universe_snapshot[col])]
        
        for feature in numeric_features:
            if feature in df_latest.columns and feature in universe_snapshot.columns:
                # Calculate z-score across universe
                universe_values = universe_snapshot[feature].dropna()
                if len(universe_values) > 1:
                    mean_val = universe_values.mean()
                    std_val = universe_values.std()
                    
                    if std_val > 0:
                        zscore = (df_latest[feature].iloc[0] - mean_val) / std_val
                        result_df.loc[df_latest.index, f'{feature}_zscore'] = zscore
                    
                    # Rank percentile
                    rank_pct = (universe_values < df_latest[feature].iloc[0]).sum() / len(universe_values) * 100
                    result_df.loc[df_latest.index, f'{feature}_rank_pct'] = rank_pct
        
        return result_df
    
    def normalize_universe(self, universe_features: Dict[str, pd.DataFrame], 
                          date: Optional[str] = None) -> pd.DataFrame:
        """
        Normalize features across universe for a given date.
        
        Args:
            universe_features: Dict of symbol -> features DataFrame
            date: Date to normalize (if None, uses latest)
            
        Returns:
            DataFrame with normalized features
        """
        # Get snapshot for all symbols at given date
        snapshots = []
        
        for symbol, df in universe_features.items():
            if 'date' in df.columns:
                if date:
                    snapshot = df[df['date'] == date].copy()
                else:
                    snapshot = df.iloc[-1:].copy()
            else:
                if date and isinstance(df.index, pd.DatetimeIndex):
                    snapshot = df.loc[df.index == pd.to_datetime(date)].copy()
                else:
                    snapshot = df.iloc[-1:].copy()
            
            if not snapshot.empty:
                snapshot['symbol'] = symbol
                snapshots.append(snapshot)
        
        if not snapshots:
            logger.warning("No snapshots found for normalization")
            return pd.DataFrame()
        
        universe_snapshot = pd.concat(snapshots, ignore_index=True)
        
        # Calculate z-scores and rank percentiles for each numeric feature
        numeric_cols = universe_snapshot.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ['symbol']]
        
        for col in numeric_cols:
            values = universe_snapshot[col].dropna()
            if len(values) > 1 and values.std() > 0:
                mean_val = values.mean()
                std_val = values.std()
                universe_snapshot[f'{col}_zscore'] = (values - mean_val) / std_val
                universe_snapshot[f'{col}_rank_pct'] = values.rank(pct=True) * 100
        
        return universe_snapshot
    
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
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Remove raw OHLCV
        to_remove = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
        to_remove = [col for col in to_remove if col in numeric_df.columns]
        numeric_df = numeric_df.drop(columns=to_remove, errors='ignore')
        
        return numeric_df.fillna(0)
    
    def get_top_features(self, df: pd.DataFrame, feature_scores: Dict[str, float], 
                        top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Get top N features by importance.
        
        Args:
            df: Features DataFrame
            feature_scores: Dictionary of feature -> importance score
            top_n: Number of top features to return
            
        Returns:
            List of (feature_name, score) tuples
        """
        sorted_features = sorted(feature_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        return sorted_features[:top_n]


if __name__ == "__main__":
    # Test feature engineering
    config = {"test": True}
    engineer = FeatureEngineer(config)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    prices = 100 + np.random.randn(100).cumsum()
    
    sample_df = pd.DataFrame({
        'date': dates,
        'open': prices + np.random.randn(100) * 0.5,
        'high': prices + abs(np.random.randn(100) * 1),
        'low': prices - abs(np.random.randn(100) * 1),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100),
        'ema_10': prices * 1.01,
        'ema_20': prices * 1.005,
        'ema_50': prices * 0.995,
        'rsi_14': 50 + np.random.randn(100).cumsum() / 5,
        'macd': np.random.randn(100) / 10,
        'macd_signal': np.random.randn(100) / 10,
        'atr_14': np.abs(np.random.randn(100)) + 1,
        'adx': 20 + np.abs(np.random.randn(100)),
        '+di': 25 + np.abs(np.random.randn(100)),
        '-di': 20 + np.abs(np.random.randn(100)),
    })
    
    # Add indicators from research library
    from core.research_library import ResearchLibrary
    library = ResearchLibrary(config)
    sample_df = sample_df.set_index('date')
    sample_df = library.calculate_all_indicators(sample_df)
    sample_df = sample_df.reset_index()
    
    # Create features
    features_df = engineer.create_features(sample_df)
    
    print(f"Created {len(engineer.feature_names)} features")
    print(f"\nSample feature names: {engineer.feature_names[:15]}")
    
    numeric_features = engineer.get_numeric_features(features_df)
    print(f"\nNumeric features shape: {numeric_features.shape}")
    print(f"\nSample derived features:")
    derived = [c for c in features_df.columns if any(x in c for x in ['slope', 'gap', 'ratio', 'spike', 'score', 'regime'])]
    print(derived[:10])
