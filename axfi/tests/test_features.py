"""
Unit tests for feature engineering and normalization
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.features import FeatureEngineer
from core.research_library import ResearchLibrary


class TestFeatures(unittest.TestCase):
    """Test feature engineering"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        # Create sample data with indicators
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        prices = 150 * (1 + np.random.randn(252) * 0.02).cumprod()
        
        cls.sample_data = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.randn(252) * 0.005),
            'high': prices * (1 + abs(np.random.randn(252) * 0.01)),
            'low': prices * (1 - abs(np.random.randn(252) * 0.01)),
            'close': prices,
            'volume': np.random.randint(50000000, 150000000, 252)
        })
        
        # Calculate indicators
        config = {'indicators': {}}
        library = ResearchLibrary(config)
        cls.sample_data = cls.sample_data.set_index('date')
        cls.sample_data = library.calculate_all_indicators(cls.sample_data)
        cls.sample_data = cls.sample_data.reset_index()
        
        cls.config = {'test': True}
        cls.engineer = FeatureEngineer(cls.config)
    
    def test_derived_features(self):
        """Test derived features creation"""
        df = self.sample_data.copy()
        result = self.engineer.create_features(df)
        
        # Check EMA slope features
        for period in [10, 20, 50]:
            col_7d = f'ema_{period}_slope_7d'
            col_21d = f'ema_{period}_slope_21d'
            if f'ema_{period}' in df.columns:
                self.assertIn(col_7d, result.columns, f"Missing {col_7d}")
                self.assertIn(col_21d, result.columns, f"Missing {col_21d}")
        
        # Check EMA gap
        if 'ema_10' in df.columns and 'ema_50' in df.columns:
            self.assertIn('ema_gap_10_50', result.columns, "Missing ema_gap_10_50")
        
        # Check price to BB width
        if 'bb_20_width' in df.columns:
            self.assertIn('price_to_bb_20_width', result.columns, "Missing price_to_bb_20_width")
        
        # Check volatility ratio
        if 'atr_14' in df.columns:
            self.assertIn('volatility_ratio', result.columns, "Missing volatility_ratio")
    
    def test_momentum_score(self):
        """Test momentum score calculation"""
        df = self.sample_data.copy()
        result = self.engineer.create_features(df)
        
        self.assertIn('momentum_score', result.columns, "Missing momentum_score")
        
        if result['momentum_score'].notna().any():
            # Momentum should be reasonable
            score_values = result['momentum_score'].dropna()
            self.assertIsInstance(score_values.iloc[0], (int, float, np.number))
    
    def test_regime_detection(self):
        """Test regime detection"""
        df = self.sample_data.copy()
        result = self.engineer.create_features(df)
        
        # Check regime columns
        self.assertIn('high_vol', result.columns, "Missing high_vol regime")
        self.assertIn('bull_trend', result.columns, "Missing bull_trend regime")
        self.assertIn('sideways', result.columns, "Missing sideways regime")
        
        # Check regime values are 0 or 1
        for regime_col in ['high_vol', 'bull_trend', 'sideways']:
            if result[regime_col].notna().any():
                unique_values = result[regime_col].dropna().unique()
                self.assertTrue(all(v in [0, 1] for v in unique_values),
                              f"{regime_col} should be binary")
    
    def test_normalization_universe(self):
        """Test cross-sectional normalization"""
        # Create universe snapshot (multiple symbols)
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        universe_data = []
        
        for symbol in symbols:
            df = self.sample_data.copy()
            df['symbol'] = symbol
            universe_data.append(df)
        
        universe_df = pd.concat(universe_data, ignore_index=True)
        
        # Test normalization
        normalized = self.engineer.normalize_universe(
            {s: df for s, df in zip(symbols, universe_data)},
            date=None  # Use latest
        )
        
        self.assertFalse(normalized.empty, "Normalized data should not be empty")
        
        # Check for z-score and rank percentile columns
        zscore_cols = [c for c in normalized.columns if '_zscore' in c]
        rank_cols = [c for c in normalized.columns if '_rank_pct' in c]
        
        self.assertGreater(len(zscore_cols), 0, "Should have z-score columns")
        self.assertGreater(len(rank_cols), 0, "Should have rank percentile columns")
        
        # Check z-scores have mean ~0 and std ~1
        for col in zscore_cols[:5]:  # Check first 5
            if normalized[col].notna().any():
                mean_val = normalized[col].mean()
                std_val = normalized[col].std()
                
                # Mean should be close to 0
                self.assertLess(abs(mean_val), 0.1, 
                              f"{col} z-score mean should be ~0, got {mean_val}")
    
    def test_feature_names(self):
        """Test feature name extraction"""
        df = self.sample_data.copy()
        result = self.engineer.create_features(df)
        
        feature_names = self.engineer.get_feature_names()
        
        self.assertGreater(len(feature_names), 0, "Should have feature names")
        self.assertIsInstance(feature_names, list)
        
        # Check feature names don't include raw OHLCV
        excluded = ['open', 'high', 'low', 'close', 'volume']
        for name in feature_names:
            self.assertNotIn(name, excluded, f"Feature {name} should not be raw OHLCV")
    
    def test_numeric_features_extraction(self):
        """Test numeric feature extraction"""
        df = self.sample_data.copy()
        result = self.engineer.create_features(df)
        
        numeric_features = self.engineer.get_numeric_features(result)
        
        self.assertFalse(numeric_features.empty, "Should have numeric features")
        
        # All columns should be numeric
        self.assertTrue(all(pd.api.types.is_numeric_dtype(numeric_features[col]) 
                           for col in numeric_features.columns))
        
        # Should not contain raw OHLCV
        self.assertNotIn('open', numeric_features.columns)
        self.assertNotIn('high', numeric_features.columns)


if __name__ == "__main__":
    unittest.main()

