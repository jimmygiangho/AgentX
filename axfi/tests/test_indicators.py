"""
Unit tests for indicator calculations
Tests all indicators with sample AAPL data
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.research_library import ResearchLibrary


class TestIndicators(unittest.TestCase):
    """Test indicator calculations"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        # Create sample AAPL data (252 trading days ~ 1 year)
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # Generate realistic price movements
        returns = np.random.randn(252) * 0.02  # 2% daily volatility
        prices = 150 * (1 + returns).cumprod()
        
        cls.sample_data = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.randn(252) * 0.005),
            'high': prices * (1 + abs(np.random.randn(252) * 0.01)),
            'low': prices * (1 - abs(np.random.randn(252) * 0.01)),
            'close': prices,
            'volume': np.random.randint(50000000, 150000000, 252)
        })
        
        cls.config = {
            'indicators': {
                'ema_periods': [10, 20, 50, 100, 200],
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'adx_period': 14,
                'rsi_periods': [7, 14, 21],
                'stochastic_k': 14,
                'stochastic_d': 3,
                'bollinger_periods': [20, 50],
                'bollinger_std': 2.0,
                'atr_periods': [14, 28],
                'donchian_period': 20,
                'keltner_period': 20,
                'keltner_multiplier': 2.0,
                'mfi_period': 14
            }
        }
        
        cls.library = ResearchLibrary(cls.config)
    
    def test_ema_calculation(self):
        """Test EMA calculation for all periods"""
        df = self.sample_data.copy().set_index('date')
        result = self.library.calculate_trend_following(df)
        
        # Check all EMA columns exist
        for period in [10, 20, 50, 100, 200]:
            col = f'ema_{period}'
            self.assertIn(col, result.columns, f"Missing {col}")
            
            # Check non-null values (after warm-up period)
            non_null = result[col].notna().sum()
            self.assertGreater(non_null, 200, f"{col} has insufficient non-null values")
            
            # Check values are reasonable
            latest_val = result[col].iloc[-1]
            self.assertGreater(latest_val, 0, f"{col} should be positive")
    
    def test_macd_calculation(self):
        """Test MACD calculation"""
        df = self.sample_data.copy().set_index('date')
        result = self.library.calculate_trend_following(df)
        
        # Check MACD components
        self.assertIn('macd', result.columns)
        self.assertIn('macd_signal', result.columns)
        self.assertIn('macd_histogram', result.columns)
        
        # Check non-null values
        for col in ['macd', 'macd_signal', 'macd_histogram']:
            non_null = result[col].notna().sum()
            self.assertGreater(non_null, 200, f"{col} has insufficient data")
        
        # Verify histogram = macd - signal
        if result['macd'].notna().any() and result['macd_signal'].notna().any():
            diff = result['macd'] - result['macd_signal'] - result['macd_histogram']
            self.assertTrue(diff.abs().max() < 0.01, "MACD histogram calculation incorrect")
    
    def test_adx_calculation(self):
        """Test ADX calculation"""
        df = self.sample_data.copy().set_index('date')
        result = self.library.calculate_trend_following(df)
        
        # Check ADX components
        self.assertIn('adx', result.columns)
        self.assertIn('+di', result.columns)
        self.assertIn('-di', result.columns)
        
        # Check values are in valid range
        for col in ['adx', '+di', '-di']:
            non_null = result[col].notna().sum()
            self.assertGreater(non_null, 200, f"{col} has insufficient data")
            
            # ADX should be 0-100
            if result[col].notna().any():
                max_val = result[col].max()
                self.assertLessEqual(max_val, 100, f"{col} should be <= 100")
                self.assertGreaterEqual(max_val, 0, f"{col} should be >= 0")
    
    def test_rsi_calculation(self):
        """Test RSI calculation for multiple periods"""
        df = self.sample_data.copy().set_index('date')
        result = self.library.calculate_mean_reversion(df)
        
        # Check RSI for all periods
        for period in [7, 14, 21]:
            col = f'rsi_{period}'
            self.assertIn(col, result.columns, f"Missing {col}")
            
            # Check values are in 0-100 range
            if result[col].notna().any():
                rsi_values = result[col].dropna()
                self.assertGreaterEqual(rsi_values.min(), 0, f"{col} should be >= 0")
                self.assertLessEqual(rsi_values.max(), 100, f"{col} should be <= 100")
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands for 20 and 50 periods"""
        df = self.sample_data.copy().set_index('date')
        result = self.library.calculate_mean_reversion(df)
        
        # Check BB 20
        for suffix in ['upper', 'lower', 'middle', 'width', 'percent']:
            col = f'bb_20_{suffix}'
            self.assertIn(col, result.columns, f"Missing {col}")
        
        # Check BB 50
        for suffix in ['upper', 'lower', 'middle', 'width']:
            col = f'bb_50_{suffix}'
            self.assertIn(col, result.columns, f"Missing {col}")
        
        # Verify upper > middle > lower
        if result['bb_20_upper'].notna().any():
            self.assertTrue(
                (result['bb_20_upper'] > result['bb_20_middle']).all(),
                "BB upper should be > middle"
            )
            self.assertTrue(
                (result['bb_20_middle'] > result['bb_20_lower']).all(),
                "BB middle should be > lower"
            )
    
    def test_atr_calculation(self):
        """Test ATR for 14 and 28 periods"""
        df = self.sample_data.copy().set_index('date')
        result = self.library.calculate_volatility(df)
        
        # Check ATR for both periods
        for period in [14, 28]:
            col = f'atr_{period}'
            self.assertIn(col, result.columns, f"Missing {col}")
            
            # ATR should be positive
            if result[col].notna().any():
                self.assertTrue((result[col].dropna() > 0).all(), f"{col} should be positive")
    
    def test_obv_vectorized(self):
        """Test OBV calculation (vectorized)"""
        df = self.sample_data.copy().set_index('date')
        result = self.library.calculate_volume_indicators(df)
        
        self.assertIn('obv', result.columns)
        
        # OBV should be non-null (after first row)
        non_null = result['obv'].notna().sum()
        self.assertGreater(non_null, 250, "OBV should have values for all rows")
    
    def test_all_indicators_together(self):
        """Test calculating all indicators together"""
        df = self.sample_data.copy().set_index('date')
        result = self.library.calculate_all_indicators(df)
        
        # Count expected indicator columns
        expected_cols = [
            'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
            'macd', 'macd_signal', 'macd_histogram',
            'adx', '+di', '-di',
            'rsi_7', 'rsi_14', 'rsi_21',
            'stoch_k', 'stoch_d',
            'bb_20_upper', 'bb_20_lower', 'bb_20_middle',
            'bb_50_upper', 'bb_50_lower', 'bb_50_middle',
            'atr_14', 'atr_28',
            'donchian_upper', 'donchian_lower',
            'keltner_upper', 'keltner_lower',
            'obv', 'mfi'
        ]
        
        for col in expected_cols:
            self.assertIn(col, result.columns, f"Missing expected column: {col}")
        
        # Verify we have data
        self.assertGreater(len(result), 0, "Result should not be empty")
        
        # Check non-null counts (allowing for warm-up periods)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['open', 'high', 'low', 'close', 'volume']:
                non_null_pct = result[col].notna().sum() / len(result)
                self.assertGreater(non_null_pct, 0.5, 
                                  f"{col} should have >50% non-null values")


if __name__ == "__main__":
    unittest.main()

