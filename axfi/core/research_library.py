"""
AXFI Research Library
Comprehensive library of technical indicators and quantitative studies
Vectorized implementation for performance
"""

import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ResearchLibrary:
    """
    Library of technical indicators and quantitative studies for financial analysis.
    
    Features:
    - Trend Following: EMA, MACD, ADX
    - Mean Reversion: RSI (7,14,21), Stochastics, Bollinger Bands (20,50)
    - Volatility: ATR (14,28), Donchian Channels, Keltner Channels
    - Volume: OBV (vectorized), MFI
    - All calculations are vectorized for performance
    """
    
    def __init__(self, config: dict):
        """
        Initialize the research library.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.indicator_config = config.get('indicators', {})
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators in one pass (vectorized).
        
        Args:
            df: Price data DataFrame with OHLCV columns
            
        Returns:
            DataFrame with all indicators added
        """
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return df
        
        result_df = df.copy()
        
        # Ensure date is index or column
        if 'date' in result_df.columns and result_df.index.name != 'date':
            result_df = result_df.set_index('date')
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in result_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Calculate all indicator families
        result_df = self.calculate_trend_following(result_df)
        result_df = self.calculate_mean_reversion(result_df)
        result_df = self.calculate_volatility(result_df)
        result_df = self.calculate_volume_indicators(result_df)
        
        logger.info(f"Calculated all indicators for {len(result_df)} rows")
        return result_df
    
    def calculate_trend_following(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend following indicators: EMA, MACD, ADX.
        
        Args:
            df: Price data DataFrame with OHLCV columns
            
        Returns:
            DataFrame with trend indicators added
        """
        result_df = df.copy()
        
        # EMA (Exponential Moving Average) - 5 periods
        ema_periods = self.indicator_config.get('ema_periods', [10, 20, 50, 100, 200])
        for period in ema_periods:
            result_df[f'ema_{period}'] = result_df['close'].ewm(span=period, adjust=False).mean()
        
        # MACD (Moving Average Convergence Divergence) - 12,26,9
        macd_fast = self.indicator_config.get('macd_fast', 12)
        macd_slow = self.indicator_config.get('macd_slow', 26)
        macd_signal = self.indicator_config.get('macd_signal', 9)
        
        ema_fast = result_df['close'].ewm(span=macd_fast, adjust=False).mean()
        ema_slow = result_df['close'].ewm(span=macd_slow, adjust=False).mean()
        
        result_df['macd'] = ema_fast - ema_slow
        result_df['macd_signal'] = result_df['macd'].ewm(span=macd_signal, adjust=False).mean()
        result_df['macd_histogram'] = result_df['macd'] - result_df['macd_signal']
        
        # ADX (Average Directional Index) - 14 default
        adx_period = self.indicator_config.get('adx_period', 14)
        result_df = self._calculate_adx(result_df, period=adx_period)
        
        logger.debug(f"Calculated trend following: EMA({ema_periods}), MACD({macd_fast},{macd_slow},{macd_signal}), ADX({adx_period})")
        
        return result_df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average Directional Index (ADX) with +DI and -DI.
        
        Args:
            df: DataFrame with OHLC data
            period: ADX period
            
        Returns:
            DataFrame with ADX, +DI, -DI columns added
        """
        result_df = df.copy()
        
        # Calculate True Range
        high_low = result_df['high'] - result_df['low']
        high_close = abs(result_df['high'] - result_df['close'].shift(1))
        low_close = abs(result_df['low'] - result_df['close'].shift(1))
        result_df['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate Directional Movement (vectorized)
        high_diff = result_df['high'].diff()
        low_diff = result_df['low'].diff()
        
        result_df['+dm'] = np.where(
            (high_diff > low_diff.abs()) & (high_diff > 0),
            high_diff, 0
        )
        result_df['-dm'] = np.where(
            (low_diff.abs() > high_diff) & (low_diff < 0),
            low_diff.abs(), 0
        )
        
        # Calculate smoothed values using Wilder's smoothing
        # First use simple moving average, then apply Wilder's
        tr_smooth = result_df['tr'].rolling(window=period).mean()
        plus_dm_smooth = result_df['+dm'].rolling(window=period).mean()
        minus_dm_smooth = result_df['-dm'].rolling(window=period).mean()
        
        # Calculate DI
        result_df['+di'] = 100 * (plus_dm_smooth / tr_smooth)
        result_df['-di'] = 100 * (minus_dm_smooth / tr_smooth)
        
        # Replace inf/nan from division by zero
        result_df['+di'] = result_df['+di'].replace([np.inf, -np.inf], np.nan).fillna(0)
        result_df['-di'] = result_df['-di'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Calculate DX and ADX
        di_sum = result_df['+di'] + result_df['-di']
        di_sum = di_sum.replace(0, np.nan)  # Avoid division by zero
        result_df['dx'] = 100 * abs(result_df['+di'] - result_df['-di']) / di_sum
        result_df['adx'] = result_df['dx'].rolling(window=period).mean()
        
        # Clean up temporary columns
        result_df = result_df.drop(columns=['tr', '+dm', '-dm', 'dx'], errors='ignore')
        
        return result_df
    
    def calculate_mean_reversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate mean reversion indicators: RSI (7,14,21), Stochastics, Bollinger Bands (20,50).
        
        Args:
            df: Price data DataFrame with OHLCV columns
            
        Returns:
            DataFrame with mean reversion indicators added
        """
        result_df = df.copy()
        
        # RSI (Relative Strength Index) - windows 7, 14, 21
        rsi_periods = self.indicator_config.get('rsi_periods', [7, 14, 21])
        for period in rsi_periods:
            result_df[f'rsi_{period}'] = self._calculate_rsi(result_df['close'], period=period)
        
        # Stochastic Oscillator - %K (14), %D (3)
        stoch_k = self.indicator_config.get('stochastic_k', 14)
        stoch_d = self.indicator_config.get('stochastic_d', 3)
        
        low_min = result_df['low'].rolling(window=stoch_k).min()
        high_max = result_df['high'].rolling(window=stoch_k).max()
        denominator = high_max - low_min
        denominator = denominator.replace(0, np.nan)  # Avoid division by zero
        
        result_df['stoch_k'] = 100 * ((result_df['close'] - low_min) / denominator)
        result_df['stoch_d'] = result_df['stoch_k'].rolling(window=stoch_d).mean()
        
        # Bollinger Bands - windows 20 & 50
        bb_periods = self.indicator_config.get('bollinger_periods', [20, 50])
        bb_std = self.indicator_config.get('bollinger_std', 2.0)
        
        for period in bb_periods:
            bb_middle = result_df['close'].rolling(window=period).mean()
            bb_std_val = result_df['close'].rolling(window=period).std()
            
            result_df[f'bb_{period}_middle'] = bb_middle
            result_df[f'bb_{period}_upper'] = bb_middle + (bb_std_val * bb_std)
            result_df[f'bb_{period}_lower'] = bb_middle - (bb_std_val * bb_std)
            result_df[f'bb_{period}_width'] = (result_df[f'bb_{period}_upper'] - result_df[f'bb_{period}_lower']) / bb_middle
            
            # %B (Percent B)
            bb_range = result_df[f'bb_{period}_upper'] - result_df[f'bb_{period}_lower']
            bb_range = bb_range.replace(0, np.nan)
            result_df[f'bb_{period}_percent'] = (result_df['close'] - result_df[f'bb_{period}_lower']) / bb_range
        
        # Legacy support: use 20-period as default if accessed without period suffix
        result_df['bb_middle'] = result_df.get('bb_20_middle', result_df['close'].rolling(window=20).mean())
        result_df['bb_upper'] = result_df.get('bb_20_upper', result_df['bb_middle'] + (result_df['close'].rolling(20).std() * 2))
        result_df['bb_lower'] = result_df.get('bb_20_lower', result_df['bb_middle'] - (result_df['close'].rolling(20).std() * 2))
        result_df['bb_width'] = result_df.get('bb_20_width', (result_df['bb_upper'] - result_df['bb_lower']) / result_df['bb_middle'])
        result_df['bb_percent'] = result_df.get('bb_20_percent', (result_df['close'] - result_df['bb_lower']) / (result_df['bb_upper'] - result_df['bb_lower']))
        
        # Default RSI (14) for backward compatibility
        if 'rsi' not in result_df.columns:
            result_df['rsi'] = result_df.get('rsi_14', self._calculate_rsi(result_df['close'], period=14))
        
        logger.debug(f"Calculated mean reversion: RSI({rsi_periods}), Stochastic({stoch_k},{stoch_d}), Bollinger({bb_periods})")
        
        return result_df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI) - vectorized.
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        # Use exponential moving average for RSI calculation
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Default to neutral if insufficient data
    
    def calculate_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility-based indicators: ATR (14,28), Donchian Channels, Keltner Channels.
        
        Args:
            df: Price data DataFrame with OHLCV columns
            
        Returns:
            DataFrame with volatility indicators added
        """
        result_df = df.copy()
        
        # ATR (Average True Range) - windows 14 & 28
        atr_periods = self.indicator_config.get('atr_periods', [14, 28])
        for period in atr_periods:
            result_df[f'atr_{period}'] = self._calculate_atr(result_df, period=period)
        
        # Default ATR (14) for backward compatibility
        if 'atr' not in result_df.columns:
            result_df['atr'] = result_df.get('atr_14', self._calculate_atr(result_df, period=14))
        
        # Donchian Channels - 20 period
        donchian_period = self.indicator_config.get('donchian_period', 20)
        result_df['donchian_upper'] = result_df['high'].rolling(window=donchian_period).max()
        result_df['donchian_lower'] = result_df['low'].rolling(window=donchian_period).min()
        result_df['donchian_middle'] = (result_df['donchian_upper'] + result_df['donchian_lower']) / 2
        
        # Keltner Channels - 20 period
        keltner_period = self.indicator_config.get('keltner_period', 20)
        keltner_mult = self.indicator_config.get('keltner_multiplier', 2.0)
        
        result_df['keltner_middle'] = result_df['close'].ewm(span=keltner_period, adjust=False).mean()
        keltner_atr = result_df.get('atr_14', result_df.get('atr', self._calculate_atr(result_df, period=14)))
        result_df['keltner_range'] = keltner_atr.rolling(window=keltner_period).mean()
        result_df['keltner_upper'] = result_df['keltner_middle'] + (result_df['keltner_range'] * keltner_mult)
        result_df['keltner_lower'] = result_df['keltner_middle'] - (result_df['keltner_range'] * keltner_mult)
        
        logger.debug(f"Calculated volatility: ATR({atr_periods}), Donchian({donchian_period}), Keltner({keltner_period},{keltner_mult})")
        
        return result_df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR) - vectorized.
        
        Args:
            df: DataFrame with OHLC data
            period: ATR period
            
        Returns:
            ATR series
        """
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()  # Use Wilder's smoothing
        
        return atr
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based indicators: OBV (vectorized), MFI.
        
        Args:
            df: Price data DataFrame with OHLCV columns
            
        Returns:
            DataFrame with volume indicators added
        """
        result_df = df.copy()
        
        # OBV (On-Balance Volume) - fully vectorized
        price_change = result_df['close'].diff()
        volume_sign = np.where(price_change > 0, 1, 
                               np.where(price_change < 0, -1, 0))
        result_df['obv'] = (result_df['volume'] * volume_sign).cumsum()
        
        # MFI (Money Flow Index) - 14 period
        mfi_period = self.indicator_config.get('mfi_period', 14)
        result_df['mfi'] = self._calculate_mfi(result_df, period=mfi_period)
        
        logger.debug(f"Calculated volume indicators: OBV, MFI({mfi_period})")
        
        return result_df
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Money Flow Index (MFI) - vectorized.
        
        Args:
            df: DataFrame with OHLCV data
            period: MFI period
            
        Returns:
            MFI series
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        # Positive and negative money flow
        price_change = typical_price.diff()
        positive_flow = money_flow.where(price_change > 0, 0)
        negative_flow = money_flow.where(price_change < 0, 0)
        
        # Rolling sums
        positive_flow_sum = positive_flow.rolling(window=period).sum()
        negative_flow_sum = negative_flow.rolling(window=period).sum()
        
        # Money flow ratio and MFI
        money_flow_ratio = positive_flow_sum / negative_flow_sum.replace(0, np.nan)
        mfi = 100 - (100 / (1 + money_flow_ratio))
        
        return mfi.fillna(50)  # Default to neutral
    
    def analyze_correlation(self, symbols_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Analyze correlation between symbols based on returns.
        
        Args:
            symbols_data: Dictionary of symbol to price DataFrame
            
        Returns:
            Correlation matrix DataFrame
        """
        logger.info("Calculating correlation matrix")
        
        if not symbols_data:
            logger.warning("No symbols data provided for correlation analysis")
            return pd.DataFrame()
        
        # Calculate returns for each symbol
        returns_dict = {}
        for symbol, df in symbols_data.items():
            if 'close' in df.columns:
                returns_dict[symbol] = df['close'].pct_change()
        
        if not returns_dict:
            logger.warning("No returns data to analyze")
            return pd.DataFrame()
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_dict)
        
        # Calculate correlation
        correlation_matrix = returns_df.corr()
        
        logger.info(f"Calculated correlation for {len(returns_dict)} symbols")
        
        return correlation_matrix
    
    def sector_rotation_analysis(self, sector_mapping: Dict[str, List[str]], 
                                  symbols_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Analyze sector rotation and relative strength.
        
        Args:
            sector_mapping: Dictionary of sector to list of symbols
            symbols_data: Dictionary of symbol to price DataFrame
            
        Returns:
            Dictionary of sector to relative strength score
        """
        logger.info("Analyzing sector rotation")
        
        sector_scores = {}
        
        for sector, symbols in sector_mapping.items():
            if not symbols:
                continue
            
            sector_returns = []
            for symbol in symbols:
                if symbol in symbols_data and 'close' in symbols_data[symbol].columns:
                    returns = symbols_data[symbol]['close'].pct_change()
                    if len(returns) > 0:
                        sector_returns.append(returns.mean() * 252)  # Annualized
            
            if sector_returns:
                sector_scores[sector] = np.mean(sector_returns)
        
        logger.info(f"Analyzed {len(sector_scores)} sectors")
        
        return sector_scores


def main():
    """Standalone test of ResearchLibrary"""
    import yaml
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize library
    library = ResearchLibrary(config)
    
    print("=" * 80)
    print("AXFI Research Library - Enhanced Test")
    print("=" * 80)
    print(f"\n[OK] Research Library initialized")
    
    # Create sample data for testing
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    np.random.seed(42)
    prices = 100 + np.random.randn(252).cumsum()
    
    sample_df = pd.DataFrame({
        'date': dates,
        'open': prices + np.random.randn(252) * 0.5,
        'high': prices + abs(np.random.randn(252) * 1),
        'low': prices - abs(np.random.randn(252) * 1),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 252)
    }).set_index('date')
    
    print(f"\n[OK] Created sample data: {len(sample_df)} rows")
    
    # Test all indicators
    print("\n1. Testing all indicators...")
    indicators_df = library.calculate_all_indicators(sample_df)
    
    # Count indicators
    ema_cols = [c for c in indicators_df.columns if c.startswith('ema_')]
    rsi_cols = [c for c in indicators_df.columns if c.startswith('rsi')]
    bb_cols = [c for c in indicators_df.columns if 'bb_' in c]
    atr_cols = [c for c in indicators_df.columns if 'atr' in c]
    
    print(f"   [OK] EMA columns: {len(ema_cols)} - {ema_cols}")
    print(f"   [OK] RSI columns: {len(rsi_cols)} - {rsi_cols}")
    print(f"   [OK] Bollinger Bands columns: {len(bb_cols)}")
    print(f"   [OK] ATR columns: {len(atr_cols)} - {atr_cols}")
    print(f"   [OK] MACD: {[c for c in indicators_df.columns if 'macd' in c]}")
    print(f"   [OK] ADX: {[c for c in indicators_df.columns if 'adx' in c or 'di' in c]}")
    print(f"   [OK] Stochastic: {[c for c in indicators_df.columns if 'stoch' in c]}")
    print(f"   [OK] Donchian: {[c for c in indicators_df.columns if 'donchian' in c]}")
    print(f"   [OK] Keltner: {[c for c in indicators_df.columns if 'keltner' in c]}")
    print(f"   [OK] OBV: {'obv' in indicators_df.columns}")
    print(f"   [OK] MFI: {'mfi' in indicators_df.columns}")
    
    # Check for non-null values
    print("\n2. Checking data quality...")
    non_null_count = indicators_df.select_dtypes(include=[np.number]).notna().sum()
    print(f"   [OK] Non-null values per indicator:")
    for col in ['ema_20', 'rsi_14', 'macd', 'atr_14', 'obv', 'mfi']:
        if col in indicators_df.columns:
            count = non_null_count[col]
            print(f"      {col}: {count}/{len(indicators_df)} ({count/len(indicators_df)*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
