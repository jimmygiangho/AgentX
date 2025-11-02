"""
AXFI Research Library
Comprehensive library of technical indicators and quantitative studies
"""

import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ResearchLibrary:
    """
    Library of technical indicators and quantitative studies for financial analysis.
    
    Features:
    - Trend Following: EMA, MACD, ADX
    - Mean Reversion: RSI, Stochastics, Bollinger Bands
    - Volatility: ATR, Donchian Channels, Keltner Channels
    - Volume: OBV, MFI
    - Correlation and sector rotation analysis
    """
    
    def __init__(self, config: dict):
        """
        Initialize the research library.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.indicator_config = config.get('indicators', {})
        
    def calculate_trend_following(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend following indicators: EMA, MACD, ADX.
        
        Args:
            df: Price data DataFrame with OHLCV columns
            
        Returns:
            DataFrame with trend indicators added
        """
        result_df = df.copy()
        
        # EMA (Exponential Moving Average)
        ema_periods = self.indicator_config.get('ema_periods', [10, 20, 50, 100, 200])
        for period in ema_periods:
            result_df[f'ema_{period}'] = result_df['close'].ewm(span=period, adjust=False).mean()
        
        # MACD (Moving Average Convergence Divergence)
        fast = self.indicator_config.get('macd_fast', [12])[0]
        slow = self.indicator_config.get('macd_slow', [26])[0]
        signal = self.indicator_config.get('macd_signal', [9])[0]
        
        result_df['macd'] = result_df['close'].ewm(span=fast, adjust=False).mean() - \
                           result_df['close'].ewm(span=slow, adjust=False).mean()
        result_df['macd_signal'] = result_df['macd'].ewm(span=signal, adjust=False).mean()
        result_df['macd_histogram'] = result_df['macd'] - result_df['macd_signal']
        
        # ADX (Average Directional Index)
        adx_period = self.indicator_config.get('adx_period', [14])[0]
        result_df = self._calculate_adx(result_df, period=adx_period)
        
        logger.info(f"Calculated trend following indicators: EMA periods={ema_periods}, MACD({fast},{slow},{signal}), ADX({adx_period})")
        
        return result_df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            df: DataFrame with OHLC data
            period: ADX period
            
        Returns:
            DataFrame with ADX, +DI, -DI columns added
        """
        result_df = df.copy()
        
        # Calculate True Range
        result_df['high_low'] = result_df['high'] - result_df['low']
        result_df['high_close'] = abs(result_df['high'] - result_df['close'].shift(1))
        result_df['low_close'] = abs(result_df['low'] - result_df['close'].shift(1))
        result_df['tr'] = result_df[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        # Calculate Directional Movement
        result_df['+dm'] = np.where(
            (result_df['high'].diff() > result_df['low'].diff().abs()) & 
            (result_df['high'].diff() > 0),
            result_df['high'].diff(), 0
        )
        result_df['-dm'] = np.where(
            (result_df['low'].diff().abs() > result_df['high'].diff()) & 
            (result_df['low'].diff() < 0),
            result_df['low'].diff().abs(), 0
        )
        
        # Calculate smoothed values
        result_df['tr_smooth'] = result_df['tr'].rolling(window=period).mean()
        result_df['+dm_smooth'] = result_df['+dm'].rolling(window=period).mean()
        result_df['-dm_smooth'] = result_df['-dm'].rolling(window=period).mean()
        
        # Calculate DI
        result_df['+di'] = 100 * (result_df['+dm_smooth'] / result_df['tr_smooth'])
        result_df['-di'] = 100 * (result_df['-dm_smooth'] / result_df['tr_smooth'])
        
        # Calculate ADX
        result_df['dx'] = 100 * abs(result_df['+di'] - result_df['-di']) / (result_df['+di'] + result_df['-di'])
        result_df['adx'] = result_df['dx'].rolling(window=period).mean()
        
        # Clean up temporary columns
        result_df = result_df.drop(columns=['high_low', 'high_close', 'low_close'])
        
        return result_df
    
    def calculate_mean_reversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate mean reversion indicators: RSI, Stochastics, Bollinger Bands.
        
        Args:
            df: Price data DataFrame with OHLCV columns
            
        Returns:
            DataFrame with mean reversion indicators added
        """
        result_df = df.copy()
        
        # RSI (Relative Strength Index)
        rsi_period = self.indicator_config.get('rsi_period', [14])[0]
        result_df['rsi'] = self._calculate_rsi(result_df['close'], period=rsi_period)
        
        # Stochastic Oscillator
        stoch_k = self.indicator_config.get('stochastic_k', [14])[0]
        stoch_d = self.indicator_config.get('stochastic_d', [3])[0]
        
        result_df['stoch_k'] = ((result_df['close'] - result_df['low'].rolling(stoch_k).min()) /
                                (result_df['high'].rolling(stoch_k).max() - result_df['low'].rolling(stoch_k).min())) * 100
        result_df['stoch_d'] = result_df['stoch_k'].rolling(window=stoch_d).mean()
        
        # Bollinger Bands
        bb_period = self.indicator_config.get('bollinger_period', [20])[0]
        bb_std = self.indicator_config.get('bollinger_std', [2])[0]
        
        result_df['bb_middle'] = result_df['close'].rolling(window=bb_period).mean()
        bb_std_val = result_df['close'].rolling(window=bb_period).std()
        result_df['bb_upper'] = result_df['bb_middle'] + (bb_std_val * bb_std)
        result_df['bb_lower'] = result_df['bb_middle'] - (bb_std_val * bb_std)
        result_df['bb_width'] = (result_df['bb_upper'] - result_df['bb_lower']) / result_df['bb_middle']
        result_df['bb_percent'] = (result_df['close'] - result_df['bb_lower']) / (result_df['bb_upper'] - result_df['bb_lower'])
        
        logger.info(f"Calculated mean reversion indicators: RSI({rsi_period}), Stochastic({stoch_k},{stoch_d}), Bollinger({bb_period},{bb_std})")
        
        return result_df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility-based indicators: ATR, Donchian Channels, Keltner Channels.
        
        Args:
            df: Price data DataFrame with OHLCV columns
            
        Returns:
            DataFrame with volatility indicators added
        """
        result_df = df.copy()
        
        # ATR (Average True Range)
        atr_period = self.indicator_config.get('atr_period', [14])[0]
        result_df['atr'] = self._calculate_atr(result_df, period=atr_period)
        
        # Donchian Channels
        donchian_period = self.indicator_config.get('donchian_period', [20])[0]
        result_df['donchian_upper'] = result_df['high'].rolling(window=donchian_period).max()
        result_df['donchian_lower'] = result_df['low'].rolling(window=donchian_period).min()
        result_df['donchian_middle'] = (result_df['donchian_upper'] + result_df['donchian_lower']) / 2
        
        # Keltner Channels
        keltner_period = self.indicator_config.get('keltner_period', [20])[0]
        keltner_mult = self.indicator_config.get('keltner_multiplier', [2.0])[0]
        
        result_df['keltner_middle'] = result_df['close'].ewm(span=keltner_period, adjust=False).mean()
        result_df['keltner_range'] = result_df['atr'].rolling(window=keltner_period).mean()
        result_df['keltner_upper'] = result_df['keltner_middle'] + (result_df['keltner_range'] * keltner_mult)
        result_df['keltner_lower'] = result_df['keltner_middle'] - (result_df['keltner_range'] * keltner_mult)
        
        logger.info(f"Calculated volatility indicators: ATR({atr_period}), Donchian({donchian_period}), Keltner({keltner_period},{keltner_mult})")
        
        return result_df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
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
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based indicators: OBV, MFI.
        
        Args:
            df: Price data DataFrame with OHLCV columns
            
        Returns:
            DataFrame with volume indicators added
        """
        result_df = df.copy()
        
        # OBV (On-Balance Volume)
        result_df['obv'] = 0
        for i in range(1, len(result_df)):
            if result_df['close'].iloc[i] > result_df['close'].iloc[i-1]:
                result_df.loc[result_df.index[i], 'obv'] = result_df['obv'].iloc[i-1] + result_df['volume'].iloc[i]
            elif result_df['close'].iloc[i] < result_df['close'].iloc[i-1]:
                result_df.loc[result_df.index[i], 'obv'] = result_df['obv'].iloc[i-1] - result_df['volume'].iloc[i]
            else:
                result_df.loc[result_df.index[i], 'obv'] = result_df['obv'].iloc[i-1]
        
        # MFI (Money Flow Index)
        mfi_period = self.indicator_config.get('mfi_period', [14])[0]
        result_df['mfi'] = self._calculate_mfi(result_df, period=mfi_period)
        
        logger.info(f"Calculated volume indicators: OBV, MFI({mfi_period})")
        
        return result_df
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Money Flow Index (MFI).
        
        Args:
            df: DataFrame with OHLCV data
            period: MFI period
            
        Returns:
            MFI series
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_flow_sum = positive_flow.rolling(window=period).sum()
        negative_flow_sum = negative_flow.rolling(window=period).sum()
        
        money_flow_ratio = positive_flow_sum / negative_flow_sum
        mfi = 100 - (100 / (1 + money_flow_ratio))
        
        return mfi
    
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
    print("AXFI Research Library - Standalone Test")
    print("=" * 80)
    print(f"\n[OK] Research Library initialized")
    
    # Load sample data
    from core.data_collector import DataCollector
    dc = DataCollector(config_path="config.yaml")
    df = dc.read_from_database(symbol="AAPL")
    
    if not df.empty:
        df.set_index('date', inplace=True)
        print(f"\n[OK] Loaded {len(df)} rows for AAPL")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # Test indicators
        print("\n1. Testing trend following indicators...")
        trend_df = library.calculate_trend_following(df)
        print(f"   [OK] EMA columns: {[col for col in trend_df.columns if col.startswith('ema_')]}")
        print(f"   [OK] MACD columns: {[col for col in trend_df.columns if 'macd' in col]}")
        print(f"   [OK] ADX columns: {[col for col in trend_df.columns if 'adx' in col or '+di' in col or '-di' in col]}")
        
        print("\n2. Testing mean reversion indicators...")
        rev_df = library.calculate_mean_reversion(df)
        print(f"   [OK] RSI calculated: mean={rev_df['rsi'].mean():.2f}, std={rev_df['rsi'].std():.2f}")
        print(f"   [OK] Stochastic calculated: {[col for col in rev_df.columns if 'stoch' in col]}")
        print(f"   [OK] Bollinger Bands calculated: {[col for col in rev_df.columns if 'bb_' in col]}")
        
        print("\n3. Testing volatility indicators...")
        vol_df = library.calculate_volatility(df)
        print(f"   [OK] ATR calculated: mean={vol_df['atr'].mean():.2f}")
        print(f"   [OK] Donchian Channels calculated: {[col for col in vol_df.columns if 'donchian' in col]}")
        print(f"   [OK] Keltner Channels calculated: {[col for col in vol_df.columns if 'keltner' in col]}")
        
        print("\n4. Testing volume indicators...")
        vol_ind_df = library.calculate_volume_indicators(df)
        print(f"   [OK] OBV calculated: final={vol_ind_df['obv'].iloc[-1]:.0f}")
        print(f"   [OK] MFI calculated: mean={vol_ind_df['mfi'].mean():.2f}")
        
        # Show sample data
        print("\n5. Sample indicator values (last 3 rows):")
        cols_to_show = ['close', 'rsi', 'macd', 'atr', 'obv']
        available_cols = [col for col in cols_to_show if col in vol_ind_df.columns]
        print(vol_ind_df[available_cols].tail(3).to_string())
        
    else:
        print("[ERROR] No data available for AAPL. Please run data_collector.py first.")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
