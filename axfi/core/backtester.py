"""
AXFI Backtester Module
Implements multiple trading strategies and calculates performance metrics
"""

import logging
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Backtester:
    """
    Implements multiple trading strategies and calculates performance metrics.
    
    Strategies:
    - SMA Crossover (Momentum): Buy when short MA crosses above long MA
    - Bollinger Band Mean Reversion: Buy/Sell based on price position relative to bands
    - ATR Volatility Breakout: Buy on high volatility breakouts
    """
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Starting capital for backtesting
            commission: Commission rate per trade (e.g., 0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        logger.info(f"Initialized Backtester with capital=${initial_capital:,.2f}, commission={commission*100:.2f}%")
    
    def calculate_returns(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        strategy_name: str
    ) -> pd.DataFrame:
        """
        Calculate returns and equity curve from trading signals.
        
        Args:
            df: DataFrame with OHLCV data (must have 'close' column)
            signals: Series with trading signals (1 for buy, -1 for sell, 0 for hold)
            strategy_name: Name of the strategy (for logging)
            
        Returns:
            DataFrame with returns, equity, and cumulative metrics
        """
        result = df.copy()
        result['signal'] = signals
        result['returns'] = result['close'].pct_change()
        
        # Position: 1 when holding, 0 when not
        result['position'] = 0
        result.loc[signals == 1, 'position'] = 1
        result.loc[signals == -1, 'position'] = 0
        
        # Calculate strategy returns (position-weighted returns)
        result['strategy_returns'] = result['returns'] * result['position'].shift(1)
        
        # Apply commission when entering/exiting positions
        result['commission_cost'] = 0.0
        position_changes = result['position'].diff().abs()
        result.loc[position_changes > 0, 'commission_cost'] = self.commission
        
        # Net returns after commission
        result['net_returns'] = result['strategy_returns'] - result['commission_cost']
        
        # Cumulative returns and equity
        result['cumulative_returns'] = (1 + result['net_returns']).cumprod()
        result['equity'] = self.initial_capital * result['cumulative_returns']
        
        # Buy and hold comparison
        result['buy_hold_returns'] = result['returns']
        result['buy_hold_cumulative'] = (1 + result['buy_hold_returns']).cumprod()
        result['buy_hold_equity'] = self.initial_capital * result['buy_hold_cumulative']
        
        logger.info(f"Strategy '{strategy_name}': Calculated returns for {len(result)} periods")
        return result
    
    def calculate_metrics(self, result_df: pd.DataFrame, strategy_name: str) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            result_df: DataFrame with returns and equity calculations
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary of performance metrics
        """
        if len(result_df) == 0:
            logger.warning(f"Empty result DataFrame for {strategy_name}")
            return {}
        
        # Filter out NaN values
        valid_data = result_df.dropna(subset=['net_returns', 'equity'])
        
        if len(valid_data) == 0:
            logger.warning(f"No valid data for metrics calculation for {strategy_name}")
            return {}
        
        # Total Return
        total_return = (valid_data['equity'].iloc[-1] / self.initial_capital) - 1
        
        # Annualized Return (CAGR) - assuming daily data
        trading_days = 252  # Approximate trading days per year
        periods = len(valid_data)
        years = periods / trading_days
        if years > 0:
            cagr = ((valid_data['equity'].iloc[-1] / self.initial_capital) ** (1 / years)) - 1
        else:
            cagr = 0
        
        # Volatility (annualized)
        volatility = valid_data['net_returns'].std() * np.sqrt(trading_days) if len(valid_data) > 1 else 0
        
        # Sharpe Ratio (assuming risk-free rate of 0)
        sharpe = (cagr / volatility) if volatility > 0 else 0
        
        # Maximum Drawdown
        rolling_max = valid_data['equity'].cummax()
        drawdown = (valid_data['equity'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win Rate
        trades = valid_data[valid_data['position'].diff() != 0].copy()
        if len(trades) > 1:
            trade_results = valid_data.loc[trades.index, 'net_returns']
            winning_trades = (trade_results > 0).sum()
            total_trades = len(trade_results)
            win_rate = (winning_trades / total_trades) if total_trades > 0 else 0
        else:
            win_rate = 0
        
        # Number of trades
        num_trades = (valid_data['position'].diff() != 0).sum()
        
        # Final equity
        final_equity = valid_data['equity'].iloc[-1]
        
        # Profit/Loss
        profit_loss = final_equity - self.initial_capital
        
        metrics = {
            'strategy_name': strategy_name,
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': int(num_trades),
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'profit_loss': profit_loss
        }
        
        logger.info(f"Calculated metrics for {strategy_name}")
        return metrics
    
    def sma_crossover(
        self,
        df: pd.DataFrame,
        short_window: int = 20,
        long_window: int = 50
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        SMA Crossover Strategy (Momentum).
        
        Buy signal: Short MA crosses above Long MA
        Sell signal: Short MA crosses below Long MA
        
        Args:
            df: DataFrame with OHLCV data (must have 'close' column)
            short_window: Short moving average window
            long_window: Long moving average window
            
        Returns:
            Tuple of (result DataFrame, metrics dictionary)
        """
        logger.info(f"Running SMA Crossover strategy (short={short_window}, long={long_window})")
        
        # Calculate moving averages
        df_copy = df.copy()
        df_copy['sma_short'] = df_copy['close'].rolling(window=short_window).mean()
        df_copy['sma_long'] = df_copy['close'].rolling(window=long_window).mean()
        
        # Generate signals
        signals = pd.Series(0, index=df_copy.index)
        signals.loc[df_copy['sma_short'] > df_copy['sma_long']] = 1
        signals.loc[df_copy['sma_short'] <= df_copy['sma_long']] = -1
        
        # Calculate returns
        result = self.calculate_returns(df, signals, f"SMA Cross ({short_window}/{long_window})")
        result['sma_short'] = df_copy['sma_short']
        result['sma_long'] = df_copy['sma_long']
        
        # Calculate metrics
        metrics = self.calculate_metrics(result, f"SMA Cross ({short_window}/{long_window})")
        
        return result, metrics
    
    def bollinger_bands_mean_reversion(
        self,
        df: pd.DataFrame,
        window: int = 20,
        num_std: float = 2.0
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Bollinger Bands Mean Reversion Strategy.
        
        Buy signal: Price touches lower band (oversold)
        Sell signal: Price touches upper band (overbought)
        
        Args:
            df: DataFrame with OHLCV data (must have 'close' column)
            window: Moving average window for Bollinger Bands
            num_std: Number of standard deviations for bands
            
        Returns:
            Tuple of (result DataFrame, metrics dictionary)
        """
        logger.info(f"Running Bollinger Bands strategy (window={window}, std={num_std})")
        
        # Calculate Bollinger Bands
        df_copy = df.copy()
        df_copy['bb_middle'] = df_copy['close'].rolling(window=window).mean()
        df_copy['bb_std'] = df_copy['close'].rolling(window=window).std()
        df_copy['bb_upper'] = df_copy['bb_middle'] + (num_std * df_copy['bb_std'])
        df_copy['bb_lower'] = df_copy['bb_middle'] - (num_std * df_copy['bb_std'])
        
        # Generate signals (mean reversion: buy low, sell high)
        signals = pd.Series(0, index=df_copy.index)
        signals.loc[df_copy['close'] <= df_copy['bb_lower']] = 1  # Buy when oversold
        signals.loc[df_copy['close'] >= df_copy['bb_upper']] = -1  # Sell when overbought
        
        # Calculate returns
        result = self.calculate_returns(df, signals, f"Bollinger Bands ({window}, {num_std} std)")
        result['bb_middle'] = df_copy['bb_middle']
        result['bb_upper'] = df_copy['bb_upper']
        result['bb_lower'] = df_copy['bb_lower']
        
        # Calculate metrics
        metrics = self.calculate_metrics(result, f"Bollinger Bands ({window}, {num_std} std)")
        
        return result, metrics
    
    def atr_volatility_breakout(
        self,
        df: pd.DataFrame,
        atr_window: int = 14,
        atr_multiplier: float = 2.0
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        ATR Volatility Breakout Strategy.
        
        Buy signal: Price breaks above previous high + (ATR * multiplier)
        Sell signal: Price breaks below previous low - (ATR * multiplier)
        
        Args:
            df: DataFrame with OHLCV data (must have 'open', 'high', 'low', 'close')
            atr_window: Window for Average True Range calculation
            atr_multiplier: Multiplier for breakout thresholds
            
        Returns:
            Tuple of (result DataFrame, metrics dictionary)
        """
        logger.info(f"Running ATR Volatility Breakout strategy (window={atr_window}, mult={atr_multiplier})")
        
        # Calculate Average True Range (ATR)
        df_copy = df.copy()
        df_copy['tr1'] = df_copy['high'] - df_copy['low']
        df_copy['tr2'] = abs(df_copy['high'] - df_copy['close'].shift(1))
        df_copy['tr3'] = abs(df_copy['low'] - df_copy['close'].shift(1))
        df_copy['true_range'] = df_copy[['tr1', 'tr2', 'tr3']].max(axis=1)
        df_copy['atr'] = df_copy['true_range'].rolling(window=atr_window).mean()
        
        # Calculate breakout levels
        df_copy['upper_band'] = df_copy['high'].shift(1) + (df_copy['atr'] * atr_multiplier)
        df_copy['lower_band'] = df_copy['low'].shift(1) - (df_copy['atr'] * atr_multiplier)
        
        # Generate signals
        signals = pd.Series(0, index=df_copy.index)
        signals.loc[df_copy['close'] > df_copy['upper_band']] = 1  # Buy on upward breakout
        signals.loc[df_copy['close'] < df_copy['lower_band']] = -1  # Sell on downward breakout
        
        # Calculate returns
        result = self.calculate_returns(df, signals, f"ATR Breakout ({atr_window}, {atr_multiplier}x)")
        result['atr'] = df_copy['atr']
        result['upper_band'] = df_copy['upper_band']
        result['lower_band'] = df_copy['lower_band']
        
        # Calculate metrics
        metrics = self.calculate_metrics(result, f"ATR Breakout ({atr_window}, {atr_multiplier}x)")
        
        return result, metrics
    
    def run_all_strategies(
        self,
        df: pd.DataFrame,
        strategy_params: Dict = None
    ) -> Dict[str, Tuple[pd.DataFrame, Dict]]:
        """
        Run all three strategies on the given data.
        
        Args:
            df: DataFrame with OHLCV data
            strategy_params: Dictionary with custom strategy parameters
                          Format: {'sma': {'short': 20, 'long': 50},
                                   'bb': {'window': 20, 'num_std': 2.0},
                                   'atr': {'atr_window': 14, 'atr_multiplier': 2.0}}
                          If None, uses default parameters
            
        Returns:
            Dictionary mapping strategy names to (result DataFrame, metrics) tuples
        """
        if strategy_params is None:
            strategy_params = {}
        
        results = {}
        
        # Run SMA Crossover
        sma_params = strategy_params.get('sma', {})
        sma_result, sma_metrics = self.sma_crossover(
            df,
            short_window=sma_params.get('short', 20),
            long_window=sma_params.get('long', 50)
        )
        results['sma_crossover'] = (sma_result, sma_metrics)
        
        # Run Bollinger Bands
        bb_params = strategy_params.get('bb', {})
        bb_result, bb_metrics = self.bollinger_bands_mean_reversion(
            df,
            window=bb_params.get('window', 20),
            num_std=bb_params.get('num_std', 2.0)
        )
        results['bollinger_bands'] = (bb_result, bb_metrics)
        
        # Run ATR Breakout
        atr_params = strategy_params.get('atr', {})
        atr_result, atr_metrics = self.atr_volatility_breakout(
            df,
            atr_window=atr_params.get('atr_window', 14),
            atr_multiplier=atr_params.get('atr_multiplier', 2.0)
        )
        results['atr_breakout'] = (atr_result, atr_metrics)
        
        logger.info(f"Completed backtesting with all strategies")
        return results


def main():
    """
    Standalone execution for testing the backtester.
    """
    print("=" * 70)
    print("AXFI Backtester - Standalone Test")
    print("=" * 70)
    
    # Import data collector to get test data
    import sys
    from pathlib import Path
    
    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.data_collector import DataCollector
    
    # Initialize data collector and backtester
    data_collector = DataCollector()
    backtester = Backtester(initial_capital=100000, commission=0.001)
    
    # Get data for the first symbol in config
    symbol = data_collector.config['symbols'][0]
    print(f"\nTesting with symbol: {symbol}")
    print("-" * 70)
    
    # Load historical data
    print("Loading historical data...")
    df = data_collector.read_from_database(symbol=symbol)
    
    if df.empty:
        print(f"No data found for {symbol}, fetching...")
        df = data_collector.fetch_symbol(symbol, period="1y")
        if df.empty:
            print(f"Failed to fetch data for {symbol}")
            return
    
    # Sort by date to ensure proper order
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    
    print(f"Loaded {len(df)} data points from {df.index.min()} to {df.index.max()}")
    
    # Run all strategies
    print("\nRunning all strategies...")
    results = backtester.run_all_strategies(df)
    
    # Display results
    print("\n" + "=" * 70)
    print("STRATEGY PERFORMANCE METRICS")
    print("=" * 70)
    
    for strategy_name, (result_df, metrics) in results.items():
        print(f"\n{strategy_name.upper().replace('_', ' ')}")
        print("-" * 70)
        
        # Pretty print metrics
        print(f"  Strategy Name:    {metrics.get('strategy_name', 'N/A')}")
        print(f"  Total Return:     {metrics.get('total_return', 0)*100:.2f}%")
        print(f"  CAGR:             {metrics.get('cagr', 0)*100:.2f}%")
        print(f"  Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  Max Drawdown:     {metrics.get('max_drawdown', 0)*100:.2f}%")
        print(f"  Win Rate:         {metrics.get('win_rate', 0)*100:.2f}%")
        print(f"  # Trades:         {metrics.get('num_trades', 0)}")
        print(f"  Initial Capital:  ${metrics.get('initial_capital', 0):,.2f}")
        print(f"  Final Equity:     ${metrics.get('final_equity', 0):,.2f}")
        print(f"  Profit/Loss:      ${metrics.get('profit_loss', 0):,.2f}")
        
        # Show first 5 rows of equity curve
        print(f"\n  First 5 Rows of Equity Curve:")
        display_cols = ['close', 'signal', 'equity', 'buy_hold_equity']
        if any(col in result_df.columns for col in ['sma_short', 'bb_upper', 'atr']):
            indicator_cols = [col for col in ['sma_short', 'sma_long', 'bb_upper', 'bb_lower', 'atr'] if col in result_df.columns]
            display_cols.extend(indicator_cols)
        
        display_df = result_df[display_cols].head(5)
        print(display_df.to_string())
    
    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

