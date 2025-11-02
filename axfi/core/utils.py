"""
AXFI Utility Functions
Common utilities for data processing, formatting, and calculations
"""

import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def format_currency(value: float, decimals: int = 2) -> str:
    """
    Format a number as currency.
    
    Args:
        value: Numeric value
        decimals: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    return f"${value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a number as percentage.
    
    Args:
        value: Numeric value (0.25 for 25%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio for a returns series.
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (default 0.0)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
    return sharpe


def calculate_max_drawdown(equity: pd.Series) -> float:
    """
    Calculate maximum drawdown from an equity curve.
    
    Args:
        equity: Series of equity values
        
    Returns:
        Maximum drawdown as a negative value
    """
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    return drawdown.min()


def calculate_win_rate(trades: pd.DataFrame) -> float:
    """
    Calculate win rate from trades DataFrame.
    
    Args:
        trades: DataFrame with 'pnl' column
        
    Returns:
        Win rate as a fraction (0.0 to 1.0)
    """
    if 'pnl' not in trades.columns or len(trades) == 0:
        return 0.0
    
    winning_trades = (trades['pnl'] > 0).sum()
    win_rate = winning_trades / len(trades)
    return win_rate


def calculate_cagr(final_value: float, initial_value: float, 
                   periods: int) -> float:
    """
    Calculate Compound Annual Growth Rate.
    
    Args:
        final_value: Ending value
        initial_value: Starting value
        periods: Number of periods
        
    Returns:
        CAGR as a decimal (0.25 for 25%)
    """
    if initial_value <= 0 or periods <= 0:
        return 0.0
    
    total_return = final_value / initial_value
    cagr = (total_return ** (252 / periods)) - 1
    return cagr


def validate_ohlcv(df: pd.DataFrame) -> bool:
    """
    Validate OHLCV DataFrame has required columns and valid data.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Missing required columns: {required_columns}")
        return False
    
    # Check for negative values
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if (df[col] < 0).any():
            logger.error(f"Negative values found in {col}")
            return False
    
    # Check OHLC relationships
    if (df['high'] < df['low']).any():
        logger.error("High < Low found in data")
        return False
    
    if (df['close'] > df['high']).any() or (df['close'] < df['low']).any():
        logger.error("Close outside High-Low range found in data")
        return False
    
    return True


def resample_data(df: pd.DataFrame, period: str = 'D') -> pd.DataFrame:
    """
    Resample price data to a different time period.
    
    Args:
        df: DataFrame with datetime index
        period: Resample period ('D' for daily, 'W' for weekly, etc.)
        
    Returns:
        Resampled DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("DataFrame index must be DatetimeIndex for resampling")
        return df
    
    resampled = pd.DataFrame()
    resampled['open'] = df['open'].resample(period).first()
    resampled['high'] = df['high'].resample(period).max()
    resampled['low'] = df['low'].resample(period).min()
    resampled['close'] = df['close'].resample(period).last()
    resampled['volume'] = df['volume'].resample(period).sum()
    
    return resampled


def main():
    """Standalone test of utils"""
    print("=" * 80)
    print("AXFI Utils - Standalone Test")
    print("=" * 80)
    
    # Test currency formatting
    print("\n1. Testing currency formatting...")
    print(f"   $1,234,567.89 = {format_currency(1234567.89)}")
    print(f"   $123.45 = {format_currency(123.45)}")
    
    # Test percentage formatting
    print("\n2. Testing percentage formatting...")
    print(f"   0.25 = {format_percentage(0.25)}")
    print(f"   0.1234 = {format_percentage(0.1234)}")
    
    # Test Sharpe ratio
    print("\n3. Testing Sharpe ratio calculation...")
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
    sharpe = calculate_sharpe_ratio(returns)
    print(f"   Sharpe ratio: {sharpe:.4f}")
    
    # Test max drawdown
    print("\n4. Testing max drawdown calculation...")
    equity = pd.Series([100, 105, 103, 110, 108, 115])
    max_dd = calculate_max_drawdown(equity)
    print(f"   Max drawdown: {format_percentage(max_dd)}")
    
    # Test win rate
    print("\n5. Testing win rate calculation...")
    trades = pd.DataFrame({'pnl': [100, -50, 200, -30, 150, -20]})
    win_rate = calculate_win_rate(trades)
    print(f"   Win rate: {format_percentage(win_rate)}")
    
    # Test CAGR
    print("\n6. Testing CAGR calculation...")
    cagr = calculate_cagr(150000, 100000, 252)
    print(f"   CAGR: {format_percentage(cagr)}")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

