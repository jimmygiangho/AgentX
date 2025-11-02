"""
AXFI Strategy Scanner
Automatically discovers and evaluates trading strategies through parameter optimization
"""

import logging
from typing import Dict, List, Tuple, Any, Callable
import pandas as pd
import numpy as np
from itertools import product

logger = logging.getLogger(__name__)


class StrategyScanner:
    """
    Scans and discovers profitable trading strategies through parameter optimization.
    
    Features:
    - Grid search optimization for strategy parameters
    - Automatic strategy discovery from predefined parameter spaces
    - Evaluation based on Sharpe, CAGR, Max DD, Win Rate
    - Ranking by composite score
    """
    
    def __init__(self, config: dict):
        """
        Initialize the strategy scanner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.optimization_method = config.get('scanner', {}).get('optimization_method', 'grid')
        self.max_strategies = config.get('scanner', {}).get('max_strategies_per_symbol', 10)
        self.min_trades = config.get('scanner', {}).get('min_trades_for_evaluation', 5)
        self.grid_samples = config.get('optimization', {}).get('grid_search_samples', 50)
        
    def discover_strategies(self, df: pd.DataFrame, symbol: str) -> List[Dict[str, Any]]:
        """
        Discover profitable strategies for a symbol through systematic parameter scanning.
        
        Args:
            df: Price data DataFrame with OHLCV
            symbol: Stock symbol
            
        Returns:
            List of discovered strategy configurations with metrics
        """
        logger.info(f"Discovering strategies for {symbol}")
        
        if df.empty:
            logger.warning(f"Empty DataFrame for {symbol}")
            return []
        
        # Generate candidate strategies
        candidate_strategies = self._generate_candidate_strategies()
        
        # Evaluate each candidate
        discovered = []
        for strategy_config in candidate_strategies:
            try:
                result = self._evaluate_strategy(df, strategy_config)
                if result:
                    discovered.append(result)
            except Exception as e:
                logger.error(f"Error evaluating strategy {strategy_config.get('name', 'unknown')}: {e}")
                continue
        
        # Rank by composite score
        discovered = sorted(discovered, key=lambda x: x.get('composite_score', 0), reverse=True)
        
        # Return top N
        top_strategies = discovered[:self.max_strategies]
        
        logger.info(f"Discovered {len(top_strategies)} strategies for {symbol}")
        
        return top_strategies
    
    def _generate_candidate_strategies(self) -> List[Dict[str, Any]]:
        """
        Generate candidate strategy configurations from config parameters.
        
        Returns:
            List of strategy configurations
        """
        candidates = []
        
        # Define strategy parameter spaces
        # SMA Crossover strategies
        sma_short_windows = [10, 20, 30, 50]
        sma_long_windows = [50, 100, 150, 200]
        
        for short, long in product(sma_short_windows, sma_long_windows):
            if short < long:
                candidates.append({
                    'type': 'sma_crossover',
                    'name': f'SMA Cross {short}/{long}',
                    'params': {'short_window': short, 'long_window': long}
                })
        
        # Bollinger Band strategies
        bb_windows = [10, 15, 20, 25]
        bb_stds = [1.5, 2.0, 2.5]
        
        for window, std in product(bb_windows, bb_stds):
            candidates.append({
                'type': 'bollinger_bands',
                'name': f'BB Mean Rev {window}/{std}',
                'params': {'window': window, 'num_std': std}
            })
        
        # ATR Breakout strategies
        atr_periods = [10, 14, 20]
        atr_multipliers = [1.5, 2.0, 2.5]
        
        for period, multiplier in product(atr_periods, atr_multipliers):
            candidates.append({
                'type': 'atr_breakout',
                'name': f'ATR Breakout {period}/{multiplier}',
                'params': {'atr_period': period, 'multiplier': multiplier}
            })
        
        logger.info(f"Generated {len(candidates)} candidate strategies")
        
        return candidates
    
    def _evaluate_strategy(self, df: pd.DataFrame, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single strategy configuration.
        
        Args:
            df: Price data DataFrame
            strategy_config: Strategy configuration dictionary
            
        Returns:
            Strategy configuration with metrics added
        """
        from core.backtester import Backtester
        
        backtester = Backtester(
            initial_capital=self.config['backtest']['initial_capital'],
            commission=self.config['backtest']['commission']
        )
        
        # Run strategy based on type
        strategy_type = strategy_config['type']
        params = strategy_config['params']
        
        try:
            if strategy_type == 'sma_crossover':
                result_df, metrics = backtester.sma_crossover(
                    df, 
                    short_window=params['short_window'],
                    long_window=params['long_window']
                )
            elif strategy_type == 'bollinger_bands':
                result_df, metrics = backtester.bollinger_bands_mean_reversion(
                    df,
                    window=params['window'],
                    num_std=params['num_std']
                )
            elif strategy_type == 'atr_breakout':
                result_df, metrics = backtester.atr_volatility_breakout(
                    df,
                    atr_window=params['atr_period'],
                    atr_multiplier=params['multiplier']
                )
            else:
                return None
            
            # Check minimum trades requirement
            if metrics.get('num_trades', 0) < self.min_trades:
                return None
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(metrics)
            
            # Add to strategy config
            result = {
                **strategy_config,
                'metrics': metrics,
                'composite_score': composite_score
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error running {strategy_config['name']}: {e}")
            return None
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate composite score for ranking strategies.
        
        Uses weighted average of normalized metrics.
        
        Args:
            metrics: Metrics dictionary
            
        Returns:
            Composite score
        """
        sharpe = metrics.get('sharpe_ratio', 0)
        cagr = metrics.get('cagr', 0)
        max_dd = abs(metrics.get('max_drawdown', 0))
        win_rate = metrics.get('win_rate', 0)
        
        # Normalize metrics (simple linear scaling)
        sharpe_norm = min(max(sharpe / 3.0, 0), 1.0)  # Assume max Sharpe ~3
        cagr_norm = min(max(cagr / 0.5, 0), 1.0)  # Assume max CAGR ~50%
        max_dd_norm = min(max(1 - (max_dd / 0.5), 0), 1.0)  # Penalize high drawdowns
        win_rate_norm = win_rate  # Already 0-1
        
        # Weighted composite score
        composite = (
            0.35 * sharpe_norm +
            0.30 * cagr_norm +
            0.20 * max_dd_norm +
            0.15 * win_rate_norm
        )
        
        return composite
    
    def optimize_parameters(self, strategy_func: Callable, df: pd.DataFrame, 
                           param_space: Dict[str, List]) -> Tuple[Dict, float]:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            strategy_func: Strategy function to optimize
            df: Price data DataFrame
            param_space: Dictionary of parameter names to possible values
            
        Returns:
            Tuple of (best_parameters, best_score)
        """
        logger.info(f"Optimizing parameters using grid search")
        
        best_params = None
        best_score = float('-inf')
        
        # Generate all parameter combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        total_combinations = np.prod([len(v) for v in param_values])
        
        # Limit combinations for performance
        if total_combinations > self.grid_samples:
            logger.info(f"Limiting grid search to {self.grid_samples} samples from {total_combinations} combinations")
            
            # Sample combinations randomly
            indices = []
            for _ in range(self.grid_samples):
                sample_params = {name: np.random.choice(values) for name, values in zip(param_names, param_values)}
                indices.append(tuple(sample_params.values()))
            indices = list(set(indices))  # Remove duplicates
        else:
            indices = list(product(*param_values))
        
        # Evaluate each combination
        for idx, param_vals in enumerate(indices):
            params = dict(zip(param_names, param_vals))
            
            try:
                result_df, metrics = strategy_func(df, **params)
                score = self._calculate_composite_score(metrics)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                logger.error(f"Error evaluating parameters {params}: {e}")
                continue
        
        logger.info(f"Best parameters: {best_params}, Score: {best_score:.4f}")
        
        return best_params, best_score


def main():
    """Standalone test of StrategyScanner"""
    import yaml
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize scanner
    scanner = StrategyScanner(config)
    
    print("=" * 80)
    print("AXFI Strategy Scanner - Standalone Test")
    print("=" * 80)
    print(f"\n[OK] Strategy Scanner initialized")
    print(f"Optimization method: {scanner.optimization_method}")
    print(f"Max strategies per symbol: {scanner.max_strategies}")
    print(f"Min trades for evaluation: {scanner.min_trades}")
    
    # Load sample data
    from core.data_collector import DataCollector
    dc = DataCollector(config_path="config.yaml")
    df = dc.read_from_database(symbol="AAPL")
    
    if not df.empty:
        df.set_index('date', inplace=True)
        print(f"\n[OK] Loaded {len(df)} rows for AAPL")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # Discover strategies
        print(f"\nScanning for profitable strategies...")
        discovered = scanner.discover_strategies(df, "AAPL")
        
        print(f"\n[OK] Discovered {len(discovered)} strategies")
        
        # Display top 5
        print("\nTop 5 Strategies:")
        print("-" * 80)
        for i, strategy in enumerate(discovered[:5], 1):
            metrics = strategy['metrics']
            print(f"{i}. {strategy['name']}")
            print(f"   Composite Score: {strategy['composite_score']:.4f}")
            print(f"   Sharpe: {metrics['sharpe_ratio']:.3f}, CAGR: {metrics['cagr']*100:.2f}%, "
                  f"Max DD: {metrics['max_drawdown']*100:.2f}%, Win Rate: {metrics['win_rate']*100:.1f}%")
            print(f"   Total Trades: {metrics.get('num_trades', 0)}")
            print()
    else:
        print("[ERROR] No data available for AAPL. Please run data_collector.py first.")
    
    print("=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
