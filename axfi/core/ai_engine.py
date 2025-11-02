"""
AXFI AI Signal Engine Module
Ranks and scores trading strategies based on performance metrics
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AIEngine:
    """
    Ranks and scores trading strategies based on performance metrics.
    
    Generates explainable recommendations for each symbol/strategy combination.
    Uses a weighted scoring system based on key performance indicators.
    """
    
    def __init__(
        self,
        sharpe_weight: float = 0.35,
        cagr_weight: float = 0.30,
        max_dd_weight: float = -0.20,
        win_rate_weight: float = 0.15
    ):
        """
        Initialize the AI engine with metric weights.
        
        Args:
            sharpe_weight: Weight for Sharpe Ratio (higher is better)
            cagr_weight: Weight for CAGR (higher is better)
            max_dd_weight: Weight for Max Drawdown (negative, lower is better)
            win_rate_weight: Weight for Win Rate (higher is better)
        """
        self.sharpe_weight = sharpe_weight
        self.cagr_weight = cagr_weight
        self.max_dd_weight = max_dd_weight
        self.win_rate_weight = win_rate_weight
        
        logger.info(f"Initialized AI Engine with weights: Sharpe={sharpe_weight}, "
                   f"CAGR={cagr_weight}, MaxDD={max_dd_weight}, WinRate={win_rate_weight}")
    
    def normalize_metric(self, value: float, min_val: float, max_val: float) -> float:
        """
        Normalize a metric value to [0, 1] range using min-max scaling.
        
        Args:
            value: Value to normalize
            min_val: Minimum value in the range
            max_val: Maximum value in the range
            
        Returns:
            Normalized value in [0, 1] range
        """
        if max_val == min_val:
            return 0.5  # Return middle value if all values are the same
        
        normalized = (value - min_val) / (max_val - min_val)
        
        # Clip to [0, 1] range
        return max(0, min(1, normalized))
    
    def calculate_score(self, metrics: Dict) -> float:
        """
        Calculate a composite score for a strategy based on weighted metrics.
        
        Args:
            metrics: Dictionary containing performance metrics
            
        Returns:
            Composite score (higher is better)
        """
        # Extract metrics
        sharpe = metrics.get('sharpe_ratio', 0)
        cagr = metrics.get('cagr', 0)
        max_dd = metrics.get('max_drawdown', 0)
        win_rate = metrics.get('win_rate', 0)
        
        # Calculate weighted score
        score = (
            self.sharpe_weight * sharpe +
            self.cagr_weight * cagr +
            self.max_dd_weight * max_dd +  # max_dd is negative, so negative weight makes it positive
            self.win_rate_weight * win_rate
        )
        
        return score
    
    def rank_strategies(
        self,
        strategy_results: Dict[str, Dict],
        symbol: str
    ) -> List[Dict]:
        """
        Rank strategies for a single symbol based on their metrics.
        
        Args:
            strategy_results: Dictionary mapping strategy names to (result_df, metrics) tuples
            symbol: Stock symbol
            
        Returns:
            List of ranked strategy dictionaries with scores and explanations
        """
        # Extract all metrics
        strategies = []
        all_metrics = []
        
        for strategy_name, (result_df, metrics) in strategy_results.items():
            strategies.append({
                'symbol': symbol,
                'strategy': strategy_name,
                'result_df': result_df,
                'metrics': metrics
            })
            all_metrics.append(metrics)
        
        if not all_metrics:
            logger.warning(f"No strategies to rank for {symbol}")
            return []
        
        # Calculate ranges for normalization
        sharpe_values = [m.get('sharpe_ratio', 0) for m in all_metrics]
        cagr_values = [m.get('cagr', 0) for m in all_metrics]
        max_dd_values = [m.get('max_drawdown', 0) for m in all_metrics]
        win_rate_values = [m.get('win_rate', 0) for m in all_metrics]
        
        sharpe_min, sharpe_max = min(sharpe_values), max(sharpe_values)
        cagr_min, cagr_max = min(cagr_values), max(cagr_values)
        max_dd_min, max_dd_max = min(max_dd_values), max(max_dd_values)
        win_rate_min, win_rate_max = min(win_rate_values), max(win_rate_values)
        
        # Calculate normalized scores and rankings
        ranked_strategies = []
        
        for strategy in strategies:
            metrics = strategy['metrics']
            
            # Normalize metrics
            norm_sharpe = self.normalize_metric(metrics.get('sharpe_ratio', 0), sharpe_min, sharpe_max)
            norm_cagr = self.normalize_metric(metrics.get('cagr', 0), cagr_min, cagr_max)
            norm_max_dd = self.normalize_metric(metrics.get('max_drawdown', 0), max_dd_min, max_dd_max)
            norm_win_rate = self.normalize_metric(metrics.get('win_rate', 0), win_rate_min, win_rate_max)
            
            # Calculate raw score (without normalization first to maintain interpretability)
            score = self.calculate_score(metrics)
            
            # Generate explanation
            explanation = self.generate_explanation(strategy['strategy'], symbol, metrics)
            
            ranked_strategies.append({
                'symbol': symbol,
                'strategy': strategy['strategy'],
                'score': score,
                'normalized_sharpe': norm_sharpe,
                'normalized_cagr': norm_cagr,
                'normalized_max_dd': norm_max_dd,
                'normalized_win_rate': norm_win_rate,
                'metrics': metrics,
                'explanation': explanation
            })
        
        # Sort by score (descending)
        ranked_strategies.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Ranked {len(ranked_strategies)} strategies for {symbol}")
        return ranked_strategies
    
    def generate_explanation(
        self,
        strategy_name: str,
        symbol: str,
        metrics: Dict
    ) -> str:
        """
        Generate a human-readable explanation for a strategy recommendation.
        
        Args:
            strategy_name: Name of the strategy
            symbol: Stock symbol
            metrics: Performance metrics dictionary
            
        Returns:
            Explanation string
        """
        # Extract metrics
        sharpe = metrics.get('sharpe_ratio', 0)
        cagr = metrics.get('cagr', 0)
        max_dd = metrics.get('max_drawdown', 0)
        win_rate = metrics.get('win_rate', 0)
        total_return = metrics.get('total_return', 0)
        num_trades = metrics.get('num_trades', 0)
        
        # Format values for display
        cagr_pct = cagr * 100
        sharpe_str = f"{sharpe:.2f}"
        dd_pct = abs(max_dd * 100)
        win_pct = win_rate * 100
        
        # Strategy type detection
        strategy_type = self._detect_strategy_type(strategy_name)
        
        # Signal strength assessment
        signal_strength = self._assess_signal_strength(sharpe, cagr, max_dd, win_rate)
        
        # Build explanation
        explanation_parts = []
        
        # Leading assessment
        if signal_strength == "Strong":
            leading = f"Strong {strategy_type} opportunity for {symbol}"
        elif signal_strength == "Moderate":
            leading = f"Moderate {strategy_type} signal for {symbol}"
        else:
            leading = f"Weak {strategy_type} potential for {symbol}"
        
        explanation_parts.append(leading)
        
        # Add key metrics
        explanation_parts.append(f"with {cagr_pct:.1f}% CAGR, Sharpe {sharpe_str}")
        
        # Add context based on strategy type
        if "SMA" in strategy_name or "momentum" in strategy_name.lower():
            explanation_parts.append("– bullish momentum trend detected")
        elif "Bollinger" in strategy_name or "mean reversion" in strategy_name.lower():
            explanation_parts.append("– mean reversion opportunity identified")
        elif "ATR" in strategy_name or "breakout" in strategy_name.lower():
            explanation_parts.append("– volatility breakout detected")
        
        # Add risk warning if needed
        if dd_pct > 20:
            explanation_parts.append(f"[WARNING] High drawdown risk ({dd_pct:.1f}% max DD)")
        
        # Add trade frequency info
        if num_trades > 0:
            explanation_parts.append(f"({num_trades} trades, {win_pct:.0f}% win rate)")
        
        explanation = " ".join(explanation_parts)
        
        return explanation
    
    def _detect_strategy_type(self, strategy_name: str) -> str:
        """
        Detect the general type of strategy from its name.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Strategy type string
        """
        strategy_name_lower = strategy_name.lower()
        
        if "momentum" in strategy_name_lower or "crossover" in strategy_name_lower or "sma" in strategy_name_lower:
            return "momentum"
        elif "mean reversion" in strategy_name_lower or "bollinger" in strategy_name_lower:
            return "mean reversion"
        elif "volatility" in strategy_name_lower or "breakout" in strategy_name_lower or "atr" in strategy_name_lower:
            return "volatility"
        else:
            return "trading"
    
    def _assess_signal_strength(
        self,
        sharpe: float,
        cagr: float,
        max_dd: float,
        win_rate: float
    ) -> str:
        """
        Assess overall signal strength based on multiple metrics.
        
        Args:
            sharpe: Sharpe Ratio
            cagr: CAGR
            max_dd: Maximum Drawdown (negative value)
            win_rate: Win Rate
            
        Returns:
            Signal strength string: "Strong", "Moderate", or "Weak"
        """
        score = self.calculate_score({
            'sharpe_ratio': sharpe,
            'cagr': cagr,
            'max_drawdown': max_dd,
            'win_rate': win_rate
        })
        
        if score > 0.5:
            return "Strong"
        elif score > 0:
            return "Moderate"
        else:
            return "Weak"
    
    def rank_all_symbols(
        self,
        all_symbol_results: Dict[str, Dict[str, tuple]]
    ) -> List[Dict]:
        """
        Rank strategies across all symbols and return top recommendations.
        
        Args:
            all_symbol_results: Dictionary mapping symbols to strategy_results dictionaries
                              Format: {symbol: {strategy_name: (result_df, metrics), ...}, ...}
            
        Returns:
            List of all ranked strategies across all symbols
        """
        all_ranked = []
        
        for symbol, strategy_results in all_symbol_results.items():
            ranked = self.rank_strategies(strategy_results, symbol)
            all_ranked.extend(ranked)
        
        # Sort all by score (descending)
        all_ranked.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Ranked {len(all_ranked)} strategies across {len(all_symbol_results)} symbols")
        return all_ranked
    
    def get_top_recommendations(
        self,
        all_symbol_results: Dict[str, Dict[str, tuple]],
        top_n: int = 5
    ) -> List[Dict]:
        """
        Get top N recommendations across all symbols and strategies.
        
        Args:
            all_symbol_results: Dictionary mapping symbols to strategy_results
            top_n: Number of top recommendations to return
            
        Returns:
            List of top N ranked strategies
        """
        all_ranked = self.rank_all_symbols(all_symbol_results)
        top_recommendations = all_ranked[:top_n]
        
        logger.info(f"Selected top {len(top_recommendations)} recommendations")
        return top_recommendations


def main():
    """
    Standalone execution for testing the AI engine.
    """
    print("=" * 80)
    print("AXFI AI Signal Engine - Standalone Test")
    print("=" * 80)
    
    # Import required modules
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.data_collector import DataCollector
    from core.backtester import Backtester
    
    # Initialize components
    data_collector = DataCollector()
    backtester = Backtester(initial_capital=100000, commission=0.001)
    ai_engine = AIEngine()
    
    # Get symbols from config
    symbols = data_collector.config['symbols'][:3]  # Test with first 3 symbols
    print(f"\nTesting with symbols: {', '.join(symbols)}")
    print("-" * 80)
    
    # Collect results for each symbol
    all_symbol_results = {}
    
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        
        # Load data
        df = data_collector.read_from_database(symbol=symbol)
        
        if df.empty:
            print(f"No data for {symbol}, skipping...")
            continue
        
        df = df.sort_values('date')
        df.set_index('date', inplace=True)
        
        # Run all strategies
        strategy_results = backtester.run_all_strategies(df)
        all_symbol_results[symbol] = strategy_results
    
    if not all_symbol_results:
        print("No data available for testing")
        return
    
    # Get top recommendations
    print("\n" + "=" * 80)
    print("TOP RECOMMENDATIONS")
    print("=" * 80)
    
    top_recommendations = ai_engine.get_top_recommendations(all_symbol_results, top_n=5)
    
    for i, rec in enumerate(top_recommendations, 1):
        print(f"\n[{i}] Score: {rec['score']:.4f}")
        print(f"    Symbol: {rec['symbol']}")
        print(f"    Strategy: {rec['strategy']}")
        print(f"    Explanation: {rec['explanation']}")
        print(f"\n    Detailed Metrics:")
        metrics = rec['metrics']
        print(f"      - CAGR: {metrics['cagr']*100:.2f}%")
        print(f"      - Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"      - Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"      - Win Rate: {metrics['win_rate']*100:.2f}%")
        print(f"      - Total Return: {metrics['total_return']*100:.2f}%")
        print(f"      - # Trades: {metrics['num_trades']}")
        print(f"      - Final Equity: ${metrics['final_equity']:,.2f}")
    
    print("\n" + "=" * 80)
    print("ALL SYMBOL RANKINGS")
    print("=" * 80)
    
    # Show rankings for each symbol
    for symbol in symbols:
        if symbol not in all_symbol_results:
            continue
        
        print(f"\n{symbol} Strategy Rankings:")
        print("-" * 80)
        
        ranked = ai_engine.rank_strategies(all_symbol_results[symbol], symbol)
        
        for i, rec in enumerate(ranked, 1):
            print(f"  {i}. {rec['strategy']}: Score {rec['score']:.4f}")
            print(f"     {rec['explanation']}")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

