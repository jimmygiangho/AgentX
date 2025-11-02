"""
Scanner Agent - Discovers profitable trading setups across S&P 500
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ScannerAgent(BaseAgent):
    """Agent responsible for daily market scanning"""
    
    def __init__(self, config: dict, **kwargs):
        """Initialize scanner agent"""
        super().__init__(config, **kwargs)
        
        # Import modules here to avoid circular imports
        try:
            from core.data_collector_v2 import DataCollectorV2
            from core.strategy_scanner import StrategyScanner
            from core.ai_engine import AIEngine
            from core.research_library import ResearchLibrary
            
            self.data_collector = DataCollectorV2()
            self.research_library = ResearchLibrary(config)
            self.strategy_scanner = StrategyScanner(config)
            self.ai_engine = AIEngine()
            self.modules_loaded = True
        except Exception as e:
            logger.error(f"Failed to load modules: {e}")
            self.modules_loaded = False
    
    def run(self, symbols: List[str] = None, top_n: int = 10) -> Dict[str, Any]:
        """
        Run daily S&P 500 scan.
        
        Args:
            symbols: List of symbols to scan (default: S&P 500)
            top_n: Number of top recommendations to return
            
        Returns:
            Dictionary with scan results and top recommendations
        """
        if not self.modules_loaded:
            return {
                "status": "error",
                "message": "Modules not loaded",
                "recommendations": []
            }
        
        # Get symbols to scan
        if symbols is None:
            # Try to fetch from Wikipedia first, fallback to config
            try:
                from core.data_utils import fetch_sp500_tickers
                symbols = fetch_sp500_tickers()
                logger.info(f"Fetched {len(symbols)} S&P 500 symbols from Wikipedia")
            except Exception as e:
                logger.warning(f"Could not fetch S&P 500 from Wikipedia: {e}, using config")
                symbols = self.config.get('sp500_symbols', self.config.get('symbols', []))
        
        if isinstance(symbols, list) and len(symbols) > 0 and isinstance(symbols[0], str) and ',' in symbols[0]:
            # Handle case where sp500_symbols is list of comma-separated strings
            symbols = [s.strip() for sublist in symbols for s in sublist.split(',')]
        
        logger.info(f"Scanner Agent: Scanning {len(symbols)} symbols")
        
        all_recommendations = []
        processed_count = 0
        
        for symbol in symbols:
            try:
                # Fetch data
                df = self.data_collector.read_from_database(symbol=symbol)
                if df.empty:
                    logger.info(f"No data in DB for {symbol}, fetching live...")
                    df = self.data_collector.fetch_symbol(symbol, period="1y")
                    if not df.empty:
                        self.data_collector.save_to_database(df, symbol)
                
                if df.empty:
                    logger.warning(f"No data for {symbol}, skipping...")
                    continue
                
                df.set_index('date', inplace=True)
                
                # Calculate indicators
                df = self.research_library.calculate_trend_following(df)
                df = self.research_library.calculate_mean_reversion(df)
                df = self.research_library.calculate_volatility(df)
                df = self.research_library.calculate_volume_indicators(df)
                
                # Discover strategies
                discovered_strategies = self.strategy_scanner.discover_strategies(df, symbol)
                
                if not discovered_strategies:
                    continue
                
                # Convert to backtest format
                from core.backtester import Backtester
                backtester = Backtester(
                    initial_capital=self.config.get('backtest', {}).get('initial_capital', 100000),
                    commission=self.config.get('backtest', {}).get('commission', 0.001)
                )
                
                # Run backtests
                strategy_results = {}
                for strat in discovered_strategies[:5]:  # Top 5 per symbol
                    strategy_type = strat['type']
                    params = strat['params']
                    
                    try:
                        if strategy_type == 'sma_crossover':
                            result_df, metrics = backtester.sma_crossover(df, **params)
                            strategy_results[f"sma_crossover_{params['short_window']}_{params['long_window']}"] = (result_df, metrics)
                        elif strategy_type == 'bollinger_bands':
                            result_df, metrics = backtester.bollinger_bands_mean_reversion(df, **params)
                            strategy_results[f"bollinger_bands_{params['window']}_{params['num_std']}"] = (result_df, metrics)
                        elif strategy_type == 'atr_breakout':
                            result_df, metrics = backtester.atr_volatility_breakout(df, 
                                                                                    atr_window=params['atr_period'],
                                                                                    atr_multiplier=params['multiplier'])
                            strategy_results[f"atr_breakout_{params['atr_period']}_{params['multiplier']}"] = (result_df, metrics)
                    except Exception as e:
                        logger.error(f"Error running strategy {strat['name']}: {e}")
                        continue
                
                # Rank strategies
                ranked = self.ai_engine.rank_strategies(strategy_results, symbol)
                
                if ranked:
                    # Get top recommendation for this symbol
                    top_rec = ranked[0]
                    all_recommendations.append({
                        'symbol': symbol,
                        'strategy': top_rec['strategy'],
                        'score': top_rec['score'],
                        'confidence': min(95, max(60, int(top_rec['score'] * 100))),  # Scale to 60-95%
                        'cagr': top_rec['metrics'].get('cagr', 0),
                        'sharpe': top_rec['metrics'].get('sharpe_ratio', 0),
                        'max_dd': abs(top_rec['metrics'].get('max_drawdown', 0)),
                        'win_rate': top_rec['metrics'].get('win_rate', 0),
                        'explanation': top_rec['explanation']
                    })
                    
                    processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        # Sort by score and get top N
        all_recommendations.sort(key=lambda x: x['score'], reverse=True)
        top_recommendations = all_recommendations[:top_n]
        
        # Generate recommendations with entry/stop/target
        for rec in top_recommendations:
            # Try to get current price
            try:
                quote = self.data_collector.get_latest_quote(rec['symbol'])
                current_price = quote.get('price', 100)  # Default if missing
            except:
                current_price = 100
            
            # Generate entry/stop/target based on volatility and momentum
            atr_mult = abs(rec['max_dd']) if rec['max_dd'] > 0 else 0.05
            entry = current_price * (1 - 0.01)  # Slightly below current for entry
            stop = current_price * (1 - atr_mult * 2)  # 2x ATR below
            target = current_price * (1 + atr_mult * 3)  # 3x ATR above
            
            rec['entry'] = round(entry, 2)
            rec['stop_loss'] = round(stop, 2)
            rec['take_profit'] = round(target, 2)
            rec['suggested_pct'] = round(min(10, max(2, rec['confidence'] / 10)), 1)  # 2-10% based on confidence
        
        logger.info(f"Scanner Agent: Completed scan, {processed_count} symbols processed, {len(top_recommendations)} top recommendations")
        
        return {
            "status": "success",
            "processed_count": processed_count,
            "total_symbols": len(symbols),
            "recommendations": top_recommendations
        }


if __name__ == "__main__":
    # Test scanner agent
    import yaml
    
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    agent = ScannerAgent(config)
    
    # Test with limited symbols
    test_symbols = ["AAPL", "MSFT", "GOOGL"]
    result = agent.run(symbols=test_symbols, top_n=5)
    print(f"Scanner result: {result}")
