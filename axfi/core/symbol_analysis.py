"""
AXFI Symbol Analysis Agent
Deep analysis of individual symbols with short/mid/long-term recommendations
"""

import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SymbolAnalysisAgent:
    """
    Comprehensive symbol analysis agent.
    
    Features:
    - Short-term analysis (1-3 days) - momentum and volatility signals
    - Mid-term analysis (1-4 weeks) - trend following and mean reversion
    - Long-term analysis (1-6 months) - structural trends and sector rotation
    - Actionable recommendations with confidence scores
    - Exit strategies and risk management
    """
    
    def __init__(self, config: dict):
        """
        Initialize the symbol analysis agent.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
    def analyze_symbol(
        self,
        df: pd.DataFrame,
        symbol: str,
        short_term_days: int = 3,
        mid_term_weeks: int = 4,
        long_term_months: int = 6
    ) -> Dict:
        """
        Perform comprehensive analysis across multiple timeframes.
        
        Args:
            df: Price data DataFrame with OHLCV
            symbol: Stock symbol
            short_term_days: Days for short-term analysis
            mid_term_weeks: Weeks for mid-term analysis
            long_term_months: Months for long-term analysis
            
        Returns:
            Comprehensive analysis dictionary
        """
        logger.info(f"Analyzing {symbol} across multiple timeframes")
        
        if df.empty:
            logger.warning(f"Empty DataFrame for {symbol}")
            return self._empty_analysis(symbol)
        
        # Load research library and indicators
        from core.research_library import ResearchLibrary
        library = ResearchLibrary(self.config)
        
        # Calculate all indicators
        df_with_indicators = df.copy()
        df_with_indicators = library.calculate_trend_following(df_with_indicators)
        df_with_indicators = library.calculate_mean_reversion(df_with_indicators)
        df_with_indicators = library.calculate_volatility(df_with_indicators)
        df_with_indicators = library.calculate_volume_indicators(df_with_indicators)
        
        # Perform timeframe-specific analysis
        short_term = self._analyze_short_term(df_with_indicators, short_term_days)
        mid_term = self._analyze_mid_term(df_with_indicators, mid_term_weeks)
        long_term = self._analyze_long_term(df_with_indicators, long_term_months)
        
        # Overall recommendation
        overall = self._synthesize_recommendation(short_term, mid_term, long_term)
        
        # Get latest live price for the analysis (BEFORE generating exit strategies)
        try:
            from core.data_collector_v2 import DataCollectorV2
            collector = DataCollectorV2()
            quote = collector.get_latest_quote(symbol)
            current_price = quote.get('price', df['close'].iloc[-1] if not df.empty else 0)
            price_provider = quote.get('provider', 'unknown')
            logger.info(f"Fetched live price for {symbol}: ${current_price:.2f} from {price_provider}")
        except Exception as e:
            logger.warning(f"Could not fetch live price for {symbol}: {e}")
            current_price = df['close'].iloc[-1] if not df.empty else 0
            price_provider = 'database'
        
        # Exit strategy analysis (use live price)
        exit_strategies = self._generate_exit_strategies(df_with_indicators, live_price=current_price)
        
        analysis = {
            'symbol': symbol,
            'current_price': float(current_price),  # Use live price from quote
            'price_provider': price_provider,  # Indicate data source
            'short_term': short_term,
            'mid_term': mid_term,
            'long_term': long_term,
            'overall_recommendation': overall,
            'exit_strategies': exit_strategies,
            'risk_assessment': self._assess_risk(df_with_indicators),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"Completed analysis for {symbol}")
        
        return analysis
    
    def _analyze_short_term(self, df: pd.DataFrame, days: int = 3) -> Dict:
        """
        Short-term momentum and volatility analysis.
        
        Args:
            df: DataFrame with indicators
            days: Lookback period in days
            
        Returns:
            Short-term analysis dictionary
        """
        recent = df.tail(days)
        current_price = df['close'].iloc[-1]
        
        # Momentum signals
        price_change_pct = ((current_price - df['close'].iloc[-days]) / df['close'].iloc[-days]) * 100
        rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
        macd_signal = 'bullish' if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else 'bearish'
        
        # Volatility signals
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0
        atr_pct = (atr / current_price) * 100
        
        # Volume analysis
        avg_volume = df['volume'].tail(20).mean()
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Generate signals
        buy_signals = []
        sell_signals = []
        
        if rsi < 40:
            buy_signals.append("Oversold condition (RSI < 40)")
        if rsi > 60:
            sell_signals.append("Overbought condition (RSI > 60)")
        if macd_signal == 'bullish':
            buy_signals.append("Positive MACD crossover")
        if price_change_pct > 2:
            buy_signals.append("Strong momentum")
        if volume_ratio > 1.5:
            buy_signals.append("High volume surge")
        if atr_pct > 5:
            sell_signals.append("Elevated volatility")
        
        # Recommendation
        signal_strength = len(buy_signals) - len(sell_signals)
        if signal_strength >= 2:
            recommendation = "STRONG BUY"
            confidence = 0.85
        elif signal_strength >= 1:
            recommendation = "BUY"
            confidence = 0.70
        elif signal_strength >= -1:
            recommendation = "HOLD"
            confidence = 0.60
        elif signal_strength >= -2:
            recommendation = "SELL"
            confidence = 0.70
        else:
            recommendation = "STRONG SELL"
            confidence = 0.85
        
        return {
            'timeframe_days': days,
            'price_change_pct': price_change_pct,
            'rsi': float(rsi),
            'macd_signal': macd_signal,
            'atr_pct': atr_pct,
            'volume_ratio': volume_ratio,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'recommendation': recommendation,
            'confidence': confidence,
            'take_profit': current_price * (1.02 if signal_strength > 0 else 0.98),
            'stop_loss': current_price * (0.97 if signal_strength > 0 else 1.03)
        }
    
    def _analyze_mid_term(self, df: pd.DataFrame, weeks: int = 4) -> Dict:
        """
        Mid-term trend and mean reversion analysis.
        
        Args:
            df: DataFrame with indicators
            weeks: Lookback period in weeks
            
        Returns:
            Mid-term analysis dictionary
        """
        days = weeks * 5  # Approximate trading days
        recent = df.tail(days)
        
        # Trend analysis
        ema_20 = df['ema_20'].iloc[-1] if 'ema_20' in df.columns else df['close'].iloc[-1]
        ema_50 = df['ema_50'].iloc[-1] if 'ema_50' in df.columns else df['close'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        trend = "Uptrend" if ema_20 > ema_50 else "Downtrend"
        
        # Bollinger Bands
        bb_percent = df['bb_percent'].iloc[-1] if 'bb_percent' in df.columns else 0.5
        bb_signal = "Neutral"
        if bb_percent < 0.2:
            bb_signal = "Oversold"
        elif bb_percent > 0.8:
            bb_signal = "Overbought"
        
        # Stochastic
        stoch_k = df['stoch_k'].iloc[-1] if 'stoch_k' in df.columns else 50
        stoch_signal = "Oversold" if stoch_k < 30 else ("Overbought" if stoch_k > 70 else "Neutral")
        
        # ADX strength
        adx = df['adx'].iloc[-1] if 'adx' in df.columns else 0
        trend_strength = "Strong" if adx > 25 else ("Moderate" if adx > 20 else "Weak")
        
        # Generate signals
        buy_signals = []
        sell_signals = []
        
        if trend == "Uptrend" and current_price > ema_20:
            buy_signals.append("Strong uptrend confirmation")
        if bb_signal == "Oversold":
            buy_signals.append("Bollinger Band oversold signal")
        if stoch_signal == "Oversold":
            buy_signals.append("Stochastic oversold condition")
        if trend_strength == "Strong":
            buy_signals.append(f"Strong trend strength (ADX > 25)")
        
        if trend == "Downtrend":
            sell_signals.append("Downtrend detected")
        if bb_signal == "Overbought":
            sell_signals.append("Bollinger Band overbought signal")
        
        # Recommendation
        signal_strength = len(buy_signals) - len(sell_signals)
        if signal_strength >= 3:
            recommendation = "STRONG BUY"
            confidence = 0.80
        elif signal_strength >= 2:
            recommendation = "BUY"
            confidence = 0.70
        elif signal_strength >= 1:
            recommendation = "MILD BUY"
            confidence = 0.65
        elif signal_strength >= -1:
            recommendation = "HOLD"
            confidence = 0.60
        elif signal_strength >= -2:
            recommendation = "SELL"
            confidence = 0.70
        else:
            recommendation = "STRONG SELL"
            confidence = 0.80
        
        return {
            'timeframe_weeks': weeks,
            'trend': trend,
            'ema_20': float(ema_20),
            'ema_50': float(ema_50),
            'bb_signal': bb_signal,
            'stoch_signal': stoch_signal,
            'trend_strength': trend_strength,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'recommendation': recommendation,
            'confidence': confidence,
            'target_price': current_price * (1.08 if signal_strength > 0 else 0.92)
        }
    
    def _analyze_long_term(self, df: pd.DataFrame, months: int = 6) -> Dict:
        """
        Long-term structural trend and sector analysis.
        
        Args:
            df: DataFrame with indicators
            months: Lookback period in months
            
        Returns:
            Long-term analysis dictionary
        """
        days = months * 21  # Approximate trading days per month
        recent = df.tail(min(days, len(df)))
        
        # Long-term trend
        ema_100 = df['ema_100'].iloc[-1] if 'ema_100' in df.columns else df['close'].iloc[-1]
        ema_200 = df['ema_200'].iloc[-1] if 'ema_200' in df.columns else df['close'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        long_trend = "Bullish" if ema_100 > ema_200 else "Bearish"
        
        # Return analysis
        total_return_pct = ((current_price - recent['close'].iloc[0]) / recent['close'].iloc[0]) * 100
        volatility = recent['close'].pct_change().std() * np.sqrt(252) * 100
        
        # MACD long-term signal
        macd_trend = "Bullish" if df['macd_histogram'].iloc[-1] > 0 else "Bearish"
        
        # OBV trend
        obv = df['obv'].iloc[-1] if 'obv' in df.columns else 0
        obv_trend = "Accumulation" if obv > df['obv'].iloc[-20] else "Distribution"
        
        # Generate signals
        buy_signals = []
        sell_signals = []
        
        if long_trend == "Bullish" and current_price > ema_200:
            buy_signals.append("Strong long-term bull market")
        if macd_trend == "Bullish":
            buy_signals.append("MACD long-term bullish signal")
        if obv_trend == "Accumulation":
            buy_signals.append("Volume accumulation pattern")
        if total_return_pct > 20:
            buy_signals.append("Strong momentum (20%+ returns)")
        
        if long_trend == "Bearish":
            sell_signals.append("Long-term bearish structure")
        if volatility > 40:
            sell_signals.append("High volatility risk")
        
        # Recommendation
        signal_strength = len(buy_signals) - len(sell_signals)
        if signal_strength >= 3:
            recommendation = "STRONG BUY"
            confidence = 0.85
        elif signal_strength >= 2:
            recommendation = "BUY"
            confidence = 0.75
        elif signal_strength >= 1:
            recommendation = "MILD BUY"
            confidence = 0.70
        elif signal_strength >= -1:
            recommendation = "HOLD"
            confidence = 0.60
        elif signal_strength >= -2:
            recommendation = "SELL"
            confidence = 0.75
        else:
            recommendation = "STRONG SELL"
            confidence = 0.85
        
        return {
            'timeframe_months': months,
            'long_trend': long_trend,
            'ema_100': float(ema_100),
            'ema_200': float(ema_200),
            'total_return_pct': total_return_pct,
            'volatility': volatility,
            'macd_trend': macd_trend,
            'obv_trend': obv_trend,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'recommendation': recommendation,
            'confidence': confidence,
            'target_price': current_price * (1.25 if signal_strength > 0 else 0.80)
        }
    
    def _synthesize_recommendation(self, short: Dict, mid: Dict, long: Dict) -> Dict:
        """
        Synthesize overall recommendation from all timeframes.
        
        Args:
            short: Short-term analysis
            mid: Mid-term analysis
            long: Long-term analysis
            
        Returns:
            Overall recommendation dictionary
        """
        # Weighted recommendation
        rec_map = {
            "STRONG BUY": 2,
            "BUY": 1,
            "MILD BUY": 0.5,
            "HOLD": 0,
            "SELL": -1,
            "STRONG SELL": -2
        }
        
        short_score = rec_map.get(short['recommendation'], 0)
        mid_score = rec_map.get(mid['recommendation'], 0)
        long_score = rec_map.get(long['recommendation'], 0)
        
        # Weight by timeframe importance
        total_score = (short_score * 0.3) + (mid_score * 0.4) + (long_score * 0.3)
        
        # Determine overall recommendation
        if total_score >= 1.5:
            overall_rec = "STRONG BUY"
            confidence = 0.85
        elif total_score >= 1.0:
            overall_rec = "BUY"
            confidence = 0.75
        elif total_score >= 0.5:
            overall_rec = "MILD BUY"
            confidence = 0.70
        elif total_score >= -0.5:
            overall_rec = "HOLD"
            confidence = 0.60
        elif total_score >= -1.0:
            overall_rec = "SELL"
            confidence = 0.75
        else:
            overall_rec = "STRONG SELL"
            confidence = 0.85
        
        # Explanation
        explanations = []
        if short['recommendation'] != "HOLD":
            explanations.append(f"Short-term: {short['recommendation']} (confidence: {short['confidence']:.0%})")
        if mid['recommendation'] != "HOLD":
            explanations.append(f"Mid-term: {mid['recommendation']} (confidence: {mid['confidence']:.0%})")
        if long['recommendation'] != "HOLD":
            explanations.append(f"Long-term: {long['recommendation']} (confidence: {long['confidence']:.0%})")
        
        return {
            'recommendation': overall_rec,
            'confidence': confidence,
            'reasoning': '; '.join(explanations) if explanations else "Neutral across all timeframes",
            'short_term_weight': 0.3,
            'mid_term_weight': 0.4,
            'long_term_weight': 0.3
        }
    
    def _generate_exit_strategies(self, df: pd.DataFrame, live_price: Optional[float] = None) -> List[Dict]:
        """
        Generate exit strategies based on technical levels.
        
        Args:
            df: DataFrame with indicators
            live_price: Optional live price (if available, uses this instead of last close)
            
        Returns:
            List of exit strategy dictionaries
        """
        current_price = live_price if live_price else df['close'].iloc[-1]
        strategies = []
        
        # Take profit levels
        if 'bb_upper' in df.columns:
            strategies.append({
                'type': 'Take Profit',
                'level': float(df['bb_upper'].iloc[-1]),
                'reason': 'Bollinger Band upper resistance'
            })
        
        if 'ema_200' in df.columns and df['ema_200'].iloc[-1] > current_price:
            strategies.append({
                'type': 'Take Profit',
                'level': float(df['ema_200'].iloc[-1]),
                'reason': '200-day moving average resistance'
            })
        
        # Stop loss levels
        if 'bb_lower' in df.columns:
            strategies.append({
                'type': 'Stop Loss',
                'level': float(df['bb_lower'].iloc[-1]),
                'reason': 'Bollinger Band lower support'
            })
        
        if 'atr' in df.columns:
            atr = df['atr'].iloc[-1]
            strategies.append({
                'type': 'Stop Loss',
                'level': float(current_price - (2 * atr)),
                'reason': '2x ATR below current price'
            })
        
        return strategies
    
    def _assess_risk(self, df: pd.DataFrame) -> Dict:
        """
        Assess overall risk profile.
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            Risk assessment dictionary
        """
        recent = df.tail(20)
        
        # Volatility risk
        volatility = recent['close'].pct_change().std() * np.sqrt(252) * 100
        vol_risk = "High" if volatility > 40 else ("Medium" if volatility > 25 else "Low")
        
        # Drawdown risk
        rolling_max = recent['close'].cummax()
        drawdown = ((recent['close'] - rolling_max) / rolling_max).min() * 100
        dd_risk = "High" if abs(drawdown) > 15 else ("Medium" if abs(drawdown) > 8 else "Low")
        
        # Trend risk
        adx = df['adx'].iloc[-1] if 'adx' in df.columns else 0
        trend_risk = "Low" if adx > 25 else ("Medium" if adx > 15 else "High")
        
        # Overall risk
        risk_scores = {"Low": 1, "Medium": 2, "High": 3}
        total_risk = (risk_scores[vol_risk] + risk_scores[dd_risk] + risk_scores[trend_risk]) / 3
        overall_risk = "High" if total_risk >= 2.5 else ("Medium" if total_risk >= 1.5 else "Low")
        
        return {
            'overall_risk': overall_risk,
            'volatility_risk': vol_risk,
            'volatility_pct': volatility,
            'drawdown_risk': dd_risk,
            'recent_drawdown_pct': drawdown,
            'trend_risk': trend_risk,
            'adx_value': float(adx)
        }
    
    def _empty_analysis(self, symbol: str) -> Dict:
        """
        Return empty analysis when no data available.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Empty analysis dictionary
        """
        return {
            'symbol': symbol,
            'current_price': 0,
            'short_term': {'recommendation': 'HOLD', 'confidence': 0.5},
            'mid_term': {'recommendation': 'HOLD', 'confidence': 0.5},
            'long_term': {'recommendation': 'HOLD', 'confidence': 0.5},
            'overall_recommendation': {'recommendation': 'HOLD', 'confidence': 0.5},
            'exit_strategies': [],
            'risk_assessment': {'overall_risk': 'Medium'},
            'error': 'No data available'
        }


def main():
    """Standalone test of SymbolAnalysisAgent"""
    import yaml
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize agent
    agent = SymbolAnalysisAgent(config)
    
    print("=" * 80)
    print("AXFI Symbol Analysis Agent - Standalone Test")
    print("=" * 80)
    print(f"\n[OK] Symbol Analysis Agent initialized")
    
    # Load sample data
    from core.data_collector import DataCollector
    dc = DataCollector(config_path="config.yaml")
    df = dc.read_from_database(symbol="AAPL")
    
    if not df.empty:
        df.set_index('date', inplace=True)
        print(f"\n[OK] Loaded {len(df)} rows for AAPL")
        
        # Perform analysis
        print("\nAnalyzing AAPL across all timeframes...")
        analysis = agent.analyze_symbol(df, "AAPL")
        
        # Display results
        print("\n" + "=" * 80)
        print("SYMBOL ANALYSIS RESULTS")
        print("=" * 80)
        
        print(f"\nSymbol: {analysis['symbol']}")
        print(f"Current Price: ${analysis['current_price']:.2f}")
        
        print("\n--- SHORT-TERM (3 days) ---")
        st = analysis['short_term']
        print(f"Recommendation: {st['recommendation']}")
        print(f"Confidence: {st['confidence']:.0%}")
        if st['buy_signals']:
            print(f"Buy Signals: {', '.join(st['buy_signals'][:3])}")
        
        print("\n--- MID-TERM (4 weeks) ---")
        mt = analysis['mid_term']
        print(f"Recommendation: {mt['recommendation']}")
        print(f"Confidence: {mt['confidence']:.0%}")
        print(f"Trend: {mt['trend']}, Strength: {mt['trend_strength']}")
        
        print("\n--- LONG-TERM (6 months) ---")
        lt = analysis['long_term']
        print(f"Recommendation: {lt['recommendation']}")
        print(f"Confidence: {lt['confidence']:.0%}")
        print(f"Total Return: {lt['total_return_pct']:.1f}%")
        
        print("\n--- OVERALL RECOMMENDATION ---")
        overall = analysis['overall_recommendation']
        print(f"Recommendation: {overall['recommendation']}")
        print(f"Confidence: {overall['confidence']:.0%}")
        print(f"Reasoning: {overall['reasoning']}")
        
        print("\n--- RISK ASSESSMENT ---")
        risk = analysis['risk_assessment']
        print(f"Overall Risk: {risk['overall_risk']}")
        print(f"Volatility: {risk['volatility_risk']} ({risk['volatility_pct']:.1f}%)")
        
        print("\n--- EXIT STRATEGIES ---")
        for strategy in analysis['exit_strategies'][:3]:
            print(f"{strategy['type']}: ${strategy['level']:.2f} - {strategy['reason']}")
    else:
        print("[ERROR] No data available for AAPL. Please run data_collector.py first.")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

