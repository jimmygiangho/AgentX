"""
AXFI - Agent X Financial Intelligence
FastAPI Dashboard and Main Application
"""

import json
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import plotly.graph_objects as go
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AXFI Dashboard",
    description="Agent X Financial Intelligence - Market Analysis Dashboard",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="ui/static"), name="static")

# Setup templates - using new Xetho-style UI
templates = Jinja2Templates(directory="ui/templates")
TEMPLATE_NAME = "dashboard_xetho.html"  # New modern UI

# Load configuration (will be initialized below)


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the config YAML file
        
    Returns:
        Dictionary containing configuration settings
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# Initialize config on import
config = load_config()


@app.get("/health", response_model=Dict)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Dictionary with status message
    """
    return {"status": "Enhanced AXFI Dashboard OK", "timestamp": datetime.now().isoformat()}


@app.get("/api/daily_scan", response_model=Dict)
async def api_daily_scan():
    """
    Function 1: Daily S&P 500 Scan
    Returns top ranked trade recommendations.
    """
    from core.agents.scanner_agent import ScannerAgent
    
    logger.info("Running daily scan via API")
    
    # Use the scanner agent which handles S&P 500 parsing and live data
    scanner_agent = ScannerAgent(config)
    result = scanner_agent.run(symbols=None, top_n=15)
    
    if result['status'] != 'success':
        return {
            "function": "Daily S&P 500 Scan",
            "timestamp": datetime.now().isoformat(),
            "total_recommendations": 0,
            "recommendations": [],
            "error": "Scan failed"
        }
    
    # Format response - convert decimals to percentages for dashboard
    # Also fetch live prices for each recommendation
    from core.data_collector_v2 import DataCollectorV2 as DataCollector
    data_collector_for_prices = DataCollector(config_path="config.yaml")
    
    recommendations = []
    for rec in result['recommendations']:
        # Fetch live price for this symbol
        current_price = None
        try:
            quote = data_collector_for_prices.get_latest_quote(rec['symbol'], use_fallback=True)
            if quote and 'price' in quote and quote['price']:
                current_price = float(quote['price'])
                logger.info(f"Fetched price for {rec['symbol']}: ${current_price:.2f}")
            else:
                # Fallback: try to get from historical data
                logger.warning(f"Quote missing price for {rec['symbol']}, trying historical data")
                try:
                    hist_data = data_collector_for_prices.fetch_symbol(rec['symbol'], period='1d')
                    if not hist_data.empty and 'close' in hist_data.columns:
                        current_price = float(hist_data['close'].iloc[-1])
                        logger.info(f"Got price from historical for {rec['symbol']}: ${current_price:.2f}")
                except Exception as e2:
                    logger.warning(f"Historical fallback failed for {rec['symbol']}: {e2}")
        except Exception as e:
            logger.error(f"Could not fetch live price for {rec['symbol']}: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Final fallback: try reading from database if still None
        if current_price is None:
            try:
                hist_df = data_collector_for_prices.read_from_database(symbol=rec['symbol'])
                if not hist_df.empty and 'close' in hist_df.columns:
                    current_price = float(hist_df['close'].iloc[-1])
                    logger.info(f"✅ Got price from database for {rec['symbol']}: ${current_price:.2f}")
            except Exception as e2:
                logger.warning(f"Database fallback also failed for {rec['symbol']}: {e2}")
                current_price = 0  # Use 0 as placeholder
        
        recommendations.append({
            'symbol': rec['symbol'],
            'strategy': rec['strategy'],
            'score': float(rec['score']),
            'confidence': rec.get('confidence', 70),
            'explanation': rec['explanation'],
            'cagr': float(rec['cagr']) * 100,
            'sharpe': float(rec['sharpe']),
            'max_dd': float(rec['max_dd']) * 100,
            'win_rate': float(rec['win_rate']) * 100,
            'current_price': current_price if current_price is not None else 0
        })
    
    # Save to cache for persistence
    timestamp = datetime.now().isoformat()
    from core.scan_cache import save_scan_results
    save_scan_results(recommendations, timestamp)
    
    return {
        "function": "Daily S&P 500 Scan",
        "timestamp": timestamp,
        "total_recommendations": len(recommendations),
        "recommendations": recommendations
    }


@app.get("/api/symbol_analysis", response_model=Dict)
async def api_symbol_analysis(symbol: str = "AAPL"):
    """
    Function 2: Single-Symbol Deep Analysis
    Returns short/mid/long-term recommendations with exit strategies.
    """
    from core.data_collector_v2 import DataCollectorV2 as DataCollector
    from core.symbol_analysis import SymbolAnalysisAgent
    
    logger.info(f"Analyzing {symbol} via API - fetching LIVE data")
    
    data_collector = DataCollector(config_path="config.yaml")
    agent = SymbolAnalysisAgent(config)
    
    # Always fetch fresh live data for symbol analysis (not from cache)
    logger.info(f"Fetching fresh live data for {symbol}...")
    try:
        df = data_collector.fetch_symbol(symbol, period='1y')
        if df.empty:
            # Try reading from database as fallback
            logger.info(f"No live data returned, trying database...")
            df = data_collector.read_from_database(symbol=symbol)
            
        if not df.empty:
            # Save fresh data to database for future use
            data_collector.save_to_database(df, symbol)
            logger.info(f"Successfully fetched {len(df)} rows of LIVE data for {symbol}")
        else:
            logger.error(f"No data available for {symbol}")
            return {
                "function": "Single-Symbol Analysis",
                "symbol": symbol,
                "error": f"Could not fetch data for {symbol}",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error fetching live data for {symbol}: {e}")
        # Try database as fallback
        try:
            df = data_collector.read_from_database(symbol=symbol)
            if df.empty:
                raise Exception("No data in database either")
            logger.warning(f"Using cached data from database for {symbol}")
        except:
            return {
                "function": "Single-Symbol Analysis",
                "symbol": symbol,
                "error": f"Could not fetch data for {symbol}: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    if df.empty:
        return {
            "function": "Single-Symbol Analysis",
            "symbol": symbol,
            "error": "No data available for this symbol",
            "timestamp": datetime.now().isoformat()
        }
    
    # Store original DataFrame for detailed info extraction (BEFORE indexing)
    df_for_details = df.copy()
    
    df.set_index('date', inplace=True)
    analysis = agent.analyze_symbol(df, symbol)
    
    # Fetch detailed stock information using both yfinance and our historical data
    import yfinance as yf
    from datetime import datetime as dt
    import pandas as pd
    
    detailed_info = {}
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Get recent trading data (intraday or latest day)
        recent_data_1d = ticker.history(period="1d", interval="1d")
        recent_data_5d = ticker.history(period="5d", interval="1d")
        
        # Use our existing DataFrame for latest data (use df_for_details which hasn't been indexed)
        if not df_for_details.empty:
            # Ensure sorted by date (ascending - oldest first)
            if 'date' in df_for_details.columns:
                df_for_details = df_for_details.sort_values('date').reset_index(drop=True)
            
            # Get the last 2 rows (most recent data)
            latest_row = df_for_details.iloc[-1]
            prev_row = df_for_details.iloc[-2] if len(df_for_details) > 1 else None
            
            # Columns are lowercase: 'open', 'high', 'low', 'close', 'volume'
            try:
                # Previous close (from second-to-last row)
                if prev_row is not None:
                    detailed_info['previous_close'] = float(prev_row['close'])
                else:
                    detailed_info['previous_close'] = None
                
                # Latest day data
                detailed_info['open'] = float(latest_row['open'])
                detailed_info['day_low'] = float(latest_row['low'])
                detailed_info['day_high'] = float(latest_row['high'])
                detailed_info['volume'] = int(latest_row['volume'])
                
                logger.info(f"Extracted from DataFrame - Open: ${detailed_info.get('open'):.2f}, Previous Close: ${detailed_info.get('previous_close'):.2f}, Volume: {detailed_info.get('volume'):,}")
            except Exception as e:
                logger.error(f"Error extracting from DataFrame: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # 52 week range from our historical data
            if len(df_for_details) >= 252:  # Trading days in a year
                year_data = df_for_details.tail(252)
            else:
                year_data = df_for_details
            
            try:
                detailed_info['52_week_low'] = float(year_data['low'].min())
                detailed_info['52_week_high'] = float(year_data['high'].max())
            except Exception as e:
                logger.warning(f"Error calculating 52-week range: {e}")
                detailed_info['52_week_low'] = None
                detailed_info['52_week_high'] = None
            
            # Average volume from last 30 days
            try:
                if len(df_for_details) >= 30:
                    detailed_info['avg_volume'] = int(df_for_details.tail(30)['volume'].mean())
                else:
                    detailed_info['avg_volume'] = int(df_for_details['volume'].mean())
            except Exception as e:
                logger.warning(f"Error calculating avg volume: {e}")
                detailed_info['avg_volume'] = None
        else:
            # Fallback to yfinance data
            if not recent_data_5d.empty and len(recent_data_5d) > 1:
                detailed_info['previous_close'] = float(recent_data_5d['Close'].iloc[-2])
                detailed_info['open'] = float(recent_data_5d['Open'].iloc[-1])
                detailed_info['day_low'] = float(recent_data_5d['Low'].iloc[-1])
                detailed_info['day_high'] = float(recent_data_5d['High'].iloc[-1])
                detailed_info['volume'] = int(recent_data_5d['Volume'].iloc[-1])
        
        # Try to get additional info from yfinance ticker.info
        try:
            info = ticker.info
            if info and len(info) > 10:  # Check if info is not empty
                # Fill in missing values from info
                if not detailed_info.get('previous_close') and info.get('previousClose'):
                    detailed_info['previous_close'] = float(info['previousClose'])
                if not detailed_info.get('open') and info.get('open'):
                    detailed_info['open'] = float(info['open'])
                if not detailed_info.get('day_low') and info.get('dayLow'):
                    detailed_info['day_low'] = float(info['dayLow'])
                if not detailed_info.get('day_high') and info.get('dayHigh'):
                    detailed_info['day_high'] = float(info['dayHigh'])
                if not detailed_info.get('volume') and info.get('volume'):
                    detailed_info['volume'] = int(info['volume'])
                
                # Get fundamental data
                detailed_info['market_cap'] = info.get('marketCap')
                detailed_info['beta'] = info.get('beta')
                detailed_info['pe_ratio'] = info.get('trailingPE') or info.get('forwardPE')
                detailed_info['eps'] = info.get('trailingEps')
                detailed_info['dividend_yield'] = info.get('dividendYield')
                detailed_info['dividend_rate'] = info.get('dividendRate')
                
                # Get 52 week range if not already set
                if not detailed_info.get('52_week_low') and info.get('fiftyTwoWeekLow'):
                    detailed_info['52_week_low'] = float(info['fiftyTwoWeekLow'])
                if not detailed_info.get('52_week_high') and info.get('fiftyTwoWeekHigh'):
                    detailed_info['52_week_high'] = float(info['fiftyTwoWeekHigh'])
                
                if not detailed_info.get('avg_volume') and info.get('averageVolume'):
                    detailed_info['avg_volume'] = int(info['averageVolume'])
                
                # Dates and targets
                ex_div_date = info.get('exDividendDate')
                if ex_div_date:
                    if isinstance(ex_div_date, (list, tuple)) and len(ex_div_date) > 0:
                        detailed_info['ex_dividend_date'] = pd.Timestamp.fromtimestamp(ex_div_date[0]).strftime('%b %d, %Y')
                    elif isinstance(ex_div_date, (int, float)):
                        detailed_info['ex_dividend_date'] = pd.Timestamp.fromtimestamp(ex_div_date).strftime('%b %d, %Y')
                
                earnings_date = info.get('earningsDate')
                if earnings_date:
                    if isinstance(earnings_date, (list, tuple)) and len(earnings_date) > 0:
                        detailed_info['earnings_date'] = pd.Timestamp.fromtimestamp(earnings_date[0]).strftime('%b %d, %Y')
                    elif isinstance(earnings_date, (int, float)):
                        detailed_info['earnings_date'] = pd.Timestamp.fromtimestamp(earnings_date).strftime('%b %d, %Y')
                
                detailed_info['target_price'] = info.get('targetMeanPrice')
                detailed_info['sector'] = info.get('sector')
                detailed_info['industry'] = info.get('industry')
        except Exception as e2:
            logger.warning(f"Could not fetch from ticker.info: {e2}")
        
        logger.info(f"Fetched detailed info for {symbol}: {len([k for k, v in detailed_info.items() if v is not None])} fields populated")
    except Exception as e:
        logger.error(f"Error fetching detailed info for {symbol}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        detailed_info = {}
    
    # Add detailed info to analysis
    analysis['detailed_info'] = detailed_info
    
    return {
        "function": "Single-Symbol Analysis",
        "timestamp": datetime.now().isoformat(),
        "analysis": analysis
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """
    Main dashboard page.
    
    Args:
        request: FastAPI request object
        
    Returns:
        HTML dashboard page
    """
    # Import modules
    from core.data_collector_v2 import DataCollectorV2 as DataCollector
    from core.backtester import Backtester
    from core.ai_engine import AIEngine
    from core.portfolio import Portfolio
    
    # Initialize components
    data_collector = DataCollector(config_path="config.yaml")
    backtester = Backtester(
        initial_capital=config['backtest']['initial_capital'],
        commission=config['backtest']['commission']
    )
    ai_engine = AIEngine()
    portfolio = Portfolio(initial_capital=config['backtest']['initial_capital'])
    
    # Load cached scan results if available
    from core.scan_cache import load_scan_results
    cached_results = load_scan_results()
    
    ranked_recommendations = []
    if cached_results and cached_results.get('recommendations'):
        # Use cached results
        logger.info(f"Loading {len(cached_results['recommendations'])} cached recommendations")
        for idx, rec in enumerate(cached_results['recommendations'][:15], 1):
            ranked_recommendations.append({
                'rank': idx,
                'symbol': rec['symbol'],
                'strategy': rec['strategy'],
                'score': rec['score'],
                'cagr': rec['cagr'],
                'sharpe': rec['sharpe'],
                'max_dd': rec['max_dd'],
                'win_rate': rec['win_rate'],
                'explanation': rec['explanation'],
                'current_price': rec.get('current_price')  # Keep existing price or None if not present
            })
    else:
        logger.info("No cached results found, processing default symbols")
        # Collect all results for initial dashboard load (limited symbols)
        all_symbol_results = {}
        symbols_to_process = config.get('symbols', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
        logger.info(f"Processing {len(symbols_to_process)} symbols for initial dashboard load")
        
        # Process each symbol
        for symbol in symbols_to_process:
            try:
                # Load data
                df = data_collector.read_from_database(symbol=symbol)
                
                if df.empty:
                    logger.warning(f"No data for {symbol}, skipping...")
                    continue
                
                df = df.sort_values('date')
                df.set_index('date', inplace=True)
                
                # Run backtesting
                strategy_results = backtester.run_all_strategies(df)
                all_symbol_results[symbol] = strategy_results
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        if all_symbol_results:
            # Get top recommendations
            top_recommendations = ai_engine.get_top_recommendations(all_symbol_results, top_n=10)
            for idx, rec in enumerate(top_recommendations, 1):
                # Extract metrics from the 'metrics' dict if present
                metrics = rec.get('metrics', rec)
                ranked_recommendations.append({
                    'rank': idx,
                    'symbol': rec['symbol'],
                    'strategy': rec['strategy'],
                    'score': rec['score'],
                    'cagr': metrics.get('cagr', 0),
                    'sharpe': metrics.get('sharpe_ratio', 0),
                    'max_dd': metrics.get('max_drawdown', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'explanation': rec['explanation'],
                    'current_price': None  # Will be filled below
                })
    
    # Fetch live prices for symbols in recommendations (refresh all prices)
    symbol_live_prices = {}
    symbols_to_price = [rec['symbol'] for rec in ranked_recommendations[:20]]  # Top 20 symbols
    if not symbols_to_price:
        symbols_to_price = config.get('symbols', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
    
    logger.info(f"Fetching live prices for {len(symbols_to_price)} symbols")
    for symbol in symbols_to_price:
        try:
            quote = data_collector.get_latest_quote(symbol, use_fallback=True)
            if quote and 'price' in quote and quote['price']:
                price = float(quote['price'])
                symbol_live_prices[symbol] = {
                    'price': price,
                    'provider': quote.get('provider', 'unknown')
                }
                # Update price in recommendations (override any cached value)
                for rec in ranked_recommendations:
                    if rec['symbol'] == symbol:
                        rec['current_price'] = price
                        logger.info(f"✅ Updated price for {symbol}: ${price:.2f}")
            else:
                # Try fallback to historical data
                try:
                    hist_df = data_collector.fetch_symbol(symbol, period='1d')
                    if not hist_df.empty and 'close' in hist_df.columns:
                        price = float(hist_df['close'].iloc[-1])
                        symbol_live_prices[symbol] = {'price': price, 'provider': 'historical'}
                        for rec in ranked_recommendations:
                            if rec['symbol'] == symbol:
                                rec['current_price'] = price
                                logger.info(f"✅ Got price from historical for {symbol}: ${price:.2f}")
                except Exception as e2:
                    logger.warning(f"Historical fallback failed for {symbol}: {e2}")
        except Exception as e:
            logger.warning(f"Could not fetch live price for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Get portfolio summary (keep for compatibility, but Position Intelligence tab removed)
    portfolio_summary = []
    portfolio_metrics = {}
    
    # Generate equity charts (simplified for performance)
    equity_charts = []
    
    # Render dashboard
    return templates.TemplateResponse(TEMPLATE_NAME, {
        "request": request,
        "ranked_recommendations": ranked_recommendations,
        "symbol_live_prices": symbol_live_prices,
        "portfolio_summary": portfolio_summary,
        "portfolio_metrics": portfolio_metrics,
        "equity_charts": equity_charts,
        "error": None
    })


if __name__ == "__main__":
    import uvicorn
    
    port = config['dashboard']['port']
    host = config['dashboard']['host']
    
    logger.info(f"Starting AXFI Dashboard on http://{host}:{port}")
    logger.info(f"Tracking {len(config['symbols'])} symbols: {', '.join(config['symbols'])}")
    uvicorn.run(app, host=host, port=port, reload=False)
