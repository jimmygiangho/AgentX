"""
Utility functions for fetching S&P 500 symbols and additional stock data
"""

import logging
import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def fetch_sp500_tickers() -> List[str]:
    """
    Fetch the current list of S&P 500 tickers from Wikipedia.
    
    Returns:
        List of ticker symbols (e.g., ['AAPL', 'MSFT', ...])
    """
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        logger.info("Fetching S&P 500 ticker list from Wikipedia...")
        
        sp500_table = pd.read_html(url)[0]  # First table contains the data
        tickers = sp500_table['Symbol'].tolist()
        
        # Yahoo Finance uses '-' instead of '.' for some tickers like BRK.B
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        
        logger.info(f"Fetched {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        logger.error(f"Error fetching S&P 500 list: {e}")
        # Return a fallback list of top 100 if Wikipedia fails
        return get_fallback_sp500_tickers()


def get_fallback_sp500_tickers() -> List[str]:
    """Fallback S&P 500 list if Wikipedia fetch fails"""
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'V', 'JNJ',
        'WMT', 'MA', 'UNH', 'XOM', 'PG', 'JPM', 'HD', 'CVX', 'COST', 'ABBV',
        'AVGO', 'PFE', 'MRK', 'ADBE', 'CSCO', 'KO', 'PEP', 'TMO', 'DHR', 'ACN',
        'WFC', 'COIN', 'T', 'CMCSA', 'VZ', 'PM', 'TXN', 'INTU', 'QCOM', 'HON',
        'NFLX', 'DIS', 'INTC', 'AMD', 'BA', 'GS', 'AXP', 'DELL', 'UBER', 'CRM',
        'ORCL', 'IBM', 'NKE', 'LOW', 'RTX', 'BKNG', 'MDT', 'ADP', 'C', 'GILD',
        'GE', 'CAT', 'TJX', 'SBUX', 'AMGN', 'ISRG', 'VRTX', 'REGN', 'SNPS', 'CDNS',
        'CRWD', 'FANG', 'MPC', 'ON', 'MCHP', 'FTNT', 'NXPI', 'KLAC', 'LRCX', 'EXPD',
        'HSY', 'CTVA', 'AME', 'EQT', 'TECH', 'WAT', 'FOXA', 'FOX', 'ENPH', 'MORN',
        'FAST', 'KEYS', 'CACI', 'NDAQ', 'ETN', 'AEE', 'MTCH', 'TCOM', 'IDXX', 'ZS'
    ]


def get_stock_fundamentals_yfinance(symbol: str) -> Dict[str, Optional[float]]:
    """
    Fetch additional stock data using yfinance including:
    - Market cap, P/E ratio, EPS, dividend yield, beta, etc.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Dictionary with fundamental data
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return {
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'eps': info.get('trailingEps'),
            'dividend_yield': info.get('dividendYield'),
            'beta': info.get('beta'),
            '52_week_high': info.get('fiftyTwoWeekHigh'),
            '52_week_low': info.get('fiftyTwoWeekLow'),
            'avg_volume': info.get('averageVolume'),
            'volume': info.get('volume'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'enterprise_value': info.get('enterpriseValue'),
            'revenue': info.get('totalRevenue'),
            'profit_margin': info.get('profitMargins'),
        }
    except Exception as e:
        logger.warning(f"Error fetching fundamentals for {symbol}: {e}")
        return {}


def fetch_latest_prices_yfinance(symbols: List[str], period: str = "1d") -> Dict[str, float]:
    """
    Fetch latest prices for multiple symbols using yfinance.
    
    Args:
        symbols: List of ticker symbols
        period: yfinance period (1d, 5d, 1mo, etc.)
        
    Returns:
        Dictionary mapping symbol to latest price
    """
    latest_prices = {}
    
    try:
        # Download data for all symbols
        data = yf.download(symbols, period=period, group_by='ticker', progress=False)
        
        for ticker in symbols:
            if ticker in data.columns.get_level_values(0):
                try:
                    ticker_data = data[ticker]
                    if 'Adj Close' in ticker_data.columns:
                        latest_close = ticker_data['Adj Close'].dropna()
                        if not latest_close.empty:
                            latest_prices[ticker] = float(latest_close.iloc[-1])
                except (IndexError, KeyError, AttributeError):
                    latest_prices[ticker] = None
            else:
                latest_prices[ticker] = None
                
    except Exception as e:
        logger.error(f"Error fetching prices for multiple symbols: {e}")
        # Fallback: fetch one at a time
        for ticker in symbols:
            try:
                ticker_obj = yf.Ticker(ticker)
                hist = ticker_obj.history(period=period)
                if not hist.empty:
                    latest_prices[ticker] = float(hist['Close'].iloc[-1])
                else:
                    latest_prices[ticker] = None
            except Exception as e2:
                logger.warning(f"Error fetching price for {ticker}: {e2}")
                latest_prices[ticker] = None
    
    # Remove None values
    return {k: v for k, v in latest_prices.items() if v is not None}


if __name__ == "__main__":
    # Test functions
    print("Testing S&P 500 ticker fetch...")
    tickers = fetch_sp500_tickers()
    print(f"Fetched {len(tickers)} tickers")
    print(f"First 10: {tickers[:10]}")
    
    print("\nTesting latest prices fetch...")
    sample_tickers = tickers[:5]
    prices = fetch_latest_prices_yfinance(sample_tickers)
    print(f"Prices: {prices}")
    
    print("\nTesting fundamentals fetch...")
    fundamentals = get_stock_fundamentals_yfinance("AAPL")
    print(f"AAPL fundamentals: {fundamentals}")

