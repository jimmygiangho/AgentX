"""
AXFI Data Provider Abstraction
Supports Polygon, Alpaca, and yfinance (fallback) providers
"""

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """Base class for data providers"""
    
    @abstractmethod
    def get_latest_quote(self, symbol: str) -> dict:
        """Get latest quote for symbol"""
        pass
    
    @abstractmethod
    def get_historical_ohlcv(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Get historical OHLCV data"""
        pass
    
    @abstractmethod
    def get_intraday_ohlcv(self, symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
        """Get intraday OHLCV data"""
        pass


class PolygonProvider(BaseProvider):
    """Polygon.io data provider"""
    
    def __init__(self):
        self.api_key = os.getenv("POLYGON_API_KEY")
        self.base_url = "https://api.polygon.io"
        
        if not self.api_key or self.api_key == "your_polygon_api_key_here":
            logger.warning("POLYGON_API_KEY not found in .env")
            self.api_key = None
    
    def get_latest_quote(self, symbol: str) -> dict:
        """Get latest quote from Polygon"""
        if not self.api_key:
            logger.error("Polygon API key not configured, cannot use Polygon provider")
            raise ValueError("Polygon API key not configured")
        
        # Try NBBO endpoint first (requires paid plan)
        try:
            url = f"{self.base_url}/v2/last/nbbo/{symbol}"
            response = requests.get(url, params={"apikey": self.api_key})
            response.raise_for_status()
            
            data = response.json()
            if data.get("status") == "OK":
                result = data.get("results", {})
                return {
                    "symbol": symbol,
                    "price": result.get("P"),
                    "timestamp": result.get("t"),
                    "provider": "polygon"
                }
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [401, 403]:
                # NBBO requires paid plan or has restrictions, try aggregates endpoint instead
                logger.info(f"NBBO endpoint not available (status {e.response.status_code}), using aggregates endpoint for {symbol}")
                try:
                    # Use previous close endpoint (works on free tier)
                    url = f"{self.base_url}/v2/aggs/ticker/{symbol}/prev"
                    response = requests.get(url, params={"apikey": self.api_key})
                    response.raise_for_status()
                    
                    data = response.json()
                    if data.get("status") == "OK" and data.get("resultsCount", 0) > 0:
                        result = data.get("results", [{}])[0]
                        return {
                            "symbol": symbol,
                            "price": result.get("c"),  # Close price
                            "timestamp": result.get("t") / 1000,  # Convert ms to seconds
                            "provider": "polygon"
                        }
                except Exception as e2:
                    logger.warning(f"Error fetching previous close for {symbol}: {e2}")
            else:
                raise
        
        return {"symbol": symbol, "error": "No data"}
    
    def get_historical_ohlcv(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Get historical daily bars"""
        if not self.api_key:
            raise ValueError("Polygon API key not configured")
        
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
        response = requests.get(url, params={"apikey": self.api_key})
        response.raise_for_status()
        
        data = response.json()
        if data.get("status") == "OK":
            bars = data.get("results", [])
            df = pd.DataFrame(bars)
            
            # Map Polygon columns to standard OHLCV
            df = df.rename(columns={
                "t": "date",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume"
            })
            
            # Convert timestamp to date
            df["date"] = pd.to_datetime(df["date"], unit="ms").dt.date
            
            # Add adj_close if not present
            if "adj_close" not in df.columns:
                df["adj_close"] = df["close"]
            
            return df
        return pd.DataFrame()
    
    def get_intraday_ohlcv(self, symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
        """Get intraday bars"""
        if not self.api_key:
            raise ValueError("Polygon API key not configured")
        
        # Map interval to Polygon multiplier
        interval_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}
        multiplier = interval_map.get(interval, 1)
        timespan = "minute"
        
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start}/{end}"
        response = requests.get(url, params={"apikey": self.api_key})
        response.raise_for_status()
        
        data = response.json()
        if data.get("status") == "OK":
            bars = data.get("results", [])
            df = pd.DataFrame(bars)
            
            df = df.rename(columns={
                "t": "date",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume"
            })
            df["date"] = pd.to_datetime(df["date"], unit="ms")
            
            if "adj_close" not in df.columns:
                df["adj_close"] = df["close"]
            
            return df
        return pd.DataFrame()


class AlpacaProvider(BaseProvider):
    """Alpaca Markets data provider"""
    
    def __init__(self):
        self.api_key = os.getenv("ALPACA_KEY")
        self.api_secret = os.getenv("ALPACA_SECRET")
        self.base_url = "https://data.alpaca.markets/v2"
        
        if not self.api_key or not self.api_secret:
            logger.warning("ALPACA_KEY or ALPACA_SECRET not found in .env")
    
    def get_latest_quote(self, symbol: str) -> dict:
        """Get latest quote from Alpaca"""
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials not configured")
        
        url = f"{self.base_url}/stocks/{symbol}/quotes/latest"
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        quote = data.get("quote", {})
        return {
            "symbol": symbol,
            "price": quote.get("p"),
            "timestamp": quote.get("t"),
            "provider": "alpaca"
        }
    
    def get_historical_ohlcv(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Get historical bars from Alpaca"""
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials not configured")
        
        url = f"{self.base_url}/stocks/{symbol}/bars"
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret
        }
        
        params = {
            "start": start,
            "end": end,
            "timeframe": "1Day"
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        bars = data.get("bars", [])
        
        df = pd.DataFrame(bars)
        if not df.empty:
            df = df.rename(columns={
                "t": "date",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume"
            })
            df["date"] = pd.to_datetime(df["date"])
            
            if "adj_close" not in df.columns:
                df["adj_close"] = df["close"]
        
        return df
    
    def get_intraday_ohlcv(self, symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
        """Get intraday bars from Alpaca"""
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials not configured")
        
        url = f"{self.base_url}/stocks/{symbol}/bars"
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret
        }
        
        params = {
            "start": start,
            "end": end,
            "timeframe": interval
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        bars = data.get("bars", [])
        
        df = pd.DataFrame(bars)
        if not df.empty:
            df = df.rename(columns={
                "t": "date",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume"
            })
            df["date"] = pd.to_datetime(df["date"])
            
            if "adj_close" not in df.columns:
                df["adj_close"] = df["close"]
        
        return df


class FallbackProvider(BaseProvider):
    """yfinance fallback provider - used when Polygon/Alpaca unavailable"""
    
    def __init__(self):
        logger.warning("USING FALLBACK DATA PROVIDER: yfinance (delayed data)")
        self.provider = "yfinance_fallback"
    
    def get_latest_quote(self, symbol: str) -> dict:
        """Get latest quote from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            
            if not data.empty:
                latest = data.iloc[-1]
                return {
                    "symbol": symbol,
                    "price": float(latest["Close"]),
                    "timestamp": data.index[-1].timestamp(),
                    "provider": "yfinance_fallback"
                }
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
        
        return {"symbol": symbol, "error": "No data"}
    
    def get_historical_ohlcv(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Get historical data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end)
            
            if not df.empty:
                df = df.reset_index()
                
                # Map columns
                if "Date" in df.columns:
                    df = df.rename(columns={"Date": "date"})
                
                df = df.rename(columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume"
                })
                
                # Handle Adjusted Close
                if "Adj Close" in df.columns:
                    df = df.rename(columns={"Adj Close": "adj_close"})
                else:
                    df["adj_close"] = df["close"]
                
                return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
        
        return pd.DataFrame()
    
    def get_intraday_ohlcv(self, symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
        """Get intraday data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval=interval)
            
            if not df.empty:
                df = df.reset_index()
                df = df.rename(columns={"Date": "date"})
                df = df.rename(columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                    "Adj Close": "adj_close"
                })
                return df
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
        
        return pd.DataFrame()


def get_data_provider(config: dict) -> BaseProvider:
    """
    Factory function to get appropriate data provider based on config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Provider instance
    """
    provider_name = config.get("data", {}).get("provider", "polygon")
    
    try:
        if provider_name == "polygon":
            polygon_provider = PolygonProvider()
            # Test if API key is available
            if polygon_provider.api_key:
                return polygon_provider
            else:
                logger.warning("Polygon API key not found, using fallback")
                return FallbackProvider()
        elif provider_name == "alpaca":
            alpaca_provider = AlpacaProvider()
            if alpaca_provider.api_key and alpaca_provider.api_secret:
                return alpaca_provider
            else:
                logger.warning("Alpaca API keys not found, using fallback")
                return FallbackProvider()
        else:
            logger.warning(f"Unknown provider '{provider_name}', using fallback")
            return FallbackProvider()
    except Exception as e:
        logger.error(f"Error initializing {provider_name} provider: {e}")
        logger.warning("Falling back to yfinance")
        return FallbackProvider()


if __name__ == "__main__":
    # Test providers
    import yaml
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    provider = get_data_provider(config)
    
    # Test quote
    quote = provider.get_latest_quote("AAPL")
    print(f"Latest quote: {quote}")

