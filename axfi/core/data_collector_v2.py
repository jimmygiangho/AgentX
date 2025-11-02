"""
AXFI Data Collector v2
Enhanced version using provider abstraction and storage layer
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_providers import get_data_provider
from core.storage import Storage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCollectorV2:
    """
    Enhanced data collector using provider abstraction and storage layer.
    
    Features:
    - Supports Polygon, Alpaca, yfinance providers
    - Uses DuckDB/SQLite storage
    - Intelligent caching
    - Live data fetching
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the DataCollector.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.provider = get_data_provider(self.config)
        self.storage = self._init_storage()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Import yfinance fallback utilities
        try:
            from core.data_utils import FallbackProvider
            self.fallback_provider = FallbackProvider() if hasattr(self.provider, 'provider') and self.provider.provider == "yfinance_fallback" else None
        except ImportError:
            self.fallback_provider = None
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    
    def _init_storage(self) -> Storage:
        """Initialize storage layer"""
        db_path = self.config.get("database", {}).get("market_data_path", "./db/market_data.db")
        use_duckdb = self.config.get("database", {}).get("engine", "duckdb") == "duckdb"
        return Storage(db_path, use_duckdb=use_duckdb)
    
    def fetch_symbol(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = "1y"
    ) -> pd.DataFrame:
        """
        Fetch historical data for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            period: yfinance period (used as fallback)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Calculate dates if not provided
        if not end:
            end = datetime.now().strftime("%Y-%m-%d")
        
        if not start:
            # Calculate start based on period
            period_days = {
                "1d": 1,
                "5d": 5,
                "1mo": 30,
                "3mo": 90,
                "6mo": 180,
                "1y": 365,
                "2y": 730,
                "5y": 1825
            }
            days = period_days.get(period, 365)
            start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        self.logger.info(f"Fetching {symbol} from {start} to {end} using {self.provider.__class__.__name__}")
        
        # Try primary provider first
        df = self.provider.get_historical_ohlcv(symbol, start, end)
        
        # If primary provider fails or returns empty, try yfinance fallback
        if df.empty or "error" in str(df).lower():
            self.logger.info(f"Primary provider failed for {symbol}, trying yfinance fallback...")
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start, end=end)
                
                if not df.empty:
                    df = df.reset_index()
                    if "Date" in df.columns:
                        df = df.rename(columns={"Date": "date"})
                    df = df.rename(columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume"
                    })
                    if "Adj Close" in df.columns:
                        df = df.rename(columns={"Adj Close": "adj_close"})
                    else:
                        df["adj_close"] = df["close"]
                    self.logger.info(f"Successfully fetched {len(df)} rows using yfinance fallback")
            except Exception as e:
                self.logger.error(f"Fallback provider also failed: {e}")
        
        if df.empty:
            self.logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()
        
        # Ensure required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            self.logger.error(f"Missing columns: {missing}")
            return pd.DataFrame()
        
        self.logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
        return df
    
    def save_to_database(
        self,
        df: pd.DataFrame,
        symbol: str,
        mode: str = "replace"
    ):
        """
        Save DataFrame to storage.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock ticker symbol
            mode: 'replace' or 'append'
        """
        if df.empty:
            self.logger.warning(f"Empty DataFrame for {symbol}")
            return
        
        # Add symbol column
        df_copy = df.copy()
        df_copy['symbol'] = symbol
        
        # Ensure date is string
        if df_copy['date'].dtype == 'datetime64[ns]':
            df_copy['date'] = df_copy['date'].dt.strftime('%Y-%m-%d')
        
        table_name = f"ohlcv_{symbol}"
        self.storage.write_table(table_name, df_copy, if_exists=mode)
        self.logger.info(f"Saved {len(df_copy)} rows for {symbol}")
    
    def read_from_database(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Read data from storage.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            
        Returns:
            DataFrame with OHLCV data
        """
        table_name = f"ohlcv_{symbol}"
        
        # Build query
        if start_date and end_date:
            query = f"""
                SELECT * FROM {table_name}
                WHERE date >= '{start_date}' AND date <= '{end_date}'
                ORDER BY date
            """
        elif start_date:
            query = f"""
                SELECT * FROM {table_name}
                WHERE date >= '{start_date}'
                ORDER BY date
            """
        elif end_date:
            query = f"""
                SELECT * FROM {table_name}
                WHERE date <= '{end_date}'
                ORDER BY date
            """
        else:
            query = f"""
                SELECT * FROM {table_name}
                ORDER BY date
            """
        
        df = self.storage.read_table(table_name, query=query)
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            self.logger.info(f"Read {len(df)} rows for {symbol}")
        
        return df
    
    def get_latest_quote(self, symbol: str, use_fallback: bool = True) -> dict:
        """
        Get latest quote for a symbol with automatic fallback to yfinance.
        
        Args:
            symbol: Stock ticker symbol
            use_fallback: Whether to use yfinance if primary fails
            
        Returns:
            Dictionary with quote data
        """
        self.logger.info(f"Fetching latest quote for {symbol} using {self.provider.__class__.__name__}")
        
        try:
            quote = self.provider.get_latest_quote(symbol)
            
            # Check if quote is valid (has price and no error)
            if quote and "price" in quote and "error" not in str(quote).lower():
                self.logger.info(f"Quote for {symbol}: ${quote.get('price', 0):.2f} from {quote.get('provider', 'unknown')}")
                return quote
        except Exception as e:
            self.logger.warning(f"Primary provider failed for {symbol}: {e}")
        
        # Fallback to yfinance if enabled and primary failed
        if use_fallback:
            self.logger.info(f"Trying yfinance fallback for {symbol}...")
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d")
                
                if not data.empty:
                    latest = data.iloc[-1]
                    quote = {
                        "symbol": symbol,
                        "price": float(latest["Close"]),
                        "timestamp": data.index[-1].timestamp(),
                        "provider": "yfinance_fallback"
                    }
                    self.logger.info(f"Quote from yfinance fallback: ${quote['price']:.2f}")
                    return quote
            except Exception as e:
                self.logger.error(f"yfinance fallback also failed for {symbol}: {e}")
        
        return {"symbol": symbol, "error": "No data available"}
    
    def fetch_and_save(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = "1y",
        mode: str = "replace"
    ):
        """
        Fetch data and save to database.
        
        Args:
            symbol: Stock ticker symbol
            start: Start date
            end: End date
            period: Period string
            mode: Save mode
        """
        df = self.fetch_symbol(symbol, start=start, end=end, period=period)
        
        if not df.empty:
            self.save_to_database(df, symbol, mode=mode)
        else:
            self.logger.error(f"Failed to fetch data for {symbol}")


def main():
    """Test the data collector"""
    collector = DataCollectorV2()
    
    # Test latest quote
    print("\n=== Testing Latest Quote ===")
    quote = collector.get_latest_quote("AAPL")
    print(f"Quote: {quote}")
    
    # Test historical fetch
    print("\n=== Testing Historical Fetch ===")
    df = collector.fetch_symbol("AAPL", period="1y")
    print(f"Fetched {len(df)} rows")
    if not df.empty:
        print(df.head())
    
    # Test save
    print("\n=== Testing Save ===")
    collector.save_to_database(df, "AAPL", mode="replace")
    
    # Test read
    print("\n=== Testing Read ===")
    df_read = collector.read_from_database("AAPL")
    print(f"Read {len(df_read)} rows")
    if not df_read.empty:
        print(df_read.head())


if __name__ == "__main__":
    main()

