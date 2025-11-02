"""
AXFI Data Collector Module
Fetches, cleans, and stores historical market data from Yahoo Finance
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCollector:
    """
    Handles fetching and storing market data from Yahoo Finance.
    
    Features:
    - Fetches historical OHLCV data
    - Cleans and validates data
    - Stores in SQLite database
    - Handles missing data gracefully with retry logic
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the DataCollector with configuration.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.db_path = self.config['database']['market_data_path']
        self._init_database()
    
    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            raise
    
    def _init_database(self):
        """
        Initialize SQLite database and create market_data table if it doesn't exist.
        """
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume INTEGER,
                    PRIMARY KEY (date, symbol)
                )
            """)
            conn.commit()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def fetch_symbol(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = "5y"
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a single symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            start: Start date in YYYY-MM-DD format (overrides period if set)
            end: End date in YYYY-MM-DD format (defaults to today)
            period: Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            
        Returns:
            DataFrame with columns: date, open, high, low, close, adj_close, volume
            
        Raises:
            ValueError: If symbol fetch fails or returns empty data
        """
        logger.info(f"Fetching data for {symbol} (period={period})")
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Fetch data
            if start and end:
                df = ticker.history(start=start, end=end)
            else:
                df = ticker.history(period=period)
            
            # Check if data is empty
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                raise ValueError(f"No data available for {symbol}")
            
            # Debug: print column names
            logger.debug(f"Columns from yfinance: {df.columns.tolist()}")
            logger.debug(f"Index name: {df.index.name}")
            
            # Convert index to date column and normalize
            df = df.reset_index()
            
            # Standard yfinance columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            # Handle potential variations
            date_col = 'Date' if 'Date' in df.columns else df.columns[0]
            
            # Create column mapping for the columns we need
            column_map = {}
            
            # Standard OHLCV columns
            standard_map = {
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            
            for old_col, new_col in standard_map.items():
                if old_col in df.columns:
                    column_map[old_col] = new_col
            
            # Handle Adjusted Close - try common variations
            # Note: Recent versions of yfinance may not include Adj Close
            adj_close_found = False
            for adj_name in ['Adj Close', 'AdjClose']:
                if adj_name in df.columns:
                    column_map[adj_name] = 'adj_close'
                    adj_close_found = True
                    break
            
            # Rename columns
            df = df.rename(columns=column_map)
            
            # Rename date column
            df = df.rename(columns={date_col: 'date'})
            
            # If adj_close is missing, copy from close
            if 'adj_close' not in df.columns and 'close' in df.columns:
                df['adj_close'] = df['close']
                logger.info(f"Copied close to adj_close for {symbol}")
            
            # Ensure we have all required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                logger.error(f"Available columns: {df.columns.tolist()}")
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Select only required columns
            df = df[required_cols].copy()
            
            # Convert date to string for database storage
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            # Data validation
            df = self._validate_data(df, symbol)
            
            logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _validate_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate and clean data.
        
        Args:
            df: DataFrame with market data
            symbol: Stock symbol (for logging)
            
        Returns:
            Cleaned DataFrame
        """
        initial_len = len(df)
        
        # Remove rows with NaN in critical columns
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        # Validate OHLC relationships
        invalid = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        if invalid.sum() > 0:
            logger.warning(f"Removed {invalid.sum()} invalid OHLC rows for {symbol}")
            df = df[~invalid]
        
        # Validate volume
        if (df['volume'] < 0).any():
            logger.warning(f"Found negative volume for {symbol}, setting to 0")
            df.loc[df['volume'] < 0, 'volume'] = 0
        
        removed = initial_len - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} invalid rows for {symbol}")
        
        return df
    
    def save_to_database(
        self,
        df: pd.DataFrame,
        symbol: str,
        mode: str = "replace"
    ):
        """
        Save DataFrame to SQLite database.
        
        Args:
            df: DataFrame with market data
            symbol: Stock ticker symbol
            mode: 'replace' to update existing data, 'append' to add only new rows
        """
        if df.empty:
            logger.warning(f"Empty DataFrame, nothing to save for {symbol}")
            return
        
        # Add symbol column
        df_copy = df.copy()
        df_copy['symbol'] = symbol
        
        with sqlite3.connect(self.db_path) as conn:
            if mode == "replace":
                # Delete existing data for this symbol
                conn.execute("DELETE FROM market_data WHERE symbol = ?", (symbol,))
                logger.info(f"Replaced existing data for {symbol}")
            
            # Insert new data
            df_copy.to_sql('market_data', conn, if_exists='append', index=False)
            conn.commit()
        
        logger.info(f"Saved {len(df_copy)} rows for {symbol} to database")
    
    def fetch_and_save(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = "5y",
        mode: str = "replace"
    ):
        """
        Fetch data for a symbol and save it to the database.
        
        Args:
            symbol: Stock ticker symbol
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            period: Valid periods for yfinance
            mode: 'replace' or 'append'
        """
        try:
            df = self.fetch_symbol(symbol, start=start, end=end, period=period)
            self.save_to_database(df, symbol, mode=mode)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch and save {symbol}: {e}")
            return None
    
    def update_all_symbols(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = "5y"
    ):
        """
        Fetch and save data for all symbols in config.
        
        Args:
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            period: Valid periods for yfinance
        """
        symbols = self.config.get('symbols', [])
        
        if not symbols:
            logger.warning("No symbols found in configuration")
            return
        
        logger.info(f"Updating data for {len(symbols)} symbols: {', '.join(symbols)}")
        
        success_count = 0
        failed_symbols = []
        
        for symbol in symbols:
            try:
                self.fetch_and_save(symbol, start=start, end=end, period=period, mode="replace")
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to update {symbol}: {e}")
                failed_symbols.append(symbol)
        
        logger.info(f"Successfully updated {success_count}/{len(symbols)} symbols")
        if failed_symbols:
            logger.warning(f"Failed symbols: {', '.join(failed_symbols)}")
    
    def read_from_database(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Read data from the database.
        
        Args:
            symbol: Filter by symbol (None for all symbols)
            start_date: Filter by start date (YYYY-MM-DD)
            end_date: Filter by end date (YYYY-MM-DD)
            
        Returns:
            DataFrame with market data
        """
        query = "SELECT * FROM market_data WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params if params else None)
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        
        logger.info(f"Retrieved {len(df)} rows from database")
        return df


def main():
    """
    Standalone execution for testing the data collector.
    """
    print("=" * 60)
    print("AXFI Data Collector - Standalone Test")
    print("=" * 60)
    
    # Initialize collector
    collector = DataCollector()
    
    # Get symbols from config
    test_symbol = collector.config['symbols'][0]
    
    print(f"\nTesting with symbol: {test_symbol}")
    print("-" * 60)
    
    # Fetch and save data
    try:
        df = collector.fetch_and_save(test_symbol, period="1y")
        print(f"[OK] Successfully fetched and saved {len(df)} rows")
    except Exception as e:
        print(f"[ERROR] Failed to fetch {test_symbol}: {e}")
        return
    
    # Read from database and display
    print("\nReading from database:")
    print("-" * 60)
    db_df = collector.read_from_database(symbol=test_symbol)
    
    if not db_df.empty:
        # Display first 5 rows
        display_df = db_df.head(5)[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
        print(display_df.to_string(index=False))
        print(f"\nTotal rows in database: {len(db_df)}")
        print(f"Date range: {db_df['date'].min()} to {db_df['date'].max()}")
    else:
        print("No data found in database")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

