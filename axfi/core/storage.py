"""
AXFI Storage Layer
DuckDB wrapper for efficient indicator and feature persistence
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Try to import DuckDB
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    logger.warning("DuckDB not available, using pandas parquet fallback")
    DUCKDB_AVAILABLE = False


class Storage:
    """
    Storage layer for AXFI data using DuckDB.
    Handles raw OHLCV, indicators, features, and scan results.
    
    Tables:
    - raw_ohlcv: Raw price data (parquet)
    - indicators_snapshot: Daily indicator snapshots
    - features_daily: Daily feature vectors with normalization
    - scan_results: Scan recommendations
    """
    
    def __init__(self, db_path: str, use_duckdb: bool = True):
        """
        Initialize storage connection.
        
        Args:
            db_path: Path to database file
            use_duckdb: If True, use DuckDB; else use parquet files
        """
        self.db_path = Path(db_path)
        self.use_duckdb = use_duckdb and DUCKDB_AVAILABLE
        self.conn = None
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create connection
        self.connect()
        self._init_tables()
    
    def connect(self):
        """Create database connection"""
        if self.use_duckdb:
            self.conn = duckdb.connect(str(self.db_path))
            logger.info(f"Connected to DuckDB: {self.db_path}")
        else:
            self.conn = None
            logger.info(f"Using parquet file storage: {self.db_path.parent}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")
    
    def _init_tables(self):
        """Initialize database tables if they don't exist"""
        if not self.use_duckdb:
            return
        
        try:
            # Raw OHLCV table (partitioned by symbol, date)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_ohlcv (
                    symbol VARCHAR,
                    date DATE,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume BIGINT,
                    adj_close DOUBLE,
                    PRIMARY KEY (symbol, date)
                )
            """)
            
            # Indicators snapshot (daily snapshots)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS indicators_snapshot (
                    symbol VARCHAR,
                    date DATE,
                    ema_10 DOUBLE,
                    ema_20 DOUBLE,
                    ema_50 DOUBLE,
                    ema_100 DOUBLE,
                    ema_200 DOUBLE,
                    macd DOUBLE,
                    macd_signal DOUBLE,
                    macd_histogram DOUBLE,
                    adx DOUBLE,
                    plus_di DOUBLE,
                    minus_di DOUBLE,
                    rsi_7 DOUBLE,
                    rsi_14 DOUBLE,
                    rsi_21 DOUBLE,
                    stoch_k DOUBLE,
                    stoch_d DOUBLE,
                    bb_20_upper DOUBLE,
                    bb_20_lower DOUBLE,
                    bb_20_middle DOUBLE,
                    bb_20_width DOUBLE,
                    bb_50_upper DOUBLE,
                    bb_50_lower DOUBLE,
                    bb_50_middle DOUBLE,
                    bb_50_width DOUBLE,
                    atr_14 DOUBLE,
                    atr_28 DOUBLE,
                    donchian_upper DOUBLE,
                    donchian_lower DOUBLE,
                    keltner_upper DOUBLE,
                    keltner_lower DOUBLE,
                    obv DOUBLE,
                    mfi DOUBLE,
                    PRIMARY KEY (symbol, date)
                )
            """)
            
            # Features daily (with normalization)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS features_daily (
                    symbol VARCHAR,
                    date DATE,
                    -- Add feature columns dynamically
                    PRIMARY KEY (symbol, date)
                )
            """)
            
            # Scan results
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS scan_results (
                    scan_date DATE,
                    symbol VARCHAR,
                    rank INTEGER,
                    strategy VARCHAR,
                    score DOUBLE,
                    confidence DOUBLE,
                    explanation TEXT,
                    top_features TEXT,
                    timestamp TIMESTAMP,
                    PRIMARY KEY (scan_date, symbol)
                )
            """)
            
            logger.info("Initialized database tables")
            
        except Exception as e:
            logger.error(f"Error initializing tables: {e}")
    
    def write_raw_ohlcv(self, symbol: str, df: pd.DataFrame, mode: str = "replace") -> bool:
        """
        Write raw OHLCV data to storage.
        
        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data
            mode: 'replace' or 'append'
            
        Returns:
            True if successful
        """
        if df.empty:
            logger.warning(f"Empty DataFrame for {symbol}")
            return False
        
        try:
            df_copy = df.copy()
            
            # Ensure date column
            if 'date' not in df_copy.columns and df_copy.index.name == 'date':
                df_copy = df_copy.reset_index()
            
            if 'date' in df_copy.columns:
                df_copy['date'] = pd.to_datetime(df_copy['date']).dt.date
            
            df_copy['symbol'] = symbol
            
            # Select required columns
            required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
            if 'adj_close' in df_copy.columns:
                required_cols.append('adj_close')
            
            df_copy = df_copy[[c for c in required_cols if c in df_copy.columns]]
            
            if self.use_duckdb:
                if mode == "replace":
                    # Delete existing data for symbol
                    self.conn.execute(
                        "DELETE FROM raw_ohlcv WHERE symbol = ?",
                        [symbol]
                    )
                
                # Register and insert
                self.conn.register("temp_ohlcv", df_copy)
                self.conn.execute("""
                    INSERT INTO raw_ohlcv 
                    SELECT * FROM temp_ohlcv
                """)
                self.conn.unregister("temp_ohlcv")
            else:
                # Save as parquet
                parquet_path = self.db_path.parent / "raw_ohlcv" / f"{symbol}.parquet"
                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                
                if mode == "replace" or not parquet_path.exists():
                    df_copy.to_parquet(parquet_path, index=False)
                else:
                    # Append mode: read existing and merge
                    existing = pd.read_parquet(parquet_path)
                    combined = pd.concat([existing, df_copy]).drop_duplicates(
                        subset=['date'], keep='last'
                    ).sort_values('date')
                    combined.to_parquet(parquet_path, index=False)
            
            logger.info(f"Wrote {len(df_copy)} rows of OHLCV for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing raw OHLCV for {symbol}: {e}")
            return False
    
    def write_indicators_snapshot(self, symbol: str, df: pd.DataFrame, date: Optional[str] = None) -> bool:
        """
        Write daily indicator snapshot.
        
        Args:
            symbol: Stock symbol
            df: DataFrame with indicators
            date: Date for snapshot (if None, uses latest)
            
        Returns:
            True if successful
        """
        if df.empty:
            logger.warning(f"Empty DataFrame for {symbol}")
            return False
        
        try:
            # Get snapshot row
            if date:
                if 'date' in df.columns:
                    snapshot = df[df['date'] == date].iloc[-1:].copy()
                else:
                    snapshot = df.loc[df.index == pd.to_datetime(date)].copy()
            else:
                snapshot = df.iloc[-1:].copy()
            
            if snapshot.empty:
                logger.warning(f"No data found for snapshot {symbol}")
                return False
            
            # Extract date
            if 'date' in snapshot.columns:
                snapshot_date = pd.to_datetime(snapshot['date'].iloc[0]).date()
            else:
                snapshot_date = snapshot.index[0].date() if isinstance(snapshot.index, pd.DatetimeIndex) else datetime.now().date()
            
            # Prepare indicator columns
            indicator_cols = {
                'symbol': symbol,
                'date': snapshot_date,
            }
            
            # Map indicator columns
            indicator_mapping = {
                'ema_10': 'ema_10', 'ema_20': 'ema_20', 'ema_50': 'ema_50',
                'ema_100': 'ema_100', 'ema_200': 'ema_200',
                'macd': 'macd', 'macd_signal': 'macd_signal', 'macd_histogram': 'macd_histogram',
                'adx': 'adx', '+di': 'plus_di', '-di': 'minus_di',
                'rsi_7': 'rsi_7', 'rsi_14': 'rsi_14', 'rsi_21': 'rsi_21',
                'stoch_k': 'stoch_k', 'stoch_d': 'stoch_d',
                'bb_20_upper': 'bb_20_upper', 'bb_20_lower': 'bb_20_lower',
                'bb_20_middle': 'bb_20_middle', 'bb_20_width': 'bb_20_width',
                'bb_50_upper': 'bb_50_upper', 'bb_50_lower': 'bb_50_lower',
                'bb_50_middle': 'bb_50_middle', 'bb_50_width': 'bb_50_width',
                'atr_14': 'atr_14', 'atr_28': 'atr_28',
                'donchian_upper': 'donchian_upper', 'donchian_lower': 'donchian_lower',
                'keltner_upper': 'keltner_upper', 'keltner_lower': 'keltner_lower',
                'obv': 'obv', 'mfi': 'mfi'
            }
            
            for df_col, db_col in indicator_mapping.items():
                if df_col in snapshot.columns:
                    val = snapshot[df_col].iloc[0]
                    indicator_cols[db_col] = float(val) if pd.notna(val) else None
            
            # Create DataFrame
            snapshot_df = pd.DataFrame([indicator_cols])
            
            if self.use_duckdb:
                # Delete existing snapshot for this symbol/date
                self.conn.execute(
                    "DELETE FROM indicators_snapshot WHERE symbol = ? AND date = ?",
                    [symbol, snapshot_date]
                )
                
                # Insert new snapshot
                self.conn.register("temp_snapshot", snapshot_df)
                self.conn.execute("""
                    INSERT INTO indicators_snapshot 
                    SELECT * FROM temp_snapshot
                """)
                self.conn.unregister("temp_snapshot")
            else:
                # Save as parquet
                parquet_path = self.db_path.parent / "indicators_snapshot" / f"{symbol}_{snapshot_date}.parquet"
                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                snapshot_df.to_parquet(parquet_path, index=False)
            
            logger.debug(f"Wrote indicator snapshot for {symbol} on {snapshot_date}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing indicator snapshot for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def write_features_daily(self, symbol: str, df: pd.DataFrame, date: Optional[str] = None) -> bool:
        """
        Write daily features with normalization.
        
        Args:
            symbol: Stock symbol
            df: DataFrame with features
            date: Date for snapshot
            
        Returns:
            True if successful
        """
        if df.empty:
            return False
        
        try:
            # Get snapshot
            if date:
                if 'date' in df.columns:
                    snapshot = df[df['date'] == date].iloc[-1:].copy()
                else:
                    snapshot = df.loc[df.index == pd.to_datetime(date)].copy()
            else:
                snapshot = df.iloc[-1:].copy()
            
            snapshot['symbol'] = symbol
            
            if 'date' in snapshot.columns:
                snapshot['date'] = pd.to_datetime(snapshot['date']).dt.date
            elif isinstance(snapshot.index, pd.DatetimeIndex):
                snapshot = snapshot.reset_index()
                snapshot['date'] = snapshot['date'].dt.date
            
            # Store all feature columns
            if self.use_duckdb:
                # Get existing columns
                try:
                    existing_cols = set(self.conn.execute(
                        "DESCRIBE features_daily"
                    ).fetchdf()['column_name'].tolist())
                except:
                    existing_cols = {'symbol', 'date'}
                
                # Add missing columns
                for col in snapshot.columns:
                    if col not in existing_cols and col not in ['symbol', 'date']:
                        try:
                            self.conn.execute(f"ALTER TABLE features_daily ADD COLUMN {col} DOUBLE")
                        except:
                            pass  # Column might already exist
                
                # Delete existing and insert
                snapshot_date = snapshot['date'].iloc[0]
                self.conn.execute(
                    "DELETE FROM features_daily WHERE symbol = ? AND date = ?",
                    [symbol, snapshot_date]
                )
                
                self.conn.register("temp_features", snapshot)
                self.conn.execute("""
                    INSERT INTO features_daily 
                    SELECT * FROM temp_features
                """)
                self.conn.unregister("temp_features")
            else:
                # Parquet storage
                snapshot_date = snapshot['date'].iloc[0]
                parquet_path = self.db_path.parent / "features_daily" / f"{symbol}_{snapshot_date}.parquet"
                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                snapshot.to_parquet(parquet_path, index=False)
            
            logger.debug(f"Wrote features for {symbol} on {snapshot_date}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing features for {symbol}: {e}")
            return False
    
    def read_raw_ohlcv(self, symbol: Optional[str] = None, 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Read raw OHLCV data.
        
        Args:
            symbol: Symbol to filter (None for all)
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if self.use_duckdb:
                query = "SELECT * FROM raw_ohlcv WHERE 1=1"
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
                
                query += " ORDER BY date"
                
                if params:
                    result = self.conn.execute(query, params).fetchdf()
                else:
                    result = self.conn.execute(query).fetchdf()
            else:
                # Read from parquet
                if symbol:
                    parquet_path = self.db_path.parent / "raw_ohlcv" / f"{symbol}.parquet"
                    if parquet_path.exists():
                        result = pd.read_parquet(parquet_path)
                    else:
                        result = pd.DataFrame()
                else:
                    # Read all symbols
                    parquet_dir = self.db_path.parent / "raw_ohlcv"
                    if parquet_dir.exists():
                        files = list(parquet_dir.glob("*.parquet"))
                        results = [pd.read_parquet(f) for f in files]
                        result = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
                    else:
                        result = pd.DataFrame()
            
            if not result.empty and 'date' in result.columns:
                result['date'] = pd.to_datetime(result['date'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error reading raw OHLCV: {e}")
            return pd.DataFrame()
    
    def read_indicators_snapshot(self, symbol: str, date: Optional[str] = None) -> pd.DataFrame:
        """
        Read indicator snapshot.
        
        Args:
            symbol: Stock symbol
            date: Date for snapshot (None for latest)
            
        Returns:
            DataFrame with indicators
        """
        try:
            if self.use_duckdb:
                if date:
                    query = "SELECT * FROM indicators_snapshot WHERE symbol = ? AND date = ?"
                    result = self.conn.execute(query, [symbol, date]).fetchdf()
                else:
                    query = "SELECT * FROM indicators_snapshot WHERE symbol = ? ORDER BY date DESC LIMIT 1"
                    result = self.conn.execute(query, [symbol]).fetchdf()
            else:
                # Read from parquet
                if date:
                    parquet_path = self.db_path.parent / "indicators_snapshot" / f"{symbol}_{date}.parquet"
                else:
                    # Find latest
                    parquet_dir = self.db_path.parent / "indicators_snapshot"
                    if parquet_dir.exists():
                        files = list(parquet_dir.glob(f"{symbol}_*.parquet"))
                        if files:
                            files.sort(reverse=True)
                            parquet_path = files[0]
                        else:
                            return pd.DataFrame()
                    else:
                        return pd.DataFrame()
                
                if parquet_path.exists():
                    result = pd.read_parquet(parquet_path)
                else:
                    result = pd.DataFrame()
            
            return result
            
        except Exception as e:
            logger.error(f"Error reading indicator snapshot: {e}")
            return pd.DataFrame()
    
    def table_exists(self, name: str) -> bool:
        """Check if table exists"""
        if not self.use_duckdb:
            return False
        
        try:
            result = self.conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_name = ?",
                [name]
            ).fetchone()
            return result is not None
        except:
            # Fallback: try to query the table
            try:
                self.conn.execute(f"SELECT 1 FROM {name} LIMIT 1")
                return True
            except:
                return False


if __name__ == "__main__":
    # Test storage
    storage = Storage("./db/test_storage.duckdb")
    
    # Create test data
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    test_df = pd.DataFrame({
        'date': dates,
        'open': np.random.randn(10).cumsum() + 100,
        'high': np.random.randn(10).cumsum() + 101,
        'low': np.random.randn(10).cumsum() + 99,
        'close': np.random.randn(10).cumsum() + 100,
        'volume': np.random.randint(1000000, 5000000, 10)
    })
    
    # Test write/read
    storage.write_raw_ohlcv("TEST", test_df)
    result = storage.read_raw_ohlcv("TEST")
    
    print(f"Wrote and read {len(result)} rows")
    print(result.head())
    
    storage.close()
