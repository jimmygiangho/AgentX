"""
AXFI Storage Layer
DuckDB wrapper with SQLite fallback
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Try to import DuckDB, fallback to SQLite if not available
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    logger.warning("DuckDB not available, using SQLite fallback")
    DUCKDB_AVAILABLE = False


class Storage:
    """
    Storage layer for AXFI data.
    Uses DuckDB by default, falls back to SQLite on Windows or if DuckDB unavailable.
    """
    
    def __init__(self, db_path: str, use_duckdb: bool = True):
        """
        Initialize storage connection.
        
        Args:
            db_path: Path to database file
            use_duckdb: If True, use DuckDB; else SQLite
        """
        self.db_path = db_path
        self.use_duckdb = use_duckdb and DUCKDB_AVAILABLE
        self.conn = None
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create connection
        self.connect()
    
    def connect(self):
        """Create database connection"""
        if self.use_duckdb:
            self.conn = duckdb.connect(self.db_path)
            logger.info(f"Connected to DuckDB: {self.db_path}")
        else:
            self.conn = sqlite3.connect(self.db_path)
            logger.info(f"Connected to SQLite: {self.db_path}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")
    
    def write_table(self, name: str, df: pd.DataFrame, if_exists: str = "replace") -> bool:
        """
        Write DataFrame to table.
        
        Args:
            name: Table name
            df: DataFrame to write
            if_exists: What to do if table exists ('replace', 'append', 'fail')
            
        Returns:
            True if successful
        """
        if df.empty:
            logger.warning(f"Empty DataFrame for table {name}")
            return False
        
        try:
            if self.use_duckdb:
                # DuckDB syntax
                if if_exists == "replace":
                    self.conn.execute(f"DROP TABLE IF EXISTS {name}")
                
                self.conn.register("temp_df", df)
                self.conn.execute(f"CREATE TABLE {name} AS SELECT * FROM temp_df")
                self.conn.unregister("temp_df")
            else:
                # SQLite syntax
                df.to_sql(name, self.conn, if_exists=if_exists, index=False)
            
            logger.info(f"Wrote {len(df)} rows to table {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing table {name}: {e}")
            return False
    
    def read_table(self, name: str, query: Optional[str] = None) -> pd.DataFrame:
        """
        Read table or execute query.
        
        Args:
            name: Table name or query
            query: Optional SQL query string
            
        Returns:
            DataFrame with results
        """
        try:
            if query:
                if self.use_duckdb:
                    result = self.conn.execute(query).fetchdf()
                else:
                    result = pd.read_sql_query(query, self.conn)
            else:
                if self.use_duckdb:
                    result = self.conn.execute(f"SELECT * FROM {name}").fetchdf()
                else:
                    result = pd.read_sql_query(f"SELECT * FROM {name}", self.conn)
            
            logger.info(f"Read {len(result)} rows from {name}")
            return result
            
        except Exception as e:
            logger.error(f"Error reading table {name}: {e}")
            return pd.DataFrame()
    
    def execute(self, sql: str) -> bool:
        """
        Execute SQL statement.
        
        Args:
            sql: SQL statement
            
        Returns:
            True if successful
        """
        try:
            if self.use_duckdb:
                self.conn.execute(sql)
            else:
                self.conn.execute(sql).commit()
            
            logger.info(f"Executed SQL: {sql[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            return False
    
    def table_exists(self, name: str) -> bool:
        """
        Check if table exists.
        
        Args:
            name: Table name
            
        Returns:
            True if table exists
        """
        try:
            if self.use_duckdb:
                result = self.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (name,)
                ).fetchone()
            else:
                result = self.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (name,)
                ).fetchone()
            
            return result is not None
        except Exception as e:
            logger.error(f"Error checking table {name}: {e}")
            return False


if __name__ == "__main__":
    # Test storage
    storage = Storage("./db/test.db")
    
    # Test write
    df = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "price": [150.0, 200.0]})
    storage.write_table("test_table", df)
    
    # Test read
    result = storage.read_table("test_table")
    print(f"Read data:\n{result}")
    
    storage.close()

