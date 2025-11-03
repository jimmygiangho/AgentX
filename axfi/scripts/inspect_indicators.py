#!/usr/bin/env python
"""
Inspect Indicators Script
Print indicator values and top features for a symbol on a given date
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.research_library import ResearchLibrary
from core.features import FeatureEngineer
from core.storage import Storage


def format_value(val, decimals=2):
    """Format value for display"""
    if pd.isna(val):
        return "N/A"
    if isinstance(val, (int, np.integer)):
        return str(val)
    return f"{float(val):.{decimals}f}"


def print_indicator_table(symbol: str, date: str, indicators_df: pd.DataFrame):
    """Print formatted indicator table"""
    print("\n" + "=" * 80)
    print(f"INDICATORS FOR {symbol} ON {date}")
    print("=" * 80)
    
    # Filter to date
    if 'date' in indicators_df.columns:
        snapshot = indicators_df[indicators_df['date'] == date]
    else:
        snapshot = indicators_df.loc[indicators_df.index == pd.to_datetime(date)]
    
    if snapshot.empty:
        print(f"\n❌ No data found for {symbol} on {date}")
        return
    
    snapshot = snapshot.iloc[0]
    
    # Group indicators by family
    indicator_groups = {
        "Trend Following": {
            "EMAs": [f'ema_{p}' for p in [10, 20, 50, 100, 200] if f'ema_{p}' in snapshot.index],
            "MACD": ['macd', 'macd_signal', 'macd_histogram'],
            "ADX": ['adx', '+di', '-di']
        },
        "Mean Reversion": {
            "RSI": [c for c in snapshot.index if c.startswith('rsi')],
            "Stochastic": ['stoch_k', 'stoch_d'],
            "Bollinger Bands (20)": [c for c in snapshot.index if c.startswith('bb_20_')],
            "Bollinger Bands (50)": [c for c in snapshot.index if c.startswith('bb_50_')]
        },
        "Volatility": {
            "ATR": [c for c in snapshot.index if 'atr' in c],
            "Donchian": [c for c in snapshot.index if 'donchian' in c],
            "Keltner": [c for c in snapshot.index if 'keltner' in c]
        },
        "Volume": {
            "Volume Indicators": ['obv', 'mfi']
        }
    }
    
    for group_name, subgroups in indicator_groups.items():
        print(f"\n📊 {group_name}")
        print("-" * 80)
        
        for subgroup_name, cols in subgroups.items():
            available_cols = [c for c in cols if c in snapshot.index]
            if available_cols:
                print(f"\n  {subgroup_name}:")
                for col in available_cols:
                    val = snapshot[col]
                    if pd.notna(val):
                        # Format based on indicator type
                        if 'rsi' in col or 'stoch' in col or 'mfi' in col or 'di' in col or col == 'adx':
                            print(f"    {col:25s}: {format_value(val, 2)}")
                        elif 'percent' in col or 'width' in col or 'gap' in col or 'ratio' in col:
                            print(f"    {col:25s}: {format_value(val, 4)}")
                        else:
                            print(f"    {col:25s}: {format_value(val, 2)}")


def get_top_features(symbol: str, date: str, features_df: pd.DataFrame, 
                     feature_scores: dict = None, top_n: int = 3):
    """Get and display top features"""
    print("\n" + "=" * 80)
    print(f"TOP {top_n} FEATURES FOR {symbol} ON {date}")
    print("=" * 80)
    
    if feature_scores:
        # Sort by absolute importance
        sorted_features = sorted(feature_scores.items(), 
                               key=lambda x: abs(x[1]), reverse=True)
        top_features = sorted_features[:top_n]
        
        print("\nTop Features by Importance:")
        for i, (feature, score) in enumerate(top_features, 1):
            # Get value from features_df
            if 'date' in features_df.columns:
                snapshot = features_df[features_df['date'] == date]
            else:
                snapshot = features_df.loc[features_df.index == pd.to_datetime(date)]
            
            if not snapshot.empty and feature in snapshot.columns:
                val = snapshot[feature].iloc[0]
                print(f"  {i}. {feature:30s} | Value: {format_value(val, 4)} | Importance: {score:.4f}")
            else:
                print(f"  {i}. {feature:30s} | Importance: {score:.4f}")
    else:
        # Use default feature scores (equal weights)
        print("\n(No feature scores provided - showing sample features)")
        sample_features = ['momentum_score', 'volatility_ratio', 'volume_spike',
                          'rsi_14', 'macd', 'adx']
        available = [f for f in sample_features if f in features_df.columns]
        
        if 'date' in features_df.columns:
            snapshot = features_df[features_df['date'] == date]
        else:
            snapshot = features_df.loc[features_df.index == pd.to_datetime(date)]
        
        if not snapshot.empty:
            print("\nSample Feature Values:")
            for feature in available[:top_n]:
                val = snapshot[feature].iloc[0]
                print(f"  • {feature:30s}: {format_value(val, 4)}")


def main():
    parser = argparse.ArgumentParser(description='Inspect indicators for a symbol on a date')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol (e.g., AAPL)')
    parser.add_argument('--date', type=str, required=True, 
                       help='Date in YYYY-MM-DD format')
    parser.add_argument('--db-path', type=str, default='./db/axfi.duckdb',
                       help='Path to DuckDB database')
    parser.add_argument('--data-source', type=str, choices=['db', 'compute'], 
                       default='compute',
                       help='Get data from DB or compute fresh')
    
    args = parser.parse_args()
    
    # Validate date
    try:
        date_obj = datetime.strptime(args.date, '%Y-%m-%d')
        args.date = date_obj.strftime('%Y-%m-%d')
    except ValueError:
        print(f"❌ Invalid date format: {args.date}. Use YYYY-MM-DD")
        sys.exit(1)
    
    print(f"\n🔍 Inspecting indicators for {args.symbol} on {args.date}")
    print("=" * 80)
    
    # Load config
    import yaml
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    library = ResearchLibrary(config)
    engineer = FeatureEngineer(config)
    
    # Get data
    if args.data_source == 'db':
        storage = Storage(args.db_path)
        df = storage.read_raw_ohlcv(args.symbol)
        if df.empty:
            print(f"❌ No data found in database for {args.symbol}")
            sys.exit(1)
        
        if 'date' in df.columns:
            df = df.set_index('date')
        
        # Filter to date range (need history for indicators)
        df = df.sort_index()
        end_date = pd.to_datetime(args.date)
        start_date = end_date - pd.Timedelta(days=252)  # Need 1 year for indicators
        df = df[(df.index >= start_date) & (df.index <= end_date)]
    else:
        # Compute from scratch (load from data collector)
        from core.data_collector_v2 import DataCollectorV2
        collector = DataCollectorV2(config_path=str(config_path))
        
        try:
            df = collector.fetch_symbol(args.symbol, period='1y')
            if df.empty:
                print(f"❌ Could not fetch data for {args.symbol}")
                sys.exit(1)
            
            if 'date' in df.columns:
                df = df.set_index('date')
            
            df = df.sort_index()
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            sys.exit(1)
    
    if df.empty:
        print(f"❌ No data available for {args.symbol}")
        sys.exit(1)
    
    print(f"\n✓ Loaded {len(df)} rows of data")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
    
    # Calculate indicators
    print("\n📊 Calculating indicators...")
    indicators_df = library.calculate_all_indicators(df)
    
    # Calculate features
    print("🔧 Calculating features...")
    features_df = engineer.create_features(indicators_df.reset_index())
    
    # Print indicator table
    print_indicator_table(args.symbol, args.date, features_df)
    
    # Get top features (simulated scores if not available)
    feature_scores = {}
    for feature in engineer.get_feature_names():
        # Simple scoring: use absolute value of latest feature
        if feature in features_df.columns:
            latest_val = features_df[feature].iloc[-1]
            if pd.notna(latest_val):
                feature_scores[feature] = abs(float(latest_val))
    
    get_top_features(args.symbol, args.date, features_df, feature_scores, top_n=3)
    
    print("\n" + "=" * 80)
    print("✓ Inspection complete")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

