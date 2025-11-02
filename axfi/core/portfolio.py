"""
AXFI Portfolio Tracker Module
Tracks positions, calculates portfolio-level metrics, and manages trades
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Portfolio:
    """
    Tracks portfolio positions and calculates portfolio-level metrics.
    
    Features:
    - Position management (add, update, remove)
    - Real-time P&L calculations
    - Portfolio exposure tracking
    - Equity curve generation
    - Risk management (stop loss, take profit)
    """
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize the portfolio tracker.
        
        Args:
            initial_capital: Starting capital for the portfolio
        """
        self.initial_capital = initial_capital
        self.positions = {}  # Dictionary to store positions: {symbol: Position dict}
        self.trade_history = []  # List of all trades
        logger.info(f"Initialized Portfolio with starting capital ${initial_capital:,.2f}")
    
    def add_position(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        entry_date: Optional[str] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        sector: Optional[str] = None
    ) -> bool:
        """
        Add a new position to the portfolio.
        
        Args:
            symbol: Stock ticker symbol
            quantity: Number of shares/units
            entry_price: Entry price per share
            entry_date: Entry date (YYYY-MM-DD) or None for today
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            sector: Optional sector classification
            
        Returns:
            True if position added successfully, False otherwise
        """
        if symbol in self.positions:
            logger.warning(f"Position for {symbol} already exists. Use update_position() to modify.")
            return False
        
        if quantity <= 0:
            logger.error(f"Invalid quantity: {quantity}. Must be positive.")
            return False
        
        if entry_price <= 0:
            logger.error(f"Invalid entry price: {entry_price}. Must be positive.")
            return False
        
        # Validate stop loss and take profit
        if stop_loss is not None and stop_loss <= 0:
            logger.error(f"Invalid stop loss: {stop_loss}. Must be positive.")
            return False
        
        if take_profit is not None and take_profit <= 0:
            logger.error(f"Invalid take profit: {take_profit}. Must be positive.")
            return False
        
        # Set default entry date
        if entry_date is None:
            entry_date = datetime.now().strftime('%Y-%m-%d')
        
        # Create position record
        position = {
            'symbol': symbol,
            'quantity': quantity,
            'entry_price': entry_price,
            'entry_date': entry_date,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'sector': sector,
            'cost_basis': quantity * entry_price
        }
        
        self.positions[symbol] = position
        self.trade_history.append({
            **position,
            'action': 'BUY',
            'timestamp': datetime.now()
        })
        
        logger.info(f"Added position: {symbol} {quantity} shares @ ${entry_price:.2f}")
        return True
    
    def update_position(
        self,
        symbol: str,
        quantity: Optional[float] = None,
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        sector: Optional[str] = None
    ) -> bool:
        """
        Update an existing position.
        
        Args:
            symbol: Stock ticker symbol
            quantity: New quantity (if provided)
            entry_price: New entry price (if provided)
            stop_loss: New stop loss (if provided, None to clear)
            take_profit: New take profit (if provided, None to clear)
            sector: New sector (if provided)
            
        Returns:
            True if position updated successfully, False otherwise
        """
        if symbol not in self.positions:
            logger.warning(f"Position for {symbol} does not exist. Use add_position() to create.")
            return False
        
        position = self.positions[symbol]
        
        # Update fields if provided
        if quantity is not None and quantity > 0:
            position['quantity'] = quantity
            position['cost_basis'] = position['entry_price'] * quantity
            logger.info(f"Updated {symbol} quantity to {quantity}")
        
        if entry_price is not None and entry_price > 0:
            old_entry = position['entry_price']
            position['entry_price'] = entry_price
            position['cost_basis'] = position['quantity'] * entry_price
            logger.info(f"Updated {symbol} entry price from ${old_entry:.2f} to ${entry_price:.2f}")
        
        if stop_loss is not None:
            position['stop_loss'] = stop_loss if stop_loss > 0 else None
            logger.info(f"Updated {symbol} stop loss to ${stop_loss:.2f}" if stop_loss else f"Cleared {symbol} stop loss")
        
        if take_profit is not None:
            position['take_profit'] = take_profit if take_profit > 0 else None
            logger.info(f"Updated {symbol} take profit to ${take_profit:.2f}" if take_profit else f"Cleared {symbol} take profit")
        
        if sector is not None:
            position['sector'] = sector
            logger.info(f"Updated {symbol} sector to {sector}")
        
        return True
    
    def remove_position(self, symbol: str, exit_price: float) -> bool:
        """
        Close a position.
        
        Args:
            symbol: Stock ticker symbol
            exit_price: Exit price per share
            
        Returns:
            True if position removed successfully, False otherwise
        """
        if symbol not in self.positions:
            logger.warning(f"Position for {symbol} does not exist.")
            return False
        
        position = self.positions[symbol]
        
        # Calculate trade result
        quantity = position['quantity']
        entry_price = position['entry_price']
        pnl = (exit_price - entry_price) * quantity
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        
        # Record trade
        self.trade_history.append({
            **position,
            'action': 'SELL',
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'timestamp': datetime.now()
        })
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"Closed position: {symbol} {quantity} shares @ ${exit_price:.2f}, P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
        return True
    
    def get_position_pnl(
        self,
        symbol: str,
        current_price: float
    ) -> Dict:
        """
        Calculate P&L for a single position.
        
        Args:
            symbol: Stock ticker symbol
            current_price: Current market price
            
        Returns:
            Dictionary with P&L metrics
        """
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return {}
        
        position = self.positions[symbol]
        entry_price = position['entry_price']
        quantity = position['quantity']
        
        # Calculate P&L
        pnl = (current_price - entry_price) * quantity
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        market_value = quantity * current_price
        
        # Check stop loss and take profit
        hit_stop_loss = False
        hit_take_profit = False
        
        if position.get('stop_loss') and current_price <= position['stop_loss']:
            hit_stop_loss = True
        
        if position.get('take_profit') and current_price >= position['take_profit']:
            hit_take_profit = True
        
        return {
            'symbol': symbol,
            'quantity': quantity,
            'entry_price': entry_price,
            'current_price': current_price,
            'cost_basis': position['cost_basis'],
            'market_value': market_value,
            'unrealized_pnl': pnl,
            'unrealized_pnl_pct': pnl_pct,
            'hit_stop_loss': hit_stop_loss,
            'hit_take_profit': hit_take_profit
        }
    
    def calculate_portfolio_metrics(
        self,
        current_prices: Dict[str, float]
    ) -> Dict:
        """
        Calculate portfolio-level metrics.
        
        Args:
            current_prices: Dictionary mapping symbols to current prices
            
        Returns:
            Dictionary with portfolio metrics
        """
        if not self.positions:
            return {
                'total_positions': 0,
                'total_cost_basis': 0,
                'total_market_value': 0,
                'total_unrealized_pnl': 0,
                'total_unrealized_pnl_pct': 0,
                'positions_with_alerts': []
            }
        
        total_cost_basis = 0
        total_market_value = 0
        positions_with_alerts = []
        
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position['entry_price'])
            pnl_metrics = self.get_position_pnl(symbol, current_price)
            
            total_cost_basis += pnl_metrics['cost_basis']
            total_market_value += pnl_metrics['market_value']
            
            # Track alert conditions
            if pnl_metrics['hit_stop_loss']:
                positions_with_alerts.append(f"{symbol}: STOP LOSS triggered")
            elif pnl_metrics['hit_take_profit']:
                positions_with_alerts.append(f"{symbol}: TAKE PROFIT triggered")
        
        total_unrealized_pnl = total_market_value - total_cost_basis
        total_unrealized_pnl_pct = (total_unrealized_pnl / total_cost_basis * 100) if total_cost_basis > 0 else 0
        
        metrics = {
            'total_positions': len(self.positions),
            'total_cost_basis': total_cost_basis,
            'total_market_value': total_market_value,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_unrealized_pnl_pct': total_unrealized_pnl_pct,
            'positions_with_alerts': positions_with_alerts,
            'available_capital': self.initial_capital - total_cost_basis,
            'total_capital_employed': total_cost_basis,
            'portfolio_return': total_unrealized_pnl_pct / 100
        }
        
        return metrics
    
    def get_positions_summary(
        self,
        current_prices: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Get a summary DataFrame of all positions with current P&L.
        
        Args:
            current_prices: Dictionary mapping symbols to current prices
            
        Returns:
            DataFrame with position summary
        """
        if not self.positions:
            return pd.DataFrame()
        
        summary_data = []
        
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position['entry_price'])
            pnl_metrics = self.get_position_pnl(symbol, current_price)
            
            summary_data.append({
                'symbol': symbol,
                'quantity': position['quantity'],
                'entry_price': position['entry_price'],
                'current_price': current_price,
                'cost_basis': position['cost_basis'],
                'market_value': pnl_metrics['market_value'],
                'unrealized_pnl': pnl_metrics['unrealized_pnl'],
                'unrealized_pnl_pct': pnl_metrics['unrealized_pnl_pct'],
                'entry_date': position.get('entry_date', 'N/A'),
                'stop_loss': position.get('stop_loss', 'N/A'),
                'take_profit': position.get('take_profit', 'N/A'),
                'sector': position.get('sector', 'N/A')
            })
        
        df = pd.DataFrame(summary_data)
        
        # Sort by unrealized P&L (descending)
        df = df.sort_values('unrealized_pnl', ascending=False)
        df = df.reset_index(drop=True)
        
        return df
    
    def get_sector_exposure(self) -> Dict:
        """
        Calculate portfolio exposure by sector.
        
        Returns:
            Dictionary with sector exposure percentages
        """
        if not self.positions:
            return {}
        
        sector_values = {}
        total_value = 0
        
        for position in self.positions.values():
            sector = position.get('sector', 'Unknown')
            value = position['cost_basis']
            
            sector_values[sector] = sector_values.get(sector, 0) + value
            total_value += value
        
        # Convert to percentages
        sector_exposure = {
            sector: (value / total_value * 100) if total_value > 0 else 0
            for sector, value in sector_values.items()
        }
        
        return sector_exposure
    
    def get_trade_history(self) -> pd.DataFrame:
        """
        Get trade history as a DataFrame.
        
        Returns:
            DataFrame with all trade history
        """
        if not self.trade_history:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.trade_history)
        
        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp', ascending=False)
        
        return df
    
    def calculate_equity_curve(
        self,
        historical_prices: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Calculate portfolio equity curve over time.
        
        Args:
            historical_prices: Dictionary mapping symbols to DataFrames with 'date' and 'close' columns
            
        Returns:
            DataFrame with equity curve
        """
        # Collect all unique dates
        all_dates = set()
        for df in historical_prices.values():
            if 'date' in df.columns:
                all_dates.update(df['date'].unique())
        
        if not all_dates:
            logger.warning("No historical data provided")
            return pd.DataFrame()
        
        all_dates = sorted(list(all_dates))
        
        # Initialize equity curve
        equity_curve = {
            'date': all_dates,
            'cost_basis': self.initial_capital,
            'market_value': self.initial_capital,
            'unrealized_pnl': 0,
            'cumulative_return': 0
        }
        
        # Calculate market value for each date
        for i, date in enumerate(all_dates):
            total_cost_basis = self.initial_capital
            total_market_value = self.initial_capital
            
            # Calculate based on positions that existed at that date
            for symbol, position in self.positions.items():
                entry_date = position.get('entry_date')
                
                # Check if position existed at this date
                if entry_date and entry_date <= date:
                    # Get price for this date
                    if symbol in historical_prices:
                        df = historical_prices[symbol]
                        df_date = df[df['date'] == date]
                        
                        if not df_date.empty:
                            price = df_date.iloc[0]['close']
                            quantity = position['quantity']
                            
                            total_cost_basis -= quantity * position['entry_price']
                            total_market_value -= quantity * position['entry_price']
                            total_market_value += quantity * price
            
            equity_curve['market_value'] = total_market_value
            equity_curve['unrealized_pnl'] = total_market_value - total_cost_basis
            equity_curve['cumulative_return'] = (total_market_value - self.initial_capital) / self.initial_capital
        
        df = pd.DataFrame(equity_curve)
        return df


def main():
    """
    Standalone execution for testing the portfolio tracker.
    """
    print("=" * 80)
    print("AXFI Portfolio Tracker - Standalone Test")
    print("=" * 80)
    
    # Create portfolio
    portfolio = Portfolio(initial_capital=100000)
    
    # Add some positions
    print("\nAdding positions...")
    portfolio.add_position('AAPL', quantity=100, entry_price=175.50, entry_date='2024-10-01', sector='Technology')
    portfolio.add_position('MSFT', quantity=50, entry_price=380.00, entry_date='2024-10-05', sector='Technology')
    portfolio.add_position('GOOGL', quantity=75, entry_price=140.00, entry_date='2024-10-10', sector='Technology')
    
    # Add stop loss and take profit to one position
    portfolio.update_position('AAPL', stop_loss=165.00, take_profit=190.00)
    
    # Display initial positions
    print("\n" + "=" * 80)
    print("INITIAL POSITIONS")
    print("=" * 80)
    print(f"Total positions: {len(portfolio.positions)}")
    
    # Simulate current prices
    current_prices = {
        'AAPL': 185.00,  # In profit
        'MSFT': 375.00,  # In loss
        'GOOGL': 145.00  # In profit
    }
    
    # Calculate portfolio metrics
    print("\n" + "=" * 80)
    print("PORTFOLIO METRICS")
    print("=" * 80)
    metrics = portfolio.calculate_portfolio_metrics(current_prices)
    
    print(f"Total Positions:        {metrics['total_positions']}")
    print(f"Total Cost Basis:       ${metrics['total_cost_basis']:,.2f}")
    print(f"Total Market Value:     ${metrics['total_market_value']:,.2f}")
    print(f"Total Unrealized P&L:   ${metrics['total_unrealized_pnl']:,.2f} ({metrics['total_unrealized_pnl_pct']:.2f}%)")
    print(f"Available Capital:      ${metrics['available_capital']:,.2f}")
    print(f"Capital Employed:       ${metrics['total_capital_employed']:,.2f}")
    print(f"Portfolio Return:       {metrics['portfolio_return']*100:.2f}%")
    
    if metrics['positions_with_alerts']:
        print(f"\nAlerts:")
        for alert in metrics['positions_with_alerts']:
            print(f"  - {alert}")
    
    # Display positions summary
    print("\n" + "=" * 80)
    print("POSITIONS SUMMARY")
    print("=" * 80)
    summary_df = portfolio.get_positions_summary(current_prices)
    
    # Format for display
    display_cols = ['symbol', 'quantity', 'entry_price', 'current_price', 
                    'unrealized_pnl', 'unrealized_pnl_pct', 'entry_date']
    print(summary_df[display_cols].to_string(index=False))
    
    # Show sector exposure
    print("\n" + "=" * 80)
    print("SECTOR EXPOSURE")
    print("=" * 80)
    sector_exposure = portfolio.get_sector_exposure()
    for sector, exposure_pct in sector_exposure.items():
        print(f"{sector}: {exposure_pct:.2f}%")
    
    # Show trade history
    print("\n" + "=" * 80)
    print("TRADE HISTORY")
    print("=" * 80)
    history_df = portfolio.get_trade_history()
    if not history_df.empty:
        history_cols = ['symbol', 'action', 'quantity', 'entry_price', 'timestamp']
        available_cols = [col for col in history_cols if col in history_df.columns]
        print(history_df[available_cols].to_string(index=False))
    
    # Close a position
    print("\n" + "=" * 80)
    print("CLOSING POSITION")
    print("=" * 80)
    portfolio.remove_position('MSFT', exit_price=370.00)
    
    # Recalculate metrics
    metrics_after = portfolio.calculate_portfolio_metrics(current_prices)
    print(f"\nAfter closing MSFT:")
    print(f"Total Positions:      {metrics_after['total_positions']}")
    print(f"Available Capital:    ${metrics_after['available_capital']:,.2f}")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

