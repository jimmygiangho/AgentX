"""
AXFI Reports Generator Module
Generates CSV and HTML reports from backtest results, AI rankings, and portfolio data
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReportsGenerator:
    """
    Generates CSV and HTML reports from analysis results.
    
    Features:
    - Trade recommendations with AI explanations
    - Portfolio summary
    - Strategy backtest metrics
    - Timestamped reports
    """
    
    def __init__(self, reports_dir: str = "./reports"):
        """
        Initialize the reports generator.
        
        Args:
            reports_dir: Directory to save reports
        """
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized Reports Generator with output dir: {reports_dir}")
    
    def generate_timestamp_filename(self, base_name: str, extension: str) -> str:
        """
        Generate a timestamped filename.
        
        Args:
            base_name: Base name for the file
            extension: File extension (without dot)
            
        Returns:
            Timestamped filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_name}_{timestamp}.{extension}"
        return str(self.reports_dir / filename)
    
    def generate_trade_recommendations_csv(
        self,
        ranked_recommendations: List[Dict],
        filename: Optional[str] = None
    ) -> str:
        """
        Generate a CSV report of trade recommendations.
        
        Args:
            ranked_recommendations: List of ranked recommendations from AI engine
            filename: Optional custom filename
            
        Returns:
            Path to the generated CSV file
        """
        if not ranked_recommendations:
            logger.warning("No recommendations to export")
            return None
        
        # Prepare DataFrame
        df_data = []
        for rec in ranked_recommendations:
            metrics = rec.get('metrics', {})
            df_data.append({
                'rank': ranked_recommendations.index(rec) + 1,
                'symbol': rec.get('symbol', 'N/A'),
                'strategy': rec.get('strategy', 'N/A'),
                'score': rec.get('score', 0),
                'explanation': rec.get('explanation', 'N/A'),
                'cagr_pct': metrics.get('cagr', 0) * 100,
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown_pct': metrics.get('max_drawdown', 0) * 100,
                'win_rate_pct': metrics.get('win_rate', 0) * 100,
                'total_return_pct': metrics.get('total_return', 0) * 100,
                'num_trades': metrics.get('num_trades', 0),
                'final_equity': metrics.get('final_equity', 0)
            })
        
        df = pd.DataFrame(df_data)
        
        # Generate filename
        if filename is None:
            filename = self.generate_timestamp_filename("trade_recommendations", "csv")
        else:
            filename = str(self.reports_dir / filename)
        
        # Save CSV
        df.to_csv(filename, index=False)
        logger.info(f"Generated trade recommendations CSV: {filename}")
        
        return filename
    
    def generate_portfolio_summary_csv(
        self,
        portfolio_summary: pd.DataFrame,
        portfolio_metrics: Dict,
        filename: Optional[str] = None
    ) -> str:
        """
        Generate a CSV report of portfolio summary.
        
        Args:
            portfolio_summary: DataFrame with position details
            portfolio_metrics: Dictionary with portfolio-level metrics
            filename: Optional custom filename
            
        Returns:
            Path to the generated CSV file
        """
        # Generate filename
        if filename is None:
            filename = self.generate_timestamp_filename("portfolio_summary", "csv")
        else:
            filename = str(self.reports_dir / filename)
        
        # Save positions summary
        if not portfolio_summary.empty:
            portfolio_summary.to_csv(filename, index=False)
            logger.info(f"Generated portfolio summary CSV: {filename}")
        
        # Also save metrics to a separate file
        metrics_filename = filename.replace("portfolio_summary", "portfolio_metrics")
        metrics_df = pd.DataFrame([portfolio_metrics])
        metrics_df.to_csv(metrics_filename, index=False)
        logger.info(f"Generated portfolio metrics CSV: {metrics_filename}")
        
        return filename
    
    def generate_backtest_metrics_csv(
        self,
        backtest_results: Dict,
        filename: Optional[str] = None
    ) -> str:
        """
        Generate a CSV report of backtest metrics for all strategies.
        
        Args:
            backtest_results: Dictionary mapping strategy names to (result_df, metrics) tuples
            filename: Optional custom filename
            
        Returns:
            Path to the generated CSV file
        """
        if not backtest_results:
            logger.warning("No backtest results to export")
            return None
        
        # Prepare DataFrame
        df_data = []
        for strategy_name, (result_df, metrics) in backtest_results.items():
            df_data.append({
                'strategy': strategy_name,
                'cagr_pct': metrics.get('cagr', 0) * 100,
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown_pct': metrics.get('max_drawdown', 0) * 100,
                'win_rate_pct': metrics.get('win_rate', 0) * 100,
                'total_return_pct': metrics.get('total_return', 0) * 100,
                'volatility_pct': metrics.get('volatility', 0) * 100,
                'num_trades': metrics.get('num_trades', 0),
                'initial_capital': metrics.get('initial_capital', 0),
                'final_equity': metrics.get('final_equity', 0),
                'profit_loss': metrics.get('profit_loss', 0)
            })
        
        df = pd.DataFrame(df_data)
        
        # Generate filename
        if filename is None:
            filename = self.generate_timestamp_filename("backtest_metrics", "csv")
        else:
            filename = str(self.reports_dir / filename)
        
        # Save CSV
        df.to_csv(filename, index=False)
        logger.info(f"Generated backtest metrics CSV: {filename}")
        
        return filename
    
    def generate_combined_html_report(
        self,
        ranked_recommendations: Optional[List[Dict]] = None,
        portfolio_summary: Optional[pd.DataFrame] = None,
        portfolio_metrics: Optional[Dict] = None,
        backtest_results: Optional[Dict] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive HTML report with all sections.
        
        Args:
            ranked_recommendations: List of ranked recommendations
            portfolio_summary: DataFrame with position details
            portfolio_metrics: Dictionary with portfolio-level metrics
            backtest_results: Dictionary of backtest results
            filename: Optional custom filename
            
        Returns:
            Path to the generated HTML file
        """
        # Generate filename
        if filename is None:
            filename = self.generate_timestamp_filename("axfi_report", "html")
        else:
            filename = str(self.reports_dir / filename)
        
        # Build HTML content
        html_parts = []
        
        # HTML Header
        html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AXFI Financial Intelligence Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        tr:hover {
            background-color: #e8f4f8;
        }
        .positive { color: #27ae60; font-weight: bold; }
        .negative { color: #e74c3c; font-weight: bold; }
        .metric-box {
            display: inline-block;
            background-color: #ecf0f1;
            padding: 15px;
            margin: 10px;
            border-radius: 5px;
            min-width: 150px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-label {
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
        }
        .timestamp {
            color: #95a5a6;
            font-size: 14px;
            text-align: right;
        }
        .recommendation {
            padding: 10px;
            margin: 5px 0;
            border-left: 4px solid #3498db;
            background-color: #ebf5fb;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>AXFI Financial Intelligence Report</h1>
        <div class="timestamp">Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</div>
""")
        
        # Trade Recommendations Section
        if ranked_recommendations:
            html_parts.append("""
        <h2>Top Trade Recommendations</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Symbol</th>
                    <th>Strategy</th>
                    <th>Score</th>
                    <th>CAGR (%)</th>
                    <th>Sharpe</th>
                    <th>Max DD (%)</th>
                    <th>Explanation</th>
                </tr>
            </thead>
            <tbody>
""")
            
            for i, rec in enumerate(ranked_recommendations[:10], 1):  # Top 10
                metrics = rec.get('metrics', {})
                cagr_pct = metrics.get('cagr', 0) * 100
                sharpe = metrics.get('sharpe_ratio', 0)
                max_dd_pct = metrics.get('max_drawdown', 0) * 100
                
                html_parts.append(f"""
                <tr>
                    <td>{i}</td>
                    <td><strong>{rec.get('symbol', 'N/A')}</strong></td>
                    <td>{rec.get('strategy', 'N/A')}</td>
                    <td>{rec.get('score', 0):.4f}</td>
                    <td class="{'positive' if cagr_pct > 0 else 'negative'}">{cagr_pct:.2f}</td>
                    <td class="{'positive' if sharpe > 0 else 'negative'}">{sharpe:.3f}</td>
                    <td class="negative">{max_dd_pct:.2f}</td>
                    <td>{rec.get('explanation', 'N/A')}</td>
                </tr>
""")
            
            html_parts.append("""
            </tbody>
        </table>
""")
        
        # Portfolio Summary Section
        if portfolio_summary is not None and not portfolio_summary.empty:
            html_parts.append("""
        <h2>Portfolio Summary</h2>
""")
            
            # Portfolio Metrics
            if portfolio_metrics:
                html_parts.append("""
        <div>
""")
                for metric_name, metric_value in portfolio_metrics.items():
                    if metric_name not in ['positions_with_alerts', 'timestamp']:
                        if isinstance(metric_value, (int, float)):
                            if 'pct' in metric_name.lower() or 'return' in metric_name.lower():
                                display_value = f"{metric_value*100:.2f}%"
                                css_class = "positive" if metric_value > 0 else "negative"
                            elif '$' in str(metric_value) or metric_name in ['total_market_value', 'total_cost_basis', 'total_unrealized_pnl']:
                                display_value = f"${metric_value:,.2f}"
                                css_class = "positive" if metric_value > 0 else "negative"
                            else:
                                display_value = f"{metric_value:,.2f}"
                                css_class = ""
                            metric_label = metric_name.replace('_', ' ').title()
                            html_parts.append(f"""
            <div class="metric-box">
                <div class="metric-label">{metric_label}</div>
                <div class="metric-value {' '.join([css_class]) if css_class else ''}">{display_value}</div>
            </div>
""")
                html_parts.append("""
        </div>
""")
            
            # Positions Table
            html_parts.append("""
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Quantity</th>
                    <th>Entry Price</th>
                    <th>Current Price</th>
                    <th>Cost Basis</th>
                    <th>Market Value</th>
                    <th>Unrealized P&L</th>
                    <th>Unrealized P&L %</th>
                </tr>
            </thead>
            <tbody>
""")
            
            for _, row in portfolio_summary.iterrows():
                pnl = row.get('unrealized_pnl', 0)
                pnl_pct = row.get('unrealized_pnl_pct', 0)
                pnl_class = "positive" if pnl > 0 else "negative"
                
                html_parts.append(f"""
                <tr>
                    <td><strong>{row.get('symbol', 'N/A')}</strong></td>
                    <td>{row.get('quantity', 0):.2f}</td>
                    <td>${row.get('entry_price', 0):.2f}</td>
                    <td>${row.get('current_price', 0):.2f}</td>
                    <td>${row.get('cost_basis', 0):.2f}</td>
                    <td>${row.get('market_value', 0):.2f}</td>
                    <td class="{pnl_class}">${pnl:.2f}</td>
                    <td class="{pnl_class}">{pnl_pct:.2f}%</td>
                </tr>
""")
            
            html_parts.append("""
            </tbody>
        </table>
""")
        
        # Backtest Metrics Section
        if backtest_results:
            html_parts.append("""
        <h2>Strategy Backtest Metrics</h2>
        <table>
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>CAGR (%)</th>
                    <th>Sharpe Ratio</th>
                    <th>Max DD (%)</th>
                    <th>Win Rate (%)</th>
                    <th>Total Return (%)</th>
                    <th># Trades</th>
                    <th>Final Equity</th>
                </tr>
            </thead>
            <tbody>
""")
            
            for strategy_name, (result_df, metrics) in backtest_results.items():
                cagr_pct = metrics.get('cagr', 0) * 100
                sharpe = metrics.get('sharpe_ratio', 0)
                max_dd_pct = metrics.get('max_drawdown', 0) * 100
                win_rate_pct = metrics.get('win_rate', 0) * 100
                total_return_pct = metrics.get('total_return', 0) * 100
                
                html_parts.append(f"""
                <tr>
                    <td><strong>{strategy_name}</strong></td>
                    <td class="{'positive' if cagr_pct > 0 else 'negative'}">{cagr_pct:.2f}</td>
                    <td class="{'positive' if sharpe > 0 else 'negative'}">{sharpe:.3f}</td>
                    <td class="negative">{max_dd_pct:.2f}</td>
                    <td class="{'positive' if win_rate_pct > 50 else 'negative'}">{win_rate_pct:.2f}</td>
                    <td class="{'positive' if total_return_pct > 0 else 'negative'}">{total_return_pct:.2f}</td>
                    <td>{metrics.get('num_trades', 0)}</td>
                    <td>${metrics.get('final_equity', 0):,.2f}</td>
                </tr>
""")
            
            html_parts.append("""
            </tbody>
        </table>
""")
        
        # HTML Footer
        html_parts.append("""
        <div class="timestamp">
            <p><em>Generated by AXFI - Agent X Financial Intelligence</em></p>
        </div>
    </div>
</body>
</html>
""")
        
        # Write HTML file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(''.join(html_parts))
        
        logger.info(f"Generated combined HTML report: {filename}")
        return filename
    
    def generate_all_reports(
        self,
        ranked_recommendations: Optional[List[Dict]] = None,
        portfolio_summary: Optional[pd.DataFrame] = None,
        portfolio_metrics: Optional[Dict] = None,
        backtest_results: Optional[Dict] = None,
        generate_csv: bool = True,
        generate_html: bool = True
    ) -> Dict[str, str]:
        """
        Generate all reports (CSV and HTML).
        
        Args:
            ranked_recommendations: List of ranked recommendations
            portfolio_summary: DataFrame with position details
            portfolio_metrics: Dictionary with portfolio-level metrics
            backtest_results: Dictionary of backtest results
            generate_csv: Whether to generate CSV reports
            generate_html: Whether to generate HTML report
            
        Returns:
            Dictionary mapping report type to file path
        """
        generated_files = {}
        
        if generate_csv:
            # Generate individual CSV reports
            if ranked_recommendations:
                csv_file = self.generate_trade_recommendations_csv(ranked_recommendations)
                generated_files['trade_recommendations_csv'] = csv_file
            
            if backtest_results:
                csv_file = self.generate_backtest_metrics_csv(backtest_results)
                generated_files['backtest_metrics_csv'] = csv_file
            
            if portfolio_summary is not None and not portfolio_summary.empty:
                csv_file = self.generate_portfolio_summary_csv(portfolio_summary, portfolio_metrics or {})
                generated_files['portfolio_summary_csv'] = csv_file
        
        if generate_html:
            # Generate combined HTML report
            html_file = self.generate_combined_html_report(
                ranked_recommendations=ranked_recommendations,
                portfolio_summary=portfolio_summary,
                portfolio_metrics=portfolio_metrics,
                backtest_results=backtest_results
            )
            generated_files['combined_html'] = html_file
        
        logger.info(f"Generated {len(generated_files)} report files")
        return generated_files


def main():
    """
    Standalone execution for testing the reports generator.
    """
    print("=" * 80)
    print("AXFI Reports Generator - Standalone Test")
    print("=" * 80)
    
    # Import required modules
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.data_collector import DataCollector
    from core.backtester import Backtester
    from core.ai_engine import AIEngine
    
    # Initialize components
    data_collector = DataCollector()
    backtester = Backtester(initial_capital=100000, commission=0.001)
    ai_engine = AIEngine()
    reports_gen = ReportsGenerator()
    
    # Get data for AAPL
    symbol = 'AAPL'
    print(f"\nProcessing {symbol}...")
    
    df = data_collector.read_from_database(symbol=symbol)
    
    if df.empty:
        print(f"No data for {symbol}")
        return
    
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    
    # Run backtesting
    print("Running backtests...")
    backtest_results = backtester.run_all_strategies(df)
    
    # Get AI rankings
    print("Generating AI rankings...")
    ranked = ai_engine.rank_strategies(backtest_results, symbol)
    
    # Simulate portfolio data
    from core.portfolio import Portfolio
    portfolio = Portfolio(initial_capital=100000)
    portfolio.add_position('AAPL', 100, df['close'].iloc[-1], sector='Technology')
    
    current_prices = {symbol: df['close'].iloc[-1]}
    portfolio_summary = portfolio.get_positions_summary(current_prices)
    portfolio_metrics = portfolio.calculate_portfolio_metrics(current_prices)
    
    # Generate all reports
    print("\nGenerating reports...")
    generated_files = reports_gen.generate_all_reports(
        ranked_recommendations=ranked,
        portfolio_summary=portfolio_summary,
        portfolio_metrics=portfolio_metrics,
        backtest_results=backtest_results,
        generate_csv=True,
        generate_html=True
    )
    
    # Display results
    print("\n" + "=" * 80)
    print("GENERATED REPORTS")
    print("=" * 80)
    
    for report_type, filepath in generated_files.items():
        print(f"{report_type}: {filepath}")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

