# AXFI - Agent X Financial Intelligence

An **enhanced, intelligent, local-first** Python application for financial data aggregation, quantitative backtesting, AI-powered signal scoring, and automated strategy discovery.

## Features

- **Three Core Functions**: Daily S&P 500 scan, single-symbol analysis, position analysis
- **Market Data Aggregation**: Collect real-time and historical market data from Yahoo Finance
- **Quantitative Backtesting**: Test momentum, mean reversion, and volatility breakout strategies
- **Strategy Scanner**: Automatically discover profitable strategies through parameter optimization (36+ candidates)
- **Research Library**: Comprehensive technical indicators (EMA, MACD, ADX, RSI, Stochastics, Bollinger Bands, ATR, Donchian, Keltner Channels, OBV, MFI)
- **Symbol Analysis Agent**: Multi-timeframe deep analysis (short/mid/long-term recommendations)
- **Scheduler Agent**: Automated pipeline orchestration (scan → test → rank → report)
- **Correlation & Sector Analysis**: Analyze relationships between symbols and sector rotation
- **AI Signal Scoring**: Rank trading signals using machine learning with explainable reasoning
- **Interactive Dashboard**: FastAPI + Jinja2 + Plotly visualization with equity curves
- **Automation Ready**: Daily/weekly scheduling for complete pipeline
- **Local-First**: SQLite for data storage, no cloud required

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Collect market data (first time only):
```bash
python core/data_collector.py
```

4. Run the dashboard:
```bash
python main.py
```

The dashboard will be available at: http://127.0.0.1:8081/

5. Test components:
```bash
# Test scaffold
python test_scaffold.py

# Test strategy scanning
python core\strategy_scanner.py

# Test symbol analysis
python core\symbol_analysis.py

# Test automated pipeline
python core\scheduler_agent.py
```

Expected outputs:
- Scaffold: "Enhanced AXFI Scaffold OK"
- Scanner: 10+ strategies discovered per symbol
- Symbol: Multi-timeframe recommendations
- Scheduler: 15 recommendations in ~10 seconds

## Project Structure

```
axfi/
 ├── main.py                 # Entry point + FastAPI dashboard
 ├── config.yaml             # Enhanced configuration
 ├── requirements.txt        # Dependencies
 ├── test_scaffold.py        # Scaffold verification
├── core/                    # Core modules & agents
│    ├── data_collector.py   # Market data collection
│    ├── backtester.py       # Strategy backtesting
│    ├── ai_engine.py        # AI signal ranking
│    ├── portfolio.py        # Position tracking
│    ├── reports.py          # Report generation
│    ├── strategy_scanner.py # Strategy discovery & optimization
│    ├── research_library.py # Technical indicators & analysis
│    ├── symbol_analysis.py  # Multi-timeframe symbol analysis ⭐
│    ├── scheduler_agent.py  # Pipeline automation ⭐
│    └── utils.py            # Utility functions
 ├── ui/                     # Web UI
 │    ├── templates/
 │    │   └── dashboard.html
 │    └── static/
 │        └── style.css
 ├── data/                   # Market data cache
 ├── db/                     # SQLite database files
 └── reports/                # Generated reports
```

## License

MIT

