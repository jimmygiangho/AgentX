# AXFI v2.0 - Agent X Financial Intelligence

A powerful, local-first, multi-agent trading research and advisory platform powered by AI and quantitative analysis.

## 🚀 Features

### Core Functions
1. **Daily S&P 500 Scanner** - Automated scanning of all S&P 500 stocks with AI-powered ranking
2. **Symbol Intelligence** - Deep analysis with short/mid/long-term forecasts
3. **Position Analyzer** - Real-time portfolio analysis with exit strategies

### Key Highlights
- ✅ **Multi-Agent Architecture** - Modular, intelligent agents for scanning, analysis, ranking
- ✅ **Explainable AI** - Every recommendation includes confidence scores and reasoning
- ✅ **Live Data Integration** - Support for Polygon.io, Alpaca, and yfinance fallback
- ✅ **Automated Scheduling** - Daily scans at 23:00 local time
- ✅ **Beautiful Dashboard** - Modern, responsive web UI with Plotly charts
- ✅ **Comprehensive Reports** - CSV and HTML export with timestamped artifacts

## 📋 Requirements

- Python 3.10+
- Virtual environment (recommended)

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd AgentX
```

### 2. Create Virtual Environment
```bash
cd axfi
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
# Copy example env file
copy env.example .env  # Windows
cp env.example .env    # Linux/Mac

# Edit .env and add your API keys
# POLYGON_API_KEY=your_key_here
```

### 5. Initialize Database
```bash
# Collect initial data for default symbols
python core/data_collector.py
```

## 🎯 Quick Start

### Run the Dashboard
```bash
cd axfi
python main.py
```

Then open: http://localhost:8081

### Run Daily Scan Manually
```bash
python -m pytest tests/test_live_data.py::test_daily_scan_manual -v
```

Or trigger via API:
```bash
curl http://localhost:8081/api/daily_scan
```

### Analyze a Symbol
```bash
curl http://localhost:8081/api/symbol_analysis?symbol=AAPL
```

## 📚 Documentation

- [QUICKSTART.md](axfi/QUICKSTART.md) - Complete getting started guide
- [README_USER_GUIDE.md](axfi/README_USER_GUIDE.md) - User guide
- [HOW_TO_USE.md](axfi/HOW_TO_USE.md) - Detailed usage instructions
- [LIVE_DATA_SETUP.md](axfi/LIVE_DATA_SETUP.md) - API setup guide

## 🏗️ Architecture

### Multi-Agent System
- **Scanner Agent** - S&P 500 market scanning
- **Strategy Agent** - Trading strategy evaluation
- **AI Ranking Agent** - Intelligent signal scoring
- **Symbol Agent** - Deep symbol analysis
- **Report Agent** - Report generation
- **Scheduler Agent** - Automated scheduling

### Core Components
- **Data Collector** - Multi-provider data aggregation
- **Research Library** - Technical indicators & studies
- **Backtester** - Strategy performance evaluation
- **AI Engine** - Explainable ranking & scoring
- **Storage Layer** - SQLite/DuckDB persistence

## 🔑 API Keys Setup

### Polygon.io (Recommended)
1. Sign up at https://polygon.io
2. Get your API key from dashboard
3. Add to `.env`: `POLYGON_API_KEY=your_key`

### Alpaca (Alternative)
1. Sign up at https://alpaca.markets
2. Get API key and secret
3. Add to `.env`:
   ```
   ALPACA_KEY=your_key
   ALPACA_SECRET=your_secret
   ```

**Note:** The system falls back to yfinance if API keys are not configured.

## 📊 Features

### Technical Indicators
- Trend Following: EMA, MACD, ADX
- Mean Reversion: RSI, Stochastics, Bollinger Bands
- Volatility: ATR, Donchian, Keltner Channels
- Volume: OBV, MFI

### Trading Strategies
- SMA Crossover
- Bollinger Band Mean Reversion
- ATR Volatility Breakout
- Volume Spike Filter

### Metrics & Analysis
- CAGR, Sharpe Ratio, Max Drawdown
- Win Rate, Total Returns
- Confidence Scores
- Explainable Reasoning

## 🤝 Contributing

This is a personal project, but suggestions and feedback are welcome!

## 📄 License

Private - All rights reserved

## 🙏 Acknowledgments

Built with:
- FastAPI - Modern web framework
- Plotly - Interactive visualizations
- yfinance - Market data
- Pandas/NumPy - Data processing
- APScheduler - Task scheduling

---

**Built by Jimmy Ho | Agent X Financial Intelligence v2.0**

