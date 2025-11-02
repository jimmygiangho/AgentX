# 🚀 AXFI - User Guide

## Quick Start

### 1️⃣ Start the Dashboard
```powershell
cd C:\Users\jimmy\AgentX\axfi
.\venv\Scripts\python.exe main.py
```

Wait for: `Starting AXFI Dashboard on http://127.0.0.1:8081`

### 2️⃣ Open Browser
```
http://127.0.0.1:8081
```

### 3️⃣ Use the Three Tabs

---

## 📊 Tab 1: Market Scanner

**What it does:** Shows you the best trade setups from today's market scan.

**How to use:**
1. Click "📊 Market Scanner" tab
2. View top 15 ranked recommendations
3. Scroll down to see equity curves
4. Click "🔄 Refresh Daily Scan" to re-run

**What you'll see:**
- **Rank** - How good this setup is (1 = best)
- **Symbol** - Stock ticker (e.g., GOOGL, AAPL, TSLA)
- **Strategy** - Trading approach (e.g., SMA Crossover, Bollinger Bands)
- **Score** - AI quality score (higher = better)
- **CAGR** - Annual return %
- **Sharpe** - Risk-adjusted return (higher = better)
- **Max DD** - Worst drawdown %
- **Win Rate** - % of winning trades
- **Explanation** - Why this is a good setup

**Example:**
```
Rank 1: GOOGL - SMA Crossover
Score: 1.0988 | CAGR: 55.80% | Sharpe: 2.58
Explanation: Strong momentum opportunity
```

---

## 🔍 Tab 2: Stock Analyzer

**What it does:** Deep AI analysis for any stock you want to research.

**How to use:**
1. Click "🔍 Stock Analyzer" tab
2. Type symbol (e.g., AAPL, TSLA, NVDA, MSFT)
3. Click "Analyze Symbol" or press Enter
4. Wait 2-3 seconds for analysis

**What you'll see:**

**Short-Term (3 days):**
- Buy/Hold/Sell recommendation
- Confidence %
- RSI (oversold/overbought)
- MACD signal
- Buy/Sell signals

**Mid-Term (4 weeks):**
- Trend direction (Uptrend/Downtrend)
- Trend strength (Strong/Weak)
- Stochastic signals
- Target price

**Long-Term (6 months):**
- Overall trend
- Total return %
- Volatility
- Distribution/Accumulation

**Overall Recommendation:**
- Final call (BUY/HOLD/MILD BUY/SELL)
- Overall confidence %
- Reasoning

**Risk Assessment:**
- Overall risk (Low/Medium/High)
- Volatility risk
- Drawdown risk

**Exit Strategies:**
- Take Profit levels (when to sell)
- Stop Loss levels (when to cut losses)
- Reasoning for each level

**Example:**
```
AAPL Analysis:
Short-term: BUY (70%) - Oversold RSI < 40
Mid-term: MILD BUY (65%) - Downtrend but oversold
Long-term: HOLD (60%) - Bearish structure
Overall: MILD BUY (70% confidence)

Risk: Low overall
Take Profit: $235.86 (Bollinger Band upper)
Stop Loss: $213.73 (2x ATR below current)
```

---

## 💼 Tab 3: Position Intelligence

**What it does:** Analyzes your current portfolio and gives you exit/hedge advice.

**How to use:**
1. Click "💼 Position Intelligence" tab
2. View portfolio summary at top
3. Scroll down to see position-by-position analysis
4. Click "🔄 Refresh Position Analysis" for latest

**What you'll see:**

**Portfolio Summary:**
- Total positions
- Total cost basis
- Market value
- Unrealized P&L
- Return %
- Available capital

**Position Table:**
- Symbol, Quantity, Entry Price
- Current Price, Cost Basis, Market Value
- P&L ($ and %)

**Position-by-Position Analysis:**
- AI recommendation for each position
- Confidence %
- Risk level (Low/Medium/High)
- Exit strategies (Take Profit, Stop Loss)

**Example:**
```
AAPL Position:
Recommendation: MILD BUY (70% confidence)
Risk: Low
Take Profit: $235.86
Stop Loss: $213.73

MSFT Position:
Recommendation: HOLD (60% confidence)
Risk: Low
Take Profit: $427.53
Stop Loss: $389.44
```

---

## 🎯 Daily Workflow Examples

### **Example 1: Morning Market Scan**
1. Open dashboard → Market Scanner tab
2. Review top 5 ranked setups
3. Click on symbols you like
4. Switch to Stock Analyzer tab
5. Analyze those symbols in depth
6. Make trading decisions

### **Example 2: Researching a Stock**
1. Open dashboard → Stock Analyzer tab
2. Enter symbol (e.g., NVDA)
3. Review multi-timeframe analysis
4. Check risk assessment
5. Note exit strategies
6. Decide if it's a good buy

### **Example 3: Managing Positions**
1. Open dashboard → Position Intelligence tab
2. Review portfolio summary
3. Check each position's recommendation
4. Review exit strategies
5. Adjust stops/targets as needed
6. Decide which positions to close

---

## 🔍 Understanding the Outputs

### **Recommendation Levels**
- **STRONG BUY** - Very high confidence, strong signals
- **BUY** - High confidence, good opportunity
- **MILD BUY** - Moderate confidence, cautiously bullish
- **HOLD** - Neutral, wait and see
- **MILD SELL** - Cautiously bearish
- **SELL** - High confidence to exit
- **STRONG SELL** - Very high confidence to exit

### **Risk Levels**
- **Low** - Safe, minimal downside risk
- **Medium** - Moderate risk, watch closely
- **High** - Risky, use strict stops

### **Confidence Scores**
- **90-100%** - Extremely confident
- **70-89%** - High confidence
- **50-69%** - Moderate confidence
- **< 50%** - Low confidence, be cautious

### **Metrics Explained**
- **CAGR** - Compound Annual Growth Rate (avg yearly return)
- **Sharpe Ratio** - Risk-adjusted return (2.0+ is excellent, 1.0+ is good)
- **Max Drawdown** - Worst peak-to-trough decline
- **Win Rate** - % of profitable trades

---

## ⚙️ Troubleshooting

### **Dashboard won't start**
- Check if port 8081 is in use
- Make sure virtualenv is activated
- Run: `python -c "import fastapi; print('OK')"`

### **No data in Market Scanner**
- Run: `python core/data_collector.py`
- Check `db/market_data.db` exists
- Verify config.yaml has symbols

### **Symbol Analyzer shows "No data"**
- That symbol may not be in database
- Run data collection for that symbol
- Check symbol ticker is correct (e.g., AAPL not apple)

### **Position Intelligence empty**
- Current positions are sample data
- Replace with your actual positions in `core/portfolio.py`
- Or configure portfolio in UI (future feature)

---

## 📞 Need Help?

- **API Docs:** http://127.0.0.1:8081/docs
- **Health Check:** http://127.0.0.1:8081/health
- **Documentation:** See `README.md`, `QUICKSTART.md`, `STATUS.md`
- **Logs:** Check terminal output for errors

---

## 🎓 Tips & Best Practices

1. **Run daily scans** - Update your watchlist every morning
2. **Use multiple timeframes** - Consider short/mid/long-term signals
3. **Check risk levels** - Never ignore stop losses
4. **Review exit strategies** - Plan your exits before entering
5. **Combine signals** - Use multiple confirmation signals
6. **Trust confidence scores** - Higher confidence = better setups
7. **Paper trade first** - Test recommendations before real money

---

**Happy Trading! 🚀**

*AXFI - AI-Powered Multi-Agent Trading Assistant*

