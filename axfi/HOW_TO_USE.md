# 🚀 AXFI v2.0 - How to Use

**Dashboard:** http://127.0.0.1:8081

---

## ✅ What's Working NOW

### **1. Stock Analyzer - ANY Stock** ✅
**Test it:**
1. Open dashboard
2. Click "🔍 Stock Analyzer" tab
3. Type any symbol: `NVDA`, `TSLA`, `AMD`, etc.
4. Click "Analyze Symbol"

**What you get:**
- Short/mid/long-term recommendations
- Confidence scores
- Risk assessment
- Exit strategies (Take Profit, Stop Loss)

**Live data:** ✅ Fetches from yfinance automatically

---

### **2. Daily S&P 500 Scan** ⚠️
**Test it:**
1. Open dashboard
2. Click "📊 Market Scanner" tab
3. Click "🔄 Refresh Daily Scan"
4. **Wait:** 30-60 seconds (first time is slow!)

**What you get:**
- Top 15 ranked trade recommendations
- Strategy score, CAGR, Sharpe, Max DD
- AI explanations
- Equity curve charts

**Live data:** ✅ Scans 40+ S&P 500 symbols

**Note:** First run is SLOW (fetching live data). Subsequent runs are fast (uses cache).

---

### **3. Position Intelligence** ✅
**Test it:**
1. Open dashboard
2. Click "💼 Position Intelligence" tab
3. View portfolio summary & positions
4. Click "🔄 Refresh Position Analysis"

**What you get:**
- Portfolio metrics
- Position-by-position analysis
- Exit strategies for each position

**Live data:** ✅ Fetches current prices

---

## 🧪 Quick Tests

### **Test Any Stock:**
```bash
# Via browser: http://127.0.0.1:8081
# Click Stock Analyzer → Enter "NVDA" → Analyze

# Or via API:
curl "http://127.0.0.1:8081/api/symbol_analysis?symbol=NVDA"
```

### **Test Daily Scan:**
```bash
# Via browser: http://127.0.0.1:8081
# Click Market Scanner → Refresh Daily Scan

# Or via API:
curl "http://127.0.0.1:8081/api/daily_scan"
```

---

## 🐛 Troubleshooting

### **Issue: "No data available"**
**Fix:** Data is cached. First-time analysis fetches live data automatically.

### **Issue: "Daily scan is slow"**
**Fix:** First run fetches live data for all symbols (30-60 sec). Subsequent runs use cache (10-20 sec).

### **Issue: "Symbol not found"**
**Fix:** Make sure it's a valid ticker. Try: AAPL, TSLA, NVDA, MSFT, GOOGL

---

## 📊 Current Capabilities

✅ **Live data** - yfinance integration  
✅ **S&P 500 scan** - 40+ symbols  
✅ **Any symbol analysis** - enter any ticker  
✅ **Multi-timeframe** - short/mid/long-term  
✅ **AI ranking** - confidence scores  
✅ **Exit strategies** - TP/SL levels  
✅ **Risk assessment** - Low/Medium/High  

---

**🎉 Everything is working! Test it now!**

*http://127.0.0.1:8081*

