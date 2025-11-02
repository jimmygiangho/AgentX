# Live Data Setup Guide

## Current Status: ⚠️ NOT USING LIVE DATA

AXFI is currently using **yfinance fallback** which provides **delayed data** (not real-time).

**Evidence:**
- Data is **60+ hours old**
- No Polygon/Alpaca API keys configured
- System automatically falls back to yfinance

---

## How to Get LIVE Data

### Option 1: Polygon.io (Recommended - Free Tier Available)

1. **Get API Key:**
   - Visit: https://polygon.io/
   - Sign up for free account
   - Get your API key from dashboard

2. **Configure:**
   - Open `axfi/.env` file
   - Add your key:
     ```
     POLYGON_API_KEY=your_actual_api_key_here
     ```

3. **Test:**
   ```bash
   cd axfi
   python test_provider.py
   ```
   - Should show "Provider: polygon" and data age < 1 minute

---

### Option 2: Alpaca Markets (Free Tier Available)

1. **Get API Keys:**
   - Visit: https://alpaca.markets/
   - Sign up for free paper trading account
   - Get API Key ID and Secret from dashboard

2. **Configure:**
   - Open `axfi/.env` file
   - Add your keys:
     ```
     ALPACA_KEY=your_api_key_id
     ALPACA_SECRET=your_secret_key
     ```

3. **Update config.yaml:**
   ```yaml
   data:
     provider: "alpaca"  # Change from "polygon"
   ```

---

### Option 3: Keep Using yfinance (Delayed Data)

If you don't need real-time data:
- yfinance provides free data with 15-20 minute delay
- Good for backtesting and historical analysis
- NOT suitable for live trading signals

---

## Verification Steps

1. **Check current provider:**
   ```bash
   python test_provider.py
   ```

2. **Look for:**
   - ✅ "Provider: polygon" or "Provider: alpaca" = LIVE DATA
   - ❌ "Provider: yfinance_fallback" = DELAYED DATA
   - ❌ Data age > 15 minutes = NOT LIVE

3. **Test in dashboard:**
   - Open: http://127.0.0.1:8081
   - Click "Refresh Daily Scan"
   - Check timestamps on recommendations

---

## API Limits (Free Tiers)

### Polygon.io Free Tier:
- 5 API calls/minute
- Historical data available
- Suitable for S&P 500 daily scans

### Alpaca Free Tier:
- Unlimited paper trading data
- Real-time quotes
- Good for testing

---

## Troubleshooting

**Issue:** "Polygon API key not found"
- Make sure `.env` file exists in `axfi/` directory
- Check no spaces around `=` in `.env`
- Restart the server after adding key

**Issue:** "Still showing old data"
- Clear cache: `rm -rf db/market_data.db`
- Restart server
- Run test_provider.py to verify

**Issue:** "401 Unauthorized"
- API key is invalid or expired
- Get a new key from provider dashboard
- Make sure you're not using "your_api_key_here" placeholder

---

## Next Steps

Once you have live data configured:
1. Restart the dashboard: `python main.py`
2. Run full S&P 500 scan
3. Verify recommendations have current timestamps
4. Use Stock Analyzer with confidence

**Questions?** Check `axfi/env.example` for template.

