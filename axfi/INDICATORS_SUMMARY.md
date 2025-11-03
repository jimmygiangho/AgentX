# AXFI Technical Indicators & Studies Summary

## Total: **11 Primary Indicator Types** + **Multiple Derived Metrics**

---

## 📊 **1. Trend Following Indicators** (5 types)

### Exponential Moving Averages (EMA)
- **5 EMAs**: 10, 20, 50, 100, 200 periods
- Used for trend identification and support/resistance

### MACD (Moving Average Convergence Divergence)
- **3 components**:
  - MACD Line (12, 26)
  - Signal Line (9)
  - Histogram
- Measures momentum and trend changes

### ADX (Average Directional Index)
- **3 components**:
  - ADX (14 period)
  - +DI (Positive Directional Indicator)
  - -DI (Negative Directional Indicator)
- Measures trend strength

---

## 🔄 **2. Mean Reversion Indicators** (3 types)

### RSI (Relative Strength Index)
- **1 indicator**: 14-period RSI
- Identifies overbought/oversold conditions

### Stochastic Oscillator
- **2 components**:
  - %K (14 period)
  - %D (3 period smoothing)
- Momentum oscillator for entry/exit signals

### Bollinger Bands
- **5 components**:
  - Upper Band
  - Lower Band
  - Middle Band (SMA 20)
  - Band Width
  - %B (Percent B)
- Volatility-based mean reversion indicator

---

## 📈 **3. Volatility Indicators** (3 types)

### ATR (Average True Range)
- **1 indicator**: 14-period ATR
- Measures market volatility

### Donchian Channels
- **3 components**:
  - Upper Channel (20 period high)
  - Lower Channel (20 period low)
  - Middle Channel
- Breakout and volatility indicator

### Keltner Channels
- **3 components**:
  - Upper Channel (EMA + 2×ATR)
  - Lower Channel (EMA - 2×ATR)
  - Middle Channel (EMA 20)
- Volatility-based trend indicator

---

## 📊 **4. Volume Indicators** (2 types)

### OBV (On-Balance Volume)
- **1 indicator**: Cumulative volume
- Measures buying/selling pressure

### MFI (Money Flow Index)
- **1 indicator**: 14-period MFI
- Volume-weighted RSI

---

## 🎯 **Additional Features & Metrics**

### From Feature Engineering:
- **Rolling Returns**: Multiple timeframes (5, 10, 20, 50 days)
- **Volatility Measures**: Multiple rolling windows
- **Momentum Features**: 5-day, 10-day, 20-day momentum
- **Volume Ratios**: Volume vs. average volume
- **Price Position**: Within Bollinger/Keltner bands
- **Regime Detection**: Volatility and trend regimes

### Analysis Metrics:
- **Correlation Analysis**: Cross-asset correlation
- **Sector Rotation**: Sector strength analysis
- **Backtest Metrics**: Sharpe Ratio, CAGR, Max Drawdown, Win Rate

---

## 📋 **Summary Count**

| Category | Indicator Count |
|----------|----------------|
| **Primary Indicators** | **11 types** |
| **EMA Variations** | 5 periods |
| **Total Indicator Values** | **30+ individual metrics** |
| **Derived Features** | 20+ additional features |

---

## 🔍 **Usage in Analysis**

### During Stock Scanning:
- All 11 indicator types are calculated
- Features are engineered from indicators
- Strategies are tested using indicator combinations
- Stocks are ranked using composite scores

### During Individual Stock Analysis:
- All indicators are calculated for multi-timeframe analysis:
  - **Short-term** (3 days): Momentum, RSI, MACD signals
  - **Mid-term** (4 weeks): Trend following, ADX, Bollinger
  - **Long-term** (6 months): Structural trends, volume analysis
- Indicators feed into AI-powered recommendations

---

## 🎯 **Indicator Categories Breakdown**

1. **Trend Following**: EMA (5), MACD (3), ADX (3) = **11 metrics**
2. **Mean Reversion**: RSI (1), Stochastic (2), Bollinger (5) = **8 metrics**
3. **Volatility**: ATR (1), Donchian (3), Keltner (3) = **7 metrics**
4. **Volume**: OBV (1), MFI (1) = **2 metrics**

**Total: 28+ core indicator metrics** + **20+ derived features** = **48+ analysis points per stock**

