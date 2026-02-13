# VIX Trading Strategy

---

## Project Overview

This project demonstrates the **critical importance of market microstructure** in quantitative finance through an iterative backtesting process. What started as a seemingly profitable VIX trading strategy evolved through three versions, with returns dropping from **61.5%** to **2.7%** annualized - a **98.5% reduction** - as I incorporated increasingly realistic modeling assumptions.

**Key Takeaway:** *Small modeling errors can lead to massively inflated backtest results. Understanding derivatives market structure is just as important as the trading signal itself.*

---

## The Versions and Lessons

### Version 1: The Naive Implementation
**File:** `backtesting.py`  
**Result:** 61.5% annualized return in the regime-based strategy 
**The Mistake:** Used raw VIX index returns as a trading proxy

```python
# V1 Code
df['Strategy_Return'] = df['Position'].shift(1) * df['VIX_Return']
```

**The Problem:**  
VIX is an **index**, not a tradeable security. You can't actually buy or sell VIX directly. In reality, you must trade VIX futures or VIX ETFs.

**What I Learned:**  
Always verify that what you're backtesting is actually tradeable in the real world.

---

### Version 2: Adding VIX ETF Decay
**File:** `backtesting_v2.py`  
**Result:** 49.9% annualized return in regime-based strategy
**The Fix:** Added ~5% monthly structural decay to model VIX ETF contango

```python
# V2 Code
MONTHLY_DECAY_RATE = 0.05  # 5% per month
DAILY_DECAY_RATE = MONTHLY_DECAY_RATE / 21

df['VIX_ETF_Return'] = df['VIX_Return'] - DAILY_DECAY_RATE
```

**The Improvement:**  
VIX ETFs like VXX suffer from **contango** - the VIX futures curve is typically upward-sloping, causing constant value erosion. VXX has lost 99.9% of its value since 2009, requiring multiple reverse splits.

**Still A Problem:**  
After adding decay, returns were still suspiciously high at 49.9%. Further investigation revealed two critical flaws:

1. **Assumed 1:1 beta** - Code assumed VXX moves identically to VIX
2. **Constant decay** - Used a fixed 5% monthly decay regardless of market conditions

**What I Learned:**  
VIX ETFs don't behave like the VIX index

---

### Version 3: Market Microstructure Reality
**File:** `regime_based_v3.py`  
**Result:** 2.7% annualized return using regime-based strategy 
**The Final Fixes:** Added VIX beta and regime-dependent decay

```python
# V3 Code
VIX_BETA = 0.45  # VXX has ~0.45 beta to VIX index

# Regime-dependent decay
if vix_level < 15:
    decay = 0.10 / 21  # 10% monthly (steep contango)
elif vix_level > 30:
    decay = -0.02 / 21  # Negative 2% monthly (backwardation)
else:
    decay = 0.05 / 21  # 5% monthly (normal contango)

df['VIX_ETF_Return'] = df['VIX_Return'] * VIX_BETA - decay
```

**Why This Matters:**

#### 1. VIX Beta (~0.45)
VIX ETFs track VIX **futures**, not the VIX index. Due to term structure effects:
- When VIX spikes 12 → 20 (+67%), VXX only gains ~30%
- When VIX drops 20 → 12 (-40%), VXX only loses ~18%

**Impact:** On 247 big VIX spike days averaging ~+18.9%, V2 captured ~18.6% while V3 only captured ~8.3% (due to beta).

#### 2. Regime-Dependent Contango
Contango steepness varies with VIX level:
- **VIX < 15** (complacency): Steep contango, ~10% monthly decay
- **VIX 15-30** (normal): Moderate contango, ~5% monthly decay  
- **VIX > 30** (fear): Often backwardation, decay reverses to negative

**Impact:** The regime strategy spends most time long VIX when it's < 15 (paying highest decay rate).

**What I Learned:**  
The difference between 61.5% and 2.7% returns demonstrates that **market microstructure details are more important than the trading signal**. Even with perfect code, wrong assumptions yield meaningless results.

---

## Results Comparison

| Version | Best Annualized Return | Key Assumptions |
|---------|-------------------|-----------------|
| **V1** | 61.52% | • Raw VIX index<br>• No decay<br>• 1.0 beta |
| **V2** | 49.94% | • Constant 5% decay<br>• 1.0 beta |
| **V3** | **2.70%** | • 0.45 beta<br>• Regime-dependent decay |

**98.5% reduction from V1 to V3** - all from fixing market microstructure assumptions!

---

## What is VIX?

The **CBOE Volatility Index (VIX)** measures expected 30-day volatility of the S&P 500, derived from options prices. Often called the "fear gauge":
- **Rises** during market stress (uncertainty, selloffs)
- **Falls** during calm markets (confidence, bullishness)  
- **Mean-reverts** around 15-20 over the long term

### The Trading Challenge

**VIX is an index** - you can't buy it directly. To trade volatility, you must use:

1. **VIX Futures** (requires futures account, margin)
2. **VIX ETFs (VXX, UVXY, SVXY)** (accessible, but decay heavily)
3. **S&P 500 Options** (capital-intensive, complex)

### Why VIX ETFs Lose Money

VIX ETFs suffer from **structural decay** due to:

- **Contango:** VIX futures curve typically upward-sloping (M1 < M2 < M3)
- **Daily Rebalancing:** ETFs roll futures daily, selling low and buying high  
- **Roll Yield:** Constant cost of rolling from expiring futures to next month

**Historical Example:**  
VXX launched in 2009 at equivalent price of $400,000 (after reverse splits). Today it trades at ~$40. That's **99.99% decay** over 15 years, even though VIX returned to similar levels.

---

## Trading Strategies Implemented

### 1. Mean Reversion
**Logic:** VIX tends to revert to its long-term mean (15-20)

**Signals:**
- SHORT VIX when it spikes > 25 (volatility expensive)
- COVER when VIX falls < 18 (mean reversion complete)
- STOP LOSS at 35 (protect against volatility explosions)

**V2 Impact:** Short positions BENEFIT from decay (earn while waiting for VIX to fall)

---

### 2. Moving Average Crossover  
**Logic:** Trend-following using technical indicators

**Signals:**
- LONG VIX when 10-day MA crosses above 30-day MA
- SHORT VIX when 10-day MA crosses below 30-day MA

**V2 Impact:** Beta reduction hurts both long and short, whipsaw losses dominate

---

### 3. Regime-Based (Focus of Analysis)
**Logic:** Trade based on VIX level categories

**Signals:**
- LONG VIX when < 15 (complacency, expect reversion up)
- SHORT VIX when > 30 (fear, expect reversion down)
- FLAT when 15-30 (normal range, no edge)

**Why Returns Varied So Much:**

| Version | Return | Explanation |
|---------|--------|-------------|
| V1 | 61.5% | Long positions during 2024-2025 captured full VIX spikes (12 → 20 = 67% gain) |
| V2 | 49.9% | Added 5% monthly decay but still assumed 1:1 VIX movement |
| V3 | 2.7% | Beta (0.45) cut spike gains in half + higher decay when VIX < 15 |

**Key Insight:** Strategy was long VIX 37% of the time (1,037 days), mostly when VIX < 15. During these periods, V3 paid 10% monthly decay AND only captured 45% of VIX moves. The combination destroyed returns.

---

### 4. Contrarian
**Logic:** Buy volatility after market drops

**Signals:**
- LONG VIX when SPY drops > 2% in one day
- HOLD for 5 days to capture volatility expansion

**V2 Impact:** Short 5-day holding period still suffers decay cost

---

## Installation & Usage

### Prerequisites
- Python 3.8+
- pip package manager

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 1: Collect Data
```bash
python data_collection.py
```
Downloads 10 years of VIX/SPY data and creates exploratory visualizations.

**Output:**
- `vix_spy_data.csv` - Processed dataset
- `exploratory_analysis.png` - EDA charts

### Step 2: Run All Three Versions

#### V1 - Naive Implementation
```bash
python backtesting.py
```
**Output:** `backtest_results.png` showing unrealistic 61.5% returns

#### V2 - With Constant Decay
```bash
python backtesting_v2.py
```
**Output:** `backtest_results_v2.png` showing 49.9% returns (better but still high)

#### V3 - Realistic Modeling
```bash
python regime_based_v3.py
```
**Output:** Console analysis showing 2.7% returns with detailed breakdown

---

## Project Structure

```
VIX-Trading-Strategy/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
│
├── data_collection.py             # Download & process VIX/SPY data
├── backtesting.py                 # V1 - Naive (raw VIX index)
├── backtesting_v2.py              # V2 - Constant decay
├── regime_based_v3.py             # V3 - Beta + regime decay
│
├── compare_versions.py            # Side-by-side comparison
│
├── data/
    ├── vix_spy_data.csv           # Generated: Market data
    ├── v1_v2_comparison.csv       # Comparison of V1 and V2
    └── v2_v3_comparison.csv       # Comparison of V2 and V3

└── images/
    ├── exploratory_analysis.png       # Generated: EDA visualizations
    ├── backtest_results.png           # Generated: V1 results
    └── backtest_results_v2.png        # Generated: V2 result
```

---

## Limitations & Disclaimers

### What V3 Still Doesn't Include

1. **Actual VIX Futures Data**
   - V3 uses VIX index with beta adjustment
   - Reality: Should use actual VIX futures term structure data
   - Impact: Beta varies over time and by expiration

2. **Transaction Costs**
   - V3 uses simplified 0.1% entry/exit costs
   - Reality: VIX ETFs have wide bid-ask spreads, especially during volatility
   - Impact: Real costs likely 0.2-0.5% per round-trip

3. **Leverage & Margin**
   - V3 doesn't model leveraged VIX ETFs (UVXY = 2x)
   - Reality: Higher leverage = higher decay rates
   - Impact: UVXY loses ~20-30% per month in contango

4. **Market Impact**
   - V3 assumes perfect fills at market price
   - Reality: Large orders move the market
   - Impact: Slippage reduces returns, especially on spike days

5. **Survivorship Bias**
   - V3 backtests over 2015-2026
   - Reality: Different periods would show different results
   - Impact: Single-period backtests can be misleading

### Educational Purpose

**This project is for educational purposes only and is not investment advice.**

The strategies shown are NOT recommended for actual trading because:
- VIX ETFs have severe structural decay
- Transaction costs and slippage are higher than modeled
- Past performance ≠ future results
- Volatility trading requires significant capital and risk management

---

## License

This project is for educational purposes. Feel free to use and modify with attribution.

---

## Project Evolution Summary

| Version | Annualized Return | Key Fix | Lesson Learned |
|---------|-------------------|---------|----------------|
| V1 | 61.52% | N/A (baseline) | VIX isn't tradeable |
| V2 | 49.94% | Added 5% monthly decay | VIX ETFs suffer contango |
| V3 | **2.70%** | Beta (0.45) + regime decay | Details matter enormously |

**98.5% reduction from V1 to V3** - all from fixing assumptions about market structure!
