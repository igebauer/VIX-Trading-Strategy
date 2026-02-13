# VIX Trading Strategy - V3: Even more realistic VIX ETF modeling
# This version adds:
# 1. VIX beta (VXX moves ~40-50% as much as VIX)
# 2. Regime-dependent decay (higher when VIX is low)

import pandas as pd
import numpy as np

# Configuration
VIX_BETA = 0.45  # VXX typically has 0.4-0.5 beta to VIX index

# Regime-dependent decay rates
DECAY_RATES = {
    'low_vix': 0.10 / 21,      # 10% monthly when VIX < 15 (steep contango)
    'normal_vix': 0.05 / 21,   # 5% monthly when VIX 15-30 (normal contango)  
    'high_vix': -0.02 / 21     # Negative 2% monthly when VIX > 30 (backwardation)
}

def calculate_realistic_etf_return(vix_return, vix_level):
    # Calculate realistic VIX ETF return accounting for:
    # 1. Beta to VIX index (VXX doesn't move 1:1 with VIX)
    # 2. Regime-dependent decay (contango varies with VIX level)

    # Apply beta
    etf_return = vix_return * VIX_BETA
    
    # Apply regime-dependent decay
    if vix_level < 15:
        decay = DECAY_RATES['low_vix']
    elif vix_level > 30:
        decay = DECAY_RATES['high_vix']
    else:
        decay = DECAY_RATES['normal_vix']
    
    return etf_return - decay


# Load data
data = pd.read_csv('data/vix_spy_data.csv', index_col=0, parse_dates=True)

print("="*80)
print("V3: MOST REALISTIC VIX ETF MODELING")
print("="*80)
print(f"\nVIX Beta: {VIX_BETA}")
print(f"Decay when VIX < 15: {DECAY_RATES['low_vix']*21*100:.1f}% per month")
print(f"Decay when VIX 15-30: {DECAY_RATES['normal_vix']*21*100:.1f}% per month")
print(f"Decay when VIX > 30: {DECAY_RATES['high_vix']*21*100:.1f}% per month (backwardation)")
print("="*80)

# Calculate returns
data['VIX_Return'] = data['VIX'].pct_change()

# V2: Constant decay, no beta
CONSTANT_DECAY = 0.05 / 21
data['V2_ETF_Return'] = data['VIX_Return'] - CONSTANT_DECAY

# V3: Beta + regime-dependent decay
data['V3_ETF_Return'] = data.apply(
    lambda row: calculate_realistic_etf_return(row['VIX_Return'], row['VIX']),
    axis=1
)

# Simulate regime strategy with both methods
data['Position'] = 0
data.loc[data['VIX'] < 15, 'Position'] = 1
data.loc[data['VIX'] > 30, 'Position'] = -1

POSITION_SIZE = 0.5

# Calculate strategy returns
data['V2_Strategy_Return'] = data['Position'].shift(1) * data['V2_ETF_Return'] * POSITION_SIZE
data['V3_Strategy_Return'] = data['Position'].shift(1) * data['V3_ETF_Return'] * POSITION_SIZE

# Calculate cumulative returns
data['V2_Cumulative'] = (1 + data['V2_Strategy_Return'].fillna(0)).cumprod()
data['V3_Cumulative'] = (1 + data['V3_Strategy_Return'].fillna(0)).cumprod()

# Compare results
v2_final = data['V2_Cumulative'].iloc[-1]
v3_final = data['V3_Cumulative'].iloc[-1]

print("\nREGIME STRATEGY COMPARISON:")
print("-"*80)
print(f"V2 (Constant 5% decay, 1.0 beta): ${100000 * v2_final:,.2f}")
print(f"V3 (Regime decay, {VIX_BETA} beta):  ${100000 * v3_final:,.2f}")
print(f"\nDifference: {((v2_final - v3_final) / v2_final * 100):.1f}% lower with realistic modeling")

# Calculate annualized returns
years = (data.index[-1] - data.index[0]).days / 365.25
v2_ann = ((v2_final) ** (1/years) - 1) * 100
v3_ann = ((v3_final) ** (1/years) - 1) * 100

print(f"\nAnnualized Returns:")
print(f"V2: {v2_ann:.2f}%")
print(f"V3: {v3_ann:.2f}%")

# Analyze why V3 is different
print("\n" + "="*80)
print("WHY V3 SHOWS DIFFERENT RESULTS:")
print("="*80)

# Compare long position returns in low VIX regime
low_vix_periods = data[data['VIX'] < 15]
print("\nWhen VIX < 15 (long position):")
print(f"  Days: {len(low_vix_periods)}")
print(f"  Avg VIX daily return: {low_vix_periods['VIX_Return'].mean()*100:.3f}%")
print(f"  V2 ETF return: {low_vix_periods['V2_ETF_Return'].mean()*100:.3f}%")
print(f"  V3 ETF return: {low_vix_periods['V3_ETF_Return'].mean()*100:.3f}%")
print("  Difference: V3 has beta reduction AND higher decay")

# Compare big spike days
big_spike_days = data[data['VIX_Return'] > 0.10]  # VIX up > 10%
print("\nBig VIX spike days (> 10%):")
print(f"  Count: {len(big_spike_days)}")
print(f"  Avg VIX return: {big_spike_days['VIX_Return'].mean()*100:.1f}%")
print(f"  V2 capture: {big_spike_days['V2_ETF_Return'].mean()*100:.1f}%")
print(f"  V3 capture: {big_spike_days['V3_ETF_Return'].mean()*100:.1f}%")
print(f"  V3 captures {VIX_BETA*100:.0f}% of VIX moves (beta effect)")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("\nV2 was still too optimistic because:")
print(f"1. Assumed VXX moves 1:1 with VIX (really ~{VIX_BETA}x)")
print(f"2. Used constant {CONSTANT_DECAY*21*100:.1f}% monthly decay")
print(f"   (Should be {DECAY_RATES['low_vix']*21*100:.1f}% when VIX < 15)")
print("\nV3 is more realistic:")
print(f"• {VIX_BETA} beta means VIX spike from 12→20 gives {(20/12-1)*VIX_BETA*100:.0f}% VXX gain, not {(20/12-1)*100:.0f}%")
print("• Higher decay when VIX low means you pay more while waiting")
print("• Result: Returns come down to more realistic levels")

# Save for comparison
comparison = pd.DataFrame({
    'V2_Final_Value': [round(v2_final * 100000, 3)],
    'V3_Final_Value': [round(v3_final * 100000, 3)],
    'V2_Ann_Return': [round(v2_ann, 3)],
    'V3_Ann_Return': [round(v3_ann, 3)],
    'Difference': [round(v2_ann - v3_ann, 3)]
})

comparison.to_csv('data/v2_vs_v3_comparison.csv', index=False)
print("\nSaved comparison to 'data/v2_vs_v3_comparison.csv'")
