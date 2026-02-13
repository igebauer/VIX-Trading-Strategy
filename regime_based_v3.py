# VIX Trading Strategy - V3: Most Realistic VIX ETF Modeling with Visualizations
# This version adds:
# 1. VIX beta (VXX moves ~40-50% as much as VIX)
# 2. Regime-dependent decay (higher when VIX is low)
# 3. Comprehensive visualizations comparing V2 vs V3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

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
data.loc[data['VIX'] < 15, 'Position'] = 1   # Long VIX when low
data.loc[data['VIX'] > 30, 'Position'] = -1  # Short VIX when high

POSITION_SIZE = 0.5
INITIAL_CAPITAL = 100000

# Calculate strategy returns
data['V2_Strategy_Return'] = data['Position'].shift(1) * data['V2_ETF_Return'] * POSITION_SIZE
data['V3_Strategy_Return'] = data['Position'].shift(1) * data['V3_ETF_Return'] * POSITION_SIZE

# Calculate cumulative returns
data['V2_Cumulative'] = (1 + data['V2_Strategy_Return'].fillna(0)).cumprod()
data['V3_Cumulative'] = (1 + data['V3_Strategy_Return'].fillna(0)).cumprod()

# Calculate portfolio values
data['V2_Portfolio'] = INITIAL_CAPITAL * data['V2_Cumulative']
data['V3_Portfolio'] = INITIAL_CAPITAL * data['V3_Cumulative']

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

# Save comparison data
comparison = pd.DataFrame({
    'V2_Final_Value': [round(v2_final * 100000, 2)],
    'V3_Final_Value': [round(v3_final * 100000, 2)],
    'V2_Ann_Return': [round(v2_ann, 2)],
    'V3_Ann_Return': [round(v3_ann, 2)],
    'Difference': [round(v2_ann - v3_ann, 2)]
})

comparison.to_csv('data/v2_v3_comparison.csv', index=False)
print("\nSaved comparison to 'data/v2_v3_comparison.csv'")


# CREATING VISUALIZATIONS

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Create comprehensive comparison figure
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Portfolio Value Comparison Over Time
ax1 = fig.add_subplot(gs[0, :])

ax1.plot(data.index, data['V2_Portfolio'], 
         label='V2 (Constant 5% decay, 1.0 beta)', 
         linewidth=2, color='#FF6B6B', alpha=0.8)
ax1.plot(data.index, data['V3_Portfolio'], 
         label=f'V3 (Regime decay, {VIX_BETA} beta)', 
         linewidth=2, color='#4ECDC4', alpha=0.8)

ax1.axhline(y=INITIAL_CAPITAL, color='black', linestyle=':', alpha=0.5, label='Initial Capital')

ax1.set_title('V2 vs V3: Portfolio Value Over Time', fontsize=14, fontweight='bold')
ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
ax1.set_xlabel('Date', fontsize=11)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# Format y-axis as currency
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Add annotation with final values
textstr = f'V2 Final: ${data["V2_Portfolio"].iloc[-1]:,.0f}\nV3 Final: ${data["V3_Portfolio"].iloc[-1]:,.0f}\nDifference: {((v2_final - v3_final) / v2_final * 100):.1f}%'
ax1.text(0.98, 0.97, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. VIX with Regime Zones
ax2 = fig.add_subplot(gs[1, 0])

ax2.plot(data.index, data['VIX'], linewidth=1, color='darkred', alpha=0.7)
ax2.axhline(y=15, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Long threshold (15)')
ax2.axhline(y=30, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Short threshold (30)')
ax2.fill_between(data.index, 0, 15, alpha=0.1, color='green', label='Long VIX zone')
ax2.fill_between(data.index, 30, 100, alpha=0.1, color='red', label='Short VIX zone')

ax2.set_title('VIX Levels with Regime Strategy Zones', fontsize=12, fontweight='bold')
ax2.set_ylabel('VIX Level', fontsize=11)
ax2.set_xlabel('Date', fontsize=11)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. Cumulative Returns Comparison
ax3 = fig.add_subplot(gs[1, 1])

ax3.plot(data.index, data['V2_Cumulative'], 
         label='V2', linewidth=2, color='#FF6B6B', alpha=0.8)
ax3.plot(data.index, data['V3_Cumulative'], 
         label='V3', linewidth=2, color='#4ECDC4', alpha=0.8)
ax3.axhline(y=1, color='black', linestyle=':', alpha=0.5)

ax3.set_title('Cumulative Return Multiple: V2 vs V3', fontsize=12, fontweight='bold')
ax3.set_ylabel('Return Multiple', fontsize=11)
ax3.set_xlabel('Date', fontsize=11)
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.3)

# Add annotation
textstr = f'V2: {v2_final:.2f}x ({v2_ann:.1f}% ann.)\nV3: {v3_final:.2f}x ({v3_ann:.1f}% ann.)'
ax3.text(0.98, 0.97, textstr, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Daily Returns Distribution
ax4 = fig.add_subplot(gs[2, 0])

v2_returns = data['V2_Strategy_Return'].dropna() * 100
v3_returns = data['V3_Strategy_Return'].dropna() * 100

ax4.hist(v2_returns, bins=50, alpha=0.5, color='#FF6B6B', label='V2', edgecolor='black')
ax4.hist(v3_returns, bins=50, alpha=0.5, color='#4ECDC4', label='V3', edgecolor='black')
ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)

ax4.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
ax4.set_xlabel('Daily Return (%)', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

# Add stats
v2_mean = v2_returns.mean()
v3_mean = v3_returns.mean()
textstr = f'V2 Mean: {v2_mean:.3f}%\nV3 Mean: {v3_mean:.3f}%'
ax4.text(0.02, 0.97, textstr, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 5. Performance Metrics Comparison
ax5 = fig.add_subplot(gs[2, 1])

# Calculate metrics
def calculate_metrics(returns, cumulative):
    total_return = (cumulative.iloc[-1] - 1) * 100
    volatility = returns.std() * np.sqrt(252) * 100
    sharpe = (total_return / years) / volatility if volatility > 0 else 0
    
    # Max drawdown
    portfolio = INITIAL_CAPITAL * cumulative
    running_max = portfolio.expanding().max()
    drawdown = (portfolio - running_max) / running_max
    max_dd = drawdown.min() * 100
    
    return {
        'Ann. Return (%)': round(total_return / years, 2),
        'Volatility (%)': round(volatility, 2),
        'Sharpe Ratio': round(sharpe, 2),
        'Max Drawdown (%)': round(max_dd, 2)
    }

v2_metrics = calculate_metrics(data['V2_Strategy_Return'].dropna(), data['V2_Cumulative'])
v3_metrics = calculate_metrics(data['V3_Strategy_Return'].dropna(), data['V3_Cumulative'])

metrics_df = pd.DataFrame({
    'V2': v2_metrics,
    'V3': v3_metrics
})

# Create bar chart
x = np.arange(len(metrics_df.index))
width = 0.35

bars1 = ax5.bar(x - width/2, metrics_df['V2'], width, label='V2', color='#FF6B6B', alpha=0.8)
bars2 = ax5.bar(x + width/2, metrics_df['V3'], width, label='V3', color='#4ECDC4', alpha=0.8)

ax5.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
ax5.set_ylabel('Value', fontsize=11)
ax5.set_xticks(x)
ax5.set_xticklabels(metrics_df.index, rotation=45, ha='right', fontsize=9)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3, axis='y')
ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Add value labels on bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=8)

autolabel(bars1)
autolabel(bars2)

# Overall title
fig.suptitle('VIX Regime Strategy: V2 vs V3 Comprehensive Comparison', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()

# Save figure
plt.savefig('images/v2_v3_comparison.png', dpi=300, bbox_inches='tight')
print("\nSaved comprehensive comparison chart to 'images/v2_v3_comparison.png'")

# Show plot
plt.show()

print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
print("\nGenerated visualization:")
print("1. images/v2_vs_v3_comparison.png - Comprehensive 6-panel comparison")
print("="*80)
