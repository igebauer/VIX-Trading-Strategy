# VIX Trading Strategy - V1 vs V2 Comparison 

# This script runs both V1 (naive) and V2 (more realistic) backtests and creates
# a side-by-side comparison to highlight the impact of VIX ETF decay modeling.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import both versions
import sys
import importlib.util

def load_module_from_file(module_name, file_path):
    # Load a Python module from a file path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def run_comparison():
    # Run both V1 and V2 backtests and compare results
    
    print("="*80)
    print("VIX TRADING STRATEGY - V1 vs V2 COMPARISON")
    print("="*80)
    print("\nThis script runs both versions and compares the results.")
    print("V1 = Naive implementation (raw VIX index)")
    print("V2 = Realistic implementation (VIX ETF with decay)")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    try:
        data = pd.read_csv('data/vix_spy_data.csv', index_col=0, parse_dates=True)
        print(f"Loaded {len(data)} days of data")
    except FileNotFoundError:
        print("Error: Data file not found. Please run data_collection.py first!")
        return
    
    # Import backtesting modules
    print("\nImporting backtesting modules...")
    bt_v1 = load_module_from_file("backtesting_v1", "backtesting.py")
    bt_v2 = load_module_from_file("backtesting_v2", "backtesting_v2.py")
    
    # Run V1
    print("\n" + "="*80)
    print("RUNNING V1 (NAIVE IMPLEMENTATION)")
    print("="*80)
    backtester_v1 = bt_v1.VIXBacktester(data, initial_capital=100000, transaction_cost=0.001)
    
    # Run all strategies in V1
    backtester_v1.mean_reversion_strategy(entry_threshold=25, exit_threshold=18, 
                                          stop_loss=35, position_size=0.5)
    backtester_v1.ma_crossover_strategy(fast_window=10, slow_window=30, position_size=0.5)
    backtester_v1.regime_based_strategy(position_size=0.5)
    backtester_v1.contrarian_spy_strategy(spy_threshold=-2.0, holding_period=5, position_size=0.5)
    
    comparison_v1 = backtester_v1.compare_strategies()
    
    # Run V2
    print("\n" + "="*80)
    print("RUNNING V2 (REALISTIC VIX ETF MODELING)")
    print("="*80)
    backtester_v2 = bt_v2.VIXBacktesterV2(data, initial_capital=100000, transaction_cost=0.001)
    
    # Run all strategies in V2
    backtester_v2.mean_reversion_strategy(entry_threshold=25, exit_threshold=18, 
                                          stop_loss=35, position_size=0.5)
    backtester_v2.ma_crossover_strategy(fast_window=10, slow_window=30, position_size=0.5)
    backtester_v2.regime_based_strategy(position_size=0.5)
    backtester_v2.contrarian_spy_strategy(spy_threshold=-2.0, holding_period=5, position_size=0.5)
    
    comparison_v2 = backtester_v2.compare_strategies()
    
    # Create comparison table
    print("\n" + "="*80)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*80)
    
    comparison_table = pd.DataFrame()
    
    for strategy in comparison_v1.index:
        v1_data = comparison_v1.loc[strategy]
        v2_data = comparison_v2.loc[strategy]
        
        row = {
            'Strategy': strategy,
            'V1_Return': v1_data['Annualized Return (%)'],
            'V2_Return': v2_data['Annualized Return (%)'],
            'Difference': round((v2_data['Annualized Return (%)'] - v1_data['Annualized Return (%)']), 3),
            'V1_Sharpe': v1_data['Sharpe Ratio'],
            'V2_Sharpe': v2_data['Sharpe Ratio'],
            'V1_MaxDD': v1_data['Max Drawdown (%)'],
            'V2_MaxDD': v2_data['Max Drawdown (%)'],
        }
        comparison_table = pd.concat([comparison_table, pd.DataFrame([row])], ignore_index=True)
    
    comparison_table = comparison_table.set_index('Strategy')
    print("\n" + comparison_table.to_string())
    
    # Save comparison table
    comparison_table.to_csv('data/v1_v2_comparison.csv')
    print("\nSaved comparison table to 'data/v1_v2_comparison.csv'")
    
    # Create visualization
    create_comparison_plots(comparison_table)
    
    # Print analysis
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Find biggest differences
    biggest_loser = comparison_table['Difference'].idxmin()
    biggest_winner = comparison_table['Difference'].idxmax()
    
    print("\nMost Impacted by VIX ETF Decay:")
    print(f"   {biggest_loser}")
    print(f"   V1: {comparison_table.loc[biggest_loser, 'V1_Return']:.2f}%")
    print(f"   V2: {comparison_table.loc[biggest_loser, 'V2_Return']:.2f}%")
    print(f"   Difference: {comparison_table.loc[biggest_loser, 'Difference']:.2f}%")
    print("   Long VIX exposure hurt by structural decay")
    
    print("\nLeast/Most Improved:")
    print(f"   {biggest_winner}")
    print(f"   V1: {comparison_table.loc[biggest_winner, 'V1_Return']:.2f}%")
    print(f"   V2: {comparison_table.loc[biggest_winner, 'V2_Return']:.2f}%")
    print(f"   Difference: {comparison_table.loc[biggest_winner, 'Difference']:.2f}%")
    if comparison_table.loc[biggest_winner, 'Difference'] > 0:
        print("   Short VIX exposure benefited from decay")
    else:
        print("   This strategy still underperformed")
    
    print("\nKey Insights:")
    print("   • Strategies with long VIX exposure show dramatically lower returns in V2")
    print("   • Strategies with short VIX exposure may improve in V2 (decay helps shorts)")
    print("   • V1 results were unrealistic due to ignoring VIX ETF structural costs")
    print("   • V2 provides more realistic expectations for VIX trading")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print("\nFiles generated:")
    print("• v1_v2_comparison.csv - Detailed comparison table")
    print("• v1_v2_comparison.png - Visual comparison charts")


def create_comparison_plots(comparison_table):
    # Create visual comparison of V1 vs V2 results
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('V1 vs V2 Comparison - Impact of VIX ETF Decay Modeling', fontsize=16)
    
    strategies = comparison_table.index
    x = np.arange(len(strategies))
    width = 0.35
    
    # 1. Annualized Returns Comparison
    ax1 = axes[0, 0]
    v1_returns = comparison_table['V1_Return']
    v2_returns = comparison_table['V2_Return']
    
    bars1 = ax1.bar(x - width/2, v1_returns, width, label='V1 (Naive)', color='lightcoral', alpha=0.8)
    bars2 = ax1.bar(x + width/2, v2_returns, width, label='V2 (Realistic)', color='lightblue', alpha=0.8)
    
    ax1.set_ylabel('Annualized Return (%)')
    ax1.set_title('Annualized Returns: V1 vs V2')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=45, ha='right')
    ax1.legend()
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    # 2. Return Difference (V2 - V1)
    ax2 = axes[0, 1]
    differences = comparison_table['Difference']
    colors = ['green' if x > 0 else 'red' for x in differences]
    
    bars = ax2.bar(x, differences, color=colors, alpha=0.7)
    ax2.set_ylabel('Return Difference (%)')
    ax2.set_title('Impact of Adding VIX ETF Decay (V2 - V1)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, differences)):
        ax2.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontsize=9)
    
    # 3. Sharpe Ratio Comparison
    ax3 = axes[1, 0]
    v1_sharpe = comparison_table['V1_Sharpe']
    v2_sharpe = comparison_table['V2_Sharpe']
    
    bars1 = ax3.bar(x - width/2, v1_sharpe, width, label='V1 (Naive)', color='lightcoral', alpha=0.8)
    bars2 = ax3.bar(x + width/2, v2_sharpe, width, label='V2 (Realistic)', color='lightblue', alpha=0.8)
    
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_title('Risk-Adjusted Returns: V1 vs V2')
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies, rotation=45, ha='right')
    ax3.legend()
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.axhline(y=1, color='green', linestyle='--', linewidth=0.8, alpha=0.3)
    ax3.grid(True, alpha=0.3)
    
    # 4. Max Drawdown Comparison
    ax4 = axes[1, 1]
    v1_dd = comparison_table['V1_MaxDD']
    v2_dd = comparison_table['V2_MaxDD']
    
    bars1 = ax4.bar(x - width/2, v1_dd, width, label='V1 (Naive)', color='lightcoral', alpha=0.8)
    bars2 = ax4.bar(x + width/2, v2_dd, width, label='V2 (Realistic)', color='lightblue', alpha=0.8)
    
    ax4.set_ylabel('Max Drawdown (%)')
    ax4.set_title('Maximum Drawdown: V1 vs V2')
    ax4.set_xticks(x)
    ax4.set_xticklabels(strategies, rotation=45, ha='right')
    ax4.legend()
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/v1_v2_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved comparison visualization to 'images/v1_v2_comparison.png'")
    plt.show()


if __name__ == "__main__":
    run_comparison()
