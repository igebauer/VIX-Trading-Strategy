# VIX Trading Strategy - Backtesting Framework

# This script implements and backtests various VIX trading strategies.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class VIXBacktester:
    # Backtesting framework for VIX trading strategies.
    
    # Please Note: This backtest trades a hypothetical VIX ETF, which tracks 
    # VIX futures. In reality, you'd need VIX futures or VXX data, but I'm going to 
    # use the VIX index as a proxy for educational purposes
    
    def __init__(self, data, initial_capital=100000, transaction_cost=0.001):
        # Initialize backtester
        
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.results = {}
        
    def mean_reversion_strategy(self, entry_threshold=25, exit_threshold=18, 
                                 stop_loss=35, position_size=0.5):
        # Mean reversion strategy: Short VIX when elevated, cover when it falls
        
        # Logic:
        # - When VIX > entry_threshold: Enter short position (sell volatility)
        # - Exit when VIX < exit_threshold OR VIX > stop_loss
        
        df = self.data.copy()
        
        # Initialize signals
        df['Signal'] = 0  # 0 = no position, -1 = short VIX
        df['Position'] = 0
        
        in_position = False
        
        for i in range(1, len(df)):
            if not in_position:
                # Check for entry signal
                if df['VIX'].iloc[i] > entry_threshold:
                    df.loc[df.index[i], 'Signal'] = -1  # Short signal
                    in_position = True
            else:
                # Check for exit signal
                if (df['VIX'].iloc[i] < exit_threshold or  # Take profit
                    df['VIX'].iloc[i] > stop_loss):  # Stop loss
                    df.loc[df.index[i], 'Signal'] = 0  # Exit signal
                    in_position = False
                else:
                    df.loc[df.index[i], 'Signal'] = -1  # Maintain position
        
        # Calculate positions
        df['Position'] = df['Signal']
        
        strategy_name = f"Mean_Reversion_Entry{entry_threshold}_Exit{exit_threshold}"
        return self._calculate_returns(df, strategy_name, position_size)
    
    def ma_crossover_strategy(self, fast_window=10, slow_window=30, position_size=0.5):
        # Moving average crossover strategy.
            
        # Logic:
        # - When VIX crosses above slow MA: Long VIX (expect volatility expansion)
        # - When VIX crosses below slow MA: Short VIX (expect mean reversion)
        
        df = self.data.copy()
        
        # Calculate moving averages
        df['VIX_Fast_MA'] = df['VIX'].rolling(window=fast_window).mean()
        df['VIX_Slow_MA'] = df['VIX'].rolling(window=slow_window).mean()
        
        # Generate signals
        df['Signal'] = 0
        df.loc[df['VIX_Fast_MA'] > df['VIX_Slow_MA'], 'Signal'] = 1  # Long VIX
        df.loc[df['VIX_Fast_MA'] < df['VIX_Slow_MA'], 'Signal'] = -1  # Short VIX
        
        df['Position'] = df['Signal']
        
        strategy_name = f"MA_Crossover_{fast_window}_{slow_window}"
        return self._calculate_returns(df, strategy_name, position_size)
    
    def regime_based_strategy(self, position_size=0.5):
        # Trade based on VIX regime
        
        # Logic:
        # - Low regime (VIX < 15): Long VIX (expect mean reversion up)
        # - High regime (VIX > 30): Short VIX (expect mean reversion down)
        # - Normal regime: No position
        
        df = self.data.copy()
        
        df['Signal'] = 0
        df.loc[df['VIX'] < 15, 'Signal'] = 1   # Long VIX when low
        df.loc[df['VIX'] > 30, 'Signal'] = -1  # Short VIX when high
        
        df['Position'] = df['Signal']
        
        strategy_name = "Regime_Based"
        return self._calculate_returns(df, strategy_name, position_size)
    
    def contrarian_spy_strategy(self, spy_threshold=-2.0, holding_period=5, position_size=0.5):
        # Contrarian strategy: Buy VIX after the market drops
            
        # Logic:
        # - When SPY drops > threshold: Long VIX (expect vol spike)
        # - Hold for holding_period days
        
        df = self.data.copy()
        
        df['Signal'] = 0
        df['Days_Held'] = 0
        
        for i in range(1, len(df)):
            # Check if we should enter
            if df['Days_Held'].iloc[i-1] == 0:
                if df['SPY_Return'].iloc[i] * 100 < spy_threshold:
                    df.loc[df.index[i], 'Signal'] = 1  # Long VIX
                    df.loc[df.index[i], 'Days_Held'] = 1
            # Check if we're in a position
            elif df['Days_Held'].iloc[i-1] > 0 and df['Days_Held'].iloc[i-1] < holding_period:
                df.loc[df.index[i], 'Signal'] = 1  # Maintain position
                df.loc[df.index[i], 'Days_Held'] = df['Days_Held'].iloc[i-1] + 1
            else:
                df.loc[df.index[i], 'Signal'] = 0
                df.loc[df.index[i], 'Days_Held'] = 0
        
        df['Position'] = df['Signal']
        
        strategy_name = f"Contrarian_SPY_Threshold{abs(spy_threshold)}"
        return self._calculate_returns(df, strategy_name, position_size)
    
    def _calculate_returns(self, df, strategy_name, position_size):
        # Calculate strategy returns and performance metrics
        
        # Calculate VIX returns (proxy for VIX ETF)
        df['VIX_Return'] = df['VIX'].pct_change()
        
        # Calculate position changes
        df['Position_Change'] = df['Position'].diff()
        
        # Calculate transaction costs
        df['Transaction_Cost'] = abs(df['Position_Change']) * self.transaction_cost
        
        # Calculate strategy returns
        # Positive position (long) benefits from VIX going up
        # Negative position (short) benefits from VIX going down
        df['Strategy_Return'] = (df['Position'].shift(1) * df['VIX_Return'] * position_size - 
                                 df['Transaction_Cost'])
        
        # Calculate cumulative returns
        df['Cumulative_Market_Return'] = (1 + df['VIX_Return']).cumprod()
        df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()
        
        # Calculate portfolio value
        df['Portfolio_Value'] = self.initial_capital * df['Cumulative_Strategy_Return']
        
        # Store results
        self.results[strategy_name] = df
        
        return df
    
    def calculate_metrics(self, df, strategy_name):
        # Calculate performance metrics for a strategy
        
        # Drop NaN values for calculations
        returns = df['Strategy_Return'].dropna()
        
        # Total return
        total_return = (df['Portfolio_Value'].iloc[-1] / self.initial_capital - 1) * 100
        
        # Annualized return
        years = (df.index[-1] - df.index[0]).days / 365.25
        annualized_return = ((df['Portfolio_Value'].iloc[-1] / self.initial_capital) ** (1/years) - 1) * 100
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = (annualized_return / volatility) if volatility != 0 else 0
        
        # Maximum drawdown
        cumulative = df['Portfolio_Value']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Win rate
        winning_trades = (returns > 0).sum()
        total_trades = (returns != 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Number of trades
        num_trades = (df['Position'].diff() != 0).sum()
        
        # Calmar ratio (return / max drawdown)
        calmar_ratio = (annualized_return / abs(max_drawdown)) if max_drawdown != 0 else 0
        
        metrics = {
            'Strategy': strategy_name,
            'Total Return (%)': round(total_return, 2),
            'Annualized Return (%)': round(annualized_return, 2),
            'Volatility (%)': round(volatility, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Calmar Ratio': round(calmar_ratio, 2),
            'Win Rate (%)': round(win_rate, 2),
            'Number of Trades': int(num_trades),
            'Final Portfolio Value ($)': round(df['Portfolio_Value'].iloc[-1], 2)
        }
        
        return metrics
    
    def compare_strategies(self):
        # Compare all strategies and display results
        
        if not self.results:
            print("No strategies have been run yet!")
            return
        
        # Calculate metrics for all strategies
        all_metrics = []
        for strategy_name, df in self.results.items():
            metrics = self.calculate_metrics(df, strategy_name)
            all_metrics.append(metrics)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_metrics)
        comparison_df = comparison_df.set_index('Strategy')
        
        print("\n" + "="*80)
        print("STRATEGY COMPARISON")
        print("="*80)
        print(comparison_df.to_string())
        print("="*80)
        
        return comparison_df
    
    def plot_results(self, save_plots=True):
        # Create a visualization of all strategy results
        
        if not self.results:
            print("No strategies have been run yet!")
            return
        
        num_strategies = len(self.results)
        fig, axes = plt.subplots(num_strategies + 1, 1, figsize=(14, 4 * (num_strategies + 1)))
        
        if num_strategies == 1:
            axes = [axes]
        
        fig.suptitle('VIX Trading Strategies - Backtest Results', fontsize=16, y=0.995)
        
        # Plot buy-and-hold VIX benchmark on first subplot
        first_df = list(self.results.values())[0]
        benchmark_value = self.initial_capital * first_df['Cumulative_Market_Return']
        axes[0].plot(benchmark_value.index, benchmark_value, 
                    label='Buy & Hold VIX (Benchmark)', 
                    linewidth=2, color='gray', linestyle='--', alpha=0.7)
        
        # Plot all strategies on first subplot for comparison
        colors = plt.cm.Set2(np.linspace(0, 1, num_strategies))
        for (strategy_name, df), color in zip(self.results.items(), colors):
            axes[0].plot(df.index, df['Portfolio_Value'], 
                        label=strategy_name, linewidth=2, color=color)
        
        axes[0].set_title('All Strategies Comparison')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].legend(loc='best', fontsize=8)
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=self.initial_capital, color='black', linestyle=':', alpha=0.5)
        
        # Plot individual strategies with positions
        for idx, ((strategy_name, df), color) in enumerate(zip(self.results.items(), colors), start=1):
            ax = axes[idx]
            
            # Plot portfolio value
            ax.plot(df.index, df['Portfolio_Value'], linewidth=2, color=color, label='Portfolio Value')
            ax.set_ylabel('Portfolio Value ($)', color=color)
            ax.tick_params(axis='y', labelcolor=color)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=self.initial_capital, color='black', linestyle=':', alpha=0.5)
            
            # Create second y-axis for positions
            ax2 = ax.twinx()
            
            # Plot positions as shaded regions
            long_positions = df['Position'] > 0
            short_positions = df['Position'] < 0
            
            ax2.fill_between(df.index, 0, 1, where=long_positions, 
                            alpha=0.2, color='green', label='Long VIX', step='post')
            ax2.fill_between(df.index, 0, -1, where=short_positions, 
                            alpha=0.2, color='red', label='Short VIX', step='post')
            ax2.set_ylabel('Position', color='black')
            ax2.set_ylim(-1.5, 1.5)
            ax2.set_yticks([-1, 0, 1])
            ax2.set_yticklabels(['Short', 'Flat', 'Long'])
            
            # Calculate metrics for title
            metrics = self.calculate_metrics(df, strategy_name)
            title = (f"{strategy_name} | Return: {metrics['Annualized Return (%)']}% | "
                    f"Sharpe: {metrics['Sharpe Ratio']:.2f} | "
                    f"Max DD: {metrics['Max Drawdown (%)']}%")
            ax.set_title(title, fontsize=10)
            
            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('images/backtest_results.png', 
                       dpi=300, bbox_inches='tight')
            print("\nSaved backtest visualization to 'backtest_results.png'")
        
        plt.show()


def main():    
    print("="*80)
    print("VIX TRADING STRATEGY - BACKTESTING")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    try:
        data = pd.read_csv('data/vix_spy_data.csv', 
                          index_col=0, parse_dates=True)
        print(f"Loaded {len(data)} days of data")
    except FileNotFoundError:
        print("Error: Data file not found. Please run data_collection.py first!")
        return
    
    # Initialize backtester
    backtester = VIXBacktester(data, initial_capital=100000, transaction_cost=0.001)
    
    print("\nRunning strategies...")
    print("-" * 80)
    
    # Strategy 1: Mean Reversion
    print("1. Mean Reversion Strategy (Short VIX when high, cover when low)")
    backtester.mean_reversion_strategy(entry_threshold=25, exit_threshold=18, 
                                       stop_loss=35, position_size=0.5)
    
    # Strategy 2: MA Crossover
    print("2. Moving Average Crossover Strategy")
    backtester.ma_crossover_strategy(fast_window=10, slow_window=30, position_size=0.5)
    
    # Strategy 3: Regime-Based
    print("3. Regime-Based Strategy")
    backtester.regime_based_strategy(position_size=0.5)
    
    # Strategy 4: Contrarian
    print("4. Contrarian Strategy (Buy VIX after market drops)")
    backtester.contrarian_spy_strategy(spy_threshold=-2.0, holding_period=5, position_size=0.5)
    
    print("\nAll strategies executed")
    
    # Compare strategies
    comparison = backtester.compare_strategies()
    
    # Plot results
    backtester.plot_results(save_plots=True)
    
    print("\n" + "="*80)
    print("BACKTESTING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
