# VIX Trading Strategy - Data Collection

# This script downloads VIX and SPY data, performs exploratory analysis,
# and creates visualizations to understand volatility patterns

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class VIXDataCollector:
    # Class to handle VIX and market data collection and analysis
    
    def __init__(self, start_date='2015-01-01', end_date=None):
        # Initialize data collector
        
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime('%Y-%m-%d')
        self.vix_data = None
        self.spy_data = None
        self.combined_data = None
        
    def download_data(self):
        # Download VIX and SPY data from Yahoo Finance
        print(f"Downloading data from {self.start_date} to {self.end_date}...")
        
        # Download VIX (CBOE Volatility Index)
        print("Downloading VIX data...")
        vix = yf.download('^VIX', start=self.start_date, end=self.end_date, progress=False)
        self.vix_data = vix[['Close']].rename(columns={'Close': 'VIX'})
        
        # Download SPY (S&P 500 ETF)
        print("Downloading SPY data...")
        spy = yf.download('SPY', start=self.start_date, end=self.end_date, progress=False)
        self.spy_data = spy[['Close', 'Volume']].rename(columns={'Close': 'SPY', 'Volume': 'SPY_Volume'})
        
        # Combine data
        self.combined_data = pd.concat([self.vix_data, self.spy_data], axis=1)
        self.combined_data = self.combined_data.dropna()

        # Flatten any MultiIndex columns (for different yfinance versions)
        if isinstance(self.combined_data.columns, pd.MultiIndex):
            self.combined_data.columns = self.combined_data.columns.get_level_values(0)
        
        print(f"Downloaded {len(self.combined_data)} days of data")
        print(f"Date range: {self.combined_data.index[0]} to {self.combined_data.index[-1]}")
        
        return self.combined_data
    
    def calculate_features(self):
        # Calculate additional features for analysis
        df = self.combined_data.copy()

        # Helper function to ensure we have Series, not DataFrame
        def ensure_series(data):
            if isinstance(data, pd.DataFrame):
                return data.squeeze()
            return data

        df['VIX'] = ensure_series(df['VIX'])
        df['SPY'] = ensure_series(df['SPY'])
        df['SPY_Volume'] = ensure_series(df['SPY_Volume'])
        
        # SPY returns
        df['SPY_Return'] = df['SPY'].pct_change()
        df['SPY_Return_5d'] = df['SPY'].pct_change(5)
        df['SPY_Return_20d'] = df['SPY'].pct_change(20)
        
        # Realized volatility (20-day rolling std of daily returns, annualized)
        df['Realized_Vol_20d'] = df['SPY_Return'].rolling(window=20).std() * np.sqrt(252)
        
        # VIX changes
        df['VIX_Change'] = df['VIX'].diff()
        df['VIX_Pct_Change'] = df['VIX'].pct_change()
        
        # VIX moving averages
        df['VIX_MA_10'] = df['VIX'].rolling(window=10).mean()
        df['VIX_MA_30'] = df['VIX'].rolling(window=30).mean()
        df['VIX_MA_50'] = df['VIX'].rolling(window=50).mean()
        
        # VIX relative to moving averages
        df['VIX_vs_MA10'] = (df['VIX'] - df['VIX_MA_10']) / df['VIX_MA_10']
        df['VIX_vs_MA30'] = (df['VIX'] - df['VIX_MA_30']) / df['VIX_MA_30']
        
        # Identify VIX regimes
        df['VIX_Regime'] = pd.cut(df['VIX'], 
                                   bins=[0, 15, 20, 30, 100],
                                   labels=['Low', 'Normal', 'Elevated', 'High'])
        
        self.combined_data = df
        print("Calculated features and regimes")
        
        return df
    
    def get_summary_statistics(self):
        # Print summary statistics
        df = self.combined_data
        
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        print("\nVIX Statistics:")
        print(f"Mean: {df['VIX'].mean():.2f}")
        print(f"Median: {df['VIX'].median():.2f}")
        print(f"Std Dev: {df['VIX'].std():.2f}")
        print(f"Min: {df['VIX'].min():.2f}")
        print(f"Max: {df['VIX'].max():.2f}")
        
        print("\nVIX Regime Distribution:")
        print(df['VIX_Regime'].value_counts(normalize=True).sort_index() * 100)
        
        print("\nCorrelation between VIX and SPY Returns:")
        print(f"{df[['VIX', 'SPY_Return']].corr().iloc[0,1]:.4f}")
        
        return df.describe()
    
    def create_visualizations(self, save_plots=True):
        # Create exploratory visualizations
        df = self.combined_data.dropna()
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('VIX Trading Strategy - Exploratory Data Analysis', fontsize=16, y=1.00)
        
        # VIX over time
        axes[0, 0].plot(df.index, df['VIX'], linewidth=0.8, color='darkred')
        axes[0, 0].axhline(y=15, color='green', linestyle='--', alpha=0.5, label='Low (15)')
        axes[0, 0].axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='Normal (20)')
        axes[0, 0].axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Elevated (30)')
        axes[0, 0].set_title('VIX Index Over Time')
        axes[0, 0].set_ylabel('VIX Level')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # SPY price over time
        axes[0, 1].plot(df.index, df['SPY'], linewidth=0.8, color='darkblue')
        axes[0, 1].set_title('SPY Price Over Time')
        axes[0, 1].set_ylabel('SPY Price ($)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # VIX distribution
        axes[1, 0].hist(df['VIX'], bins=50, edgecolor='black', alpha=0.7, color='darkred')
        axes[1, 0].axvline(df['VIX'].mean(), color='blue', linestyle='--', 
                          label=f'Mean: {df["VIX"].mean():.1f}')
        axes[1, 0].axvline(df['VIX'].median(), color='green', linestyle='--', 
                          label=f'Median: {df["VIX"].median():.1f}')
        axes[1, 0].set_title('VIX Distribution')
        axes[1, 0].set_xlabel('VIX Level')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # VIX vs SPY Returns scatter
        axes[1, 1].scatter(df['SPY_Return'] * 100, df['VIX'], alpha=0.3, s=10, color='purple')
        axes[1, 1].set_title('VIX vs SPY Daily Returns')
        axes[1, 1].set_xlabel('SPY Daily Return (%)')
        axes[1, 1].set_ylabel('VIX Level')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add correlation text
        corr = df[['VIX', 'SPY_Return']].corr().iloc[0, 1]
        axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                       transform=axes[1, 1].transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        #  VIX with moving averages
        axes[2, 0].plot(df.index, df['VIX'], linewidth=0.8, label='VIX', color='darkred', alpha=0.7)
        axes[2, 0].plot(df.index, df['VIX_MA_10'], linewidth=1, label='10-day MA', color='blue')
        axes[2, 0].plot(df.index, df['VIX_MA_30'], linewidth=1, label='30-day MA', color='green')
        axes[2, 0].set_title('VIX with Moving Averages')
        axes[2, 0].set_ylabel('VIX Level')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Implied vs Realized Volatility
        axes[2, 1].plot(df.index, df['VIX'], linewidth=0.8, label='Implied Vol (VIX)', color='darkred')
        axes[2, 1].plot(df.index, df['Realized_Vol_20d'] * 100, linewidth=0.8, 
                       label='Realized Vol (20d)', color='darkblue')
        axes[2, 1].set_title('Implied vs Realized Volatility')
        axes[2, 1].set_ylabel('Volatility (%)')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('images/exploratory_analysis.png', 
                       dpi=300, bbox_inches='tight')
            print("\nSaved visualization to 'exploratory_analysis.png'")
        
        plt.show()
        
    def save_data(self, filename='data/vix_spy_data.csv'):
        # Save processed data to CSV
        self.combined_data.to_csv(filename)
        print(f"\nSaved data to '{filename}'")


def main():
    print("="*60)
    print("VIX TRADING STRATEGY - DATA COLLECTION")
    print("="*60)
    
    # Initialize collector
    collector = VIXDataCollector(start_date='2015-01-01')
    
    # Download data
    data = collector.download_data()
    
    # Calculate features
    data = collector.calculate_features()
    
    # Get summary statistics
    collector.get_summary_statistics()
    
    # Create visualizations
    collector.create_visualizations(save_plots=True)
    
    # Save data
    collector.save_data()
    
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
