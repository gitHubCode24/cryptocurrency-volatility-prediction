import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class EDAAnalyzer:
    def __init__(self, df):
        self.df = df
        
    def basic_statistics(self):
        """Generate basic dataset statistics"""
        print("Dataset Shape:", self.df.shape)
        print("\nCryptocurrency Count:", self.df['crypto_name'].nunique())
        print("Cryptocurrencies:", self.df['crypto_name'].unique()[:10])
        print("\nDate Range:", self.df['date'].min(), "to", self.df['date'].max())
        print("\nBasic Statistics:")
        return self.df.describe()
    
    def missing_values_analysis(self):
        """Analyze missing values"""
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_percent
        })
        return missing_df[missing_df['Missing Count'] > 0]
    
    def price_trends_analysis(self):
        """Analyze price trends for top cryptocurrencies"""
        # Get top 5 cryptocurrencies by market cap
        top_cryptos = self.df.groupby('crypto_name')['marketCap'].mean().nlargest(5).index
        
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Price Trends', 'Volume Trends'))
        
        for crypto in top_cryptos:
            crypto_data = self.df[self.df['crypto_name'] == crypto]
            
            fig.add_trace(go.Scatter(x=crypto_data['date'], y=crypto_data['close'],
                                   name=f'{crypto} Price', mode='lines'), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=crypto_data['date'], y=crypto_data['volume'],
                                   name=f'{crypto} Volume', mode='lines'), row=2, col=1)
        
        fig.update_layout(height=800, title_text="Top 5 Cryptocurrencies Analysis")
        return fig
    
    def volatility_analysis(self):
        """Analyze volatility patterns"""
        if 'volatility' in self.df.columns:
            plt.figure(figsize=(15, 10))
            
            # Volatility distribution
            plt.subplot(2, 2, 1)
            self.df['volatility'].hist(bins=50)
            plt.title('Volatility Distribution')
            plt.xlabel('Volatility')
            
            # Volatility by cryptocurrency
            plt.subplot(2, 2, 2)
            crypto_count = self.df['crypto_name'].nunique()
            if crypto_count > 1:
                top_cryptos = self.df.groupby('crypto_name')['marketCap'].mean().nlargest(10).index
                volatility_by_crypto = self.df[self.df['crypto_name'].isin(top_cryptos)].groupby('crypto_name')['volatility'].mean()
                volatility_by_crypto.plot(kind='bar')
                plt.title(f'Average Volatility by Cryptocurrency ({len(volatility_by_crypto)} cryptos)')
            else:
                # Show volatility over time for single crypto
                monthly_vol = self.df.groupby(self.df['date'].dt.to_period('M'))['volatility'].mean()
                monthly_vol.plot(kind='line')
                plt.title(f'Volatility Over Time - {self.df["crypto_name"].iloc[0]}')
            plt.xticks(rotation=45)
            
            # Volatility over time
            plt.subplot(2, 2, 3)
            monthly_volatility = self.df.groupby(self.df['date'].dt.to_period('M'))['volatility'].mean()
            monthly_volatility.plot()
            plt.title('Average Volatility Over Time')
            plt.xlabel('Date')
            
            # Volatility vs Volume correlation
            plt.subplot(2, 2, 4)
            plt.scatter(self.df['volume'], self.df['volatility'], alpha=0.5)
            plt.title('Volatility vs Volume')
            plt.xlabel('Volume')
            plt.ylabel('Volatility')
            
            plt.tight_layout()
            return plt
    
    def technical_indicators_analysis(self):
        """Analyze technical indicators"""
        fig = make_subplots(rows=2, cols=2, 
                           subplot_titles=('RSI Trends', 'Moving Averages', 'Bollinger Bands', 'ATR'))
        
        top_crypto = self.df.groupby('crypto_name')['marketCap'].mean().nlargest(1).index[0]
        crypto_data = self.df[self.df['crypto_name'] == top_crypto].head(200)
        
        # RSI
        if 'rsi' in crypto_data.columns:
            fig.add_trace(go.Scatter(x=crypto_data['date'], y=crypto_data['rsi'],
                                   name='RSI', line=dict(color='purple')), row=1, col=1)
        
        # Moving Averages
        if 'ma_7' in crypto_data.columns and 'ma_30' in crypto_data.columns:
            fig.add_trace(go.Scatter(x=crypto_data['date'], y=crypto_data['ma_7'],
                                   name='MA 7', line=dict(color='blue')), row=1, col=2)
            fig.add_trace(go.Scatter(x=crypto_data['date'], y=crypto_data['ma_30'],
                                   name='MA 30', line=dict(color='red')), row=1, col=2)
        
        # Bollinger Bands
        if 'bb_high' in crypto_data.columns and 'bb_low' in crypto_data.columns:
            fig.add_trace(go.Scatter(x=crypto_data['date'], y=crypto_data['bb_high'],
                                   name='BB High', line=dict(color='green')), row=2, col=1)
            fig.add_trace(go.Scatter(x=crypto_data['date'], y=crypto_data['bb_low'],
                                   name='BB Low', line=dict(color='orange')), row=2, col=1)
        
        # ATR
        if 'atr' in crypto_data.columns:
            fig.add_trace(go.Scatter(x=crypto_data['date'], y=crypto_data['atr'],
                                   name='ATR', line=dict(color='brown')), row=2, col=2)
        
        fig.update_layout(height=800, title_text=f"Technical Indicators - {top_crypto}")
        return fig
    
    def market_cap_analysis(self):
        """Analyze market cap trends"""
        plt.figure(figsize=(15, 10))
        
        # Market cap distribution
        plt.subplot(2, 2, 1)
        self.df['marketCap'].hist(bins=50, log=True)
        plt.title('Market Cap Distribution (Log Scale)')
        plt.xlabel('Market Cap')
        
        # Top cryptocurrencies by market cap
        plt.subplot(2, 2, 2)
        crypto_count = self.df['crypto_name'].nunique()
        if crypto_count > 1:
            top_cryptos = self.df.groupby('crypto_name')['marketCap'].mean().nlargest(10)
            top_cryptos.plot(kind='bar')
            plt.title(f'Top {len(top_cryptos)} Cryptocurrencies by Market Cap')
        else:
            # Show market cap over time for single crypto
            monthly_cap = self.df.groupby(self.df['date'].dt.to_period('M'))['marketCap'].mean()
            monthly_cap.plot(kind='line')
            plt.title(f'Market Cap Over Time - {self.df["crypto_name"].iloc[0]}')
        plt.xticks(rotation=45)
        
        # Market cap vs volatility
        plt.subplot(2, 2, 3)
        if 'volatility' in self.df.columns:
            plt.scatter(self.df['marketCap'], self.df['volatility'], alpha=0.5)
            plt.xscale('log')
            plt.title('Market Cap vs Volatility')
            plt.xlabel('Market Cap (Log Scale)')
            plt.ylabel('Volatility')
        
        # Market cap over time
        plt.subplot(2, 2, 4)
        monthly_market_cap = self.df.groupby(self.df['date'].dt.to_period('M'))['marketCap'].mean()
        monthly_market_cap.plot()
        plt.title('Average Market Cap Over Time')
        plt.xlabel('Date')
        
        plt.tight_layout()
        return plt
    
    def generate_eda_report(self, save_path='docs/eda_report.html'):
        """Generate comprehensive EDA report"""
        print("=== CRYPTOCURRENCY VOLATILITY PREDICTION - EDA REPORT ===\n")
        
        # Basic statistics
        print("1. BASIC DATASET STATISTICS")
        print("=" * 40)
        stats = self.basic_statistics()
        
        # Missing values
        print("\n2. MISSING VALUES ANALYSIS")
        print("=" * 40)
        missing = self.missing_values_analysis()
        if len(missing) > 0:
            print(missing)
        else:
            print("No missing values found!")
        
        # Generate visualizations
        print("\n3. GENERATING VISUALIZATIONS...")
        print("=" * 40)
        
        # Price trends
        price_fig = self.price_trends_analysis()
        price_fig.write_html(f"{save_path.replace('.html', '_price_trends.html')}")
        
        # Volatility analysis
        volatility_plt = self.volatility_analysis()
        if volatility_plt:
            volatility_plt.savefig(f"{save_path.replace('.html', '_volatility.png')}")
        
        # Technical indicators analysis
        tech_fig = self.technical_indicators_analysis()
        tech_fig.write_html(f"{save_path.replace('.html', '_technical.html')}")
        
        # Market cap analysis
        market_plt = self.market_cap_analysis()
        if market_plt:
            market_plt.savefig(f"{save_path.replace('.html', '_market.png')}")
        
        print("EDA Report generated successfully!")
        return stats, missing

if __name__ == "__main__":
    # df = pd.read_csv('data/processed_data.csv')
    # analyzer = EDAAnalyzer(df)
    # analyzer.generate_eda_report()
    print("EDA module ready!")