import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import ta library, if not available use simple alternatives
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("Warning: 'ta' library not found. Using simplified technical indicators.")

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        """Load cryptocurrency dataset"""
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values(['crypto_name', 'date'])
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Forward fill missing values within each crypto
        df = df.groupby('crypto_name').apply(lambda x: x.fillna(method='ffill')).reset_index(drop=True)
        # Drop rows with remaining missing values
        return df.dropna()
    
    def engineer_features(self, df):
        """Create volatility and technical indicator features"""
        features_df = df.copy()
        
        for crypto in df['crypto_name'].unique():
            mask = df['crypto_name'] == crypto
            crypto_data = df[mask].copy().reset_index(drop=True)
            
            # Price-based features
            crypto_data['price_change'] = crypto_data['close'].pct_change()
            crypto_data['high_low_ratio'] = crypto_data['high'] / crypto_data['low']
            crypto_data['open_close_ratio'] = crypto_data['open'] / crypto_data['close']
            
            # Volatility features (target variable)
            crypto_data['volatility'] = crypto_data['price_change'].rolling(window=7).std()
            crypto_data['volatility_30d'] = crypto_data['price_change'].rolling(window=30).std()
            
            # Moving averages
            crypto_data['ma_7'] = crypto_data['close'].rolling(window=7).mean()
            crypto_data['ma_30'] = crypto_data['close'].rolling(window=30).mean()
            crypto_data['ma_ratio'] = crypto_data['ma_7'] / crypto_data['ma_30']
            
            # Volume features
            crypto_data['volume_ma'] = crypto_data['volume'].rolling(window=7).mean()
            crypto_data['volume_ratio'] = crypto_data['volume'] / crypto_data['volume_ma']
            crypto_data['liquidity_ratio'] = crypto_data['volume'] / crypto_data['marketCap']
            
            # Technical indicators
            if TA_AVAILABLE:
                # Use ta library if available
                if len(crypto_data) >= 14:  # Need at least 14 days for RSI
                    crypto_data['rsi'] = ta.momentum.RSIIndicator(crypto_data['close']).rsi()
                else:
                    crypto_data['rsi'] = 50.0  # Default RSI
                    
                if len(crypto_data) >= 20:  # Need at least 20 days for Bollinger Bands
                    crypto_data['bb_high'] = ta.volatility.BollingerBands(crypto_data['close']).bollinger_hband()
                    crypto_data['bb_low'] = ta.volatility.BollingerBands(crypto_data['close']).bollinger_lband()
                else:
                    crypto_data['bb_high'] = crypto_data['high']
                    crypto_data['bb_low'] = crypto_data['low']
                    
                if len(crypto_data) >= 14:  # Need at least 14 days for ATR
                    crypto_data['atr'] = ta.volatility.AverageTrueRange(crypto_data['high'], crypto_data['low'], crypto_data['close']).average_true_range()
                else:
                    crypto_data['atr'] = crypto_data['high'] - crypto_data['low']
            else:
                # Simple alternatives when ta library is not available
                crypto_data['rsi'] = self._calculate_simple_rsi(crypto_data['close'])
                crypto_data['bb_high'], crypto_data['bb_low'] = self._calculate_simple_bollinger_bands(crypto_data['close'])
                crypto_data['atr'] = self._calculate_simple_atr(crypto_data['high'], crypto_data['low'], crypto_data['close'])
            
            # Update the main dataframe with the new features
            original_indices = df[mask].index
            for col in crypto_data.columns:
                if col not in df.columns:
                    features_df[col] = np.nan
                features_df.loc[mask, col] = crypto_data[col].values
            
        return features_df.dropna()
    
    def normalize_features(self, df, feature_columns):
        """Normalize numerical features"""
        df_normalized = df.copy()
        df_normalized[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        return df_normalized
    
    def _calculate_simple_rsi(self, prices, window=14):
        """Simple RSI calculation without ta library"""
        if len(prices) < window:
            return pd.Series([50.0] * len(prices), index=prices.index)
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)
    
    def _calculate_simple_bollinger_bands(self, prices, window=20, std_dev=2):
        """Simple Bollinger Bands calculation without ta library"""
        if len(prices) < window:
            return prices * 1.1, prices * 0.9  # Simple approximation
        
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        return upper_band.fillna(prices * 1.1), lower_band.fillna(prices * 0.9)
    
    def _calculate_simple_atr(self, high, low, close, window=14):
        """Simple ATR calculation without ta library"""
        if len(high) < 2:
            return high - low
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr.fillna(high - low)
    
    def prepare_data(self, filepath):
        """Complete data preprocessing pipeline"""
        print("Loading data...")
        df = self.load_data(filepath)
        
        print("Handling missing values...")
        df = self.handle_missing_values(df)
        
        print("Engineering features...")
        df = self.engineer_features(df)
        
        # Define feature columns for normalization (only use columns that exist)
        all_feature_columns = ['open', 'high', 'low', 'close', 'volume', 'marketCap',
                              'price_change', 'high_low_ratio', 'open_close_ratio',
                              'ma_7', 'ma_30', 'ma_ratio', 'volume_ma', 'volume_ratio',
                              'liquidity_ratio', 'rsi', 'bb_high', 'bb_low', 'atr']
        
        # Only use columns that actually exist in the dataframe
        feature_columns = [col for col in all_feature_columns if col in df.columns]
        missing_columns = [col for col in all_feature_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Missing feature columns: {missing_columns}")
        
        print(f"Normalizing {len(feature_columns)} features...")
        df = self.normalize_features(df, feature_columns)
        
        # Ensure we have the target variable
        if 'volatility' not in df.columns:
            print("Error: Target variable 'volatility' not found!")
            return None
            
        print(f"* Data preprocessing completed successfully!")
        print(f"* Final dataset shape: {df.shape}")
        print(f"* Features available: {len(feature_columns)}")
        
        return df

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    # df = preprocessor.prepare_data('data/dataset.csv')
    # df.to_csv('data/processed_data.csv', index=False)
    print("Data preprocessing module ready!")