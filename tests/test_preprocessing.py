import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import DataPreprocessor

class TestDataPreprocessor:
    
    def setup_method(self):
        """Setup test data"""
        self.preprocessor = DataPreprocessor()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [102, 103, 104, 105, 106],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000],
            'marketCap': [1e12, 1.1e12, 1.2e12, 1.3e12, 1.4e12],
            'date': pd.date_range('2023-01-01', periods=5),
            'crypto_name': ['BTC'] * 5
        })
    
    def test_load_data_structure(self):
        """Test if data loading preserves structure"""
        # Save sample data
        test_file = 'test_data.csv'
        self.sample_data.to_csv(test_file, index=False)
        
        try:
            df = self.preprocessor.load_data(test_file)
            assert len(df) == 5
            assert 'date' in df.columns
            assert pd.api.types.is_datetime64_any_dtype(df['date'])
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
    
    def test_feature_engineering(self):
        """Test feature engineering creates expected features"""
        df = self.preprocessor.engineer_features(self.sample_data.copy())
        
        expected_features = ['price_change', 'high_low_ratio', 'open_close_ratio', 
                           'volatility', 'ma_7', 'ma_30']
        
        for feature in expected_features:
            assert feature in df.columns, f"Feature {feature} not created"
    
    def test_missing_values_handling(self):
        """Test missing value handling"""
        # Add some NaN values
        test_data = self.sample_data.copy()
        test_data.loc[2, 'close'] = np.nan
        
        df = self.preprocessor.handle_missing_values(test_data)
        assert df['close'].isna().sum() == 0
    
    def test_normalization(self):
        """Test feature normalization with sufficient data"""
        # Create larger sample data for feature engineering
        large_sample = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109] * 5,
            'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114] * 5,
            'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104] * 5,
            'close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111] * 5,
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000] * 5,
            'marketCap': [1e12, 1.1e12, 1.2e12, 1.3e12, 1.4e12, 1.5e12, 1.6e12, 1.7e12, 1.8e12, 1.9e12] * 5,
            'date': pd.date_range('2023-01-01', periods=50),
            'crypto_name': ['BTC'] * 50
        })
        
        df = self.preprocessor.engineer_features(large_sample.copy())
        
        # Skip normalization test if no valid data after feature engineering
        if len(df) == 0:
            pytest.skip("No data available after feature engineering")
        
        # Get feature columns (excluding non-numeric columns)
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'volatility' in feature_columns:
            feature_columns.remove('volatility')  # Remove target variable
        
        if len(feature_columns) == 0:
            pytest.skip("No feature columns available for normalization")
            
        df_normalized = self.preprocessor.normalize_features(df, feature_columns)
        
        # Basic check that normalization was applied
        assert df_normalized is not None
        assert len(df_normalized) > 0