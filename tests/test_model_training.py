import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_training import VolatilityPredictor

class TestVolatilityPredictor:
    
    def setup_method(self):
        """Setup test data"""
        self.predictor = VolatilityPredictor()
        
        # Create sample processed data
        np.random.seed(42)
        n_samples = 100
        
        self.sample_data = pd.DataFrame({
            'open': np.random.normal(100, 10, n_samples),
            'high': np.random.normal(105, 10, n_samples),
            'low': np.random.normal(95, 10, n_samples),
            'close': np.random.normal(102, 10, n_samples),
            'volume': np.random.normal(1000000, 100000, n_samples),
            'marketCap': np.random.normal(1e12, 1e11, n_samples),
            'price_change': np.random.normal(0.01, 0.05, n_samples),
            'high_low_ratio': np.random.normal(1.1, 0.1, n_samples),
            'open_close_ratio': np.random.normal(1.0, 0.05, n_samples),
            'ma_7': np.random.normal(100, 10, n_samples),
            'ma_30': np.random.normal(100, 10, n_samples),
            'ma_ratio': np.random.normal(1.0, 0.1, n_samples),
            'volume_ma': np.random.normal(1000000, 100000, n_samples),
            'volume_ratio': np.random.normal(1.0, 0.2, n_samples),
            'liquidity_ratio': np.random.normal(0.001, 0.0001, n_samples),
            'rsi': np.random.uniform(20, 80, n_samples),
            'bb_high': np.random.normal(110, 10, n_samples),
            'bb_low': np.random.normal(90, 10, n_samples),
            'atr': np.random.normal(5, 1, n_samples),
            'volatility': np.random.normal(0.03, 0.01, n_samples)  # Target
        })
    
    def test_prepare_features_target(self):
        """Test feature-target separation"""
        X, y = self.predictor.prepare_features_target(self.sample_data)
        
        assert len(X) == len(self.sample_data)
        assert len(y) == len(self.sample_data)
        assert 'volatility' not in X.columns
        assert isinstance(y, pd.Series)
    
    def test_model_training(self):
        """Test if models can be trained"""
        X, y = self.predictor.prepare_features_target(self.sample_data)
        X_train, X_test, y_train, y_test = self.predictor.split_data(X, y)
        
        models = self.predictor.train_models(X_train, y_train)
        
        assert len(models) > 0
        assert 'linear_regression' in models
        
        # Test if models can make predictions
        for name, model in models.items():
            predictions = model.predict(X_test)
            assert len(predictions) == len(y_test)
    
    def test_model_evaluation(self):
        """Test model evaluation metrics"""
        X, y = self.predictor.prepare_features_target(self.sample_data)
        X_train, X_test, y_train, y_test = self.predictor.split_data(X, y)
        models = self.predictor.train_models(X_train, y_train)
        
        results = self.predictor.evaluate_models(models, X_test, y_test)
        
        assert len(results) > 0
        for model_name, metrics in results.items():
            assert 'rmse' in metrics
            assert 'mae' in metrics
            assert 'r2' in metrics
            assert metrics['rmse'] > 0
            assert metrics['mae'] > 0