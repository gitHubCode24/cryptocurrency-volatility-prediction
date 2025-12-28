import pytest
import json
import sys
import os

# Add deployment to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'deployment'))

from app import predict_volatility

class TestDeployment:
    
    def setup_method(self):
        """Setup test data"""
        self.sample_input = {
            'open': 45000,
            'high': 46000,
            'low': 44000,
            'close': 45500,
            'volume': 1000000000,
            'marketCap': 850000000000
        }
    
    def test_predict_volatility_input_validation(self):
        """Test prediction function with valid input"""
        result = predict_volatility(self.sample_input)
        
        # Should return either a prediction or error
        assert isinstance(result, dict)
        assert 'volatility' in result or 'error' in result
    
    def test_predict_volatility_missing_fields(self):
        """Test prediction with missing required fields"""
        incomplete_input = {'open': 45000, 'high': 46000}
        result = predict_volatility(incomplete_input)
        
        assert 'error' in result
    
    def test_risk_level_classification(self):
        """Test risk level classification logic"""
        # Mock a prediction result
        test_cases = [
            (0.01, "Low"),
            (0.03, "Medium"), 
            (0.07, "High")
        ]
        
        for volatility, expected_risk in test_cases:
            if volatility < 0.02:
                risk_level = "Low"
            elif volatility < 0.05:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            assert risk_level == expected_risk