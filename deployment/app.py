import joblib
import pandas as pd
import numpy as np
import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
model = None
feature_columns = ['open', 'high', 'low', 'close', 'volume', 'marketCap',
                  'price_change', 'high_low_ratio', 'open_close_ratio',
                  'ma_7', 'ma_30', 'ma_ratio', 'volume_ma', 'volume_ratio',
                  'liquidity_ratio', 'rsi', 'bb_high', 'bb_low', 'atr']

def load_model():
    """Load the best trained model"""
    global model
    model_files = [f for f in os.listdir(MODEL_PATH) if f.startswith('best_model_') and f.endswith('.pkl')]
    if model_files:
        model_path = os.path.join(MODEL_PATH, model_files[0])
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return True
    else:
        print("No trained model found!")
        return False

def predict_volatility(data):
    """Make volatility prediction"""
    try:
        if model is None:
            return {'error': 'Model not loaded'}
        
        # Calculate derived features
        price_change = (float(data['close']) - float(data['open'])) / float(data['open'])
        high_low_ratio = float(data['high']) / float(data['low'])
        open_close_ratio = float(data['open']) / float(data['close'])
        
        # Create feature vector
        features = {
            'open': float(data['open']),
            'high': float(data['high']),
            'low': float(data['low']),
            'close': float(data['close']),
            'volume': float(data['volume']),
            'marketCap': float(data['marketCap']),
            'price_change': price_change,
            'high_low_ratio': high_low_ratio,
            'open_close_ratio': open_close_ratio,
            'ma_7': float(data['close']),
            'ma_30': float(data['close']),
            'ma_ratio': 1.0,
            'volume_ma': float(data['volume']),
            'volume_ratio': 1.0,
            'liquidity_ratio': float(data['volume']) / float(data['marketCap']),
            'rsi': 50.0,
            'bb_high': float(data['high']),
            'bb_low': float(data['low']),
            'atr': float(data['high']) - float(data['low'])
        }
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Make prediction
        volatility = model.predict(df)[0]
        
        # Determine risk level
        if volatility < 0.02:
            risk_level = "Low"
        elif volatility < 0.05:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            'volatility': float(volatility),
            'risk_level': risk_level
        }
        
    except Exception as e:
        return {'error': str(e)}

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Cryptocurrency Volatility Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .form-group { margin: 15px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        .result { margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 4px; }
        .error { background-color: #f8d7da; color: #721c24; }
        .success { background-color: #d4edda; color: #155724; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Cryptocurrency Volatility Predictor</h1>
        <p>Enter cryptocurrency data to predict volatility:</p>
        
        <form id="predictionForm">
            <div class="form-group">
                <label>Open Price:</label>
                <input type="number" step="any" name="open" required>
            </div>
            <div class="form-group">
                <label>High Price:</label>
                <input type="number" step="any" name="high" required>
            </div>
            <div class="form-group">
                <label>Low Price:</label>
                <input type="number" step="any" name="low" required>
            </div>
            <div class="form-group">
                <label>Close Price:</label>
                <input type="number" step="any" name="close" required>
            </div>
            <div class="form-group">
                <label>Volume:</label>
                <input type="number" step="any" name="volume" required>
            </div>
            <div class="form-group">
                <label>Market Cap:</label>
                <input type="number" step="any" name="marketCap" required>
            </div>
            <button type="submit">Predict Volatility</button>
        </form>
        
        <div id="result"></div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                const resultDiv = document.getElementById('result');
                
                if (result.error) {
                    resultDiv.innerHTML = `<div class="result error">Error: ${result.error}</div>`;
                } else {
                    resultDiv.innerHTML = `<div class="result success">
                        <h3>Prediction Result</h3>
                        <p><strong>Predicted Volatility:</strong> ${result.volatility.toFixed(6)}</p>
                        <p><strong>Risk Level:</strong> ${result.risk_level}</p>
                    </div>`;
                }
            } catch (error) {
                document.getElementById('result').innerHTML = `<div class="result error">Error: ${error.message}</div>`;
            }
        });
    </script>
</body>
</html>
"""

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'status': 'healthy', 'model_loaded': model is not None}
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                result = predict_volatility(data)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                error_response = {'error': str(e)}
                self.wfile.write(json.dumps(error_response).encode())
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    if load_model():
        server = HTTPServer(('localhost', 5000), RequestHandler)
        print("Cryptocurrency Volatility Predictor running on http://localhost:5000")
        print("Press Ctrl+C to stop the server")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
            server.server_close()
    else:
        print("Failed to load model. Exiting.")