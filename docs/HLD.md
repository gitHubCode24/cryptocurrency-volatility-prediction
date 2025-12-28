# Pipeline Architecture Document
## Cryptocurrency Volatility Prediction System

### 1. Pipeline Overview

```
Raw Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Deployment
    ↓           ↓              ↓                 ↓              ↓           ↓
  CSV File   Clean Data    19 Features      5+ Models      Best Model   Web API
```

### 2. Detailed Data Flow

#### Stage 1: Data Ingestion
```
Input: dataset.csv
├── Columns: open, high, low, close, volume, marketCap, timestamp, crypto_name, date
├── Format: Time-series data with multiple cryptocurrencies
└── Size: Variable (tested with 70K+ records)
```

#### Stage 2: Data Preprocessing
```
Raw Data Processing:
├── Load CSV with pandas
├── Convert date column to datetime
├── Sort by crypto_name and date
├── Handle missing values (forward fill)
└── Group by cryptocurrency for processing
```

#### Stage 3: Feature Engineering
```
For each cryptocurrency:
├── Price Features:
│   ├── price_change = (close - open) / open
│   ├── high_low_ratio = high / low
│   └── open_close_ratio = open / close
├── Volatility Features (Target):
│   ├── volatility = rolling_std(price_change, 7)
│   └── volatility_30d = rolling_std(price_change, 30)
├── Moving Averages:
│   ├── ma_7 = rolling_mean(close, 7)
│   ├── ma_30 = rolling_mean(close, 30)
│   └── ma_ratio = ma_7 / ma_30
├── Volume Features:
│   ├── volume_ma = rolling_mean(volume, 7)
│   ├── volume_ratio = volume / volume_ma
│   └── liquidity_ratio = volume / marketCap
└── Technical Indicators:
    ├── rsi = Relative Strength Index
    ├── bb_high/bb_low = Bollinger Bands
    └── atr = Average True Range
```

#### Stage 4: Data Normalization
```
Feature Scaling:
├── StandardScaler for all numerical features
├── Fit on training data
├── Transform both train and test sets
└── Preserve original target variable
```

#### Stage 5: Model Training Pipeline
```
Model Training Flow:
├── Feature-Target Separation
├── Train-Test Split (80-20)
├── Model Initialization:
│   ├── Linear Regression
│   ├── Ridge Regression
│   ├── Random Forest
│   ├── Gradient Boosting
│   └── XGBoost (if available)
├── Model Training
├── Performance Evaluation
├── Hyperparameter Tuning (top 2 models)
└── Best Model Selection (lowest RMSE)
```

#### Stage 6: Model Evaluation
```
Evaluation Metrics:
├── RMSE (Root Mean Square Error)
├── MAE (Mean Absolute Error)
├── R² Score (Coefficient of Determination)
└── Model Comparison Report
```

#### Stage 7: Deployment Pipeline
```
Deployment Flow:
├── Model Serialization (joblib.dump)
├── HTTP Server Setup
├── API Endpoint Creation:
│   ├── GET / → Web Interface
│   ├── POST /predict → Volatility Prediction
│   └── GET /health → System Status
└── Real-time Prediction Service
```

### 3. Component Interactions

#### 3.1 Main Pipeline (main.py)
```python
CryptoPipeline:
├── DataPreprocessor.prepare_data()
├── EDAAnalyzer.generate_eda_report()
├── VolatilityPredictor.train_and_evaluate()
└── generate_final_report()
```

#### 3.2 Preprocessing Pipeline
```python
DataPreprocessor.prepare_data():
├── load_data() → Raw DataFrame
├── handle_missing_values() → Clean DataFrame
├── engineer_features() → Feature-rich DataFrame
└── normalize_features() → Scaled DataFrame
```

#### 3.3 Training Pipeline
```python
VolatilityPredictor.train_and_evaluate():
├── prepare_features_target() → X, y
├── split_data() → X_train, X_test, y_train, y_test
├── train_models() → trained_models dict
├── evaluate_models() → results dict
├── hyperparameter_tuning() → tuned_models dict
└── select_best_model() → best_model
```

### 4. Data Validation & Quality Checks

#### 4.1 Input Validation
- Required columns presence check
- Data type validation
- Date format verification
- Missing value assessment

#### 4.2 Feature Validation
- Feature creation success verification
- NaN handling in engineered features
- Feature correlation analysis
- Target variable availability check

#### 4.3 Model Validation
- Training data sufficiency check
- Model convergence verification
- Performance threshold validation
- Cross-validation consistency

### 5. Error Handling Strategy

#### 5.1 Library Dependencies
- Optional library imports (ta, lightgbm)
- Fallback implementations for missing libraries
- Graceful degradation of functionality

#### 5.2 Runtime Errors
- File path resolution across OS
- Unicode character handling
- Memory management for large datasets
- Model loading validation

### 6. Performance Considerations

#### 6.1 Computational Efficiency
- Single-threaded processing (n_jobs=1)
- Batch processing by cryptocurrency
- Memory-efficient data operations
- Selective feature computation

#### 6.2 Scalability
- Modular component design
- Configurable parameters
- Extensible model framework
- Flexible deployment options

### 7. Security & Reliability

#### 7.1 Input Sanitization
- Numeric input validation
- Range checking for financial data
- SQL injection prevention (no database)
- XSS protection in web interface

#### 7.2 Model Reliability
- Multiple model comparison
- Cross-validation for robustness
- Performance monitoring
- Graceful failure handling