import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import optional libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available")

class VolatilityPredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_columns = None
        
    def prepare_features_target(self, df):
        """Prepare features and target variable"""
        # Define feature columns (only use columns that exist)
        all_feature_columns = ['open', 'high', 'low', 'close', 'volume', 'marketCap',
                              'price_change', 'high_low_ratio', 'open_close_ratio',
                              'ma_7', 'ma_30', 'ma_ratio', 'volume_ma', 'volume_ratio',
                              'liquidity_ratio', 'rsi', 'bb_high', 'bb_low', 'atr']
        
        # Only use columns that actually exist in the dataframe
        self.feature_columns = [col for col in all_feature_columns if col in df.columns]
        missing_columns = [col for col in all_feature_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Missing feature columns: {missing_columns}")
        
        print(f"Using {len(self.feature_columns)} features for training")
        
        # Remove rows where target is missing
        df_clean = df.dropna(subset=['volatility'])
        
        X = df_clean[self.feature_columns]
        y = df_clean['volatility']
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def initialize_models(self):
        """Initialize different ML models"""
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=1)
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1, n_jobs=1)
    
    def train_models(self, X_train, y_train):
        """Train all models"""
        self.initialize_models()
        trained_models = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
            
        return trained_models
    
    def evaluate_models(self, models, X_test, y_test):
        """Evaluate all models and return performance metrics"""
        results = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            
            results[name] = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            
        return results
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for best models"""
        tuned_models = {}
        
        # XGBoost tuning (if available)
        if XGBOOST_AVAILABLE:
            print("Tuning XGBoost...")
            xgb_params = {
                'n_estimators': [50, 100],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1]
            }
            
            xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=1)
            xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=3, scoring='neg_mean_squared_error')
            xgb_grid.fit(X_train, y_train)
            tuned_models['xgboost_tuned'] = xgb_grid.best_estimator_
        
        # Random Forest tuning
        print("Tuning Random Forest...")
        rf_params = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
        }
        
        rf_model = RandomForestRegressor(random_state=42, n_jobs=1)
        rf_grid = GridSearchCV(rf_model, rf_params, cv=3, scoring='neg_mean_squared_error')
        rf_grid.fit(X_train, y_train)
        tuned_models['random_forest_tuned'] = rf_grid.best_estimator_
        
        return tuned_models
    
    def select_best_model(self, results):
        """Select best model based on RMSE"""
        best_model_name = min(results.keys(), key=lambda x: results[x]['rmse'])
        return best_model_name
    
    def save_model(self, model, model_name, filepath):
        """Save trained model"""
        joblib.dump(model, f"{filepath}/{model_name}.pkl")
        print(f"Model saved as {model_name}.pkl")
    
    def train_and_evaluate(self, df):
        """Complete training and evaluation pipeline"""
        print("Preparing features and target...")
        X, y = self.prepare_features_target(df)
        
        print("Splitting data...")
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        print("Training models...")
        trained_models = self.train_models(X_train, y_train)
        
        print("Evaluating models...")
        results = self.evaluate_models(trained_models, X_test, y_test)
        
        print("Performing hyperparameter tuning...")
        tuned_models = self.hyperparameter_tuning(X_train, y_train)
        
        # Evaluate tuned models
        tuned_results = self.evaluate_models(tuned_models, X_test, y_test)
        
        # Combine results
        all_results = {**results, **tuned_results}
        
        # Select best model
        best_model_name = self.select_best_model(all_results)
        
        if best_model_name in tuned_models:
            self.best_model = tuned_models[best_model_name]
        else:
            self.best_model = trained_models[best_model_name]
        
        print(f"\nBest Model: {best_model_name}")
        print(f"RMSE: {all_results[best_model_name]['rmse']:.4f}")
        print(f"MAE: {all_results[best_model_name]['mae']:.4f}")
        print(f"R²: {all_results[best_model_name]['r2']:.4f}")
        
        # Save best model
        self.save_model(self.best_model, f"best_model_{best_model_name}", "models")
        
        return all_results, self.best_model

if __name__ == "__main__":
    # df = pd.read_csv('data/processed_data.csv')
    # predictor = VolatilityPredictor()
    # results, best_model = predictor.train_and_evaluate(df)
    print("Model training module ready!")