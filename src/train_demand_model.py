"""
Demand Forecasting Model
Trains Linear Regression to predict product demand
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

def train_demand_model(data_path='data/supply_chain_data.csv', model_path='models/demand_model.pkl'):
    """
    Train Linear Regression model for demand forecasting
    
    Parameters:
    -----------
    data_path : str
        Path to the supply chain dataset
    model_path : str
        Path to save the trained model
    
    Returns:
    --------
    dict : Model performance metrics
    """
    
    # Load data
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Features for demand prediction
    feature_cols = ['base_demand', 'seasonality_factor', 'promotion', 'price', 'competitor_price']
    X = df[feature_cols].copy()
    y = df['actual_demand'].copy()
    
    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Evaluation metrics
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    metrics = {
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'mae_test': mae_test,
        'r2_test': r2_test,
        'feature_importance': dict(zip(feature_cols, model.coef_))
    }
    
    # Save model and scaler
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, model_path.replace('.pkl', '_scaler.pkl'))
    
    print("=" * 60)
    print("DEMAND FORECASTING MODEL - LINEAR REGRESSION")
    print("=" * 60)
    print(f"✓ Model trained and saved: {model_path}")
    print(f"\nPerformance Metrics:")
    print(f"  RMSE (Train): {rmse_train:.2f}")
    print(f"  RMSE (Test):  {rmse_test:.2f}")
    print(f"  MAE (Test):   {mae_test:.2f}")
    print(f"  R² Score:     {r2_test:.4f}")
    print(f"\nFeature Coefficients:")
    for feat, coef in metrics['feature_importance'].items():
        print(f"  {feat}: {coef:.4f}")
    
    return metrics, model, scaler, X_test, y_test, y_pred_test

if __name__ == '__main__':
    train_demand_model()
