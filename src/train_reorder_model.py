"""
Inventory Reorder Model
Trains classification models to predict reorder necessity
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib
import os

def train_reorder_model(data_path='data/supply_chain_data.csv', 
                       demand_model_path='models/demand_model.pkl',
                       model_path='models/reorder_model.pkl'):
    """
    Train classification models for reorder decisions
    
    Parameters:
    -----------
    data_path : str
        Path to the supply chain dataset
    demand_model_path : str
        Path to the trained demand model
    model_path : str
        Path to save the trained reorder model
    
    Returns:
    --------
    dict : Model performance metrics
    """
    
    # Load data
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Load demand model for predictions
    demand_model = joblib.load(demand_model_path)
    scaler = joblib.load(demand_model_path.replace('.pkl', '_scaler.pkl'))
    
    # Predict demand
    feature_cols = ['base_demand', 'seasonality_factor', 'promotion', 'price', 'competitor_price']
    X_demand = df[feature_cols].copy()
    X_demand_scaled = scaler.transform(X_demand)
    df['predicted_demand'] = demand_model.predict(X_demand_scaled)
    
    # Create target: needs_reorder
    # Reorder if inventory < predicted_demand * lead_time (safety stock)
    df['needs_reorder'] = (df['inventory_level'] < df['predicted_demand'] * df['lead_time']).astype(int)
    
    # Features for reorder prediction
    reorder_features = ['inventory_level', 'predicted_demand', 'lead_time', 'stock_out']
    X = df[reorder_features].copy()
    y = df['needs_reorder'].copy()
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature scaling
    reorder_scaler = StandardScaler()
    X_train_scaled = reorder_scaler.fit_transform(X_train)
    X_test_scaled = reorder_scaler.transform(X_test)
    
    # Train Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    
    # Train Decision Tree
    dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    y_pred_dt = dt_model.predict(X_test_scaled)
    
    # Evaluation - Logistic Regression
    lr_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr, zero_division=0),
        'recall': recall_score(y_test, y_pred_lr, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred_lr)
    }
    
    # Evaluation - Decision Tree
    dt_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_dt),
        'precision': precision_score(y_test, y_pred_dt, zero_division=0),
        'recall': recall_score(y_test, y_pred_dt, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred_dt)
    }
    
    # Save best model (Decision Tree typically better for this task)
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
    joblib.dump(dt_model, model_path)
    joblib.dump(reorder_scaler, model_path.replace('.pkl', '_scaler.pkl'))
    
    print("=" * 60)
    print("REORDER DECISION MODEL - CLASSIFICATION")
    print("=" * 60)
    print(f"✓ Models trained. Best model saved: {model_path}")
    
    print(f"\nLogistic Regression Performance:")
    print(f"  Accuracy:  {lr_metrics['accuracy']:.4f}")
    print(f"  Precision: {lr_metrics['precision']:.4f}")
    print(f"  Recall:    {lr_metrics['recall']:.4f}")
    print(f"  Confusion Matrix:\n{lr_metrics['confusion_matrix']}")
    
    print(f"\nDecision Tree Performance:")
    print(f"  Accuracy:  {dt_metrics['accuracy']:.4f}")
    print(f"  Precision: {dt_metrics['precision']:.4f}")
    print(f"  Recall:    {dt_metrics['recall']:.4f}")
    print(f"  Confusion Matrix:\n{dt_metrics['confusion_matrix']}")
    
    return {
        'lr_metrics': lr_metrics,
        'dt_metrics': dt_metrics,
        'dt_model': dt_model,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'y_pred': y_pred_dt
    }

if __name__ == '__main__':
    train_reorder_model()
