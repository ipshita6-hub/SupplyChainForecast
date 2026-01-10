"""
Visualization Module
Generates plots for demand forecasting and inventory analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

def plot_actual_vs_predicted(data_path='data/supply_chain_data.csv',
                            demand_model_path='models/demand_model.pkl',
                            output_dir='outputs/plots'):
    """
    Plot actual vs predicted demand over time
    """
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Load model
    demand_model = joblib.load(demand_model_path)
    scaler = joblib.load(demand_model_path.replace('.pkl', '_scaler.pkl'))
    
    # Predict
    feature_cols = ['base_demand', 'seasonality_factor', 'promotion', 'price', 'competitor_price']
    X = df[feature_cols].copy()
    X_scaled = scaler.transform(X)
    df['predicted_demand'] = demand_model.predict(X_scaled)
    
    # Plot for first 3 products
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    for idx, product_id in enumerate(df['product_id'].unique()[:3]):
        product_df = df[df['product_id'] == product_id].sort_values('date')
        
        ax = axes[idx]
        ax.plot(product_df['date'], product_df['actual_demand'], 
               label='Actual Demand', linewidth=2, marker='o', markersize=3, alpha=0.7)
        ax.plot(product_df['date'], product_df['predicted_demand'], 
               label='Predicted Demand', linewidth=2, linestyle='--', marker='s', markersize=3, alpha=0.7)
        
        ax.set_title(f'Product {int(product_id)}: Actual vs Predicted Demand', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Demand (units)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/actual_vs_predicted_demand.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/actual_vs_predicted_demand.png")
    plt.close()

def plot_inventory_vs_demand(data_path='data/supply_chain_data.csv',
                            output_dir='outputs/plots'):
    """
    Plot inventory level vs demand
    """
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    for idx, product_id in enumerate(df['product_id'].unique()[:3]):
        product_df = df[df['product_id'] == product_id].sort_values('date')
        
        ax = axes[idx]
        ax.bar(product_df['date'], product_df['actual_demand'], 
              label='Demand', alpha=0.6, color='coral', width=0.8)
        ax.plot(product_df['date'], product_df['inventory_level'], 
               label='Inventory Level', linewidth=2.5, color='green', marker='o', markersize=4)
        
        # Highlight stockouts
        stockout_dates = product_df[product_df['stock_out'] == 1]['date']
        if len(stockout_dates) > 0:
            ax.scatter(stockout_dates, [0] * len(stockout_dates), 
                      color='red', s=100, marker='X', label='Stockout', zorder=5)
        
        ax.set_title(f'Product {int(product_id)}: Inventory vs Demand', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Units')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/inventory_vs_demand.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/inventory_vs_demand.png")
    plt.close()

def plot_confusion_matrix(data_path='data/supply_chain_data.csv',
                         demand_model_path='models/demand_model.pkl',
                         reorder_model_path='models/reorder_model.pkl',
                         output_dir='outputs/plots'):
    """
    Plot confusion matrix for reorder classification
    """
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Load models
    demand_model = joblib.load(demand_model_path)
    demand_scaler = joblib.load(demand_model_path.replace('.pkl', '_scaler.pkl'))
    reorder_model = joblib.load(reorder_model_path)
    reorder_scaler = joblib.load(reorder_model_path.replace('.pkl', '_scaler.pkl'))
    
    # Predict demand
    feature_cols = ['base_demand', 'seasonality_factor', 'promotion', 'price', 'competitor_price']
    X_demand = df[feature_cols].copy()
    X_demand_scaled = demand_scaler.transform(X_demand)
    df['predicted_demand'] = demand_model.predict(X_demand_scaled)
    
    # Create target
    df['needs_reorder'] = (df['inventory_level'] < df['predicted_demand'] * df['lead_time']).astype(int)
    
    # Predict reorder
    reorder_features = ['inventory_level', 'predicted_demand', 'lead_time', 'stock_out']
    X_reorder = df[reorder_features].copy()
    X_reorder_scaled = reorder_scaler.transform(X_reorder)
    y_pred = reorder_model.predict(X_reorder_scaled)
    
    # Confusion matrix
    cm = confusion_matrix(df['needs_reorder'], y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
               xticklabels=['No Reorder', 'Reorder'],
               yticklabels=['No Reorder', 'Reorder'])
    
    ax.set_title('Reorder Decision - Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/confusion_matrix.png")
    plt.close()

def plot_feature_importance(data_path='data/supply_chain_data.csv',
                           demand_model_path='models/demand_model.pkl',
                           output_dir='outputs/plots'):
    """
    Plot feature importance for demand model
    """
    
    demand_model = joblib.load(demand_model_path)
    
    feature_cols = ['base_demand', 'seasonality_factor', 'promotion', 'price', 'competitor_price']
    coefficients = demand_model.coef_
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['green' if c > 0 else 'red' for c in coefficients]
    bars = ax.barh(feature_cols, coefficients, color=colors, alpha=0.7)
    
    ax.set_title('Demand Model - Feature Coefficients', fontsize=14, fontweight='bold')
    ax.set_xlabel('Coefficient Value')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    # Add value labels
    for i, (bar, coef) in enumerate(zip(bars, coefficients)):
        ax.text(coef, i, f' {coef:.4f}', va='center', ha='left' if coef > 0 else 'right')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/feature_importance.png")
    plt.close()

def plot_seasonality_analysis(data_path='data/supply_chain_data.csv',
                             output_dir='outputs/plots'):
    """
    Plot seasonality patterns
    """
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Weekly seasonality
    weekly_demand = df.groupby('day_of_week')['actual_demand'].mean()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_demand = weekly_demand.reindex(day_order)
    
    axes[0].bar(range(len(weekly_demand)), weekly_demand.values, color='skyblue', alpha=0.7)
    axes[0].set_xticks(range(len(weekly_demand)))
    axes[0].set_xticklabels(weekly_demand.index, rotation=45)
    axes[0].set_title('Weekly Seasonality Pattern', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Average Demand (units)')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Monthly seasonality
    monthly_demand = df.groupby('month')['actual_demand'].mean()
    axes[1].plot(monthly_demand.index, monthly_demand.values, marker='o', linewidth=2.5, markersize=8, color='darkgreen')
    axes[1].fill_between(monthly_demand.index, monthly_demand.values, alpha=0.3, color='green')
    axes[1].set_title('Yearly Seasonality Pattern', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Average Demand (units)')
    axes[1].set_xticks(range(1, 13))
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/seasonality_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/seasonality_analysis.png")
    plt.close()

def generate_all_visualizations(data_path='data/supply_chain_data.csv',
                               demand_model_path='models/demand_model.pkl',
                               reorder_model_path='models/reorder_model.pkl',
                               output_dir='outputs/plots'):
    """
    Generate all visualizations
    """
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_actual_vs_predicted(data_path, demand_model_path, output_dir)
    plot_inventory_vs_demand(data_path, output_dir)
    plot_confusion_matrix(data_path, demand_model_path, reorder_model_path, output_dir)
    plot_feature_importance(data_path, demand_model_path, output_dir)
    plot_seasonality_analysis(data_path, output_dir)
    
    print("✓ All visualizations generated successfully!")

if __name__ == '__main__':
    generate_all_visualizations()
