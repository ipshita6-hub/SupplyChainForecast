"""
Prediction Module
Makes predictions on new data and generates business recommendations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os

def predict_future_demand(days=30, 
                         data_path='data/supply_chain_data.csv',
                         demand_model_path='models/demand_model.pkl'):
    """
    Predict demand for the next N days
    
    Parameters:
    -----------
    days : int
        Number of days to forecast
    data_path : str
        Path to the supply chain dataset
    demand_model_path : str
        Path to the trained demand model
    
    Returns:
    --------
    pd.DataFrame : Future demand predictions
    """
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Load models
    demand_model = joblib.load(demand_model_path)
    scaler = joblib.load(demand_model_path.replace('.pkl', '_scaler.pkl'))
    
    # Get last date and product info
    last_date = df['date'].max()
    products = df['product_id'].unique()
    
    # Generate future dates
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    
    predictions = []
    
    for product_id in products:
        product_data = df[df['product_id'] == product_id].iloc[-1]
        
        for future_date in future_dates:
            # Calculate seasonality for future date
            day_of_year = future_date.timetuple().tm_yday
            yearly_seasonality = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
            
            day_of_week = future_date.weekday()
            weekly_seasonality = 1.2 if day_of_week >= 4 else 0.9
            
            seasonality_factor = yearly_seasonality * weekly_seasonality
            
            # Assume no promotion by default
            promotion = 0
            price = product_data['price']
            competitor_price = product_data['competitor_price']
            base_demand = product_data['base_demand']
            
            # Create feature vector
            features = np.array([[base_demand, seasonality_factor, promotion, price, competitor_price]])
            features_scaled = scaler.transform(features)
            
            # Predict
            predicted_demand = demand_model.predict(features_scaled)[0]
            
            predictions.append({
                'date': future_date,
                'product_id': product_id,
                'predicted_demand': max(0, predicted_demand),
                'seasonality_factor': seasonality_factor
            })
    
    future_df = pd.DataFrame(predictions)
    print(f"✓ Generated {len(future_df)} future demand predictions for {days} days")
    
    return future_df

def generate_reorder_recommendations(data_path='data/supply_chain_data.csv',
                                    reorder_model_path='models/reorder_model.pkl',
                                    demand_model_path='models/demand_model.pkl',
                                    output_path='outputs/recommendations.txt'):
    """
    Generate intelligent reorder recommendations
    
    Parameters:
    -----------
    data_path : str
        Path to the supply chain dataset
    reorder_model_path : str
        Path to the trained reorder model
    demand_model_path : str
        Path to the trained demand model
    output_path : str
        Path to save recommendations
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
    
    # Predict reorder need
    reorder_features = ['inventory_level', 'predicted_demand', 'lead_time', 'stock_out']
    X_reorder = df[reorder_features].copy()
    X_reorder_scaled = reorder_scaler.transform(X_reorder)
    df['needs_reorder'] = reorder_model.predict(X_reorder_scaled)
    
    # Get latest data per product
    latest_df = df.sort_values('date').groupby('product_id').tail(1)
    
    # Categorize products
    at_risk = latest_df[latest_df['needs_reorder'] == 1]
    overstocked = latest_df[(latest_df['inventory_level'] > latest_df['predicted_demand'] * 5) & 
                            (latest_df['needs_reorder'] == 0)]
    optimal = latest_df[(~latest_df['product_id'].isin(at_risk['product_id'])) & 
                        (~latest_df['product_id'].isin(overstocked['product_id']))]
    
    # Calculate metrics
    total_inventory = latest_df['inventory_level'].sum()
    total_predicted_demand = latest_df['predicted_demand'].sum()
    stockout_rate = df['stock_out'].mean() * 100
    
    # Generate report
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SUPPLY CHAIN DEMAND FORECASTING & INVENTORY OPTIMIZATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Period: {df['date'].min().date()} to {df['date'].max().date()}\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Products Analyzed: {len(latest_df)}\n")
        f.write(f"Total Current Inventory: {total_inventory:,.2f} units\n")
        f.write(f"Total Predicted Daily Demand: {total_predicted_demand:,.2f} units\n")
        f.write(f"Historical Stockout Rate: {stockout_rate:.2f}%\n")
        f.write(f"Days of Stock Available: {total_inventory / total_predicted_demand:.2f} days\n\n")
        
        f.write("PRODUCTS AT RISK OF STOCKOUT\n")
        f.write("-" * 70 + "\n")
        if len(at_risk) > 0:
            for _, row in at_risk.iterrows():
                reorder_qty = max(0, row['predicted_demand'] * row['lead_time'] * 1.5 - row['inventory_level'])
                f.write(f"\nProduct ID: {int(row['product_id'])}\n")
                f.write(f"  Category: {row['category']}\n")
                f.write(f"  Current Inventory: {row['inventory_level']:.2f} units\n")
                f.write(f"  Predicted Daily Demand: {row['predicted_demand']:.2f} units\n")
                f.write(f"  Lead Time: {int(row['lead_time'])} days\n")
                f.write(f"  RECOMMENDED REORDER QUANTITY: {reorder_qty:.2f} units\n")
                f.write(f"  Risk Level: HIGH\n")
        else:
            f.write("No products at immediate risk.\n")
        
        f.write("\n\nOVERSTOCKED PRODUCTS\n")
        f.write("-" * 70 + "\n")
        if len(overstocked) > 0:
            for _, row in overstocked.iterrows():
                excess = row['inventory_level'] - (row['predicted_demand'] * 3)
                f.write(f"\nProduct ID: {int(row['product_id'])}\n")
                f.write(f"  Category: {row['category']}\n")
                f.write(f"  Current Inventory: {row['inventory_level']:.2f} units\n")
                f.write(f"  Predicted Daily Demand: {row['predicted_demand']:.2f} units\n")
                f.write(f"  Excess Inventory: {excess:.2f} units\n")
                f.write(f"  Recommendation: Consider promotions or clearance sales\n")
        else:
            f.write("No overstocked products identified.\n")
        
        f.write("\n\nOPTIMAL INVENTORY PRODUCTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Number of Products: {len(optimal)}\n")
        f.write(f"Average Inventory Level: {optimal['inventory_level'].mean():.2f} units\n")
        f.write(f"Average Days of Stock: {(optimal['inventory_level'] / optimal['predicted_demand']).mean():.2f} days\n")
        
        f.write("\n\nBUSINESS IMPACT & COST SAVINGS\n")
        f.write("-" * 70 + "\n")
        
        # Calculate potential savings
        current_holding_cost = total_inventory * 0.05  # 5% annual holding cost
        optimized_inventory = total_predicted_demand * 3  # Target 3 days of stock
        optimized_holding_cost = optimized_inventory * 0.05
        potential_savings = current_holding_cost - optimized_holding_cost
        
        f.write(f"Current Annual Holding Cost: ${current_holding_cost:,.2f}\n")
        f.write(f"Optimized Annual Holding Cost: ${optimized_holding_cost:,.2f}\n")
        f.write(f"Potential Annual Savings: ${potential_savings:,.2f}\n")
        f.write(f"Stockout Prevention Value: Reduced lost sales and customer dissatisfaction\n")
        
        f.write("\n\nACTION ITEMS\n")
        f.write("-" * 70 + "\n")
        f.write("1. IMMEDIATE: Reorder products at risk of stockout\n")
        f.write("2. SHORT-TERM: Implement promotional campaigns for overstocked items\n")
        f.write("3. MEDIUM-TERM: Optimize lead times with suppliers\n")
        f.write("4. LONG-TERM: Implement automated reorder system based on ML predictions\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"✓ Recommendations saved: {output_path}")
    
    return {
        'at_risk': len(at_risk),
        'overstocked': len(overstocked),
        'optimal': len(optimal),
        'potential_savings': potential_savings
    }

if __name__ == '__main__':
    predict_future_demand()
    generate_reorder_recommendations()
