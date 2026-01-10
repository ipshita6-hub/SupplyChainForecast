"""
Supply Chain Data Generator
Generates synthetic supply chain dataset with realistic patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_supply_chain_data(n_rows=1000, n_products=10, output_path='data/supply_chain_data.csv'):
    """
    Generate synthetic supply chain dataset with realistic patterns
    
    Parameters:
    -----------
    n_rows : int
        Number of rows to generate
    n_products : int
        Number of unique products
    output_path : str
        Path to save the CSV file
    """
    
    np.random.seed(42)
    
    # Initialize data containers
    dates = []
    product_ids = []
    categories = []
    base_demands = []
    seasonality_factors = []
    promotions = []
    prices = []
    competitor_prices = []
    inventory_levels = []
    lead_times = []
    stock_outs = []
    actual_demands = []
    
    # Product metadata
    product_metadata = {
        i: {
            'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Home', 'Sports']),
            'base_demand': np.random.uniform(50, 500),
            'price': np.random.uniform(10, 500),
            'lead_time': np.random.randint(1, 15)
        }
        for i in range(1, n_products + 1)
    }
    
    start_date = datetime(2022, 1, 1)
    
    for idx in range(n_rows):
        # Cycle through products
        product_id = (idx % n_products) + 1
        date = start_date + timedelta(days=idx // n_products)
        
        metadata = product_metadata[product_id]
        
        # Seasonality: yearly pattern (sine wave)
        day_of_year = date.timetuple().tm_yday
        yearly_seasonality = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Weekly seasonality: higher demand on weekends
        day_of_week = date.weekday()
        weekly_seasonality = 1.2 if day_of_week >= 4 else 0.9
        
        seasonality_factor = yearly_seasonality * weekly_seasonality
        
        # Promotion: 15% chance
        promotion = 1 if np.random.random() < 0.15 else 0
        
        # Price dynamics
        base_price = metadata['price']
        price = base_price * (0.85 if promotion else 1.0)
        competitor_price = base_price * np.random.uniform(0.9, 1.1)
        
        # Demand calculation
        base_demand = metadata['base_demand']
        price_elasticity = -0.5 * (price - base_price) / base_price if base_price > 0 else 0
        promotion_boost = 0.25 if promotion else 0
        competitor_effect = -0.1 * (competitor_price - base_price) / base_price if base_price > 0 else 0
        
        actual_demand = base_demand * seasonality_factor * (1 + price_elasticity + promotion_boost + competitor_effect)
        actual_demand = max(0, actual_demand + np.random.normal(0, actual_demand * 0.1))
        
        # Inventory management
        if idx == 0 or idx % n_products == 0:
            inventory_level = actual_demand * 5  # Start with 5 days of stock
        else:
            # Inventory decreases by demand, increases by restocking
            prev_inventory = inventory_levels[-1] if inventory_levels else actual_demand * 5
            restocking = np.random.choice([0, actual_demand * 3], p=[0.7, 0.3])
            inventory_level = max(0, prev_inventory - actual_demand + restocking)
        
        # Stock out indicator
        stock_out = 1 if inventory_level < actual_demand else 0
        
        # Append data
        dates.append(date)
        product_ids.append(product_id)
        categories.append(metadata['category'])
        base_demands.append(base_demand)
        seasonality_factors.append(seasonality_factor)
        promotions.append(promotion)
        prices.append(price)
        competitor_prices.append(competitor_price)
        inventory_levels.append(inventory_level)
        lead_times.append(metadata['lead_time'])
        stock_outs.append(stock_out)
        actual_demands.append(actual_demand)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'product_id': product_ids,
        'category': categories,
        'base_demand': base_demands,
        'seasonality_factor': seasonality_factors,
        'promotion': promotions,
        'price': prices,
        'competitor_price': competitor_prices,
        'inventory_level': inventory_levels,
        'lead_time': lead_times,
        'stock_out': stock_outs,
        'actual_demand': actual_demands
    })
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✓ Dataset generated: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Products: {df['product_id'].nunique()}")
    
    return df

if __name__ == '__main__':
    generate_supply_chain_data()
