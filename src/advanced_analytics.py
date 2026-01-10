"""
Advanced Analytics Module
Provides deeper insights into supply chain performance
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def calculate_abc_analysis(data_path='data/supply_chain_data.csv',
                          output_path='outputs/abc_analysis.csv'):
    """
    Perform ABC analysis on products based on demand value
    A: High value, B: Medium value, C: Low value
    
    Parameters:
    -----------
    data_path : str
        Path to supply chain dataset
    output_path : str
        Path to save ABC analysis results
    """
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate total demand value per product
    product_analysis = df.groupby('product_id').agg({
        'actual_demand': ['sum', 'mean', 'std'],
        'price': 'mean',
        'inventory_level': 'mean',
        'stock_out': 'sum',
        'category': 'first'
    }).round(2)
    
    product_analysis.columns = ['total_demand', 'avg_demand', 'demand_std', 
                                'avg_price', 'avg_inventory', 'stockout_count', 'category']
    
    # Calculate demand value (demand * price)
    product_analysis['demand_value'] = (product_analysis['total_demand'] * 
                                        product_analysis['avg_price']).round(2)
    
    # Sort by demand value
    product_analysis = product_analysis.sort_values('demand_value', ascending=False)
    product_analysis['cumulative_pct'] = (product_analysis['demand_value'].cumsum() / 
                                          product_analysis['demand_value'].sum() * 100).round(2)
    
    # Classify into ABC
    def classify_abc(cumulative_pct):
        if cumulative_pct <= 80:
            return 'A'
        elif cumulative_pct <= 95:
            return 'B'
        else:
            return 'C'
    
    product_analysis['classification'] = product_analysis['cumulative_pct'].apply(classify_abc)
    
    # Save results
    product_analysis.to_csv(output_path)
    
    print("\n" + "=" * 60)
    print("ABC ANALYSIS - PRODUCT CLASSIFICATION")
    print("=" * 60)
    print(f"\nA Products (High Value - 80% of demand):")
    a_products = product_analysis[product_analysis['classification'] == 'A']
    print(f"  Count: {len(a_products)}")
    print(f"  Total Demand Value: ${a_products['demand_value'].sum():,.2f}")
    
    print(f"\nB Products (Medium Value - 80-95% of demand):")
    b_products = product_analysis[product_analysis['classification'] == 'B']
    print(f"  Count: {len(b_products)}")
    print(f"  Total Demand Value: ${b_products['demand_value'].sum():,.2f}")
    
    print(f"\nC Products (Low Value - 95-100% of demand):")
    c_products = product_analysis[product_analysis['classification'] == 'C']
    print(f"  Count: {len(c_products)}")
    print(f"  Total Demand Value: ${c_products['demand_value'].sum():,.2f}")
    
    print(f"\n✓ ABC Analysis saved: {output_path}")
    
    return product_analysis

def calculate_inventory_metrics(data_path='data/supply_chain_data.csv'):
    """
    Calculate key inventory performance metrics
    """
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Metrics per product
    metrics = df.groupby('product_id').agg({
        'actual_demand': 'mean',
        'inventory_level': 'mean',
        'stock_out': 'sum',
        'lead_time': 'first'
    }).round(2)
    
    metrics.columns = ['avg_demand', 'avg_inventory', 'stockout_events', 'lead_time']
    
    # Calculate derived metrics
    metrics['days_of_stock'] = (metrics['avg_inventory'] / metrics['avg_demand']).round(2)
    metrics['stockout_rate'] = (metrics['stockout_events'] / len(df.groupby('product_id')) * 100).round(2)
    metrics['inventory_turnover'] = (metrics['avg_demand'] * 365 / metrics['avg_inventory']).round(2)
    
    print("\n" + "=" * 60)
    print("INVENTORY PERFORMANCE METRICS")
    print("=" * 60)
    print(f"\nAverage Days of Stock: {metrics['days_of_stock'].mean():.2f} days")
    print(f"Average Inventory Turnover: {metrics['inventory_turnover'].mean():.2f}x per year")
    print(f"Total Stockout Events: {metrics['stockout_events'].sum():.0f}")
    print(f"Average Stockout Rate: {metrics['stockout_rate'].mean():.2f}%")
    
    return metrics

def calculate_demand_variability(data_path='data/supply_chain_data.csv'):
    """
    Analyze demand variability and forecast accuracy potential
    """
    
    df = pd.read_csv(data_path)
    
    # Calculate coefficient of variation (CV) for each product
    variability = df.groupby('product_id')['actual_demand'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).round(2)
    
    variability['cv'] = (variability['std'] / variability['mean']).round(3)
    variability['range'] = (variability['max'] - variability['min']).round(2)
    
    # Classify demand patterns
    def classify_demand(cv):
        if cv < 0.5:
            return 'Stable'
        elif cv < 1.0:
            return 'Moderate'
        else:
            return 'Volatile'
    
    variability['pattern'] = variability['cv'].apply(classify_demand)
    
    print("\n" + "=" * 60)
    print("DEMAND VARIABILITY ANALYSIS")
    print("=" * 60)
    print(f"\nStable Demand (CV < 0.5): {len(variability[variability['pattern'] == 'Stable'])} products")
    print(f"Moderate Demand (0.5 ≤ CV < 1.0): {len(variability[variability['pattern'] == 'Moderate'])} products")
    print(f"Volatile Demand (CV ≥ 1.0): {len(variability[variability['pattern'] == 'Volatile'])} products")
    print(f"\nAverage Coefficient of Variation: {variability['cv'].mean():.3f}")
    
    return variability

def calculate_safety_stock_recommendations(data_path='data/supply_chain_data.csv',
                                          service_level=0.95):
    """
    Calculate recommended safety stock levels based on service level
    
    Parameters:
    -----------
    data_path : str
        Path to supply chain dataset
    service_level : float
        Desired service level (0-1), default 95%
    """
    
    df = pd.read_csv(data_path)
    
    # Z-score for service level (95% = 1.645)
    z_scores = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
    z = z_scores.get(service_level, 1.645)
    
    recommendations = []
    
    for product_id in df['product_id'].unique():
        product_df = df[df['product_id'] == product_id]
        
        demand_mean = product_df['actual_demand'].mean()
        demand_std = product_df['actual_demand'].std()
        lead_time = product_df['lead_time'].iloc[0]
        
        # Safety stock = Z * std_dev * sqrt(lead_time)
        safety_stock = z * demand_std * np.sqrt(lead_time)
        
        # Reorder point = (demand_mean * lead_time) + safety_stock
        reorder_point = (demand_mean * lead_time) + safety_stock
        
        # Economic order quantity (simplified)
        eoq = np.sqrt(2 * demand_mean * 365 * 5 / 0.05)  # 5 = order cost, 0.05 = holding cost
        
        recommendations.append({
            'product_id': product_id,
            'avg_demand': round(demand_mean, 2),
            'demand_std': round(demand_std, 2),
            'lead_time': lead_time,
            'safety_stock': round(safety_stock, 2),
            'reorder_point': round(reorder_point, 2),
            'economic_order_qty': round(eoq, 2)
        })
    
    rec_df = pd.DataFrame(recommendations)
    
    print("\n" + "=" * 60)
    print(f"SAFETY STOCK RECOMMENDATIONS (Service Level: {service_level*100:.0f}%)")
    print("=" * 60)
    print(f"\nAverage Safety Stock: {rec_df['safety_stock'].mean():.2f} units")
    print(f"Average Reorder Point: {rec_df['reorder_point'].mean():.2f} units")
    print(f"Average Economic Order Qty: {rec_df['economic_order_qty'].mean():.2f} units")
    
    return rec_df

def generate_advanced_report(data_path='data/supply_chain_data.csv',
                            output_path='outputs/advanced_analytics_report.txt'):
    """
    Generate comprehensive advanced analytics report
    """
    
    import os
    
    # Run all analyses
    abc_analysis = calculate_abc_analysis(data_path)
    inventory_metrics = calculate_inventory_metrics(data_path)
    demand_variability = calculate_demand_variability(data_path)
    safety_stock = calculate_safety_stock_recommendations(data_path)
    
    # Generate report
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ADVANCED SUPPLY CHAIN ANALYTICS REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. ABC ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write("Products classified by demand value contribution:\n\n")
        for classification in ['A', 'B', 'C']:
            products = abc_analysis[abc_analysis['classification'] == classification]
            f.write(f"{classification} Products: {len(products)} items\n")
            f.write(f"  Total Demand Value: ${products['demand_value'].sum():,.2f}\n")
            f.write(f"  Avg Demand: {products['avg_demand'].mean():.2f} units\n")
            f.write(f"  Avg Inventory: {products['avg_inventory'].mean():.2f} units\n\n")
        
        f.write("\n2. INVENTORY PERFORMANCE METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Average Days of Stock: {inventory_metrics['days_of_stock'].mean():.2f} days\n")
        f.write(f"Average Inventory Turnover: {inventory_metrics['inventory_turnover'].mean():.2f}x/year\n")
        f.write(f"Total Stockout Events: {inventory_metrics['stockout_events'].sum():.0f}\n")
        f.write(f"Average Stockout Rate: {inventory_metrics['stockout_rate'].mean():.2f}%\n")
        
        f.write("\n3. DEMAND VARIABILITY ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Stable Demand Products: {len(demand_variability[demand_variability['pattern'] == 'Stable'])}\n")
        f.write(f"Moderate Demand Products: {len(demand_variability[demand_variability['pattern'] == 'Moderate'])}\n")
        f.write(f"Volatile Demand Products: {len(demand_variability[demand_variability['pattern'] == 'Volatile'])}\n")
        f.write(f"Average Coefficient of Variation: {demand_variability['cv'].mean():.3f}\n")
        
        f.write("\n4. SAFETY STOCK RECOMMENDATIONS (95% Service Level)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Average Safety Stock: {safety_stock['safety_stock'].mean():.2f} units\n")
        f.write(f"Average Reorder Point: {safety_stock['reorder_point'].mean():.2f} units\n")
        f.write(f"Average Economic Order Qty: {safety_stock['economic_order_qty'].mean():.2f} units\n")
        
        f.write("\n5. KEY INSIGHTS & RECOMMENDATIONS\n")
        f.write("-" * 70 + "\n")
        
        # Insights
        high_turnover = inventory_metrics[inventory_metrics['inventory_turnover'] > 10]
        low_turnover = inventory_metrics[inventory_metrics['inventory_turnover'] < 2]
        
        f.write(f"\nHigh Turnover Products ({len(high_turnover)}): Fast-moving items\n")
        f.write(f"  Recommendation: Increase order frequency, reduce batch size\n")
        
        f.write(f"\nLow Turnover Products ({len(low_turnover)}): Slow-moving items\n")
        f.write(f"  Recommendation: Consider promotions or discontinuation\n")
        
        volatile = demand_variability[demand_variability['pattern'] == 'Volatile']
        f.write(f"\nVolatile Demand Products ({len(volatile)}): Unpredictable demand\n")
        f.write(f"  Recommendation: Increase safety stock, improve forecasting\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"\n✓ Advanced analytics report saved: {output_path}")

if __name__ == '__main__':
    generate_advanced_report()
