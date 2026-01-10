"""
Supply Chain Forecasting API
Provides programmatic interface for predictions and recommendations
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class SupplyChainForecaster:
    """
    Main API class for supply chain forecasting and optimization
    """
    
    def __init__(self, demand_model_path='models/demand_model.pkl',
                 reorder_model_path='models/reorder_model.pkl'):
        """
        Initialize forecaster with trained models
        
        Parameters:
        -----------
        demand_model_path : str
            Path to trained demand model
        reorder_model_path : str
            Path to trained reorder model
        """
        
        self.demand_model = joblib.load(demand_model_path)
        self.demand_scaler = joblib.load(demand_model_path.replace('.pkl', '_scaler.pkl'))
        self.reorder_model = joblib.load(reorder_model_path)
        self.reorder_scaler = joblib.load(reorder_model_path.replace('.pkl', '_scaler.pkl'))
        
        self.feature_cols = ['base_demand', 'seasonality_factor', 'promotion', 'price', 'competitor_price']
        self.reorder_features = ['inventory_level', 'predicted_demand', 'lead_time', 'stock_out']
    
    def predict_demand(self, base_demand: float, seasonality_factor: float,
                      promotion: int, price: float, competitor_price: float) -> float:
        """
        Predict demand for a product
        
        Parameters:
        -----------
        base_demand : float
            Baseline demand
        seasonality_factor : float
            Seasonal multiplier
        promotion : int
            Promotion flag (0 or 1)
        price : float
            Product price
        competitor_price : float
            Competitor price
        
        Returns:
        --------
        float : Predicted demand
        """
        
        features = np.array([[base_demand, seasonality_factor, promotion, price, competitor_price]])
        features_scaled = self.demand_scaler.transform(features)
        predicted_demand = self.demand_model.predict(features_scaled)[0]
        
        return max(0, predicted_demand)
    
    def predict_reorder_need(self, inventory_level: float, predicted_demand: float,
                            lead_time: int, stock_out: int) -> Tuple[int, float]:
        """
        Predict if product needs reorder
        
        Parameters:
        -----------
        inventory_level : float
            Current inventory
        predicted_demand : float
            Predicted demand
        lead_time : int
            Supplier lead time (days)
        stock_out : int
            Historical stockout indicator
        
        Returns:
        --------
        tuple : (reorder_needed, reorder_probability)
        """
        
        features = np.array([[inventory_level, predicted_demand, lead_time, stock_out]])
        features_scaled = self.reorder_scaler.transform(features)
        
        reorder_needed = self.reorder_model.predict(features_scaled)[0]
        reorder_probability = self.reorder_model.predict_proba(features_scaled)[0][1]
        
        return int(reorder_needed), float(reorder_probability)
    
    def forecast_demand_period(self, product_data: Dict, days: int = 30) -> List[Dict]:
        """
        Forecast demand for a period
        
        Parameters:
        -----------
        product_data : dict
            Product information (base_demand, price, competitor_price, etc.)
        days : int
            Number of days to forecast
        
        Returns:
        --------
        list : Daily demand forecasts
        """
        
        forecasts = []
        start_date = datetime.now()
        
        for day in range(days):
            forecast_date = start_date + timedelta(days=day)
            
            # Calculate seasonality
            day_of_year = forecast_date.timetuple().tm_yday
            yearly_seasonality = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
            
            day_of_week = forecast_date.weekday()
            weekly_seasonality = 1.2 if day_of_week >= 4 else 0.9
            
            seasonality_factor = yearly_seasonality * weekly_seasonality
            
            # Predict demand
            predicted_demand = self.predict_demand(
                base_demand=product_data.get('base_demand', 100),
                seasonality_factor=seasonality_factor,
                promotion=product_data.get('promotion', 0),
                price=product_data.get('price', 50),
                competitor_price=product_data.get('competitor_price', 50)
            )
            
            forecasts.append({
                'date': forecast_date.date(),
                'predicted_demand': round(predicted_demand, 2),
                'seasonality_factor': round(seasonality_factor, 3),
                'day_of_week': forecast_date.strftime('%A')
            })
        
        return forecasts
    
    def get_reorder_recommendation(self, product_id: int, inventory_level: float,
                                  predicted_demand: float, lead_time: int,
                                  stock_out: int = 0) -> Dict:
        """
        Get reorder recommendation for a product
        
        Parameters:
        -----------
        product_id : int
            Product identifier
        inventory_level : float
            Current inventory
        predicted_demand : float
            Predicted daily demand
        lead_time : int
            Supplier lead time (days)
        stock_out : int
            Historical stockout indicator
        
        Returns:
        --------
        dict : Reorder recommendation
        """
        
        # Predict reorder need
        needs_reorder, probability = self.predict_reorder_need(
            inventory_level, predicted_demand, lead_time, stock_out
        )
        
        # Calculate reorder quantity
        safety_stock = predicted_demand * lead_time * 1.5
        reorder_qty = max(0, safety_stock - inventory_level)
        
        # Determine urgency
        days_of_stock = inventory_level / predicted_demand if predicted_demand > 0 else float('inf')
        
        if days_of_stock < lead_time:
            urgency = 'CRITICAL'
        elif days_of_stock < lead_time * 1.5:
            urgency = 'HIGH'
        elif days_of_stock < lead_time * 2:
            urgency = 'MEDIUM'
        else:
            urgency = 'LOW'
        
        return {
            'product_id': product_id,
            'needs_reorder': bool(needs_reorder),
            'reorder_probability': round(probability, 3),
            'current_inventory': round(inventory_level, 2),
            'predicted_daily_demand': round(predicted_demand, 2),
            'days_of_stock': round(days_of_stock, 2),
            'lead_time_days': lead_time,
            'safety_stock_target': round(safety_stock, 2),
            'recommended_reorder_qty': round(reorder_qty, 2),
            'urgency_level': urgency,
            'recommendation': self._get_recommendation_text(needs_reorder, urgency, reorder_qty)
        }
    
    def _get_recommendation_text(self, needs_reorder: int, urgency: str, reorder_qty: float) -> str:
        """Generate recommendation text"""
        
        if not needs_reorder:
            return "No reorder needed at this time. Monitor inventory levels."
        
        if urgency == 'CRITICAL':
            return f"URGENT: Reorder {reorder_qty:.0f} units immediately to avoid stockout!"
        elif urgency == 'HIGH':
            return f"Reorder {reorder_qty:.0f} units soon to maintain service level."
        elif urgency == 'MEDIUM':
            return f"Plan to reorder {reorder_qty:.0f} units in the next few days."
        else:
            return f"Consider reordering {reorder_qty:.0f} units when convenient."
    
    def batch_reorder_recommendations(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get reorder recommendations for multiple products
        
        Parameters:
        -----------
        products_df : pd.DataFrame
            DataFrame with columns: product_id, inventory_level, predicted_demand, 
                                   lead_time, stock_out
        
        Returns:
        --------
        pd.DataFrame : Recommendations for all products
        """
        
        recommendations = []
        
        for _, row in products_df.iterrows():
            rec = self.get_reorder_recommendation(
                product_id=int(row['product_id']),
                inventory_level=float(row['inventory_level']),
                predicted_demand=float(row['predicted_demand']),
                lead_time=int(row['lead_time']),
                stock_out=int(row.get('stock_out', 0))
            )
            recommendations.append(rec)
        
        return pd.DataFrame(recommendations)
    
    def get_inventory_summary(self, products_df: pd.DataFrame) -> Dict:
        """
        Get inventory summary statistics
        
        Parameters:
        -----------
        products_df : pd.DataFrame
            DataFrame with product inventory data
        
        Returns:
        --------
        dict : Summary statistics
        """
        
        total_inventory = products_df['inventory_level'].sum()
        total_demand = products_df['predicted_demand'].sum()
        
        at_risk = len(products_df[products_df['inventory_level'] < 
                                  products_df['predicted_demand'] * products_df['lead_time']])
        
        return {
            'total_products': len(products_df),
            'total_inventory_value': round(total_inventory, 2),
            'total_daily_demand': round(total_demand, 2),
            'days_of_stock': round(total_inventory / total_demand, 2) if total_demand > 0 else 0,
            'products_at_risk': at_risk,
            'products_optimal': len(products_df) - at_risk,
            'avg_inventory_per_product': round(products_df['inventory_level'].mean(), 2),
            'avg_demand_per_product': round(products_df['predicted_demand'].mean(), 2)
        }

# Example usage
if __name__ == '__main__':
    # Initialize forecaster
    forecaster = SupplyChainForecaster()
    
    # Example 1: Predict demand
    demand = forecaster.predict_demand(
        base_demand=100,
        seasonality_factor=1.1,
        promotion=1,
        price=50,
        competitor_price=55
    )
    print(f"Predicted Demand: {demand:.2f} units")
    
    # Example 2: Get reorder recommendation
    rec = forecaster.get_reorder_recommendation(
        product_id=1,
        inventory_level=150,
        predicted_demand=50,
        lead_time=7,
        stock_out=0
    )
    print(f"\nReorder Recommendation:")
    for key, value in rec.items():
        print(f"  {key}: {value}")
    
    # Example 3: Forecast demand for period
    product_data = {
        'base_demand': 100,
        'price': 50,
        'competitor_price': 55,
        'promotion': 0
    }
    forecasts = forecaster.forecast_demand_period(product_data, days=7)
    print(f"\n7-Day Demand Forecast:")
    for forecast in forecasts:
        print(f"  {forecast['date']}: {forecast['predicted_demand']:.2f} units")
