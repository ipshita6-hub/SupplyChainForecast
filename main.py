"""
Supply Chain Demand Forecasting & Inventory Optimization System
Main Pipeline - Orchestrates the entire ML workflow
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import generate_supply_chain_data
from train_demand_model import train_demand_model
from train_reorder_model import train_reorder_model
from predict import predict_future_demand, generate_reorder_recommendations
from visualize import generate_all_visualizations
from advanced_analytics import generate_advanced_report
from model_evaluation import generate_model_comparison_report

def main():
    """
    Execute the complete supply chain forecasting pipeline
    """
    
    print("\n" + "=" * 70)
    print("SUPPLY CHAIN DEMAND FORECASTING & INVENTORY OPTIMIZATION SYSTEM")
    print("=" * 70 + "\n")
    
    # Step 1: Generate Data
    print("STEP 1: GENERATING SYNTHETIC DATA")
    print("-" * 70)
    df = generate_supply_chain_data(n_rows=1000, n_products=10, 
                                   output_path='data/supply_chain_data.csv')
    print()
    
    # Step 2: Train Demand Model
    print("STEP 2: TRAINING DEMAND FORECASTING MODEL")
    print("-" * 70)
    demand_metrics, demand_model, scaler, X_test, y_test, y_pred = train_demand_model(
        data_path='data/supply_chain_data.csv',
        model_path='models/demand_model.pkl'
    )
    print()
    
    # Step 3: Train Reorder Model
    print("STEP 3: TRAINING REORDER DECISION MODEL")
    print("-" * 70)
    reorder_results = train_reorder_model(
        data_path='data/supply_chain_data.csv',
        demand_model_path='models/demand_model.pkl',
        model_path='models/reorder_model.pkl'
    )
    print()
    
    # Step 4: Generate Predictions
    print("STEP 4: GENERATING FUTURE PREDICTIONS")
    print("-" * 70)
    future_demand = predict_future_demand(
        days=30,
        data_path='data/supply_chain_data.csv',
        demand_model_path='models/demand_model.pkl'
    )
    print()
    
    # Step 5: Generate Recommendations
    print("STEP 5: GENERATING BUSINESS RECOMMENDATIONS")
    print("-" * 70)
    recommendations = generate_reorder_recommendations(
        data_path='data/supply_chain_data.csv',
        reorder_model_path='models/reorder_model.pkl',
        demand_model_path='models/demand_model.pkl',
        output_path='outputs/recommendations.txt'
    )
    print()
    
    # Step 6: Generate Visualizations
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("-" * 70)
    generate_all_visualizations(
        data_path='data/supply_chain_data.csv',
        demand_model_path='models/demand_model.pkl',
        reorder_model_path='models/reorder_model.pkl',
        output_dir='outputs/plots'
    )
    print()
    
    # Step 7: Advanced Analytics
    print("STEP 7: ADVANCED ANALYTICS")
    print("-" * 70)
    generate_advanced_report(
        data_path='data/supply_chain_data.csv',
        output_path='outputs/advanced_analytics_report.txt'
    )
    print()
    
    # Step 8: Model Evaluation
    print("STEP 8: MODEL EVALUATION & COMPARISON")
    print("-" * 70)
    generate_model_comparison_report(
        data_path='data/supply_chain_data.csv',
        demand_model_path='models/demand_model.pkl',
        reorder_model_path='models/reorder_model.pkl',
        output_path='outputs/model_comparison.txt'
    )
    print()
    
    # Summary
    print("=" * 70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nGenerated Artifacts:")
    print(f"  ✓ Dataset: data/supply_chain_data.csv ({df.shape[0]} rows)")
    print(f"  ✓ Demand Model: models/demand_model.pkl (R² = {demand_metrics['r2_test']:.4f})")
    print(f"  ✓ Reorder Model: models/reorder_model.pkl (Accuracy = {reorder_results['dt_metrics']['accuracy']:.4f})")
    print(f"  ✓ Future Predictions: {len(future_demand)} demand forecasts")
    print(f"  ✓ Recommendations: outputs/recommendations.txt")
    print(f"  ✓ Visualizations: 5 plots in outputs/plots/")
    print("\nBusiness Impact:")
    print(f"  • Products at Risk: {recommendations['at_risk']}")
    print(f"  • Overstocked Products: {recommendations['overstocked']}")
    print(f"  • Optimal Inventory: {recommendations['optimal']}")
    print(f"  • Potential Annual Savings: ${recommendations['potential_savings']:,.2f}")
    print("\n" + "=" * 70 + "\n")

if __name__ == '__main__':
    main()
