# Supply Chain Demand Forecasting

A complete machine learning system for supply chain demand forecasting and inventory optimization. This project demonstrates end-to-end ML workflows with practical business applications.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete system
python main.py
```

## 📋 What It Does

- **Demand Forecasting**: Predicts future product demand using Linear Regression
- **Reorder Optimization**: Smart inventory reorder recommendations using Logistic Regression & Decision Trees
- **Data Generation**: Creates realistic supply chain data with seasonal patterns
- **Visualizations**: Generates charts showing trends, seasonality, and model performance
- **Actionable Insights**: Provides inventory management recommendations

## 📊 Features

✅ Demand prediction with performance metrics  
✅ Intelligent reorder decision system  
✅ Comprehensive data visualizations  
✅ Realistic synthetic data generation  
✅ End-to-end ML pipeline  
✅ RESTful API for predictions  
✅ Advanced analytics and insights  

## 📁 Project Structure

```
.
├── main.py                      # Main entry point
├── config.yaml                  # Configuration settings
├── requirements.txt             # Python dependencies
├── data/
│   └── supply_chain_data.csv   # Generated dataset
├── models/
│   ├── demand_model.pkl        # Trained demand forecasting model
│   ├── demand_model_scaler.pkl # Feature scaler for demand model
│   ├── reorder_model.pkl       # Trained reorder decision model
│   └── reorder_model_scaler.pkl# Feature scaler for reorder model
├── outputs/
│   ├── plots/                  # Generated visualizations
│   └── recommendations.txt     # Inventory recommendations
└── src/
    ├── data_generator.py       # Synthetic data generation
    ├── train_demand_model.py   # Demand model training
    ├── train_reorder_model.py  # Reorder model training
    ├── predict.py              # Prediction engine
    ├── visualize.py            # Visualization utilities
    ├── advanced_analytics.py   # Advanced analysis tools
    └── api.py                  # REST API server
```

## 🎯 Model Performance

- **Demand Forecasting**: MSE ~12-15, captures seasonal trends
- **Reorder Accuracy**: 90%+ precision on inventory decisions
- **Feature Importance**: Identifies key demand drivers

## 📈 Output Examples

Generated visualizations include:
- Actual vs Predicted demand trends
- Feature importance analysis
- Seasonality patterns
- Confusion matrix for reorder decisions
- Inventory vs demand correlation

Recommendations file includes:
- Priority reorder items
- Predicted demand vs current inventory
- Reorder probability scores

## 🛠 Requirements

- Python 3.7+
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- joblib >= 1.1.0
- flask >= 2.0.0 (for API)

## 🔧 Configuration

Edit `config.yaml` to customize:
- Data generation parameters
- Model hyperparameters
- Visualization settings
- Reorder thresholds
- API settings

## 📚 Data Format

Input CSV should contain:
- `date`: Record date
- `product_id`: Product identifier
- `demand`: Historical demand quantity
- `inventory`: Current stock level
- `price`: Product price

## 🎓 Learning Outcomes

This project demonstrates:
- Time series feature engineering
- Model training and evaluation
- Hyperparameter tuning
- Data visualization best practices
- API development for ML models
- Production-ready code structure

## License

Educational use only.
