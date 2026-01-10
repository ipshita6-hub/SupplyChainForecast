# Supply Chain Demand Forecasting

A simple, complete machine learning system for supply chain demand forecasting and inventory optimization.

## 🚀 Quick Start

Run the complete system:
```bash
python3 main.py
```

Or run a quick demo:
```bash
python3 demo.py
```

## 📋 What It Does

- **Creates realistic sample data** with seasonal patterns
- **Predicts demand** using Linear Regression
- **Recommends reorders** using Logistic Regression & Decision Trees
- **Generates visualizations** showing trends and model performance
- **Provides actionable recommendations** for inventory management

## 📊 Features

✅ **Demand Forecasting**: Predicts future product demand  
✅ **Reorder Decisions**: Smart inventory reorder recommendations  
✅ **Visualizations**: Charts showing demand trends and model performance  
✅ **Sample Data**: Generates realistic supply chain data for testing  
✅ **Complete Pipeline**: End-to-end ML workflow  

## 📁 Project Structure

```
SupplyChain-Demand-Forecasting/
├── main.py                    # Complete forecasting system
├── demo.py                    # Quick demonstration
├── data/
│   ├── raw/                   # Raw data files
│   └── processed/             # Processed data
├── src/                       # Individual modules (optional)
├── reports/                   # Generated reports and charts
├── notebooks/                 # Jupyter notebooks
└── requirements.txt           # Dependencies
```

## 🎯 Results

The system provides:
- **Demand predictions** with MSE around 12-15
- **Reorder accuracy** of 90%+ 
- **Visual charts** saved to `reports/figures/`
- **CSV recommendations** saved to `reports/`

## 📈 Sample Output

```
=== INVENTORY RECOMMENDATIONS ===
Total items analyzed: 50
Items needing reorder: 11

TOP PRIORITY REORDERS:
• PROD_E: Current=17, Predicted Demand=23, Probability=0.908
• PROD_A: Current=16, Predicted Demand=22, Probability=0.904
...
```

## 🛠 Requirements

- Python 3.7+
- pandas, numpy, scikit-learn, matplotlib, seaborn

Install with:
```bash
pip install -r requirements.txt
```

## 🎓 Educational Use

This system demonstrates:
- Machine learning for business applications
- Time series feature engineering
- Model evaluation and comparison
- Inventory optimization techniques
- Data visualization best practices

Perfect for learning supply chain analytics and ML!

### Expected Data Format

The system expects CSV files with the following columns:
- `date`: Date of the record
- `product_id`: Unique product identifier
- `demand` or `sales`: Historical demand/sales quantity
- `inventory` or `stock`: Current inventory levels (optional)
- `price`: Product price (optional)

## Requirements

- Python 3.7+
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- joblib >= 1.1.0

## Configuration

The system can be configured through `supply_chain_forecasting/config.py`:
- Data directories and file paths
- Model parameters and random seeds
- Preprocessing strategies
- Visualization settings
- Reorder thresholds and business rules

## Educational Purpose

This system is designed for academic and educational use, focusing on:
- Clear, modular code structure
- Comprehensive documentation
- Step-by-step machine learning pipeline
- Interpretable models and results
- Best practices in data science and ML engineering

## License

This project is intended for educational purposes.
