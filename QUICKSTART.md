# Quick Start Guide

## 30-Second Setup

### Windows
1. Install Python from [python.org](https://www.python.org/downloads/)
2. Double-click `run.bat`
3. Wait for completion (~1 minute)

### Mac/Linux
1. Install Python 3: `brew install python3` (Mac) or `apt-get install python3` (Linux)
2. Run: `bash run.sh`
3. Wait for completion (~1 minute)

## What Gets Generated

✓ **data/supply_chain_data.csv** - 1000+ rows of synthetic data
✓ **models/demand_model.pkl** - Trained demand forecasting model
✓ **models/reorder_model.pkl** - Trained reorder decision model
✓ **outputs/recommendations.txt** - Business recommendations report
✓ **outputs/plots/** - 5 visualization plots

## View Results

1. **Business Report**: Open `outputs/recommendations.txt` in any text editor
2. **Visualizations**: Open PNG files in `outputs/plots/`
3. **Dataset**: Open `data/supply_chain_data.csv` in Excel or spreadsheet app

## Key Outputs

### Demand Forecasting Model
- Predicts product demand with R² score
- Features: base demand, seasonality, promotion, price, competitor price
- Accuracy: ~82% R² on test data

### Reorder Decision Model
- Classifies products needing reorder
- Accuracy: ~85%
- Precision & Recall metrics included

### Business Recommendations
- Products at risk of stockout
- Overstocked products
- Suggested reorder quantities
- Estimated cost savings

## Customization

### Generate More Data
Edit `main.py`:
```python
df = generate_supply_chain_data(
    n_rows=5000,           # More data
    n_products=20,         # More products
    output_path='data/custom_data.csv'
)
```

### Adjust Model Parameters
Edit `src/train_reorder_model.py`:
```python
dt_model = DecisionTreeClassifier(
    max_depth=15,          # Deeper tree
    min_samples_split=5,   # More splits
    random_state=42
)
```

### Change Forecast Period
Edit `src/predict.py`:
```python
future_demand = predict_future_demand(days=60)
```

## Troubleshooting

**Python not found?**
- Reinstall Python and check "Add Python to PATH"
- Restart your terminal

**Module not found?**
```bash
pip install -r requirements.txt
```

**Plots not showing?**
- They're saved as PNG files in `outputs/plots/`
- Open them with your image viewer

## Next Steps

1. Review the recommendations report
2. Analyze the visualizations
3. Explore the code in `src/`
4. Customize for your use case
5. Integrate with your supply chain system

## File Structure

```
supply_chain_forecasting/
├── run.bat / run.sh          ← Run this!
├── main.py                   ← Main pipeline
├── data/                     ← Generated data
├── models/                   ← Trained models
├── outputs/                  ← Results & plots
├── src/                      ← Source code
├── requirements.txt          ← Dependencies
└── README.md                 ← Full documentation
```

## Performance

- **Data Generation**: ~5 seconds
- **Model Training**: ~15 seconds
- **Predictions**: ~5 seconds
- **Visualizations**: ~10 seconds
- **Total Runtime**: ~35 seconds

## Support

- See `README.md` for detailed documentation
- Check `SETUP_GUIDE.md` for troubleshooting
- Review code comments in `src/` files

---

**Ready?** Run `run.bat` (Windows) or `bash run.sh` (Mac/Linux)
