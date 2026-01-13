#!/bin/bash

# Supply Chain Forecasting System - Unix/Linux/Mac Script

echo ""
echo "========================================"
echo "Supply Chain Forecasting System"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3 from https://www.python.org/downloads/"
    exit 1
fi

echo "Python found: $(python3 --version)"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo ""
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo ""
echo "========================================"
echo "Running Supply Chain Forecasting Pipeline"
echo "========================================"
echo ""

# Run the main pipeline
python3 main.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Pipeline execution failed"
    exit 1
fi

echo ""
echo "========================================"
echo "Pipeline completed successfully!"
echo "========================================"
echo ""
echo "Generated files:"
echo "  - Dataset: data/supply_chain_data.csv"
echo "  - Models: models/demand_model.pkl, models/reorder_model.pkl"
echo "  - Report: outputs/recommendations.txt"
echo "  - Plots: outputs/plots/"
echo ""
