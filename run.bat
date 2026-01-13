@echo off
REM Supply Chain Forecasting System - Windows Batch Script

echo.
echo ========================================
echo Supply Chain Forecasting System
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python found. Checking dependencies...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install/upgrade dependencies
echo Installing dependencies...
pip install -q -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo Running Supply Chain Forecasting Pipeline
echo ========================================
echo.

REM Run the main pipeline
python main.py

if errorlevel 1 (
    echo.
    echo ERROR: Pipeline execution failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Pipeline completed successfully!
echo ========================================
echo.
echo Generated files:
echo   - Dataset: data/supply_chain_data.csv
echo   - Models: models/demand_model.pkl, models/reorder_model.pkl
echo   - Report: outputs/recommendations.txt
echo   - Plots: outputs/plots/
echo.
pause
