# Uk-weather-flight-delays-analysis


# Reproducible pipeline analyzing Largest 10 UK airports (2010-2025).

## Features
- Monthly weather trends (daily avgs, scatter+line plots)
- ML model: Predict % flights delayed >15min (Ridge regression)
- Make all`: Generates figures/models/predictions


# UK Airport Delays: Weather ML Pipeline (2022-2024)

**Reproducible Ridge regression predicting median flight delays from Weather Stress Index + airport/airline/month.**

Data: CAA "Main (2022-2024) - Main.csv" | Train ≤2023, Test=2024 | Metrics: RMSE/MAE/R² + time series plots.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org) [![scikit-learn](https://img.shields.io/badge/scikit-learn-1.5-yellow)](https://scikit-learn.org)

## 📈 Model Summary

**Key Features**: Weather_Stress_Index (sum of 9 weather risks), reporting_airport, airline_name, month.  
**Outputs**: Coef table, 2024 actual-vs-pred time series plots.

## 🚀 Reproduce in 90 Seconds

### 1. Clone & Setup
```bash
git clone https://github.com/YOURUSERNAME/uk-weather-flight-delays.git
cd uk-weather-flight-delays
pip install -r requirements.txt

data/raw/
└── Main (2022-2024) - Main.csv   # ← Place CSV here

## Prequisites

make run

pandas
numpy
scikit-learn
matplotlib
joblib

install:
	pip install -r requirements.txt

run:
	python src/delay_prediction.py
