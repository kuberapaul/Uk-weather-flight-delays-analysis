# Uk-weather-flight-delays-analysis-GROUP F


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


## 🚀 v2: Time-Series Holdout (2024 Train → 2025 Test)

**New!** Linear Regression on Heathrow monthly data:
- **Train:** 2024 delays + weather (dew, precip, wind dir)
- **Test:** Unseen 2025 (R² 0.78, MAE ~2min)

**Run:**



**Needs CSVs:**
- `master sheet 24.csv` (2024 delays)
- `Copy of Master Punctuality sheet (2025) - processed - Sheet1.csv`
- `Weather Data Top 10 UK Airports 2010-2025 (processed) - Weather data.csv`

**Output:** Metrics + `2025_validation.png`

![v2 Plot](2025_validation.png)

pandas
numpy
scikit-learn
matplotlib
